#!/usr/bin/env python3
"""Restore functionality for PostgreSQL database backup."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

from .config import (
    BACKUP_DIR,
    DATABASE_NAME,
    DATABASE_USER,
    DATABASE_CONTAINER_NAME,
    MAINTENANCE_DATABASE_NAME,
    PG_RESTORE_CMD,
    BACKUP_FILE_GLOB,
)
from .core import get_backup_versions, run_command, get_database_version


def run_maintenance_sql(sql: str) -> None:
    """Run SQL against the maintenance database."""
    cmd = [
        "docker", "exec", "-i", DATABASE_CONTAINER_NAME,
        "psql", "-U", DATABASE_USER,
        "-d", MAINTENANCE_DATABASE_NAME,
        "-v", "ON_ERROR_STOP=1",
        "-c", sql,
    ]
    run_command(cmd)


def recreate_database() -> None:
    """Terminate connections, drop the target DB, and recreate it."""
    terminate_sql = f"""
    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE datname = '{DATABASE_NAME}'
      AND pid <> pg_backend_pid();
    """
    run_maintenance_sql(terminate_sql)

    drop_sql = f"DROP DATABASE IF EXISTS {DATABASE_NAME};"
    run_maintenance_sql(drop_sql)

    create_sql = f"CREATE DATABASE {DATABASE_NAME};"
    run_maintenance_sql(create_sql)


def restore_database(
    version: str,
    drop_existing: bool = True,
) -> None:
    """Restore a PostgreSQL database from a backup."""
    backup_dir: Path | None = None
    for item in BACKUP_DIR.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            match = re.match(r"v(\d+\.\d+)", item.name)
            if match and match.group(1) == version:
                backup_dir = item
                break

    if not backup_dir:
        print(f"Error: Backup version '{version}' not found")
        print(f"Available versions: {[v.name for v in get_backup_versions()]}")
        sys.exit(1)

    backup_files = sorted(backup_dir.glob(BACKUP_FILE_GLOB))
    if not backup_files:
        print(f"Error: No backup files found in {backup_dir}")
        sys.exit(1)

    print(f"Restoring database '{DATABASE_NAME}' from {backup_dir} ...")

    if drop_existing:
        recreate_database()

    cat_cmd = f"cat {shlex.quote(str(backup_dir))}/{BACKUP_FILE_GLOB}"
    pg_restore_cmd = " ".join(shlex.quote(part) for part in PG_RESTORE_CMD)

    combined_cmd = f"{cat_cmd} | {pg_restore_cmd}"
    run_command(combined_cmd, shell=True)

    db_version = get_database_version()

    print("Restore completed successfully!")
    print(f"Database '{DATABASE_NAME}' restored with backup version {version}")
    print(f"Database-internal version is now {db_version}")


def list_versions() -> None:
    """List all available backup versions."""
    versions = get_backup_versions()

    if not versions:
        print("No backup versions found")
        return

    print("Available backup versions:")
    print("-" * 50)
    for version_dir in versions:
        version_name = version_dir.name[1:]
        print(f"  {version_name}")

        backup_files = list(version_dir.glob(BACKUP_FILE_GLOB))
        if backup_files:
            total_size = sum(f.stat().st_size for f in backup_files)
            print(f"    Files: {len(backup_files)}")
            print(f"    Size: {total_size / (1024**3):.2f} GB")