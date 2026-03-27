#!/usr/bin/env python3
"""Restore functionality for PostgreSQL database backup."""

from __future__ import annotations

import subprocess
import sys
import re
from pathlib import Path

from .config import BACKUP_DIR, DATABASE_NAME, DATABASE_USER, PG_RESTORE_CMD, DROPDB_CMD, CREATEDB_CMD
from .core import get_backup_versions, run_command


def restore_database(
    version: str,
    drop_existing: bool = True,
) -> None:
    """Restore a PostgreSQL database from a backup."""
    backup_dir = None
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
    
    if drop_existing:
        print(f"Dropping existing database '{DATABASE_NAME}'...")
        run_command(DROPDB_CMD)
    
    print(f"Creating database '{DATABASE_NAME}'...")
    run_command(CREATEDB_CMD)
    
    print(f"Restoring from {backup_dir}/*.dump.part-*...")
    
    # Concatenate all parts and pipe to pg_restore inside Docker
    cat_cmd = ["cat", str(backup_dir) + "/*.dump.part-*"]
    pg_restore_cmd = PG_RESTORE_CMD.copy()

    combined_cmd = f"{' '.join(cat_cmd)} | {' '.join(pg_restore_cmd)}"
    run_command(combined_cmd, shell=True)
    print(f"Restore completed successfully!")
    print(f"Database '{DATABASE_NAME}' restored with version {version}")


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
        
        backup_files = list(version_dir.glob("*.dump.part-*"))
        if backup_files:
            total_size = sum(f.stat().st_size for f in backup_files)
            print(f"    Files: {len(backup_files)}")
            print(f"    Size: {total_size / (1024**3):.2f} GB")