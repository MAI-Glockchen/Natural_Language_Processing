#!/usr/bin/env python3
"""Core backup and restore functions for PostgreSQL database."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from .config import (
    BACKUP_DIR,
    DATABASE_NAME,
    DATABASE_USER,
    DATABASE_CONTAINER_NAME,
    PG_DUMP_CMD,
    DEFAULT_VERSION,
    INITIAL_DATABASE_VERSION,
    VERSION_PATTERN,
    PART_SIZE_MB,
    BACKUP_FILE_PREFIX,
    BACKUP_FILE_GLOB,
    VERSION_TABLE_NAME,
)


def run_command(cmd: list[str] | str, check: bool = True, shell: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    return subprocess.run(cmd, check=check, shell=shell)


def run_sql(sql: str, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run SQL against the configured PostgreSQL database inside Docker."""
    cmd = [
        "docker", "exec", "-i", DATABASE_CONTAINER_NAME,
        "psql", "-U", DATABASE_USER,
        "-d", DATABASE_NAME,
        "-v", "ON_ERROR_STOP=1",
    ]

    if capture_output:
        cmd.extend(["-At", "-c", sql])
    else:
        cmd.extend(["-c", sql])

    return subprocess.run(cmd, check=True, text=True, capture_output=capture_output)


def ensure_version_table() -> None:
    """Ensure the version metadata table exists and contains exactly one row."""
    sql = f"""
    CREATE TABLE IF NOT EXISTS public.{VERSION_TABLE_NAME} (
        id integer PRIMARY KEY,
        version varchar(16) NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now()
    );

    INSERT INTO public.{VERSION_TABLE_NAME} (id, version)
    VALUES (1, '{INITIAL_DATABASE_VERSION}')
    ON CONFLICT (id) DO NOTHING;
    """
    run_sql(sql)


def get_database_version() -> str:
    """Read the current database version stored inside the database."""
    ensure_version_table()

    sql = f"SELECT version FROM public.{VERSION_TABLE_NAME} WHERE id = 1;"
    result = run_sql(sql, capture_output=True)
    version = result.stdout.strip()

    if not version:
        raise RuntimeError("Could not read database version from backup_metadata table.")

    return version


def set_database_version(version: str) -> None:
    """Update the current database version stored inside the database."""
    sql = f"""
    UPDATE public.{VERSION_TABLE_NAME}
    SET version = '{version}',
        updated_at = now()
    WHERE id = 1;
    """
    run_sql(sql)


def increment_version(version: str) -> str:
    """Increase a version of the form major.minor by 0.01."""
    parts = version.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid version format: {version}")

    major = int(parts[0])
    minor = int(parts[1]) + 1

    if minor >= 100:
        major += 1
        minor = 0

    return f"{major}.{minor:02d}"


def get_latest_backup() -> Path | None:
    """Get the path to the latest backup version directory."""
    versions = get_backup_versions()
    if not versions:
        return None
    return versions[-1]


def get_backup_versions() -> list[Path]:
    """Get all backup version directories sorted by version number."""
    if not BACKUP_DIR.exists():
        return []

    version_dirs: list[tuple[Path, str]] = []
    for item in BACKUP_DIR.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            match = re.match(VERSION_PATTERN, item.name)
            if match:
                version_dirs.append((item, match.group(1)))

    version_dirs.sort(key=lambda x: [int(v) for v in x[1].split(".")])
    return [item[0] for item in version_dirs]


def get_current_version() -> str:
    """Get the current version stored inside the database."""
    return get_database_version()


def get_next_available_version(current_version: str, output_dir: Path) -> str:
    """Find the next version whose backup directory does not already exist."""
    candidate = increment_version(current_version)

    while (output_dir / f"v{candidate}").exists():
        candidate = increment_version(candidate)

    return candidate


def backup_current(output_dir: Path | None = None) -> None:
    """Create a new backup split into multiple files."""
    if output_dir is None:
        output_dir = BACKUP_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    current_version = get_database_version()
    new_version = get_next_available_version(current_version, output_dir)

    backup_dir = output_dir / f"v{new_version}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    backup_prefix = backup_dir / f"{BACKUP_FILE_PREFIX}{new_version}.dump.part-"

    print(f"Current database version: {current_version}")
    print(f"Next database version: {new_version}")

    set_database_version(new_version)

    try:
        print(f"Backing up database '{DATABASE_NAME}' to {backup_dir}")

        pg_dump_cmd = " ".join(shlex.quote(part) for part in PG_DUMP_CMD)
        split_cmd = f"split -b {PART_SIZE_MB}M - {shlex.quote(str(backup_prefix))}"

        combined_cmd = f"{pg_dump_cmd} | {split_cmd}"
        run_command(combined_cmd, shell=True)
    except Exception:
        set_database_version(current_version)
        raise

    created_files = sorted(backup_dir.glob(BACKUP_FILE_GLOB))

    print(f"Backup completed: {backup_dir}")
    print(f"Database version: {new_version}")
    print(f"Files created: {len(created_files)}")