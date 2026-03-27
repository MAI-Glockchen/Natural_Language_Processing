#!/usr/bin/env python3
"""Core backup and restore functions for PostgreSQL database."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from .config import (
    BACKUP_DIR,
    DATABASE_NAME,
    DATABASE_USER,
    PG_DUMP_CMD,
    DROPDB_CMD,
    CREATEDB_CMD,
    PG_RESTORE_CMD,
    DEFAULT_VERSION,
    VERSION_PATTERN,
    DATABASE_SCHEMA,
    PART_SIZE_MB,
)


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result


def get_latest_backup() -> Path | None:
    """Get the path to the latest backup version directory."""
    if not BACKUP_DIR.exists():
        return None
    
    version_dirs = sorted(BACKUP_DIR.glob("v*"), key=lambda p: p.name)
    if not version_dirs:
        return None
    
    return version_dirs[-1]


def get_backup_versions() -> list[Path]:
    """Get all backup version directories sorted by version number."""
    if not BACKUP_DIR.exists():
        return []
    
    version_dirs = []
    for item in BACKUP_DIR.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            match = re.match(VERSION_PATTERN, item.name)
            if match:
                version_dirs.append((item, match.group(1)))
    
    version_dirs.sort(key=lambda x: [int(v) for v in x[1].split(".")])
    
    return [item[0] for item in version_dirs]


def get_current_version() -> str:
    """Get the current database version from the latest backup."""
    latest = get_latest_backup()
    if latest:
        match = re.match(VERSION_PATTERN, latest.name)
        if match:
            return match.group(1)
    
    return DEFAULT_VERSION


def backup_current(output_dir: Path | None = None) -> None:
    """Create a new backup of the native split."""
    if output_dir is None:
        output_dir = BACKUP_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    versions = get_backup_versions()
    if versions:
        latest_version = versions[-1].name[1:]
        parts = latest_version.split(".")
        major = int(parts[0])
        minor = int(parts[1])
        minor += 1
        new_version = f"{major}.{minor}"
    else:
        new_version = DEFAULT_VERSION
    
    backup_file = output_dir / f"v{new_version}"
    
    print(f"Backing up database '{DATABASE_SCHEMA}' to {backup_file}")

    # Run pg_dump and pipe to split
    pg_dump_cmd = PG_DUMP_CMD.copy()
    split_cmd = ["split", "-b", f"{PART_SIZE_MB}M", "-", str(backup_file)]

    # Use shell to pipe the output
    combined_cmd = f"{' '.join(pg_dump_cmd)} | {' '.join(split_cmd)}"
    run_command(combined_cmd, shell=True)
    
    print(f"Backup completed: {backup_file}")
    print(f"Version: {new_version}")
    print(f"Files created: {backup_file}/*.dump.part-*")