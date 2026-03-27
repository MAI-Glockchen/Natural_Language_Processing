#!/usr/bin/env python3
"""
PostgreSQL backup and restore utility for the citation pipeline.

This script provides easy commands to backup and restore the PostgreSQL database
for other developers to start with a backed-up database.

Features:
- Versioned backups (e.g., "2.01")
- "backup current" - creates new backup in database backup folder
- "restore latest" - restores the most recent backup
- "restore <version>" - restores a specific version
- Always wipes database on restore (content + structure)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Configuration
BACKUP_DIR = Path("database backup")
VERSION_PATTERN = re.compile(r"citation-postgres_backup_(\d+\.\d+).dump\.part-")
DEFAULT_VERSION = "2.01"


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result


def get_latest_backup() -> Path | None:
    """Get the path to the latest backup version directory."""
    if not BACKUP_DIR.exists():
        return None
    
    # Find all version directories
    version_dirs = sorted(BACKUP_DIR.glob("v*"), key=lambda p: p.name)
    if not version_dirs:
        return None
    
    return version_dirs[-1]


def get_backup_versions() -> list[Path]:
    """Get all backup version directories sorted by version number."""
    if not BACKUP_DIR.exists():
        return []
    
    # Find all version directories matching v\d+\.\d+
    version_dirs = []
    for item in BACKUP_DIR.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            match = re.match(r"v(\d+\.\d+)", item.name)
            if match:
                version_dirs.append((item, match.group(1)))
    
    # Sort by version number
    version_dirs.sort(key=lambda x: [int(v) for v in x[1].split(".")])
    
    return [item[0] for item in version_dirs]


def get_current_version() -> str:
    """Get the current database version from the latest backup."""
    latest = get_latest_backup()
    if latest:
        match = re.match(r"v(\d+\.\d+)", latest.name)
        if match:
            return match.group(1)
    
    return DEFAULT_VERSION


def backup_current(output_dir: Path | None = None) -> None:
    """
    Create a new backup of the current database.
    
    The backup is split into 99MB parts for easier handling.
    """
    if output_dir is None:
        output_dir = BACKUP_DIR
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate version number (increment from latest or use default)
    versions = get_backup_versions()
    if versions:
        # Parse latest version and increment
        latest_version = versions[-1].name[1:]  # Remove 'v' prefix
        parts = latest_version.split(".")
        major = int(parts[0])
        minor = int(parts[1])
        minor += 1
        new_version = f"{major}.{minor}"
    else:
        new_version = DEFAULT_VERSION
    
    # Generate backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = output_dir / f"v{new_version}"
    
    # Build pg_dump command
    cmd = [
        "pg_dump", "-U", "postgres", "-d", "wiki",
        "-Fc", "-C", "-Z", "zstd:level=9"
    ]
    
    # Run backup and split into parts
    print(f"Backing up database 'wiki' to {backup_file}")
    run_command(cmd)
    
    # Split the backup into 99MB parts
    split_cmd = ["split", "-b", "99M", str(backup_file), str(backup_file)]
    run_command(split_cmd)
    
    print(f"Backup completed: {backup_file}")
    print(f"Version: {new_version}")
    print(f"Files created: {backup_file}/*.dump.part-*")


def restore_database(
    version: str,
    drop_existing: bool = True,
) -> None:
    """
    Restore a PostgreSQL database from a backup.
    
    The backup files are concatenated and piped to pg_restore.
    Always drops and recreates the database to ensure clean state.
    
    Args:
        version: Version to restore (e.g., "2.01")
        drop_existing: Whether to drop existing database before restore
    """
    # Find the backup directory
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
    
    # Drop existing database
    if drop_existing:
        print(f"Dropping existing database 'postgres'...")
        drop_cmd = ["dropdb", "-U", "postgres", "-d", "postgres"]
        run_command(drop_cmd)
    
    # Create database
    print(f"Creating database 'postgres'...")
    create_cmd = ["createdb", "-U", "postgres", "postgres"]
    run_command(create_cmd)
    
    # Find all backup parts and restore
    print(f"Restoring from {backup_dir}/*.dump.part-*...")
    
    # Build the cat command to concatenate all parts
    cat_cmd = ["cat", str(backup_dir) + "/*.dump.part-*"]
    
    # Build the pg_restore command
    restore_cmd = ["pg_restore", "-U", "postgres", "-d", "postgres"]
    
    # Run: cat parts | pg_restore
    run_command(cat_cmd + ["|"] + restore_cmd)
    
    print(f"Restore completed successfully!")
    print(f"Database 'postgres' restored with version {version}")


def list_versions() -> None:
    """List all available backup versions."""
    versions = get_backup_versions()
    
    if not versions:
        print("No backup versions found")
        return
    
    print("Available backup versions:")
    print("-" * 50)
    for version_dir in versions:
        version_name = version_dir.name[1:]  # Remove 'v' prefix
        print(f"  {version_name}")
        
        # Show backup files
        backup_files = list(version_dir.glob("*.dump.part-*"))
        if backup_files:
            total_size = sum(f.stat().st_size for f in backup_files)
            print(f"    Files: {len(backup_files)}")
            print(f"    Size: {total_size / (1024**3):.2f} GB")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backup and restore PostgreSQL database for citation pipeline"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup the current database")
    backup_parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=BACKUP_DIR,
        help=f"Directory to save backup files (default: {BACKUP_DIR})"
    )
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from a backup")
    restore_parser.add_argument(
        "version",
        type=str,
        nargs="?",
        default="latest",
        help="Version to restore (e.g., '2.01') or 'latest' for most recent"
    )
    restore_parser.add_argument(
        "--no-drop",
        action="store_true",
        help="Don't drop existing database before restore (not recommended)"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all backup versions")
    
    args = parser.parse_args()
    
    if args.command == "backup":
        backup_current(output_dir=args.output_dir)
    elif args.command == "restore":
        version = args.version
        if version == "latest":
            latest = get_latest_backup()
            if latest:
                version = latest.name[1:]  # Remove 'v' prefix
            else:
                print("Error: No backup versions found")
                sys.exit(1)
        
        restore_database(version, drop_existing=not args.no_drop)
    elif args.command == "list":
        list_versions()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()