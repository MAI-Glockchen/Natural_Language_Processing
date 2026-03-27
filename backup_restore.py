#!/usr/bin/env python3
"""
PostgreSQL backup and restore utility for the citation pipeline.

This script provides easy commands to backup and restore the PostgreSQL database
for other developers to start with a backed-up database.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result


def backup_database(
    output_dir: Path | None = None,
    database: str = "wiki",
    user: str = "postgres",
    password: str | None = None,
) -> None:
    """
    Backup the PostgreSQL database to a .dump file.
    
    Args:
        output_dir: Directory to save the backup file
        database: Database name to backup
        user: Database user
        password: Database password (if not in environment)
    """
    if output_dir is None:
        output_dir = Path.cwd()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate backup filename
    timestamp = Path.now().strftime("%Y%m%d_%H%M%S")
    backup_file = output_dir / f"citation-postgres_backup_{timestamp}.dump"
    
    # Build command
    cmd = ["pg_dump", "-U", user, "-d", database, "-Fc"]  # Custom format for easier restore
    
    # Add password if provided
    if password:
        cmd.extend(["--password", password])
    
    # Run backup
    print(f"Backing up database '{database}' to {backup_file}")
    run_command(cmd)
    
    print(f"Backup completed: {backup_file}")


def restore_database(
    backup_file: Path,
    database: str = "wiki",
    user: str = "postgres",
    password: str | None = None,
    drop_existing: bool = False,
) -> None:
    """
    Restore a PostgreSQL database from a .dump file.
    
    Args:
        backup_file: Path to the backup file
        database: Database name to restore to
        user: Database user
        password: Database password (if not in environment)
        drop_existing: Whether to drop existing database before restore
    """
    if not backup_file.exists():
        print(f"Error: Backup file not found: {backup_file}")
        sys.exit(1)
    
    # Drop existing database if requested
    if drop_existing:
        print(f"Dropping existing database '{database}'...")
        drop_cmd = ["dropdb", "-U", user, "-d", database]
        if password:
            drop_cmd.extend(["--password", password])
        run_command(drop_cmd)
    
    # Create database if it doesn't exist
    print(f"Creating database '{database}'...")
    create_cmd = ["createdb", "-U", user, database]
    if password:
        create_cmd.extend(["--password", password])
    run_command(create_cmd)
    
    # Restore from backup
    print(f"Restoring from {backup_file}...")
    restore_cmd = ["pg_restore", "-U", user, "-d", database, str(backup_file)]
    if password:
        restore_cmd.extend(["--password", password])
    run_command(restore_cmd)
    
    print(f"Restore completed successfully!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backup and restore PostgreSQL database for citation pipeline"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup the database")
    backup_parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to save backup file (default: current directory)"
    )
    backup_parser.add_argument(
        "-d", "--database",
        default="wiki",
        help="Database name to backup (default: wiki)"
    )
    backup_parser.add_argument(
        "-u", "--user",
        default="postgres",
        help="Database user (default: postgres)"
    )
    backup_parser.add_argument(
        "-p", "--password",
        help="Database password (reads from environment if not provided)"
    )
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument(
        "backup_file",
        type=Path,
        help="Path to the backup file to restore"
    )
    restore_parser.add_argument(
        "-d", "--database",
        default="wiki",
        help="Database name to restore to (default: wiki)"
    )
    restore_parser.add_argument(
        "-u", "--user",
        default="postgres",
        help="Database user (default: postgres)"
    )
    restore_parser.add_argument(
        "-p", "--password",
        help="Database password (reads from environment if not provided)"
    )
    restore_parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing database before restore"
    )
    
    args = parser.parse_args()
    
    if args.command == "backup":
        backup_database(
            output_dir=args.output_dir,
            database=args.database,
            user=args.user,
            password=args.password,
        )
    elif args.command == "restore":
        restore_database(
            backup_file=args.backup_file,
            database=args.database,
            user=args.user,
            password=args.password,
            drop_existing=args.drop_existing,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()