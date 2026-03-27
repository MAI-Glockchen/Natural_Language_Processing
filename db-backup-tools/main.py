#!/usr/bin/env python3
"""Main entry point for PostgreSQL backup and restore utility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import BACKUP_DIR
from .core import backup_current, get_latest_backup, get_current_version
from .restore import restore_database, list_versions


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backup and restore PostgreSQL database for citation pipeline"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    backup_parser = subparsers.add_parser("backup", help="Backup the current database")
    backup_parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=BACKUP_DIR,
        help=f"Directory to save backup files (default: {BACKUP_DIR})"
    )
    
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
    
    list_parser = subparsers.add_parser("list", help="List all backup versions")
    
    args = parser.parse_args()
    
    if args.command == "backup":
        backup_current(output_dir=args.output_dir)
    elif args.command == "restore":
        version = args.version
        if version == "latest":
            latest = get_latest_backup()
            if latest:
                version = latest.name[1:]
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