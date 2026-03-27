#!/usr/bin/env python3
"""Configuration constants for the PostgreSQL backup and restore utility."""

from pathlib import Path

# Directory configuration
BACKUP_DIR = Path("db-backup")

# Database configuration
DATABASE_NAME = "wiki"
DATABASE_USER = "postgres"
DATABASE_CONTAINER_NAME = "citation-postgres"
MAINTENANCE_DATABASE_NAME = "postgres"

# Backup configuration
DEFAULT_VERSION = "1.00"
INITIAL_DATABASE_VERSION = "1.00"
PART_SIZE_MB = 99
COMPRESSION_LEVEL = 9

# Version pattern for matching backup directories
VERSION_PATTERN = r"v(\d+\.\d+)"
BACKUP_FILE_PREFIX = f"{DATABASE_CONTAINER_NAME}_backup_"
BACKUP_FILE_GLOB = "*.dump.part-*"

# Version metadata table
VERSION_TABLE_NAME = "backup_metadata"

# Command configuration
PG_DUMP_CMD = [
    "docker", "exec", "-i", DATABASE_CONTAINER_NAME,
    "pg_dump", "-U", DATABASE_USER,
    "-d", DATABASE_NAME,
    "-Fc",
    "-C",
    "-Z", f"zstd:level={COMPRESSION_LEVEL}",
]

PG_RESTORE_CMD = [
    "docker", "exec", "-i", DATABASE_CONTAINER_NAME,
    "pg_restore", "-U", DATABASE_USER,
    "-d", DATABASE_NAME,
    "--clean",
    "--if-exists",
    "--no-owner",
    "--no-privileges",
]