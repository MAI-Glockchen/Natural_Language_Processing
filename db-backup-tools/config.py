#!/usr/bin/env python3
"""Configuration constants for the PostgreSQL backup and restore utility."""

from pathlib import Path

# Directory configuration
BACKUP_DIR = Path("db-backup")

# Database configuration
DATABASE_NAME = "postgres"
DATABASE_USER = "postgres"
DATABASE_SCHEMA = "wiki"
DATABASE_CONTAINER_NAME = "citation-postgres"

# Backup configuration
DEFAULT_VERSION = "2.01"
PART_SIZE_MB = 99
COMPRESSION_LEVEL = 9

# Version pattern for matching backup directories
VERSION_PATTERN = r"v(\d+\.\d+)"
BACKUP_FILE_PATTERN = rf"{DATABASE_CONTAINER_NAME}_backup_(\d+\.\d+).dump\.part-"

# Command configuration
PG_DUMP_CMD = [
    "docker", "exec", "-i", DATABASE_CONTAINER_NAME,
    "pg_dump", "-U", DATABASE_USER,
    "-d", DATABASE_SCHEMA,
    "-Fc",
    "-C",
    f"-Z", f"zstd:level={COMPRESSION_LEVEL}"
]

DROPDB_CMD = ["docker", "exec", "-i", DATABASE_CONTAINER_NAME, "dropdb", "-U", DATABASE_USER, "-d", DATABASE_NAME]
CREATEDB_CMD = ["docker", "exec", "-i", DATABASE_CONTAINER_NAME, "createdb", "-U", DATABASE_USER, DATABASE_NAME]
PG_RESTORE_CMD = ["docker", "exec", "-i", DATABASE_CONTAINER_NAME, "pg_restore", "-U", DATABASE_USER, "-d", DATABASE_NAME]
