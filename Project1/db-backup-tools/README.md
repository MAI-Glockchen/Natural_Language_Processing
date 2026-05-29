# Backup and Restore Tools

This directory contains the backup and restore utility for the PostgreSQL database.

## Prerequisites

- Docker container `citation-postgres` must be running

## Usage

### Backup the current database

```bash
cd db-backup-tools
python -m db-backup-tools backup
```

This creates a new versioned backup in the `db-backup/` folder.

### Restore from a backup

```bash
# Restore the latest backup
python -m db-backup-tools restore latest

# Restore a specific version
python -m db-backup-tools restore 2.01
```

### List available backups

```bash
python -m db-backup-tools list
```

## Notes

- Restores always drop and recreate the database to ensure a clean state
- Backups are split into 99MB parts for easier handling
- Compression uses zstd with level 9
- Versioning follows semantic versioning (e.g., 2.01)
- All backup/restore operations run inside the `citation-postgres` Docker container
