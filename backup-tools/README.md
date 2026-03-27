# Backup and Restore Tools

This directory contains the backup and restore utility for the PostgreSQL database.

## Usage

### Backup the current database

```bash
python backup_restore.py backup
```

This creates a new versioned backup in the `database backup/` folder.

### Restore from a backup

```bash
# Restore the latest backup
python backup_restore.py restore latest

# Restore a specific version
python backup_restore.py restore 2.01
```

### List available backups

```bash
python backup_restore.py list
```

## Notes

- Restores always drop and recreate the database to ensure a clean state
- Backups are split into 99MB parts for easier handling
- Compression uses zstd with level 9
- Versioning follows semantic versioning (e.g., 2.01)
