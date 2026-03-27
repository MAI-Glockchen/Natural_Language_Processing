## Useful commands

Start only Postgres:
```bash
docker compose up -d postgres
```

Open psql:
```bash
docker compose exec postgres psql -U postgres -d wiki
```

Stop everything:
```bash
docker compose down
```

Delete DB volume too:
```bash
docker compose down -v
```

## Notes

- `.env` controls both container settings and pipeline tuning.
- The app uses `host=postgres` inside Docker.
- For local Python execution outside Docker, change `POSTGRES_DSN` host to `localhost`.

## Backup and Restore

The backup and restore tools are located in the `backup-tools/` directory.

To backup the PostgreSQL database:

```bash
cd backup-tools
python backup_restore.py backup
```

To restore from a backup:

```bash
# Restore the latest backup
cd backup-tools
python backup_restore.py restore latest

# Restore a specific version
cd backup-tools
python backup_restore.py restore 2.01
```

To list available backups:

```bash
cd backup-tools
python backup_restore.py list
```

Other developers should start with a backed-up database using the restore command.
