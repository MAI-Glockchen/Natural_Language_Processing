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

Developers should start with a backed-up database using the restore command.
For more information, see `db-backup-tools\README.md`