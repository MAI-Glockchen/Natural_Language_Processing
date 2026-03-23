# Citation pipeline infra

## Start

```bash
cp .env.example .env
./run.sh
```

This will:
1. start a Postgres 16 container
2. wait until Postgres is healthy
3. run the Python pipeline container once

## Useful commands

Start only Postgres:
```bash
docker compose up -d postgres
```

Run pipeline against running Postgres:
```bash
docker compose run --rm pipeline
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
