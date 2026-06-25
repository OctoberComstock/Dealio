# Railway Deployment

Dealio is intended to deploy on Railway with the repository `Dockerfile`.

Railway provides the `PORT` environment variable automatically. The container start command uses that value and binds FastAPI to `0.0.0.0`.

Set these runtime environment variables in Railway:

```text
ANTHROPIC_API_KEY=<production Anthropic API key>
AGENT_TIMEOUT_SECONDS=140
TAVILY_API_KEY=<production Tavily API key>
DEALIO_DEMO_PASSWORD=<strong demo password>
SESSION_SECRET_KEY=<strong random session secret>
LOG_LEVEL=INFO
DATABASE_PATH=/data/dealio.db
```

Use `DATABASE_PATH=/data/dealio.db` when a Railway volume is mounted at `/data`.

Do not commit real secret values to this repository.
