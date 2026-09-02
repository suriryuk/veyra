# Veyra local SearXNG

This Compose stack runs SearXNG and its private Valkey service for Veyra web
research. SearXNG is exposed only on `127.0.0.1:8888` by default; Valkey is not
published to the host. The JSON search response required by Veyra is enabled in
`settings.yml`.

## First-time setup

Start Docker Desktop with the Linux container engine, then run from PowerShell:

```powershell
Set-Location C:\veyra\searXNG
.\setup.ps1
docker compose pull
docker compose up -d
```

`setup.ps1` creates the Git-ignored `.env` file with a random SearXNG secret.
Use `.env.example` as the reference for optional host, port, or image-version
overrides. The checked-in defaults pin the SearXNG and Valkey image versions that
passed the Veyra smoke test; updates are deliberate edits rather than implicit
`latest` changes. Keep `SEARXNG_HOST=127.0.0.1` unless other machines genuinely
need access; publishing this development instance broadly requires separate
proxy, TLS, and rate-limit hardening.

## Daily commands

```powershell
# Start or apply configuration changes
docker compose up -d

# View status and logs
docker compose ps
docker compose logs -f searxng

# Stop while retaining cache and Valkey data
docker compose down

# Update images and recreate containers
# First update SEARXNG_VERSION and/or VALKEY_VERSION in .env after reviewing
# upstream release notes, then:
docker compose pull
docker compose up -d
```

Run `docker compose down -v` only when you intentionally want to delete the
stack's cache and Valkey volume.

## Verify the JSON API

```powershell
Invoke-RestMethod "http://127.0.0.1:8888/search?q=rust&format=json"
```

`docker compose ps` should report both services as healthy. If `searxng` is
unhealthy, inspect `docker compose logs searxng`; the healthcheck requests the
local HTML root and does not expose another port.

The Veyra default already points to `http://127.0.0.1:8888/`. If the port is
changed in `.env`, set `VEYRA_SEARXNG_BASE_URL` to the matching URL before
starting Veyra.

The first search can take a little longer while engines respond. Individual
upstream engines may rate-limit or fail independently; inspect
`docker compose logs -f searxng` when the API returns no useful results.
If Compose reports that the `dockerDesktopLinuxEngine` pipe is missing, start
Docker Desktop and wait until its engine status shows as running.
