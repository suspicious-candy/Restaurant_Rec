---
title: Palate Recommender
emoji: 🍽️
colorFrom: orange
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Palate Recommender

Vector-search backend for Palate. Embeds restaurant descriptions with
`all-MiniLM-L6-v2`, stores them in a Chroma HNSW index, and ranks candidates for
one person or for a group.

This Space is an **API, not a demo** — there is no UI. The interactive docs at
`/docs` are the only thing worth opening in a browser.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | Vector count and status. 503 when the index is empty. |
| `POST` | `/recommend` | Rank candidates for one taste profile. |
| `POST` | `/recommend/group` | Least-misery / centroid ranking for a group. |
| `POST` | `/index/missing` | Write path. Requires `x-recommender-token`. |

## Cold starts

Free CPU Spaces sleep after 48 hours of inactivity, and waking costs 30–90
seconds while torch, the model and a 553MB HNSW index load. A scheduled ping
against `/health` keeps the Space awake so this rarely happens — see
`SETUP.md` in the main repo.

`/health` is the right target for that ping: it is cheap, needs no token, and
returns 200 only once the index is genuinely loaded, so the keep-alive doubles
as a monitor that catches a bad deploy.

## Writes do not persist

Spaces have no persistent disk. `POST /index/missing` succeeds and its writes
survive only until the Space restarts. Rows are never lost — they live in
Mongo — they are simply unranked until the next `rebuild_index.py` run and
redeploy. Rebuild monthly, or whenever a new city is added.
