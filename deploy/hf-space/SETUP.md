# Deploying the recommender to Hugging Face Spaces

Free CPU Basic: 2 vCPU, 16GB RAM, x86, HTTPS provided, no time limit. The costs
are a 48-hour idle sleep, a public URL, and no SLA. The sleep is the only one
that meaningfully hurts, and a scheduled ping removes it — see **Keep it awake**
below, which is the part that makes this host viable at all.

## Verify this first — it can rule HF out

Your `chroma_db/` is **553MB** and has to reach the Space somehow. Space repos
store large files through **Git LFS**, and free accounts have a storage quota.

Check your quota at <https://huggingface.co/settings/storage> before doing
anything else. If 553MB does not fit, the fallback is to put the index in a
**Dataset repo** (designed for large files, more generous quotas) and download
it at build time with `huggingface_hub.snapshot_download` instead of committing
it to the Space. That adds a build-time token if the dataset is private.

## 1. Create the Space

<https://huggingface.co/new-space>

- **Owner:** your account
- **Space name:** `palate-recommender`
- **License:** your choice
- **SDK:** **Docker** → *Blank*
- **Hardware:** CPU basic (free)
- **Visibility:** Public

> Private Spaces on a free account are rate-limited and awkward to call from
> Vercel. Public is the practical choice — which is exactly why
> `RECOMMENDER_TOKEN` matters more here than it did on Cloud Run.

## 2. Clone the Space and add the files

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/palate-recommender
```

Copy in from this repo:

| From | To (Space root) |
|---|---|
| everything `.dockerignore` does not exclude | repo root |
| `deploy/hf-space/Dockerfile` | `Dockerfile` |
| `deploy/hf-space/space-README.md` | `README.md` |

The `README.md` frontmatter is not decoration — `sdk: docker` and
`app_port: 7860` are how HF knows what to build and where to route. A Space
without them builds nothing.

Track the index with LFS **before** committing it, or git will try to store
553MB as a normal blob and the push will fail:

```bash
git lfs install && git lfs track "chroma_db/**"
```

```bash
git add .gitattributes . && git commit -m "Palate recommender" && git push
```

The first push moves 553MB. Expect several minutes.

## 3. Set the secrets

Space → **Settings → Variables and secrets**. Add as **Secrets** (not
Variables — Variables are visible to anyone who can see the Space):

| Name | Value |
|---|---|
| `mongo_url` | Atlas connection string |
| `MONGO_DB` | `test` |
| `RECOMMENDER_TOKEN` | the same hex string Vercel sends |

Copy `RECOMMENDER_TOKEN` verbatim from the old `env.yaml`. Generate a fresh one
and `/index/missing` starts 401ing the moment Vercel points here.

## 4. Atlas network access — read this before you skip it

**HF Spaces have no static egress IP.** Your Atlas cluster is IP-allowlisted, so
there is no address to add. The only workable option is allowing `0.0.0.0/0`
in Atlas → Network Access.

That is a real downgrade from the Cloud Run setup, and worth being deliberate
about rather than discovering later. What still protects the database:

- TLS on every connection (`mongodb+srv://` enforces it)
- The database password, which is now the *only* thing standing in front of
  Atlas from the open internet

So before opening it, two changes that cost minutes and matter a great deal:

1. **Rotate the Atlas password** to something long and random. It is about to
   become the only credential guarding the cluster from the open internet.

2. **Use a read-only database user.** This is verified, not assumed: the only
   Mongo access anywhere in `service.py` is `col.find(...)` at line 714. There
   is no insert, update, delete or bulk write in the service. `POST
   /index/missing` writes to *Chroma*, not to Mongo. So a user with `read` on
   the one database is sufficient, and it means a leaked connection string
   cannot be used to modify or destroy your data.

Together these turn "Atlas is open to the internet" from alarming into roughly
the same posture as any managed database behind a strong credential.

## 5. Watch the build

Space → **Logs**. First build takes 10–20 minutes: torch is a large wheel and
the model bakes into a layer.

Common failures, in the order you will hit them:

- **Permission denied on chroma_db** — the `chown -R user:user /app` before
  `USER user` is missing. Chroma opens SQLite read-write even for pure reads.
- **Space builds but never becomes reachable** — the container is not on 7860,
  or `app_port` is absent from the README frontmatter.
- **Model downloading at runtime** — `HF_HOME` was not set before the bake
  layer, so the cache landed in `/root` where UID 1000 cannot read it.

## 6. Verify

```bash
curl -s https://YOUR_USERNAME-palate-recommender.hf.space/health
```

Expect `{"status":"ok","count":15211}`. **15211 is the baseline** measured
against the local index — a materially different number means the LFS transfer
was incomplete, and the fix is to re-push, not to debug the app.

For the detail fields, send the token:

```bash
curl -s -H "x-recommender-token: YOUR_TOKEN" https://YOUR_USERNAME-palate-recommender.hf.space/health
```

Then a real query through `/docs` → `POST /recommend` → **Try it out**. A 200
with an empty array is a failure, not a pass.

## 7. Keep it awake

This is the step that makes the whole host viable. The 48-hour sleep timer
resets on **any** HTTP request, so a scheduled ping means it never sleeps.

**Use UptimeRobot or cron-job.org.** Both free, purpose-built, and they alert
you when the check fails.

- URL: `https://YOUR_USERNAME-palate-recommender.hf.space/health`
- Interval: 5 minutes (well inside the 48h window, with enormous margin)
- Alert on failure: on

Because `/health` returns 503 on an empty index, this doubles as a deploy
monitor: a bad rebuild pages you instead of silently serving zero results.

**Do not use GitHub Actions cron for this.** Scheduled workflows get delayed
under load and are auto-disabled after 60 days of repository inactivity —
precisely when you have stopped watching.

**Vercel Cron** is a reasonable backup since the app is already there, but Hobby
allows only *daily* crons, and daily against a 48-hour window leaves no margin
for a single missed run. In `vercel.json`:

```json
{ "crons": [{ "path": "/api/keep-recommender-awake", "schedule": "0 6 * * *" }] }
```

with a route that fetches `/health` and returns its status.

## 8. Warm on session start

Keep-alive handles idle sleep. It does not handle restarts after a rebuild, or
HF-side restarts, which have no SLA behind them.

For those, move the wake **off the critical path**: fire a no-await request to
`/health` when a user signs in or lands anywhere in the app, not when the
dashboard mounts. By the time they reach a screen that needs ranking, the Space
has had several seconds' head start.

Then let the existing degradation do its job. `/api/Restaurants/nearby` already
falls back to distance order at 3 seconds — keep that. Consider a client-side
refetch a few seconds later so the list silently upgrades to ranked, instead of
staying unranked for the whole session.

## 9. Cut Vercel over

Point the recommender base URL at
`https://YOUR_USERNAME-palate-recommender.hf.space`, redeploy, and exercise both
callers — the dashboard's `/nearby` and a group shortlist.

## 10. Decommission GCP

Only after the above passes:

```bash
gcloud run services delete palate-recommender --region asia-south1
```

Then delete the Artifact Registry repo (`cloud-run-source-deploy`) and empty the
`run-sources-*` staging bucket. **Those two are where the money actually was** —
deleting the service alone leaves the storage billing running.

Keep the $5 budget alert.
