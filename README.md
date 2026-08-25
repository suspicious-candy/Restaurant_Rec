# Restaurant Recommender

The recommendation service behind [Palate](https://github.com/suspicious-candy/palate). A FastAPI app that ranks
restaurants by semantic similarity — for one person, or for a group of people
who have to agree on dinner.

Originally exploratory work on the [Yelp Open Dataset](https://www.yelp.com/dataset);
that data now serves as a seed and a test fixture, and the live index is built
from Palate's own MongoDB.

> **Deployment status: live on Google Cloud Run**, serving
> [Palate on Vercel](https://palate-suspicious-candy.vercel.app/). The
> `Dockerfile`, `.dockerignore`, `.gcloudignore` and `env.yaml.example` in this
> directory are the whole deploy. See [Deployment](#deployment) — and read
> [The index ships inside the image](#the-index-ships-inside-the-image) before
> touching it, because it changes what `POST /index/missing` means in production.
>
> Two things changed after the first deploy and both are documented below:
> `GET /health` now exists, because a wrong index path produces a container that
> starts perfectly and answers every query with nothing; and the `Dockerfile`
> splits the index and the source into **separate layers**, because bundling
> them meant every one-line code change republished 553MB into Artifact Registry
> — see [What the first deploys cost](#what-the-first-deploys-cost).

---

## Table of contents

- [How it fits together](#how-it-fits-together)
- [Quick start (local)](#quick-start-local)
- [Configuration](#configuration)
- [API](#api)
- [Deployment](#deployment)
- [Other hosts evaluated](#other-hosts-evaluated)
- [Files](#files)
- [Rebuilding the index](#rebuilding-the-index)
- [The index](#the-index)
- [Design decisions, and the measurements behind them](#design-decisions-and-the-measurements-behind-them)
- [Known limitations](#known-limitations)
- [Gotchas](#gotchas)
- [Historical: the Yelp dataset](#historical-the-yelp-dataset)

---

## How it fits together

```
Foursquare API ──► MongoDB `restaurants`  ◄── source of truth for text
                          │
                          │  rebuild_index.py   (the index's ONLY writer)
                          ▼
                   Chroma `places_v2`     ◄── derived index, disposable
                          ▲  │                  (ships inside the image)
        POST /index/missing │  │  service.py    ── on Google Cloud Run
     (same build_text path) │  ▼
      POST /recommend · POST /recommend/group · GET /health
                          ▲
                          │  HTTPS  (+ x-recommender-token on the write)
                   Palate (Next.js) ── on Vercel
```

**Mongo is the source of truth; Chroma is a derived index.** That direction is
what makes the embedding model, the chunking, and the text template all
changeable later — none of which is possible if accumulated text lives only
inside the vector store. The index can be deleted and rebuilt at any time.

The invariant is **one text template**, not "the service never writes".
`/index/missing` does write, but it reads the text from Mongo and runs the same
`build_text -> chunk -> pool` pipeline `rebuild_index.py` runs — it is imported,
not reimplemented. Callers choose WHICH rows get indexed, never WHAT is
embedded. That distinction is the whole reason the old `/embed` endpoint had to
go: it let callers push their own sentence.

---

## Quick start (local)

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**The `--extra-index-url` is not optional.** Without it pip resolves the CUDA
build of torch from PyPI: roughly 2.5GB unpacked plus ~3GB of CUDA runtime that
nothing here can use, since `all-MiniLM-L6-v2` embeds a few hundred short
sentences per request on CPU comfortably. The existing dev `.venv` has the CUDA
wheel and is 6.1GB, which is exactly the mistake this flag avoids repeating in
the image.

Mongo credentials are read from `../palate/.env` (`mongo_url`) by default — one
secret in one place, rather than a copy that drifts. Override the path with
`RECOMMENDER_ENV_FILE`, or just set real environment variables, which always
win. The URI carries no database name, so the scripts default to `test`, which
is where Node's driver has been writing. Override with `MONGO_DB`.

Run the service **from this directory** in development — `CHROMA_DIR` defaults
to the relative path `chroma_db`:

```bash
./.venv/Scripts/python.exe -m uvicorn service:app --port 8000
```

Port 8000 matters locally: Palate's `RECOMMENDER_URL` defaults to
`http://localhost:8000`. Without the service running, `/api/Restaurants/nearby`
gives up after 3 seconds and silently degrades to an unranked distance-ordered
list, while the group shortlist route returns 503 and a vote cannot start at
all. Two opposite symptoms from one missing process — see
[the two timeouts](#cold-starts-and-the-two-timeouts).

Confirm it came up against a real index before doing anything else — "it
started" and "it can answer" are different claims here:

```bash
curl -s http://localhost:8000/health
```

`{"status":"ok","count":15211}`. A 503 with `"status":"empty"` means you started
it from the wrong directory, so `CHROMA_DIR` resolved somewhere with no store in
it — and Chroma creates an empty collection rather than complaining. See
[`GET /health`](#get-health).

---

## Configuration

Every path and secret comes from the environment. Nothing assumes a working
directory or a sibling checkout any more.

| Variable | Required | Default | What it does, and what breaks without it |
|---|---|---|---|
| `mongo_url` | in production | — | Atlas connection string, the **same database the Next app uses**. Only the write path needs it: `/index/missing` answers 503 without it, while `/recommend*` keep working. The connection is lazy for exactly that reason. |
| `MONGO_DB` | — | `test` | Database name. The app's URI historically carried none, so everything landed in a database literally called `test`. Rename both sides together. |
| `CHROMA_DIR` | — | `chroma_db` | Where Chroma's SQLite lives. Relative paths resolve against the working directory. `rebuild_index.py` reads the same variable, and **the two must agree** — point them at different directories and the rebuild writes an index the service never reads. |
| `RECOMMENDER_TOKEN` | on any reachable host | *(unset)* | Shared secret for `POST /index/missing`, sent as `x-recommender-token`. Unset means the endpoint is **open**, and the service prints a warning to stderr at startup saying so. Must be byte-identical to `RECOMMENDER_TOKEN` in the Next app. Generate with `openssl rand -hex 32`. |
| `RECOMMENDER_ENV_FILE` | — | `../palate/.env` | Development convenience only. A missing file is not an error, which is what should happen in production. |
| `PORT` | — | `8080` | Injected by Cloud Run. The container must listen on whatever it is given. |

`load_dotenv` runs **after** the real environment, and python-dotenv does not
override variables that are already set — so a container's `-e` flags beat
anything in a file, and the sibling `.env` is a laptop fallback.

Copy `.env.example` to `.env` for local development. `env.yaml.example` is the
production counterpart: copy it to `env.yaml`, fill it in, and pass it to
`gcloud` with `--env-vars-file`. Both `.env` and `env.yaml` are gitignored and
gcloudignored.

---

## API

Two search endpoints and a health check, open. One write endpoint, guarded.

### `GET /health`

```jsonc
// no headers
→ 200 { "status": "ok", "count": 15211 }
→ 503 { "status": "empty", "count": 0 }
→ 503 { "status": "error", "count": null, "error": "OperationalError" }
```

```jsonc
// headers: { "x-recommender-token": "<RECOMMENDER_TOKEN>" }
→ 200 {
  "collection": "places_v2",
  "chromaDir":  "/app/chroma_db",   // the RESOLVED path, not the env var
  "embedDim":   384,
  "writeGuard": "enforced",
  "status":     "ok",
  "count":      15211
}
```

**The count is the whole point.** Chroma treats a missing or empty
`persist_directory` as a *new empty collection* rather than an error, and `db` is
built at import time — so a wrong `CHROMA_DIR` produces a process that starts
cleanly, serves `/docs`, passes any "is it listening?" probe, and returns nothing
for every single query. Until this endpoint there was no `GET` route through
which to notice that.

Which is why **an empty collection is a 503, not a 200.** Returning success with
`count: 0` would put a green tick on a broken deploy. 503 surfaces in
`docker ps` as `(unhealthy)` and to an uptime check, without causing a restart
loop, since nothing acts on health status by default.

It deliberately **does not touch Mongo.** The Mongo client is lazy precisely so
an unreachable database cannot stop the read endpoints from serving searches
they are perfectly capable of serving; failing health on it would undo that and
take the service down for an outage it can ride out.

The detail fields are token-gated because the resolved filesystem path and
whether the write guard is active are not things a public endpoint should
volunteer — but `status` and `count` stay open, because a container healthcheck
and an external probe both need them and neither can hold a secret. Same
`secrets.compare_digest` as the write guard, for the same reason.

`chromaDir` being the *resolved* absolute path is what catches the most common
version of this failure: a relative `CHROMA_DIR` landing against an unexpected
working directory.

### `POST /recommend`

```jsonc
{ "query": "A Italian, Pizza restaurant", "k": 5, "candidateIds": ["4f24…"] }
→ { "businessIds": ["4f24…", …] }
```

### `POST /recommend/group`

```jsonc
{
  "queries":  ["spicy North Indian curry", "Chinese noodles"],  // or…
  "vectors":  [[0.01, …]],          // pre-computed per-person taste vectors
  "weights":  [1.0, 1.0],           // optional, defaults to equal
  "candidateIds": ["4f24…"],        // required unless strategy is "centroid"
  "strategy": "blend",              // blend | min | mean | centroid
  "alpha":    0.5,
  "k":        20,
  "debug":    false                 // also returns each member's own top 5
}
→ {
  "businessIds": [...],
  "results": [{ "businessId", "score", "memberSims", "worstMemberSim" }],
  "strategy": "blend",
  "agreement": 0.24,     // mean pairwise cosine between members
  "consensus": 0.78,     // ||mean of unit member vectors||
  "memberCount": 2
}
```

`queries` and `vectors` are concatenated in that order, and `weights` must
follow it. Accepting both is deliberate: callers send preference text today,
and can send per-person taste vectors later with no change to this endpoint.

**Always pass `candidateIds`.** Retrieval should be a *filter* — the caller
decides what is reachable (nearby, open, in budget) and the vector search only
*orders* what survives. See "Filter, then rank" below. `MAX_CANDIDATES` is 500;
more than that is a 400, because Chroma builds one `$in` predicate per id and a
huge list turns the filter into the slow path.

`score` is higher-is-better on every strategy, but **ordinal within one
response**: centroid scores are cosine similarities in [-1, 1]; blend and min
scores are built from z-scores and can exceed 1. Fine for ranking, wrong for
thresholding.

### `POST /index/missing`

**Requires `x-recommender-token` when `RECOMMENDER_TOKEN` is set.**

```jsonc
// headers: { "x-recommender-token": "<RECOMMENDER_TOKEN>" }
{ "businessIds": ["4f24…", …], "force": false }
→ { "requested": 50, "alreadyIndexed": 38, "written": 12,
    "skipped": 0, "truncated": false }
```

Embeds any of the given restaurants that Mongo has and Chroma lacks. Palate
calls this fire-and-forget from `/api/Restaurants/nearby` after syncing a cold
area, so those places have vectors by the next request — and, more to the point,
by the time a group needs a shortlist there. Capped at `MAX_INDEX_BATCH` (500)
per call, because encoding happens inside the request; a full build is still
`rebuild_index.py`'s job.

`force: true` re-embeds rows the index already holds, for when the **text**
changed rather than the row being new — a post-meal review landing in `tips[]`
is the case it exists for. It requires `businessIds`: `force` with no ids would
re-embed the whole corpus inside one request. The write is an upsert keyed on
`fsqId`, so re-embedding replaces a vector rather than duplicating it.

The guard uses `secrets.compare_digest`, not `==`. String equality in CPython
short-circuits on the first differing byte, which leaks the length of the
matching prefix through response timing and makes the token recoverable byte by
byte over enough requests.

The read endpoints stay open on purpose: they are cheap, they leak nothing a
user of the app cannot already see, and the Next server calls them on behalf of
signed-out visitors.

---

## Deployment

Runs on **Google Cloud Run** (`asia-south1`), and the `Dockerfile` is written for
its constraints specifically. The whole deploy is one command; everything
interesting is in what the image contains, in what order, and why.

### Prerequisites

- A MongoDB Atlas cluster reachable from Cloud Run. **Cloud Run's egress IP is
  not stable**, so either allow `0.0.0.0/0` in Atlas Network Access (acceptable
  when the connection string itself is the credential and it lives only in Cloud
  Run's environment) or attach a VPC connector with Cloud NAT and allowlist that
  one static IP. The second is the right answer for anything long-lived.
- `gcloud` authenticated, with Cloud Run, Cloud Build and Artifact Registry
  enabled on the project.
- **A current `chroma_db/`.** It ships inside the image, so whatever is on disk
  at build time is what production searches. Run `rebuild_index.py` first if it
  is stale.

### Deploy

```bash
cp env.yaml.example env.yaml    # then fill in mongo_url and RECOMMENDER_TOKEN
```

```bash
gcloud run deploy palate-recommender --source . --region asia-south1 --env-vars-file env.yaml --allow-unauthenticated --memory 2Gi --cpu 2 --concurrency 8 --timeout 300 --min-instances 0
```

Flag by flag, because most of these are not defaults and each one is load-bearing:

| Flag | Why |
|---|---|
| `--allow-unauthenticated` | The caller is Vercel, which has no Google identity to present. IAM cannot express "this one Next app", so the service is public at the network layer and `RECOMMENDER_TOKEN` is what actually protects the write endpoint. The reads are meant to be open anyway. |
| `--memory 2Gi` | Cloud Run's default is 512Mi, and torch plus the sentence-transformer plus Chroma's HNSW index will not fit in it. Start at 2Gi and check the revision's memory-utilisation metric before trimming — an OOM here surfaces as a generic startup failure, not as memory pressure, so it is worth over-provisioning until you have one working revision to measure. |
| `--cpu 2` | Embedding is CPU-bound and the only slow thing in a request. Also shortens the cold start, which is dominated by loading the model. |
| `--concurrency 8` | Cloud Run defaults to **80**, which is right for an I/O-bound web app and wrong for this one. Eighty concurrent CPU-bound embeds on two vCPUs queue behind each other and every request times out together. |
| `--timeout 300` | Only matters when an instance is cold *and* a large `/index/missing` batch lands on it. |
| `--min-instances 0` | Scale to zero, and pay the cold start. See below — set it to 1 if group shortlists are the priority. |

### Cold starts, and the two timeouts

A warm response is well under a second. A cold one has to start a container and
load a sentence-transformer first: **measured at 14 seconds locally**, and
reasonably slower on a shared vCPU.

The model is baked into the image precisely because of this.
`HuggingFaceEmbeddings` would otherwise download `all-MiniLM-L6-v2` (~90MB) on
first use — inside the first request, on every cold start, and only if
huggingface.co happens to be reachable from Cloud Run's network. The
`RUN python -c "SentenceTransformer(...)"` line in the Dockerfile turns that
into build-time cost paid once.

The two callers in Palate then make **deliberately opposite** choices about
waiting, and it is worth understanding both before tuning `--min-instances`:

- `/api/Restaurants/nearby` gives up after **3 seconds** and serves distance
  order. It degrades invisibly and well, so making a user watch a spinner
  through a cold start to get a barely different list is the worst option. The
  request still warms the instance in the background, so the ranking is there on
  the next load.
- `/api/user/matching/[groupId]/shortlist` waits **60 seconds**. It cannot
  degrade — it freezes a ballot that people then vote on — so a cold start that
  ran long used to produce a 503 and a group that could not start dinner.

The consequence: **with `--min-instances 0`, the first person to open the
dashboard after an idle period gets an unranked list.** That is a designed
degradation, not a bug. If group voting is the flow that matters most,
`--min-instances 1` removes both the unranked first load and the 60-second
worst case, at the cost of one always-on instance.

### The index ships inside the image

Read this before deploying, because it changes what `/index/missing` means.

Cloud Run gives a container **no persistent disk** and freezes it between
requests, so there is nowhere durable to keep a 553MB SQLite file. The
alternatives are both worse: downloading it from Cloud Storage adds ~553MB to
every cold start, and mounting a bucket over FUSE puts SQLite on a filesystem
whose locking semantics it does not trust. So `chroma_db/` is deliberately
**not** in `.dockerignore`, `CHROMA_DIR` points at `/app/chroma_db`, and the
index travels in the image.

`.gcloudignore` exists for one reason: when it is absent, `gcloud` generates one
from `.gitignore` — which excludes `chroma_db/`, because a 553MB derived
artifact has no business in git. The result is a deploy that succeeds, a service
that starts cleanly, and **every search returning nothing.** Silent and
confusing. Do not delete that file.

**What this costs you.** `POST /index/missing` still works, but its writes go to
the container's ephemeral layer and vanish when the instance scales down. So
restaurants synced from Foursquare after a deploy are searchable only until that
instance dies. They are never *lost* — the rows are in Mongo — they are simply
unranked until the next `rebuild_index.py` run and redeploy. Both callers
degrade to distance order, so this is slow staleness rather than failure.

**Rebuild and redeploy monthly, or whenever a new city is synced:**

```bash
./.venv/Scripts/python.exe rebuild_index.py --only-missing
```

On a host **with** a disk (Fly, a VM), none of this applies: delete the
`COPY chroma_db/ ./chroma_db/` line, re-add `chroma_db/` to `.dockerignore`,
point `CHROMA_DIR` at the mount, and runtime writes persist. That variant is
written out in `Dockerfile.arm64` — see
[Other hosts evaluated](#other-hosts-evaluated).

### What the first deploys cost

Worth reading before you deploy ten times in a week, because the bill does not
come from where you would look for it.

Roughly ten deploys over three days billed about **$1.04** — a ~$16/month run
rate. Almost none of it was compute. Cloud Run's free tier (240,000 vCPU-s,
450,000 GiB-s, 2M requests per month) covered the actual request handling
comfortably. The spend was **storage of repeated deploy artifacts**: Artifact
Registry gives 0.5GB free and then charges $0.10/GB/month, and nothing garbage-
collects old image layers or the `--source .` build-context tarballs that land
in a staging bucket on every deploy.

The cause was one line. The image originally ended with:

```dockerfile
COPY . .
```

A Docker layer is rebuilt whenever its inputs change, and every rebuilt layer is
pushed and stored as a **new blob**. That single `COPY` put the 553MB index and
the ~40KB of source into *one* layer, so editing a line of `service.py`
invalidated both and republished half a gigabyte. Ten deploys, ten near-identical
copies of the index, billing forever.

The fix is a two-line split, and **the order is the whole point**:

```dockerfile
COPY chroma_db/ ./chroma_db/          # stable: changes only when the index does
COPY service.py rebuild_index.py ./   # churning: changes every code edit
```

Layers cache in order, so the stable thing has to come *first* or the split buys
nothing. Now a routine code deploy pushes kilobytes and reuses the cached 553MB
layer.

Two consequences to keep in mind:

- The second `COPY` **names files explicitly** rather than using `COPY . .`,
  which would re-copy `chroma_db/` and undo the split. That list is the complete
  runtime set: `service.py` imports `build_text`, `chunk` and `pool` from
  `rebuild_index`, and nothing else in the tree is imported at runtime. **Adding
  a new runtime module means adding it to that line** — and forgetting produces
  an `ImportError` at container start, which Cloud Run reports as a generic
  "failed to start and listen".
- Rebuilding the index still republishes 553MB, by design. That is the one
  deploy where the cost is real, and it is another reason the rebuild cadence is
  monthly rather than continuous.

**If you are diagnosing a bill, check the attribution before trusting any of
this.** Billing → Cost breakdown, grouped by service. Spend under *Artifact
Registry* or *Cloud Build* is this problem. Spend under *Cloud Run* itself is a
different problem — traffic or a misconfiguration — and would follow you onto
any other host. Keep a **$5 budget alert** on the project regardless; it costs
nothing and it is the only thing watching when you have stopped.

### Shrink the image first

The store on disk is 553MB, and **more than half of it is dead**. `chroma.sqlite3`
holds two collections:

| collection | vectors | read by |
|---|---|---|
| `places_v2` | 15,211 | everything |
| `langchain` | 20,613 | nothing |

`langchain` is the abandoned L2-space collection that mixed Yelp review text
with Foursquare template sentences — two populations that were never comparable.
Nothing in `service.py` opens it. Dropping it and vacuuming is the cheapest
image-size win available, and image size is what cold start scales with:

```bash
./.venv/Scripts/python.exe -c "import chromadb; chromadb.PersistentClient(path='chroma_db').delete_collection('langchain')"
```

```bash
./.venv/Scripts/python.exe -c "import sqlite3; sqlite3.connect('chroma_db/chroma.sqlite3').execute('VACUUM')"
```

Both steps are irreversible and `VACUUM` needs free disk roughly equal to the
database size. The `langchain` vectors are **not** reproducible from Mongo alone
— they were built from the Yelp parquet — so copy `chroma_db/` aside first if
that history matters. Two orphaned segment directories from earlier rebuilds
(~25MB) are also on disk and are not referenced by the `segments` table.

> **Still not done.** The shipped index carries both collections; the count in
> `/health` is `places_v2`'s alone, so this is invisible from the outside. Doing
> it now costs one 553MB layer republish — the shrink changes the index layer's
> inputs, so that deploy pushes the whole thing — and every deploy after it
> caches a smaller one. Worth doing on the next rebuild rather than as its own
> deploy.

### Verify after deploying

Start with the index, because an image that shipped without one is the single
most likely thing to be wrong and the only failure that looks like success:

```bash
curl -s "$SERVICE_URL/health"
```

`{"status":"ok","count":15211}`. **The number is the assertion** — a materially
different count means the build context did not carry the index you expected,
and the fix is to rebuild and redeploy rather than to debug the app. A 503 with
`"status":"empty"` means `chroma_db/` never reached the image; check
`.gcloudignore` still exists.

Then the resolved paths and the guard state, which need the token:

```bash
curl -s -H "x-recommender-token: $RECOMMENDER_TOKEN" "$SERVICE_URL/health"
```

`"writeGuard": "enforced"` is the one to read. Confirm it from the other side
too — an unauthenticated write must be refused:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST "$SERVICE_URL/index/missing" -H 'Content-Type: application/json' -d '{"businessIds":["x"]}'
```

**401 is the correct answer.** A 200 means `RECOMMENDER_TOKEN` did not reach the
service, and the startup log will contain the `RECOMMENDER_TOKEN is not set`
warning to confirm it.

Health proves vectors exist; it does not prove ranking works. Finish with a real
query:

```bash
curl -s -X POST "$SERVICE_URL/recommend" -H 'Content-Type: application/json' -d '{"query":"A Italian, Pizza restaurant","k":3}'
```

**A 200 with an empty array is a failure, not a pass.** Finally, set
`RECOMMENDER_URL` in Vercel to `$SERVICE_URL` and `RECOMMENDER_TOKEN` to the same
value — see the app's README — then exercise both callers, the dashboard's
`/nearby` and a group shortlist.

> `/health` is also the right target for an external uptime check. It is cheap,
> needs no token, and returns 200 only once the index is genuinely readable — so
> a monitor pointed at it catches a bad rebuild instead of letting the service
> quietly serve zero results.

---

## Other hosts evaluated

The `deploy/` directory holds two complete, unused migration paths. They exist
because when the Artifact Registry bill appeared the obvious reading was *"Cloud
Run is expensive, move"* — and both alternatives were built out far enough to be
deployable before the actual cause turned out to be
[one `COPY` line](#what-the-first-deploys-cost).

**Cloud Run stayed, because the problem was never the host.** A ~$16/month run
rate from republishing the same 553MB layer is an artifact-storage bug with a
two-line fix; it is not a reason to take on OS administration or a 48-hour sleep
timer. Fixing the cause was cheaper than either migration, in money and in
attention.

They are kept rather than deleted because they are genuinely useful — as a
costed comparison, as a fallback if this service outgrows scale-to-zero, and
because working through the bind-mount case is what produced
[`GET /health`](#get-health), which turned out to be worth having here too.

| | **Cloud Run** (live) | `deploy/oracle/` | `deploy/hf-space/` |
|---|---|---|---|
| Host | Google Cloud Run | Oracle Ampere A1, Always Free | HF Spaces, free CPU Basic |
| Arch | x86 | **ARM64** (`Dockerfile.arm64`) | x86 |
| Idle | Scales to zero, cold start | Always on | **Sleeps after 48h**, 30–90s wake |
| `chroma_db/` | In the image | **Bind-mounted** from disk | In the image, via Git LFS |
| `/index/missing` writes | Vanish on scale-down | **Persist** | Vanish on restart |
| TLS | Provided | Caddy + Let's Encrypt (sslip.io) | Provided |
| Port | `$PORT`, injected | 8080 behind Caddy | **7860, hardcoded** |
| Runs as | root | root | **UID 1000** |
| Cost | ~$0 once layers are split | $0 | $0 |
| You administer | Nothing | OS, firewall, certs, restarts | Nothing |

Each has a full walkthrough in its own directory — `deploy/oracle/README.md` and
`deploy/hf-space/SETUP.md` — including the parts that bite:

- **Oracle:** Always Free A1 capacity is heavily contested and *"Out of host
  capacity"* is the normal first response (`retry-launch.sh` exists for that).
  Oracle also **reclaims** Always Free compute that sits under ~20% utilisation
  across a 7-day window, which a service answering occasional queries can trip.
  And Oracle's Ubuntu images ship iptables rules that REJECT inbound traffic
  regardless of what the cloud Security List allows — the symptom is a
  connection that hangs rather than refuses.
- **HF Spaces:** the 48-hour sleep is survivable only with a scheduled ping
  against `/health` (UptimeRobot or cron-job.org — *not* GitHub Actions cron,
  which is auto-disabled after 60 days of repository inactivity, precisely when
  you have stopped watching). The 553MB index has to travel through Git LFS
  against a free-account storage quota, which may rule the host out before
  anything else does.
- **Both:** neither has a static egress IP, so both require opening MongoDB
  Atlas to `0.0.0.0/0`. Cloud Run's egress is not stable either — but it *can*
  be pinned, with a VPC connector and Cloud NAT behind one allowlisted address,
  and that option is what these two give up permanently. If you take either
  path, do two things first: rotate the Atlas password, and switch to a
  **read-only** database user.

  The read-only part is verified rather than assumed. `service.py` touches Mongo
  in exactly one place — a single `col.find(...)` — and nowhere inserts, updates
  or deletes; `POST /index/missing` writes to *Chroma*, not to Mongo. So `read`
  on the one database is sufficient, and a leaked connection string then cannot
  modify or destroy anything. Worth re-checking with a grep if the service ever
  grows a write path, because nothing enforces this but the code itself.

Nothing on the Cloud Run path was deleted to make room for these. The root
`Dockerfile`, `.dockerignore` and `.gcloudignore` are the live deploy;
`Dockerfile.arm64` carries its own
`Dockerfile.arm64.dockerignore` via BuildKit's per-Dockerfile context, so the two
do not interfere.

---

## Files

| file | purpose |
|---|---|
| `service.py` | FastAPI API. Reads for `/recommend*`, liveness for `/health`; tops up vectors via `/index/missing`. |
| `rebuild_index.py` | Builds `places_v2` from Mongo. Owns `build_text` — the one template. |
| `Dockerfile` | **The live deploy** (Cloud Run, x86, index baked in). Every non-obvious line has the reasoning inline — including why the index and the source are separate `COPY` layers. |
| `requirements.txt` | Pinned to what the working `.venv` has, so a deploy reproduces the environment the index was tested against. |
| `.dockerignore` / `.gcloudignore` | What reaches the builder. `.gcloudignore` exists to override `.gitignore` — see above. |
| `env.yaml.example` | Template for the Cloud Run `--env-vars-file`. |
| `Dockerfile.arm64` | Unused. ARM build with the index **bind-mounted** instead of baked, for a host with a disk. Pairs with `Dockerfile.arm64.dockerignore` (BuildKit per-Dockerfile context, so the Cloud Run one is untouched). |
| `deploy/oracle/` | Unused. Oracle Ampere A1 walkthrough: `docker-compose.yml`, `Caddyfile`, `retry-launch.sh` for capacity retries. See [Other hosts evaluated](#other-hosts-evaluated). |
| `deploy/hf-space/` | Unused. Hugging Face Spaces walkthrough: its own `Dockerfile` (port 7860, UID 1000), `SETUP.md`, and the Space's `README.md` frontmatter. |
| `backfill_source.py` | One-time: stamps every restaurant with its provenance. |
| `experiment_template.py` | Offline comparison of embedding text templates. |
| `export_seed.py` | One-time: Yelp parquet → `palate/scripts/data/restaurants.seed.json`. |
| `save_clean_data.py` | One-time: raw Yelp JSON → cleaned parquet under `data/`. |
| `main.py` | PyCharm scaffold. Unused, safe to delete. |
| `*.ipynb` | Exploratory notebooks from the original EDA. |

---

## Rebuilding the index

```bash
./.venv/Scripts/python.exe rebuild_index.py --limit 200      # smoke test
./.venv/Scripts/python.exe rebuild_index.py --only-missing   # top-up after a sync
./.venv/Scripts/python.exe rebuild_index.py --recreate       # full rebuild
```

`--only-missing` adds restaurants that have no vector yet and exits before even
loading the model if there are none. A change to `build_text` needs
`--recreate`, since existing vectors would otherwise be left on the old template.

`rebuild_index.py` honours `CHROMA_DIR` too, so a rebuild aimed at a mounted
volume works the same way. **The rebuild and the service must agree on that
path** or the rebuild writes an index nothing reads.

**Restart the service after any rebuild.** Chroma's `PersistentClient` caches
in-process, so a running service cannot see a collection another process
dropped and rebuilt. On Cloud Run the equivalent is redeploying, because the
index is in the image.

---

## The index

`places_v2` — **15,211 vectors** as of the last rebuild plus runtime top-ups,
one per restaurant, dim 384, **cosine** space, all unit length, ids are `fsqId`,
`business_id` in metadata on every vector (which is what makes candidate
filtering work). Covers Mongo's `{ source: "foursquare" }` population.

Model: `sentence-transformers/all-MiniLM-L6-v2`, **256-token** window.

The count drifts upward between rebuilds because `/index/missing` adds newly
synced places — locally, where those writes persist. In the Cloud Run image it
does not drift, for the reason in
[The index ships inside the image](#the-index-ships-inside-the-image).

A dead `langchain` collection is still on disk with 20,613 vectors in L2 space,
mixing Yelp review text with Foursquare template sentences — **and it shipped**,
since the deployed image is a copy of this directory. Nothing reads it; it is
pure cold-start weight. Drop it on the next rebuild — see
[Shrink the image first](#shrink-the-image-first).

---

## Design decisions, and the measurements behind them

These are the non-obvious ones. Each produced plausible-looking output while
being wrong, so none of them surfaced without deliberate measurement. The corpus
was 15,169 restaurants when these ran.

### Filter, then rank

`nearby/route.ts` used to take the 50 nearest restaurants from Mongo, separately
take the top-50 from Chroma across the whole corpus, and intersect. Measured
overlap: **0**, at k=50 and k=200; 1 at k=1000. Two independent top-50 draws
from a 15,169-item corpus are expected to share 0.16 items. Because of an
`if (ranked.length > 0)` guard the route silently fell back every time — the
recommender had never affected the dashboard.

**Never intersect two independently-retrieved top-K lists.** Retrieval is a
filter; ranking is an ordering. Ranking must also never *drop* candidates: rows
with no vector are appended in distance order rather than disappearing.

### The embedded text excludes the restaurant's name

Three templates, 15,169 restaurants × 5 cuisine probes, top-20:

| template | category@20 ↑ | nameleak@20 ↓ |
|---|---|---|
| control (name + category) | 74% | 88% |
| cat_x3 (category repeated) | 78% | 87% |
| **no_name** | **95%** | **41%** |

Worst control case: `"pizza and pasta"` scored **5%** category precision with
100% name leakage — 20 results with a pizza word in the *name*, one of which
was actually a pizza place.

Repeating the category to outweigh the name barely helped. A transformer is not
bag-of-words: duplicating a token does not linearly raise its weight, and a
distinctive proper noun keeps winning. Removing it is what works.

Finding a place *by name* is a different job, and belongs in lexical search over
Mongo.

**Known cost:** without names, restaurants sharing categories and locality
produce identical text and therefore identical vectors — 15,169 places collapse
to 9,877 distinct ones, largest group 453 reading `"A Indian Restaurant."` Ties
break on Chroma's arbitrary insertion order. The fix is enrichment and
structured tie-breaking, not putting the name back.

### Group aggregation: least misery, not the average

For unit vectors in cosine space, `cos(centroid, d) == mean_i cos(v_i, d)` over
a constant — so **centroid ranking *is* mean ranking**, identically. Verified:
the `centroid` and `mean` strategies return the same top pick and the same
worst-member rank in every test.

That makes the original centroid approach the *additive utilitarian* social
choice function, adopted by accident because `similarity_search` takes exactly
one vector. Averaging is known to fail on polarized groups.

Once the candidate set is bounded, the whole N-members × M-candidates matrix
fits in microseconds and any aggregation is affordable. Measured — **worst-member
rank of the group's #1 pick**, lower is better, over 67 NYC candidates:

| group | centroid / mean | min / blend |
|---|---|---|
| vegan + steakhouse | 18 | **1** |
| vegan + steak + Sichuan | 64 | **12** |
| two Indian queries (aligned) | 17 | 9–10 |

The first row is the argument in two restaurant names: mean picked
**Sweetgreen** (a salad chain — the vegan's answer, nothing for the steak fan);
min picked **Superiority Burger** (a vegetarian burger restaurant) — the
compromise a thoughtful human would name.

Rows are z-scored per member first, or the member whose taste vector sits
furthest from the corpus centre sets `min` for every candidate — looking like
the pickiest person purely because of where their vector sits. Caveat: this
optimizes *relative* satisfaction, so raw worst-member cosine can dip slightly
for already-aligned groups even as rank improves.

`blend` defaults to α=0.5 because pure `min` is brittle here: one outlier member
swings everything, and vector collisions make minima tie constantly.

### `agreement` over `consensus`

`consensus` is `‖mean of unit member vectors‖`. `agreement` inverts it in closed
form to the mean pairwise cosine:

```
agreement = (n · consensus² − 1) / (n − 1)
```

Verified against ground-truth pairwise cosines to machine epsilon, N=2 and N=3.
Two reasons to prefer it: ~3.2× the resolution in the 0.2–0.4 band where real
sentence pairs land (the square root is very flat there), and it is comparable
across group sizes — `consensus` for unrelated members floors near `1/√N`, so
the same value means "divided" for a pair and "aligned" for six.

### One template, not one writer

A `/embed` endpoint used to let callers push their own text. That made two
templates — the caller's kept the restaurant name, embedding at 74% category
precision against this index's 95% — and wrote vectors Mongo could not
reproduce, so any `--recreate` silently changed them. Removed.

The rule it was replaced with is narrower than "only one process writes":
`/index/missing` writes too. What it does not do is accept text. It takes
`businessIds`, reads those rows from Mongo, and calls the same `build_text`
`rebuild_index.py` calls — by importing it, so the two cannot drift. **Callers
choose which rows to index, never what to embed.**

---

## Known limitations

- **Text is thin.** p50 is ~10 tokens per restaurant. The index can currently
  distinguish cuisines and little else — it cannot tell a great Italian
  restaurant from a mediocre one, because nothing in the vector encodes
  quality, atmosphere, or price. Palate's post-meal reviews now write back into
  `tips[]` and call `/index/missing` with `force`, so this improves as the app
  is used; backfilling Foursquare tips would improve it faster.
- **48% of restaurants share a vector** with at least one other. See above.
- **Runtime index writes do not survive on Cloud Run.** They land in the
  container's ephemeral layer. See
  [The index ships inside the image](#the-index-ships-inside-the-image).
- **No per-person taste vectors yet.** `/recommend/group` accepts them via
  `vectors`; nothing builds them. Palate sends one short sentence per member,
  assembled by `lib/tasteQuery.ts` from their stated cuisines, diet, and now
  their recently well-rated visits.
- **No exclusion filter.** Palate used to intersect a `disliked` cuisine list
  out of the candidates before ranking; that field is no longer tracked, so
  group shortlists are shaped by attraction alone.
- **No evaluation harness.** Every number above came from a throwaway script.
  Promoting them into a repeatable suite with a fixed probe set is what would
  make "did that change help?" answerable.
- **`/health` catches an empty index, not a stale one.** It reports whether
  `places_v2` is readable and non-empty, which closes the failure it was written
  for. It cannot tell you the index is three cities and two months behind Mongo —
  the count would be perfectly healthy and the answers quietly wrong. Comparing
  `count` against `db.restaurants.countDocuments({source: "foursquare"})` would
  make that visible; nothing does it today.

  (The narrower readiness gap *is* closed, and by accident: `EMBED_DIM` is
  computed at module scope with a live `embed_query` call, so the model is fully
  loaded before the process can accept a connection at all. A container that
  answers anything has already paid for the model.)

---

## Gotchas

- **Import `pyarrow` before torch/transformers.** On Windows they bundle
  separate native OpenMP runtimes; if pyarrow's DLL loads second the process
  dies with a native access violation (`0xC0000005`) and uvicorn never starts.
  Every script here does this first — keep it that way.
- **`SentenceTransformer.max_seq_length` is the real limit** (256 here).
  `AutoTokenizer.from_pretrained` reports `model_max_length: 512` for the same
  checkpoint and is wrong about what actually truncates — and truncation is
  silent, never an exception.
- **Chroma's distance metric is fixed at collection creation.** Changing it
  means a new collection.
- **Chroma caches per process.** Cross-process writes are invisible until a
  fresh process opens the store. This is also why the container runs a single
  uvicorn worker: N workers each load their own copy of the model, costing N×
  the memory for no throughput gain on a small instance, and a write through one
  is invisible to the others. Scale with instances, not workers.
- **Cloud Run assigns the port at runtime.** Hardcoding one means the health
  check never passes and the revision rolls back with a "failed to start and
  listen" error that says nothing about ports. The `CMD` uses the shell form so
  `${PORT}` expands, and `exec` so uvicorn becomes PID 1 and receives `SIGTERM`
  directly — without it the shell holds PID 1, swallows the signal, and every
  shutdown waits out the grace period.
- **`slim`, not `alpine`.** torch and pyarrow ship manylinux wheels that need
  glibc; on musl pip falls back to building from source, which does not finish.

---

## Historical: the Yelp dataset

`save_clean_data.py` and `export_seed.py` built the original 5,444-restaurant
seed from the Yelp Open Dataset — real review text, reshaped so Yelp's
`business_id` lives in the `fsqId` field (deliberate stopgap, not a real
Foursquare id) and stars are doubled onto the schema's 0–10 scale. Those rows
are Mongo's `source: "yelp_seed"` population and are **excluded** from
`places_v2`.

They remain the only rich-text corpus available, which makes them useful as a
development fixture — the only way to test that centroid maths and group
aggregation behave before the live data is dense enough to tell a working
implementation from a broken one.

The dataset itself is not in this repo. Download via
[kagglehub](https://github.com/Kaggle/kagglehub) or the
[Yelp dataset page](https://www.yelp.com/dataset).
