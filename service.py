"""FastAPI service for the Palate recommender.

Text comes from MongoDB and is turned into vectors by exactly one function,
rebuild_index.build_text. That is the invariant — not "the service never
writes". An older /embed endpoint let CALLERS push their own sentence, which
made a second template that kept the restaurant name and embedded at 74%
category precision against this one's 95%; it was removed for that reason.
/index/missing below writes, but it reads the text from Mongo and runs the same
build_text -> chunk -> pool pipeline rebuild_index.py runs, so there is still
one template. Callers choose WHICH rows to index, never WHAT to embed.

Three endpoints:
    POST /recommend        one text query  -> ranked businessIds
    POST /recommend/group  N members       -> ranked businessIds for a group
    POST /index/missing    top up vectors for rows Mongo has and Chroma lacks

Both take an optional `candidateIds`, and passing it is strongly preferred:
retrieval should be a FILTER (the caller decides what is reachable — nearby,
open, in budget) and the vector search should only ORDER what survives.
Retrieving top-K independently and intersecting afterwards does not work; two
50-item draws from a 15,169-item corpus are expected to share 0.16 items, and
the measured overlap was zero.

Development, from this directory (CHROMA_DIR defaults to ./chroma_db, and
mongo_url is picked up from ../palate/.env):
    ./.venv/Scripts/python.exe -m uvicorn service:app --port 8000

Production: every path and secret comes from the environment — see .env.example
and the Dockerfile. Nothing here assumes a working directory or a sibling
checkout any more. Set RECOMMENDER_TOKEN, or POST /index/missing is open.

Restart it after any rebuild_index.py run — Chroma's PersistentClient caches
in-process, so a collection dropped and rebuilt by another process is invisible
to a service that is already running. This is also why the container runs a
single worker; see the Dockerfile.
"""
# Preload pyarrow BEFORE torch/transformers get pulled in below. On Windows
# these packages each bundle their own native OpenMP runtime; if pyarrow's DLL
# loads *after* torch's, the process dies with a native access violation
# (exit code 0xC0000005) and uvicorn never starts. Importing pyarrow first
# makes it win the DLL load and avoids the crash.
import pyarrow  # noqa: F401
import pyarrow.dataset  # noqa: F401

import os

import secrets
import sys

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# The indexing pipeline, imported rather than reimplemented — this is what keeps
# "one text template" true now that this process can write. rebuild_index.py
# guards main() behind __name__, so importing it costs only its module-level
# constants.
from rebuild_index import build_text, chunk, pool

"""Configuration, all of it, read once at import.

Everything here used to be a literal or a relative path, which worked because
the service was only ever started by hand from this directory with the Next app
checked out as a sibling. None of those assumptions survive a deploy: there is
no ../palate, the working directory is whatever the process manager chose, and
the vector index is on a mounted volume rather than under the source tree.

load_dotenv is now a DEVELOPMENT CONVENIENCE ONLY, and the order matters.
os.environ wins, because python-dotenv does not override an existing variable by
default — so a real environment (Docker, Fly, Render, systemd) is authoritative
and the sibling .env is a fallback for a laptop. The path is also no longer
hardcoded: RECOMMENDER_ENV_FILE overrides it, and a missing file is not an
error, which is exactly what should happen in production.
"""
load_dotenv(os.environ.get("RECOMMENDER_ENV_FILE", os.path.join("..", "palate", ".env")))

# Where Chroma's SQLite lives. Relative paths resolve against the working
# directory, which is why this must be settable: a container mounts the index as
# a volume at an absolute path, and 553MB is not going in the image.
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")

"""Shared secret for the write endpoint.

/index/missing reads Mongo and runs a sentence-transformer over up to
MAX_INDEX_BATCH rows inside the request. Unauthenticated on a public host that
is both a way to run up CPU indefinitely and a way to keep a worker busy while
real searches queue behind it.

Unset means unauthenticated, which is right on a laptop and checked loudly at
startup below. The read endpoints stay open: they are cheap, they leak nothing
a user of the app cannot already see, and the Next app calls them on behalf of
signed-out visitors.
"""
API_TOKEN = os.environ.get("RECOMMENDER_TOKEN", "").strip()

app = FastAPI()

# normalize_embeddings makes query vectors unit length, matching what
# rebuild_index.py stored. Both the centroid maths and cosine distance assume it.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# places_v2: one normalized vector per restaurant, cosine space, built from
# Mongo. The old "langchain" collection mixed 5,444 Yelp review-text docs with
# 15,169 Foursquare template sentences in L2 space — two populations that were
# never comparable. It is still on disk and can be deleted once nothing reads it.
db = Chroma(collection_name="places_v2", persist_directory=CHROMA_DIR,
            embedding_function=embeddings)

# Probed rather than hardcoded to 384, so swapping the model cannot silently
# invalidate the dimension check in /recommend/group.
EMBED_DIM = len(embeddings.embed_query("dimension probe"))

# Chroma builds one $in predicate per id, so a huge candidate list turns the
# filter into the slow path. Callers should pre-narrow (geo, open now, price).
MAX_CANDIDATES = 500

# Aggregation functions available to /recommend/group. See _aggregate().
STRATEGIES = ("blend", "min", "mean", "centroid")

# Ceiling on one /index/missing call. Encoding is the slow part and this runs
# inside a request, so a caller that has just synced 50 places gets them indexed
# immediately while an accidental corpus-wide call is truncated rather than
# holding a worker for minutes. A full build is still rebuild_index.py's job.
MAX_INDEX_BATCH = 500

if not API_TOKEN:
    # stderr and flush=True, not a bare print. Python block-buffers stdout when
    # it is not a TTY, which is every container, so a plain print can sit in the
    # buffer indefinitely — and a security warning that appears an hour late, or
    # not at all because the process was killed first, is not a warning. uvicorn
    # logs to stderr too, so this lands in order with the rest of the boot.
    print(
        # Plain ASCII deliberately. Comments in this file use em-dashes freely,
        # but this string is emitted to a log that may be read through a console
        # or aggregator in any codepage, and a security warning should not be the
        # place a mojibake character makes someone doubt what they are reading.
        "[recommender] RECOMMENDER_TOKEN is not set - POST /index/missing is "
        "UNAUTHENTICATED. Fine on a laptop; on a reachable host anyone can drive "
        "the embedding model. Set it, and send it as x-recommender-token.",
        file=sys.stderr,
        flush=True,
    )


def require_token(x_recommender_token: str = Header(default="")) -> None:
    """Guard the write endpoint with a shared secret.

    A shared secret rather than anything richer because the only caller is the
    Next server, machine to machine. There is no user identity to model here and
    no third party to federate with, so a signature scheme would be ceremony
    around the same one secret.

    secrets.compare_digest, not ==. String equality in CPython short-circuits on
    the first differing byte, which leaks the length of the matching prefix
    through response timing and makes the token recoverable byte by byte over
    enough requests. This is the same reasoning as the constant-time bcrypt
    compare in the app's login route.

    Raises:
        HTTPException: 401 if the header is missing or wrong.
    """
    # Unset means the guard is disabled — see the startup warning above.
    if not API_TOKEN:
        return
    if not secrets.compare_digest(x_recommender_token, API_TOKEN):
        raise HTTPException(401, "Missing or invalid x-recommender-token")


_mongo = None


def _restaurants():
    """The Mongo `restaurants` collection, connected on first use.

    Lazy on purpose: the read endpoints do not need Mongo, and connecting at
    import time would mean an unreachable database stops the recommender from
    serving searches it is perfectly capable of serving.

    Returns:
        A pymongo Collection.

    Raises:
        HTTPException: 503 if mongo_url is not configured.
    """
    global _mongo
    if _mongo is None:
        url = (os.environ.get("mongo_url") or "").strip()
        if not url:
            raise HTTPException(503, "mongo_url not configured; cannot index")
        # The URI usually carries no database name and pymongo will not guess,
        # so name it explicitly. "test" is the historical default because that
        # is where the app's data actually landed; set MONGO_DB to override.
        _mongo = MongoClient(url)[os.environ.get("MONGO_DB", "test")]
    return _mongo.restaurants


def _candidate_filter(ids: list[str] | None) -> dict | None:
    """Build the Chroma metadata filter that confines a search to `ids`.

    Args:
        ids: restaurant fsqIds the caller considers reachable. Empty or None
            means "search the whole corpus".

    Returns:
        A Chroma `where` clause, or None for an unfiltered search.

    Raises:
        HTTPException: 400 if more than MAX_CANDIDATES ids were supplied.
    """
    if not ids:
        return None
    if len(ids) > MAX_CANDIDATES:
        raise HTTPException(
            400, f"{len(ids)} candidateIds exceeds MAX_CANDIDATES={MAX_CANDIDATES}"
        )
    return {"business_id": {"$in": ids}}


def _business_id(doc) -> str:
    """Recover a restaurant's fsqId from a LangChain Document.

    Every vector in places_v2 carries `business_id` in metadata, so the
    fallback only matters for the legacy "langchain" collection, whose
    bulk-loaded docs stored the id as the first token of page_content.

    Args:
        doc: a langchain_core Document returned by a similarity search.

    Returns:
        The restaurant's fsqId.
    """
    return doc.metadata.get("business_id") or doc.page_content.split(" ", 1)[0]


def _l2_normalize(rows: np.ndarray) -> np.ndarray:
    """Scale each row of a matrix to unit length.

    All-zero rows are left as zeros instead of becoming NaN — one NaN in a
    384-dim vector propagates through every mean, norm and distance downstream
    and produces an empty result with nothing pointing at the cause.

    Args:
        rows: (N, D) matrix of vectors.

    Returns:
        (N, D) matrix where every non-zero row has norm 1.
    """
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)


def _candidate_matrix(ids: list[str]) -> tuple[list[str], np.ndarray]:
    """Fetch stored vectors for specific restaurants.

    Not every candidate has a vector: places_v2 covers `source: "foursquare"`
    only, so a nearby query that catches a yelp_seed row asks for an id that
    was never indexed. Chroma silently omits those, which is why the returned
    id list — not the input — is the authority on what actually got scored.

    Args:
        ids: restaurant fsqIds to look up.

    Returns:
        (found_ids, matrix) where matrix is (len(found_ids), EMBED_DIM) and the
        two are index-aligned. Both are empty if nothing was found.
    """
    got = db.get(ids=ids, include=["embeddings"])
    found = list(got.get("ids") or [])
    if not found:
        return [], np.zeros((0, EMBED_DIM), dtype=np.float32)
    return found, np.asarray(got["embeddings"], dtype=np.float32)


def _zscore_rows(S: np.ndarray) -> np.ndarray:
    """Put every group member on a comparable scale before aggregating.

    Members' similarity distributions differ systematically: a taste vector
    sitting near the corpus centre scores uniformly high against everything,
    a sharp or unusual one scores uniformly low. Aggregate the raw numbers and
    the sharp-tasted member sets `min` for every candidate — they look like the
    pickiest person in the group purely because of where their vector sits,
    not because of anything they want.

    Zero-variance rows (a single candidate, or a member tied against all of
    them — 48% of restaurants share a vector) collapse to zeros, not NaN.

    Args:
        S: (N members, M candidates) similarity matrix.

    Returns:
        Same shape, each row standardized to mean 0 / std 1.
    """
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True)
    return np.divide(S - mu, sd, out=np.zeros_like(S), where=sd > 1e-9)


def _aggregate(S: np.ndarray, strategy: str, alpha: float,
               weights: np.ndarray) -> np.ndarray:
    """Collapse an (N members x M candidates) matrix into one score per candidate.

    These are social choice functions, and picking one is a product decision:

    mean
        Additive utilitarian — maximizes total happiness. For unit vectors in
        cosine space this is provably the SAME RANKING as the centroid, since
        cos(centroid, d) equals mean_i cos(v_i, d) divided by a constant. So
        this is what the original centroid code was doing, and it is why a
        vegan plus a steakhouse fan received five results that appeared in
        neither person's top five.
    min
        Least misery — maximizes the worst-off member. The retrieval-side twin
        of the vote's "tap everything you'd happily eat at tonight". Measured:
        moved the winner from rank 18 to rank 1 in the least-happy member's
        own ordering.
    blend
        alpha*mean + (1-alpha)*min, the default. Pure min is brittle here — a
        single outlier member swings every score, and heavy vector collisions
        make minima tie constantly.

    Args:
        S: similarity matrix, rows already z-scored by _zscore_rows.
        strategy: "mean", "min", or anything else for blend.
        alpha: blend weight on the utilitarian half, 0..1.
        weights: per-member weights summing to 1.

    Returns:
        (M,) score per candidate, higher is better.
    """
    if strategy == "min":
        return S.min(axis=0)
    if strategy == "mean":
        return np.average(S, axis=0, weights=weights)
    # Weights apply to the utilitarian half only; a minimum has no weighting.
    return alpha * np.average(S, axis=0, weights=weights) + (1 - alpha) * S.min(axis=0)


def _dedup(hits) -> list[dict]:
    """Turn LangChain (Document, distance) pairs into result dicts, deduped.

    Deduplication is cheap insurance — places_v2 holds one vector per
    restaurant, but the legacy collection did not.

    Args:
        hits: [(Document, distance)] from a similarity search, best first.

    Returns:
        [{businessId, score, distance}] in the same order. `score` is
        higher-is-better on every strategy so callers never branch on which
        one produced the result; places_v2 is cosine space, so 1 - distance is
        exactly the cosine similarity.
    """
    seen, out = set(), []
    for doc, distance in hits:
        bid = _business_id(doc)
        if bid in seen:
            continue
        seen.add(bid)
        out.append({
            "businessId": bid,
            "score": 1.0 - float(distance),
            "distance": float(distance),
        })
    return out


class RecReq(BaseModel):
    """Request body for POST /recommend.

    Attributes:
        query: free text describing what is wanted, e.g. "A Italian, Pizza
            restaurant". nearby/route.ts builds this from the user's
            likedCuisines and diet.
        k: how many results to return.
        candidateIds: restrict ranking to these restaurants. Omit for a
            corpus-wide search; pass the geo-feasible set in production.
    """
    query: str
    k: int = 5
    candidateIds: list[str] = []


class GroupRecReq(BaseModel):
    """Request body for POST /recommend/group.

    Members can arrive as text or as pre-computed vectors, and the two are
    concatenated in that order — queries first, then vectors — which is the
    order `weights` must follow. Accepting both is deliberate: today callers
    send preference text, and when per-person taste vectors exist they can be
    sent instead with no change to this endpoint.

    Attributes:
        queries: one free-text description per member.
        vectors: one pre-computed EMBED_DIM vector per member.
        weights: per-member influence, defaults to equal. Applies to the
            utilitarian half of the score only.
        k: how many results to return.
        candidateIds: the geo-feasible restaurant set. Required for every
            strategy except "centroid".
        strategy: one of STRATEGIES. See _aggregate().
        alpha: blend weight, 0..1, used only when strategy is "blend".
        debug: also return each member's own top 5, which reveals whose taste
            the group result actually reflects.
    """
    queries: list[str] = []
    vectors: list[list[float]] = []
    weights: list[float] | None = None
    k: int = 20
    candidateIds: list[str] = []
    strategy: str = "blend"
    alpha: float = 0.5
    debug: bool = False


@app.get("/health")
def health(response: Response, x_recommender_token: str = Header(default="")):
    """Report whether this process can actually answer a search.

    Added for the move to a host where the index arrives as a bind mount rather
    than inside the image. Chroma treats a missing or empty persist_directory as
    a NEW EMPTY COLLECTION rather than an error, and `db` is built at import
    time, so a wrong mount produces a process that starts cleanly, serves /docs,
    passes any liveness probe, and returns nothing for every query. Until this
    endpoint there was no GET route to notice that through.

    The count is the whole point. Returning 200 with `count: 0` would put a
    green tick on a broken deploy, so an empty collection is a 503 — which
    surfaces in `docker ps` as (unhealthy) and to an uptime check, without
    causing a restart loop, since Docker does not act on health status by
    default.

    Deliberately does NOT touch Mongo. _restaurants() is lazy precisely so an
    unreachable database cannot stop the read endpoints from serving searches
    they are perfectly capable of serving; failing health on it would undo that
    and take the service down for an outage it can ride out.

    The detail fields are token-gated. `status` and `count` stay public because
    the container healthcheck and any external probe need them, but the resolved
    filesystem path and whether the write guard is active are not things a public
    endpoint should volunteer. Same constant-time compare as require_token, for
    the same reason.

    Returns (JSON):
        status: "ok" | "empty" | "error".
        count: vectors in places_v2, or null if the collection could not be
            read. The number that matters.
        collection, chromaDir, embedDim: token-only. What THIS process actually
            resolved, for comparing against the mount when count is wrong.
        writeGuard: token-only. Whether RECOMMENDER_TOKEN is enforced, so a
            deploy that lost the secret is visible here rather than by
            discovering /index/missing is open.
    """
    body: dict = {}

    # Unset token means the guard is disabled everywhere else too, so there is
    # no secret to check against and nothing to gate on — see require_token.
    if API_TOKEN and secrets.compare_digest(x_recommender_token, API_TOKEN):
        body = {
            "collection": "places_v2",
            # The RESOLVED path, not the env var. A relative CHROMA_DIR against
            # an unexpected working directory is a normal way to end up with an
            # empty index, and only the absolute form shows that.
            "chromaDir": os.path.abspath(CHROMA_DIR),
            "embedDim": EMBED_DIM,
            "writeGuard": "enforced" if API_TOKEN else "DISABLED",
        }

    try:
        # db._collection, matching how /index/missing already reaches past the
        # LangChain wrapper. Chroma exposes no count through it.
        count = db._collection.count()
    except Exception as exc:
        # Unreadable SQLite, a permissions problem on the mount, a corrupt
        # segment. Report the exception CLASS rather than str(exc): this is a
        # public endpoint and the message routinely contains absolute paths.
        response.status_code = 503
        return {**body, "status": "error", "count": None, "error": type(exc).__name__}

    if count == 0:
        response.status_code = 503
        return {**body, "status": "empty", "count": 0}

    return {**body, "status": "ok", "count": count}


@app.post("/recommend/group")
def recommend_group(req: GroupRecReq):
    """Rank restaurants for a group of people.

    Embeds any text members, normalizes every member to unit length, then
    either searches on the group centroid (strategy "centroid") or scores the
    full member x candidate similarity matrix and aggregates it.

    Normalizing before combining matters: without it whichever member's text
    happened to produce a larger vector would silently get a louder vote.

    Returns (JSON):
        businessIds: ranked fsqIds, best first.
        results: [{businessId, score, ...}]. Non-centroid strategies also
            include `memberSims` (each member's raw cosine to that candidate,
            for explaining a pick in the UI) and `worstMemberSim`.
        strategy: echoed back.
        agreement: mean pairwise cosine between members, derived in closed form
            from `consensus` via (n*c^2 - 1)/(n - 1). Preferred over consensus:
            ~3x the resolution in the 0.2-0.4 band where real sentence pairs
            land, and comparable across group sizes, whereas consensus floors
            near 1/sqrt(N) so the same value means "divided" for a pair and
            "aligned" for six.
        consensus: ||mean of the unit member vectors||, in [0, 1].
        memberCount: N.
        perMember: only when debug is set.

    Raises:
        HTTPException: 400 for no members, wrong vector dimension, mismatched
            or negative weights, unknown strategy, alpha out of range, or a
            non-centroid strategy without candidateIds. 404 if none of the
            candidateIds are indexed. 422 if the members' vectors cancel out,
            which would make the centroid direction meaningless.

    Note:
        `score` is ordinal within one response. Centroid scores are cosine
        similarities in [-1, 1]; blend/min scores are built from z-scores and
        can exceed 1. Fine for ranking, wrong for thresholding.
    """
    members: list[list[float]] = []
    if req.queries:
        members.extend(embeddings.embed_documents(req.queries))
    members.extend(req.vectors)

    if not members:
        raise HTTPException(400, "Provide at least one of `queries` or `vectors`")
    for i, v in enumerate(members):
        if len(v) != EMBED_DIM:
            raise HTTPException(400, f"member {i} has dim {len(v)}, expected {EMBED_DIM}")

    if req.weights is None:
        weights = np.ones(len(members), dtype=np.float32)
    else:
        if len(req.weights) != len(members):
            raise HTTPException(
                400, f"{len(req.weights)} weights for {len(members)} members"
            )
        weights = np.asarray(req.weights, dtype=np.float32)
        if (weights < 0).any() or weights.sum() <= 0:
            raise HTTPException(400, "weights must be non-negative and sum above zero")
    weights = weights / weights.sum()

    raw = np.asarray(members, dtype=np.float32)
    units = _l2_normalize(raw)
    centroid = (units * weights[:, None]).sum(axis=0)

    # ||mean of unit vectors|| is 1 when everyone agrees and tends to 0 as they
    # cancel out, so it is a free measure of how aligned the group is. Capture
    # it before normalizing, because dividing by it destroys the information.
    consensus = float(np.linalg.norm(centroid))
    if consensus < 1e-6:
        raise HTTPException(422, "Member tastes cancel out; the centroid is meaningless")
    query_vec = centroid / consensus

    n = len(members)
    agreement = (n * consensus**2 - 1) / (n - 1) if n > 1 else 1.0

    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"strategy must be one of {list(STRATEGIES)}")
    if not 0.0 <= req.alpha <= 1.0:
        raise HTTPException(400, "alpha must be between 0 and 1")

    where = _candidate_filter(req.candidateIds)
    k = max(1, min(req.k, len(req.candidateIds)) if req.candidateIds else req.k)

    if req.strategy == "centroid":
        results = _dedup(db.similarity_search_by_vector_with_relevance_scores(
            query_vec.tolist(), k=k, filter=where
        ))
    else:
        # The candidate filter is what makes this possible: with a bounded set
        # the full matrix is at most 8 x 500 dot products, so any aggregation
        # is affordable. Compressing the group into one query vector was only
        # ever a workaround for similarity_search taking exactly one.
        if not req.candidateIds:
            raise HTTPException(400, f"strategy '{req.strategy}' requires candidateIds")
        ids, mat = _candidate_matrix(req.candidateIds)
        if not ids:
            raise HTTPException(404, "none of the candidateIds are in the index")

        sims = units @ mat.T  # (N, M) true cosines — both sides are unit length
        scores = _aggregate(_zscore_rows(sims), req.strategy, req.alpha, weights)

        order = np.argsort(-scores)[:k]
        results = [{
            "businessId": ids[j],
            "score": float(scores[j]),
            # Raw cosines, not z-scores: these are what a UI would show to
            # explain a pick ("everyone in your group rated this highly").
            "memberSims": [round(float(s), 4) for s in sims[:, j]],
            "worstMemberSim": float(sims[:, j].min()),
        } for j in order]

    payload = {
        "businessIds": [r["businessId"] for r in results],
        "results": results,
        "strategy": req.strategy,
        "agreement": agreement,
        "consensus": consensus,
        "memberCount": n,
    }
    if req.debug:
        payload["perMember"] = [
            [h["businessId"] for h in _dedup(
                db.similarity_search_by_vector_with_relevance_scores(
                    v.tolist(), k=5, filter=where
                )
            )]
            for v in units
        ]

    return payload


@app.post("/recommend")
def recommend(req: RecReq):
    """Rank restaurants for a single free-text query.

    Used by /api/Restaurants/nearby, which passes the geo-filtered set as
    candidateIds and reorders its own results by the returned order.

    Returns (JSON):
        businessIds: ranked fsqIds, best first.

    Raises:
        HTTPException: 400 if candidateIds exceeds MAX_CANDIDATES.
    """
    where = _candidate_filter(req.candidateIds)
    # Asking for more hits than there are candidates just wastes work.
    k = min(req.k, len(req.candidateIds)) if req.candidateIds else req.k
    hits = db.similarity_search(req.query, k=max(1, k), filter=where)
    return {"businessIds": [_business_id(h) for h in hits]}


class IndexMissingReq(BaseModel):
    """Request body for POST /index/missing.

    Attributes:
        businessIds: restrict the top-up to these fsqIds — what a caller that
            has just synced a handful of places should send. Omit to sweep
            every unindexed `source: "foursquare"` row, up to MAX_INDEX_BATCH.
        force: re-embed the listed rows even if Chroma already holds them, for
            when the TEXT changed rather than the row being new — a post-meal
            review landing in tips[] is the case this exists for. Requires
            businessIds; see index_missing for why.
    """
    businessIds: list[str] = []
    force: bool = False


@app.post("/index/missing", dependencies=[Depends(require_token)])
def index_missing(req: IndexMissingReq):
    """Embed restaurants that MongoDB has and the index does not.

    The gap this closes: a Foursquare sync writes new rows to Mongo, but until
    something embeds them they are invisible to every search. They still appear
    in a caller's candidateIds, so the symptom is a silently short result list —
    or, if a whole area is new, /recommend/group failing with "none of the
    candidateIds are in the index".

    Doing it here rather than in rebuild_index.py is what makes it usable from
    the app: the write lands in THIS process's Chroma handle, so the new vectors
    are searchable immediately. A separate process would leave them invisible
    until the service restarted, which is the caching caveat in the module
    docstring.

    Deliberately unchanged from rebuild_index.py: build_text, the chunk window,
    pool(), the metadata keys, and normalize_embeddings=True. Anything that
    diverges here produces vectors that are not comparable with the 15,169
    already stored, which no test would catch — the search would just quietly
    get worse.

    Returns (JSON):
        requested: how many ids were considered.
        alreadyIndexed: how many already had vectors.
        written: how many were embedded and stored.
        skipped: rows whose text chunked to nothing.
        truncated: True if MAX_INDEX_BATCH capped the work, meaning a further
            call (or a full rebuild_index.py run) is needed.

    Raises:
        HTTPException: 503 if Mongo is not configured or unreachable.
            400 if force is set without businessIds.
    """
    # force + no ids would re-embed the whole corpus, MAX_INDEX_BATCH rows of
    # encoding inside one request, holding a worker for minutes. A full re-embed
    # is rebuild_index.py's job; this endpoint stays a top-up.
    if req.force and not req.businessIds:
        raise HTTPException(
            status_code=400,
            detail="force requires businessIds — a full re-embed is rebuild_index.py's job",
        )

    col = _restaurants()

    query = {"source": "foursquare"}
    if req.businessIds:
        query["fsqId"] = {"$in": req.businessIds}

    projection = {"fsqId": 1, "name": 1, "categories": 1,
                  "location": 1, "rating": 1, "tips": 1}
    docs = [d for d in col.find(query, projection) if d.get("fsqId")]
    requested = len(docs)

    # Nothing matched in Mongo — unknown ids, or all of them yelp_seed. Returned
    # early because Chroma's get() rejects an empty id list outright rather than
    # returning nothing, which would surface here as a 500.
    if not docs:
        return {"requested": 0, "alreadyIndexed": 0,
                "written": 0, "skipped": 0, "truncated": False}

    # One get() for the ids we care about, not the whole collection — asking
    # Chroma for the ~20k it holds to check 50 would dominate the request.
    ids = [d["fsqId"] for d in docs]
    known = set(db._collection.get(ids=ids, include=[]).get("ids") or [])
    # Under force the filter is skipped, not the get(): alreadyIndexed stays an
    # honest count of what was overwritten. Safe because the write below is an
    # upsert keyed on fsqId, so re-embedding replaces a vector rather than
    # duplicating it — and the text still comes from Mongo through build_text,
    # so the "one template" invariant in the module docstring holds either way.
    if not req.force:
        docs = [d for d in docs if d["fsqId"] not in known]

    truncated = len(docs) > MAX_INDEX_BATCH
    if truncated:
        docs = docs[:MAX_INDEX_BATCH]

    if not docs:
        return {"requested": requested, "alreadyIndexed": len(known),
                "written": 0, "skipped": 0, "truncated": False}

    # embeddings._client is the SentenceTransformer the read path already
    # loaded. Reused rather than constructed fresh so there is one set of
    # weights in the process, and no chance of the two paths drifting onto
    # different models.
    model = embeddings._client
    tokenizer = model.tokenizer
    window = model.max_seq_length - 2

    texts = [build_text(d) for d in docs]
    chunked = [chunk(t, tokenizer, window) for t in texts]

    flat = [c for cs, _ in chunked for c in cs]
    if not flat:
        return {"requested": requested, "alreadyIndexed": len(known),
                "written": 0, "skipped": len(docs), "truncated": truncated}

    vecs = np.asarray(
        model.encode(flat, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )

    out_ids, embs, documents, metas, skipped, cursor_i = [], [], [], [], 0, 0
    for doc, text, (cs, n_tokens) in zip(docs, texts, chunked):
        if not cs:
            skipped += 1
            continue
        block = vecs[cursor_i:cursor_i + len(cs)]
        cursor_i += len(cs)

        out_ids.append(doc["fsqId"])
        embs.append(pool(block).tolist())
        documents.append(text[:2000])
        metas.append({
            "business_id": doc["fsqId"],
            "name": doc.get("name") or "",
            "n_chunks": len(cs),
            "n_tokens": n_tokens,
            "has_tips": bool(doc.get("tips")),
        })

    if out_ids:
        db._collection.upsert(ids=out_ids, embeddings=embs,
                              documents=documents, metadatas=metas)

    return {"requested": requested, "alreadyIndexed": len(known),
            "written": len(out_ids), "skipped": skipped, "truncated": truncated}
