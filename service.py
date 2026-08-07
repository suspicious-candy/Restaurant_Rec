"""FastAPI read-side of the Palate recommender.

This service only ever READS the vector index. Writing is owned entirely by
rebuild_index.py, which builds from MongoDB — one writer, one source of truth,
one text template. (An older /embed endpoint let callers push their own text
and was removed; see the note above RecReq.)

Two endpoints:
    POST /recommend        one text query  -> ranked businessIds
    POST /recommend/group  N members       -> ranked businessIds for a group

Both take an optional `candidateIds`, and passing it is strongly preferred:
retrieval should be a FILTER (the caller decides what is reachable — nearby,
open, in budget) and the vector search should only ORDER what survives.
Retrieving top-K independently and intersecting afterwards does not work; two
50-item draws from a 15,169-item corpus are expected to share 0.16 items, and
the measured overlap was zero.

Run from this directory (Chroma's persist_directory is a relative path):
    ./.venv/Scripts/python.exe -m uvicorn service:app --port 8000

Restart it after any rebuild_index.py run — Chroma's PersistentClient caches
in-process, so a collection dropped and rebuilt by another process is invisible
to a service that is already running.
"""
# Preload pyarrow BEFORE torch/transformers get pulled in below. On Windows
# these packages each bundle their own native OpenMP runtime; if pyarrow's DLL
# loads *after* torch's, the process dies with a native access violation
# (exit code 0xC0000005) and uvicorn never starts. Importing pyarrow first
# makes it win the DLL load and avoids the crash.
import pyarrow  # noqa: F401
import pyarrow.dataset  # noqa: F401

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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
db = Chroma(collection_name="places_v2", persist_directory="chroma_db",
            embedding_function=embeddings)

# Probed rather than hardcoded to 384, so swapping the model cannot silently
# invalidate the dimension check in /recommend/group.
EMBED_DIM = len(embeddings.embed_query("dimension probe"))

# Chroma builds one $in predicate per id, so a huge candidate list turns the
# filter into the slow path. Callers should pre-narrow (geo, open now, price).
MAX_CANDIDATES = 500

# Aggregation functions available to /recommend/group. See _aggregate().
STRATEGIES = ("blend", "min", "mean", "centroid")


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
