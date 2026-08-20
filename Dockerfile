# syntax=docker/dockerfile:1

# slim, not alpine: torch and pyarrow ship manylinux wheels that need glibc, and
# on musl pip falls back to building from source, which does not finish.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, as their own layer, so editing service.py does not
# reinstall several hundred megabytes of PyTorch on every build.
COPY requirements.txt .

# --extra-index-url gets the CPU torch wheel. Without it pip resolves the CUDA
# build from PyPI and the image grows by gigabytes of driver runtime that nothing
# in this service can use. Measured: the CPU wheel is 197MB, the CUDA one is
# roughly ten times that. See requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# BAKE THE MODEL INTO THE IMAGE.
#
# HuggingFaceEmbeddings downloads all-MiniLM-L6-v2 (~90MB) on first use and
# caches it under ~/.cache. Left to runtime that download happens inside the
# first request — which on a scale-to-zero host means EVERY cold start pays it,
# and the service depends on huggingface.co being reachable from inside whatever
# network the host puts it on. Doing it here makes the cold start predictable and
# the image self-contained.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Source AND the vector index. See the note below on why the index ships inside
# the image on Cloud Run; .dockerignore is what decides whether it is included.
COPY . .

# THE INDEX LIVES IN THE IMAGE, and this is a Cloud Run decision rather than a
# preference.
#
# Cloud Run gives a container no persistent disk and freezes it between
# requests, so there is nowhere durable to keep a 553MB SQLite file. The two
# alternatives are both worse: downloading it from Cloud Storage would add
# ~553MB to every cold start, and mounting a bucket over FUSE puts SQLite on a
# filesystem whose locking semantics it does not trust.
#
# The consequence is worth stating plainly. POST /index/missing still works, but
# its writes go to the container's ephemeral layer and vanish when the instance
# scales down. So restaurants synced from Foursquare after deploy are searchable
# only until that instance dies. They are never LOST — the rows are in Mongo —
# they are simply unranked until the next `rebuild_index.py` run and redeploy.
# Both callers of that endpoint degrade to distance order, so this is a slow
# staleness rather than a failure. Rebuild monthly, or whenever a new city is
# added.
#
# On a host WITH a disk (Fly, a VM), delete this line, re-add chroma_db/ to
# .dockerignore, and point CHROMA_DIR at the mount instead — then the runtime
# writes persist and none of the above applies.
ENV CHROMA_DIR=/app/chroma_db

# Config comes from the environment, never from a file in the image.
#   mongo_url            Atlas connection string
#   MONGO_DB             database name (defaults to "test")
#   RECOMMENDER_TOKEN    shared secret; POST /index/missing is open without it
#   PORT                 injected by Cloud Run; 8080 is its default

# Cloud Run assigns the port at runtime and the container MUST listen on it.
# Hardcoding one means the health check never passes and the revision is rolled
# back with a "failed to start and listen" error that says nothing about ports.
#
# `exec` so uvicorn becomes PID 1 and receives SIGTERM directly — without it the
# shell holds PID 1, swallows the signal, and every shutdown waits out the grace
# period instead of stopping cleanly. The shell form is required regardless,
# because the exec form does not expand ${PORT}.
#
# One worker on purpose. Each loads its own copy of the model, so N workers cost
# N times the memory for no throughput gain on a 1-CPU instance — and because
# Chroma caches in-process, a write through one worker is invisible to the
# others. Scale with Cloud Run instances, not workers.
CMD exec uvicorn service:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
