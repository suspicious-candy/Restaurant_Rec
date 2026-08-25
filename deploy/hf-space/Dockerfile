# Hugging Face Spaces (Docker SDK, free CPU Basic tier).
#
# Copy this to the ROOT of the Space repo as `Dockerfile`. See SETUP.md.
#
# Closer to the root Cloud Run Dockerfile than to Dockerfile.arm64, because HF
# Spaces share Cloud Run's defining constraint: NO PERSISTENT DISK. The index
# ships inside the image again, and POST /index/missing writes vanish on
# restart. That is the same designed staleness the Cloud Run image documents --
# rows are never lost, they live in Mongo, they are simply unranked until the
# next rebuild_index.py and redeploy.
#
# Three things differ from the Cloud Run image, all HF platform requirements:
#
#   1. Port 7860, hardcoded. HF routes to a fixed port rather than injecting
#      one, so the ${PORT} indirection Cloud Run needed is not just unnecessary
#      here, it is a way to bind the wrong port and get a Space that builds
#      cleanly and never becomes reachable.
#
#   2. Runs as UID 1000, not root. HF Spaces drop privileges, and anything
#      root-owned becomes unreadable at runtime. This is the single most common
#      reason a working Dockerfile fails on Spaces.
#
#   3. x86, not ARM. Spaces CPU Basic is Intel, so the --extra-index-url that
#      pulls the CPU torch wheel matters again -- without it pip resolves the
#      CUDA build and the image grows by gigabytes of driver runtime that
#      nothing here can use.

FROM python:3.12-slim

# UID 1000 specifically: HF runs the container as that user, and a mismatch
# means every file in /app is owned by someone the process is not.
RUN useradd -m -u 1000 user

WORKDIR /app

# Dependencies first, as their own layer, so editing service.py does not
# reinstall several hundred megabytes of PyTorch on every build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Point the HF cache inside /app BEFORE baking the model, so it lands somewhere
# the chown below can reach. Left at its default the model caches into /root,
# and UID 1000 cannot read it -- which surfaces at runtime as the model being
# re-downloaded inside the first request, on a Space whose whole problem is
# cold-start latency.
ENV HF_HOME=/app/.hfcache

# Bake the model rather than downloading at runtime. On a host that sleeps after
# 48h this matters more than it did on Cloud Run: every wake would otherwise pay
# the ~90MB download inside the first request.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Source AND the vector index. .dockerignore deliberately does NOT exclude
# chroma_db/ -- see its own note -- which is correct for this host.
COPY . .

# Hand everything to the runtime user, then drop privileges. Both steps matter:
# Chroma opens its SQLite file read-write even for pure reads, so a root-owned
# index fails on startup rather than silently degrading.
RUN chown -R user:user /app
USER user

ENV CHROMA_DIR=/app/chroma_db

# Must match `app_port` in the Space README frontmatter.
EXPOSE 7860

# One worker, unchanged: each loads its own copy of the model, and Chroma caches
# in-process so a write through one worker is invisible to the others.
CMD exec uvicorn service:app --host 0.0.0.0 --port 7860 --workers 1
