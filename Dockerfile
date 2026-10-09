# syntax=docker/dockerfile:1
#
# Two stages, because the build needs a compiler for wheels that the runtime does
# not need to carry. The runtime image installs the package, then builds the index
# at image-build time so the container starts ready to answer -- with no network
# access required at run time.

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY pipeline/ ./pipeline/
COPY serve/ ./serve/
COPY eval/ ./eval/
COPY scripts/ ./scripts/

# Build a wheel, and pre-download the runtime wheels, so the final stage needs no
# toolchain. ``--mount=type=cache`` keeps rebuilds fast without baking a cache into
# the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip build && \
    python -m build --wheel --outdir /dist . && \
    pip download --dest /wheels --only-binary=:all: /dist/*.whl


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Offline and loopback by default: the container serves the generated corpus and
    # is not reachable from outside the host unless -p is given.
    MINING_OFFLINE=true \
    MINING_INDEX_DIR=/app/.index \
    MINING_CACHE_DIR=/app/.cache \
    MINING_SERVE_HOST=0.0.0.0 \
    MINING_SERVE_PORT=8000

# A non-root user: the service has no authentication, so it should at least not be
# able to write outside its own directory.
RUN useradd --create-home --uid 10001 app

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/wheels mining-three-source-pipeline && \
    rm -rf /wheels

COPY --chown=app:app . /app
USER app

# Build the index during the image build: same date, same corpus, so this is
# reproducible. A container that starts with an empty index is a container whose
# health check is red on first boot.
RUN python -m pipeline.ingest --days 30 --index-dir "$MINING_INDEX_DIR" --quiet

EXPOSE 8000

# /health answers 200 even when degraded, so probe the field that matters.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import json,urllib.request,sys; \
b=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); \
sys.exit(0 if b.get('index_loaded') else 1)"

CMD ["python", "-m", "serve.app"]
