# findex: search API + web UI. Three stages:
#   builder - uv installs the locked dependencies, then the project, into /app/.venv
#   index   - downloads the corpus (Python docs + PEPs, both freely redistributable) and builds index.bin
#   runtime - python:3.14-slim with the venv and the index only; runs as a non-root user
# Build: docker build -t findex .      Run: docker run -p 8000:8000 findex
# Bring your own index instead: docker run -v $PWD/index.bin:/data/index.bin -e INDEX_PATH=/data/index.bin ...

FROM ghcr.io/astral-sh/uv:0.12.1-python3.14-trixie-slim AS builder
# no BuildKit cache mounts, so the legacy builder works too; UV_NO_CACHE keeps uv's cache out of the layers
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 UV_NO_CACHE=1
WORKDIR /app
# dependencies first: this layer is reused until uv.lock changes
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM builder AS index
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git unzip \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/get-corpus.sh scripts/get-corpus.sh
RUN ./scripts/get-corpus.sh \
    && /app/.venv/bin/findex index data/ --out /app/index.bin --executor serial \
    && rm -rf data

FROM python:3.14-slim AS runtime
# security fixes published after the base image was built (the homelab CI gates on Trivy)
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 findex \
    && useradd --system --uid 10001 --gid findex --home-dir /app --shell /usr/sbin/nologin findex
WORKDIR /app
COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=index --chown=findex:findex /app/index.bin /app/index.bin
ENV PATH="/app/.venv/bin:$PATH" \
    INDEX_PATH=/app/index.bin \
    LOG_JSON=true \
    PYTHONUNBUFFERED=1
USER findex
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
# exec form: uvicorn is PID 1 and gets SIGTERM directly -> graceful shutdown, lifespan teardown runs.
# --no-access-log: the app logs one line per request itself (with the request id), as JSON
CMD ["uvicorn", "findex.web.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*", "--no-access-log"]
