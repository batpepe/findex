#!/usr/bin/env bash
# The HTML build of the Python docs (one download from docs.python.org), served locally by
# scripts/mirror_server.py so the crawler can be benchmarked without crawling anyone's site.
set -euo pipefail
PY_DOCS_VERSION="${PY_DOCS_VERSION:-3.15}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fsSL -o "$TMP/html.tar.bz2" "https://docs.python.org/3/archives/python-${PY_DOCS_VERSION}-docs-html.tar.bz2"
rm -rf "$ROOT/mirror"
mkdir -p "$ROOT/mirror"
tar -xjf "$TMP/html.tar.bz2" -C "$ROOT/mirror" --strip-components 1
printf 'User-agent: *\nAllow: /\n' > "$ROOT/mirror/robots.txt"
find "$ROOT/mirror" -name '*.html' | wc -l | xargs echo "html pages:"
