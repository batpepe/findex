#!/usr/bin/env bash
# Download the findex corpus into data/ (which is git-ignored):
#   data/pydocs/  - the official Python documentation, plain-text build (docs.python.org)
#   data/peps/    - every PEP source file (github.com/python/peps, public domain / CC0)
# Crawled pages (Lab 6) land in data/crawl/*.jsonl via `findex crawl`.
set -euo pipefail

PY_DOCS_VERSION="${PY_DOCS_VERSION:-3.15}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$DATA"

echo "==> Python $PY_DOCS_VERSION docs (text)"
curl -fsSL -o "$TMP/docs.zip" "https://docs.python.org/3/archives/python-${PY_DOCS_VERSION}-docs-text.zip"
rm -rf "$DATA/pydocs"
unzip -q "$TMP/docs.zip" -d "$TMP/docs"
mv "$TMP/docs/python-${PY_DOCS_VERSION}-docs-text" "$DATA/pydocs"

echo "==> PEPs"
git clone -q --depth 1 --filter=blob:none --sparse https://github.com/python/peps.git "$TMP/peps"
git -C "$TMP/peps" sparse-checkout set peps
rm -rf "$DATA/peps"
mkdir -p "$DATA/peps"
cp "$TMP"/peps/peps/pep-*.rst "$DATA/peps/"

echo "==> done"
find "$DATA" -type f | wc -l | xargs echo "files:"
du -sh "$DATA"/* 
