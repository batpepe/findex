# findex

[![CI](https://github.com/batpepe/findex/actions/workflows/ci.yml/badge.svg)](https://github.com/batpepe/findex/actions/workflows/ci.yml)
[![Image](https://github.com/batpepe/findex/actions/workflows/image.yml/badge.svg)](https://github.com/batpepe/findex/actions/workflows/image.yml)

A search engine over the **Python documentation and every PEP**, built from scratch in Python: you can ask
it `"event loop" NOT asyncio`, or, in your own words, *find out how long a piece of code takes to run*.

**Live: [findex.batpepe.online](https://findex.batpepe.online)** (API browser:
[/docs](https://findex.batpepe.online/docs)) - self-hosted on my K3s homelab behind a Cloudflare Tunnel,
deployed by ArgoCD ([Lab 7](#deployed-findexbatpepeonline)).

![findex search in a terminal: ranked results with highlighted snippets](docs/demo.gif)

| The corpus | The live site under load (p95) | Ranking the most common word | Questions in other words (precision@5) |
|---|---|---|---|
| 1,297 documents, 34 MB, 4.15 M tokens | **290 ms** at 87 requests/s, 20 clients through Cloudflare | **156×** faster with NumPy | **0.40** by meaning, 0.04 by keywords |

It grew over the eight labs of the *Python - Build a Search Engine* course, one language feature and one
piece of the engine per lab; every lab is tagged (`lab-01` ... `lab-08`), so the history shows it growing.

| Lab | Feature | What it added | Tag |
|---|---|---|---|
| 1 | Iterators & generators | streaming corpus, tokenizer, stats in constant memory | `lab-01` |
| 2 | Dicts, hashing, memory | inverted index, Boolean search, two on-disk formats | `lab-02` |
| 3 | The data model, decorators | `Index` object, BM25/TF-IDF, query language, snippets, precision@5 | `lab-03` |
| 4 | Typing, testing, packaging | `findex` CLI (typer + rich), pyright strict, 77 tests, CI, v0.4.0 wheel | `lab-04` |
| 5 | Concurrency and the GIL | parallel indexing: serial / threads / processes / free-threaded, benchmarked | `lab-05` |
| 6 | asyncio | a polite async crawler (robots.txt, per-host limits, retries) that feeds the corpus | `lab-06` |
| 7 | FastAPI, Pydantic, Docker | search API + web UI, containerized, load-tested, deployed | `lab-07` |
| 8 | Profiling, NumPy, embeddings | profiled hot paths, NumPy ranking, semantic + hybrid (RRF) search, v1.0.0 | `lab-08` |

## Architecture

```mermaid
flowchart LR
    subgraph corpus [The corpus]
        F["Python docs + PEPs<br/>get-corpus.sh"]
        J["crawled pages, JSON lines<br/><i>Lab 6: async crawler</i>"]
    end
    F --> T["iter_documents + tokenize<br/><i>Lab 1: generators</i>"]
    J --> T
    T -->|"map / reduce<br/><i>Lab 5: processes or no-GIL threads</i>"| I["inverted index, mmap file<br/><i>Lab 2</i>"]
    I --> K["query parser, BM25 / TF-IDF<br/><i>Lab 3</i>; NumPy engine <i>Lab 8</i>"]
    I -->|"findex embed: chunks -> MiniLM<br/><i>Lab 8</i>"| V["chunk vectors (.npy, mmap)"]
    V --> S["E @ q, best chunk per document<br/><i>Lab 8</i>"]
    K --> H["hybrid: reciprocal rank fusion<br/><i>Lab 8</i>"]
    S --> H
    H --> C["findex CLI<br/><i>Lab 4: typer + rich</i>"]
    H --> W["FastAPI + web UI<br/><i>Lab 7</i>"]
    W --> D["Docker -> GHCR -> ArgoCD -> K3s<br/>Cloudflare Tunnel <i>Lab 7</i>"]
```

## Quick start

```bash
git clone https://github.com/batpepe/findex && cd findex
uv tool install '.[semantic]'                # the `findex` command (Python 3.14+); drop [semantic] for keyword only
./scripts/get-corpus.sh                      # ~31 MB into data/ (git-ignored)
findex index data/ --out index.bin           # positions + texts, binary format
findex search index.bin '"event loop" NOT asyncio'
findex search index.bin 'TaskGroup' --json | jq .title
findex embed index.bin                       # semantic search: downloads all-MiniLM-L6-v2 once; CPU-bound (~20 min on 2 vCPUs)
findex search index.bin 'find out how long a piece of code takes to run' --mode semantic
findex serve index.bin                       # the web UI and API on http://127.0.0.1:8000
```

Development: `uv sync --extra semantic`, then `uv run pytest`, `uv run ruff check .`, `uv run pyright`.
The lab-by-lab entry points still work: `uv run python -m findex.stats data/`, `python -m findex.index`,
`python -m findex.search`.

## The measurements, in one place

Every number below comes from a script in `scripts/` (named in each lab's section).

| Lab | Measured | Result |
|---|---|---|
| 1 | Peak memory, full corpus: lists vs generators | 261.6 MiB vs 41.2 MiB (16.6 MiB after the Lab 4 tokenizer fix) |
| 2 | Index load: pickle vs mmap-ed binary | 2,013 ms vs 25 ms (80×) |
| 2 | Bytes per posting: dataclass / `slots=True` / arrays | 118 / 78 / 37 |
| 3 | precision@5 on 10 labelled queries: BM25 vs TF-IDF | 0.72 vs 0.56 |
| 4 | Tests, coverage | 77 tests, 7 Hypothesis properties, 94% (now 196 tests, 94%) |
| 5 | Parallel index build: 8 processes / 4 free-threaded threads | 1.81× / 2.29× |
| 5 | `counter[k] += 1` from 4 threads on 3.14t | lost 44,098 of 4,152,788 increments |
| 6 | Crawler with per-host back queues instead of one FIFO | 1.2 -> 3.1 pages/s (the politeness ceiling) |
| 7 | `/search` load test: 1 worker vs 4 workers | 46 vs 150 requests/s, p50 1,058 vs 306 ms |
| 7 | `/search` on the live site, 20 clients through Cloudflare (cached path) | 87 requests/s, p50 175 ms, p95 290 ms |
| 8 | Ranking: Lab 3 Python loop vs NumPy | 34× (`python`), 156× (`the`), 0.8× (`walrus`, 16 postings) |
| 8 | A new query end to end, snippets included | 4.2× faster (`global interpreter lock`: 31.2 -> 7.5 ms) |
| 8 | Semantic search: all 32,367 passage similarities + best passage per document | 1.9 ms (brute force, exact) |
| 8 | precision@5: keyword / semantic / hybrid, 10 queries and 5 paraphrases | 10: 0.74 / 0.66 / **0.82**; paraphrases: 0.04 / **0.40** / 0.12 |

Labs 1-7 were measured on an Apple M4; Lab 8 on a 2-vCPU cloud VM (see its section), so compare ratios,
not milliseconds, across that line.

## The corpus

| Source | Where it comes from | Documents | Size on disk |
|---|---|---|---|
| Python 3.15 documentation, plain-text build | [docs.python.org/3/download.html](https://docs.python.org/3/download.html) (`python-3.15-docs-text.zip`) | 552 | 16 MB |
| All PEPs (reStructuredText sources) | [github.com/python/peps](https://github.com/python/peps), `peps/pep-*.rst` (public domain / CC0) | 745 | 15 MB |
| Packaging guide, pip, devguide, typing docs (Lab 6) | crawled by `findex crawl` into `data/crawl/python-sites.jsonl` | 267 | 2.5 MB |
| **Total** | `./scripts/get-corpus.sh` + the crawl command below | **1,564** | **34 MB** |

One file is one document. The corpus lives in `data/`, which is in `.gitignore`: corpora don't go into git,
the script that fetches them does. Why this corpus: it is the reference I actually search while learning
Python, it is large and varied enough (from a 97-byte deprecation note to the 2.4 MB changelog), and its
licences allow crawling and redistributing (Lab 6 crawls more of it).

---

## Lab 1 — Streams, not lists

**Built:** `findex.corpus.iter_documents` (lazy, sorted, one file open at a time; `.txt`/`.rst`/`.md` trees
and JSON-lines, also `.jsonl.gz`; undecodable bytes are logged and replaced instead of crashing the run),
`findex.tokenize.tokenize` (a generator over `re.finditer`), and `python -m findex.stats` — the
`iter_documents → tokenize → Counter` pipeline with `--limit N` implemented with `itertools.islice`.

Both stages are generators (`inspect.isgeneratorfunction` is asserted in `tests/test_tokenize.py`), and
`next(iter_documents(Path("data")))` returns the first document (`PEP 1 – PEP Purpose and Guidelines`) in
**0.6 ms** — before the other 1,296 files are even opened.

### Tokenizer policy

1. **Unicode:** NFC → `casefold()` → NFC again (case folding can decompose). `casefold` rather than `lower`:
   `"Straße".casefold() == "strasse"`. NFC because `"café"` and `"café"` look identical but are
   different strings. Typographic and Ukrainian apostrophes (`’` U+2019, `ʼ` U+02BC, `‘`, `＇`) become `'`.
2. **Token** = a run of Unicode word characters (`\w` — letters and digits of any script, and `_`) plus
   combining marks.
3. **Apostrophes** stay *inside* words (`don't`, `п'ять`), never at the edges (`'quoted'` → `quoted`);
   the English possessive `'s` is dropped (`Python's` → `python`).
4. **Hyphens split** (`free-threaded` → `free`, `threaded`), so a search for `threaded` finds it; phrase
   search (Lab 3) gets the compound back.
5. **Digits** are tokens; `3.15` → `3`, `15`; mixed alphanumerics stay whole (`utf8`, `py3`).
6. **Underscores** are word characters, so identifiers survive (`__init__`, `snake_case`); a token of only
   underscores is dropped.
7. **No stop words, no stemming:** IDF will down-weight `the` by itself, and phrases need every word.

All seven rules are pinned by tests (mixed case, Cyrillic, a combining accent, punctuation, empty input,
apostrophes, hyphens, digits, identifiers): `uv run pytest`.

### Corpus statistics

```text
$ uv run python -m findex.stats data/
mode:        lazy (generators)
documents:   1,297
tokens:      4,152,788
vocabulary:  78,218
elapsed:     3.09 s
peak memory: 41.2 MiB (tracemalloc)
retained:    5.6 MiB (still allocated at the end: the term table)
max RSS:     79.4 MiB (whole process, incl. interpreter)

top 50 terms:
   1. the         219,940
   2. to           97,279
   3. a            95,684
  ...
  12. python       33,934
  ...
  27. pep          14,765
  ...
  35. bpo          12,178
```

The top of the list is English function words, as Zipf predicts; the first content words are `python`,
`pep`, `type`, `module`. `bpo` (the old bugs.python.org issue prefix) is in the top 35 purely because of
the 2.4 MB changelog — the first hint that document length will matter for ranking (Lab 3, BM25).

### Eager vs lazy

The eager version (`--eager`) reads every document into a list, tokenizes each into a list, then counts.
Same tokenizer, same output; each row is the median of 3 runs in a fresh process
(`uv run python scripts/bench_stats.py data/`, Apple M4, Python 3.14):

| Version | Documents | Tokens | Peak memory (tracemalloc) | Max RSS | Elapsed |
|---|---|---|---|---|---|
| eager (lists) | 400 | 991,030 | 58.3 MiB | 160.3 MiB | 0.68 s |
| lazy (generators) | 400 | 991,030 | 3.1 MiB | 29.5 MiB | 0.69 s |
| eager (lists) | 800 | 2,214,105 | 132.1 MiB | 329.5 MiB | 1.63 s |
| lazy (generators) | 800 | 2,214,105 | 7.4 MiB | 34.7 MiB | 1.62 s |
| eager (lists) | 1,297 | 4,152,788 | 261.6 MiB | 629.0 MiB | 3.20 s |
| lazy (generators) | 1,297 | 4,152,788 | 41.2 MiB | 79.4 MiB | 3.22 s |

**Where the memory went.** The eager peak grows linearly with the corpus because everything is alive at
once. On the full corpus I accounted for it object by object: the 4,152,788 token strings are separate
`str` objects of ~46 bytes each (a 40-byte header for a 4–5 character word) = **182.5 MiB**; the 1,297
token lists hold an 8-byte pointer per token = **33.7 MiB**; the document texts themselves = **40.6 MiB**.
That is 256.8 of the 261.6 MiB peak; the rest is the `Counter` and the document tuples. The text is the
*smallest* part: tokenizing a 31 MB corpus into Python objects costs 6× the corpus. The lazy pipeline
holds one document and the `Counter` (5.6 MiB retained at the end: 78,218 terms). Its peak is not that
small on the full corpus because one document is huge: `whatsnew/changelog.txt` is 2.4 MB of text,
stored as a 2-byte-per-character string (one `→` forces UCS-2), so 4.7 MB per copy; normalization makes
several transient copies (NFC, casefold, apostrophe translation, NFC again) while it is the current
document. Without it (`--limit 1295` stops just before the changelog) the lazy peak is **7.6 MiB** for
1,295 documents — memory is *one document + the term table*, not the corpus. (The changelog alone also
adds 14,332 terms to the vocabulary: issue numbers and contributor names.) Time is identical: both versions do the same CPU work (regex + casefold), the
lazy one just never keeps what it has finished with. Max RSS is higher than tracemalloc's peak in both
columns because it includes the interpreter, tracemalloc's own bookkeeping, and freed memory the
allocator has not returned to the OS.

**What I learned.** A generator is a paused function, not a container: the pipeline never "has" the
corpus, it has the next document. The only structure allowed to grow is the sink that holds the answer.
And Python objects are expensive — a 4-letter word costs 46 bytes as a `str` plus 8 for the pointer to
it — which is exactly the problem Lab 2 has to solve for the inverted index.

---

## Lab 2 — The inverted index

**Built:** `findex.index` — `Posting(doc_id, tf, positions)` and `DocMeta(path, title)` are
`@dataclass(frozen=True, slots=True)` (hashable value objects: the tests put them in a set and check
there is no `__dict__`); `build_index` makes **one streaming pass**: each document's terms are counted
in a local `Counter` (or collected as position lists with `--positions`) and *flushed* into a
`defaultdict(list)` when the document ends. Documents arrive in increasing `doc_id` order, so every
postings list is born sorted. `findex.search` evaluates `a b` (AND), `OR`, `NOT` left to right with
**two-pointer merges**, or with Python sets (`--engine set`). `findex.store` saves and loads two formats.

```text
$ uv run python -m findex.index data/ --out index.bin
documents:   1,297
terms:       78,218
postings:    861,841
positions:   no
build:       3.82 s
save:        1.02 s (binary, 8.6 MiB)
peak memory: 97.5 MiB (tracemalloc)

$ uv run python -m findex.search index.bin "asyncio TaskGroup" --limit 4
13 matching documents for 'asyncio TaskGroup' (merge engine)
  [  472] PEP 654 – Exception Groups and except*  (peps/pep-0654.rst)
  [  601] PEP 785 – New methods for easier handling of ExceptionGroups  (peps/pep-0785.rst)
  [  605] PEP 789 – Preventing task-cancellation bugs by limiting yield in async generators  (peps/pep-0789.rst)
  [  917] High-level API Index  (pydocs/library/asyncio-api-index.txt)
  ... and 9 more
load:        107.1 ms
search:      0.18 ms
peak memory: 24.7 MiB (tracemalloc)
```

### Merge vs sets

`uv run python scripts/bench_merge.py index.bin` — best of 5 × 2,000 calls, on the two most common terms
(`the` is in 1,295 of 1,297 documents), the two rarest (document frequency 1), and one of each:

| Operands | len(a), len(b) | Op | merge (µs) | set (µs) | faster |
|---|---|---|---|---|---|
| common AND common (`the`, `in`) | 1295, 1279 | AND | 58.67 | 31.43 | set 1.9× |
| common AND common (`the`, `in`) | 1295, 1279 | OR | 58.88 | 29.30 | set 2.0× |
| common AND common (`the`, `in`) | 1295, 1279 | NOT | 56.12 | 24.92 | set 2.3× |
| rare AND rare (`aaaaa`, `aaaaaa`) | 1, 1 | AND | 0.08 | 0.13 | merge 1.6× |
| rare AND rare (`aaaaa`, `aaaaaa`) | 1, 1 | OR | 0.16 | 0.14 | set 1.1× |
| rare AND rare (`aaaaa`, `aaaaaa`) | 1, 1 | NOT | 0.10 | 0.14 | merge 1.4× |
| rare AND common (`aaaaa`, `the`) | 1, 1295 | AND | 50.72 | 4.24 | set 12.0× |
| rare AND common (`aaaaa`, `the`) | 1, 1295 | OR | 57.51 | 14.72 | set 3.9× |
| rare AND common (`aaaaa`, `the`) | 1, 1295 | NOT | 44.15 | 4.21 | set 10.5× |

Both are O(n + m), but the merge's n + m steps are **Python bytecode** (a compare, an index, an
increment — tens of nanoseconds each) while `set(a) & set(b)` hashes in **C**. So sets win on long lists
by ~2×, and by 10× when one list is tiny: the merge still walks all 1,295 entries of `the` to find one
document. The merge wins only when both lists are tiny (no set to build). What the merge buys is not
speed in CPython but shape: it streams, needs no extra memory, works on compressed or memory-mapped
postings, and is the same algorithm phrase search uses on positions (Lab 3). The obvious next step for
rare-AND-common — galloping through the long list with `bisect` (C) instead of stepping — is noted for
Lab 8.

### Two formats on disk

`findex.store` writes **pickle** or a **binary** format (default): a JSON header with the documents and
`term → (byte offset, df, total tf)`, then every postings list as raw `uint32` arrays (`doc_ids`, `tfs`,
then positions if present). Loading the binary file `mmap`s it and parses only the header; a term's
postings are decoded from the mapped bytes the first time a query asks for them. Terms are written in
sorted order, so the same index always produces the same bytes (a test checks it).
`uv run python scripts/bench_store.py data/`:

| Positions | Format | File size | Save | Load (median of 5) | Load + first query |
|---|---|---|---|---|---|
| no | pickle | 14.3 MiB | 1.81 s | 2013 ms | 2025 ms |
| no | binary | 8.6 MiB | 0.58 s | 25 ms | 25 ms |
| yes | pickle | 27.4 MiB | 2.32 s | 2546 ms | 2525 ms |
| yes | binary | 24.5 MiB | 0.56 s | 28 ms | 27 ms |

The binary file loads **80× faster** because it does not create 861,841 objects up front; it decodes only
the postings a query touches. **Positions** cost 2.85× the file size (24.5 vs 8.6 MiB: one 4-byte int
per token occurrence, 4.15 M of them, against two per posting), +44% build time (5.5 vs 3.8 s), and 2.9× the
build peak (280 vs 97.5 MiB — a tuple object per posting). They stay optional (`--positions`) until
phrase search needs them in Lab 3.

**Why pickle is dangerous.** `pickle.load` is not a parser, it is a small virtual machine: a pickle can
tell it to import any callable (`os.system`, `subprocess.Popen`) and call it with arguments — that is how
`__reduce__` works. Loading a downloaded index file is therefore remote code execution. `findex` only
unpickles files it wrote itself (`store.load_pickle` says so in a comment), and the binary format, which
can only ever produce numbers and strings, is the default.

### The memory study

The same postings in three representations, each built in a fresh process under tracemalloc
(`uv run python scripts/bench_memory.py data/`):

| Postings representation | Peak memory (build) | Held by postings | Bytes / posting | Pickle file size | Pickle load time |
|---|---|---|---|---|---|
| `list[Posting]`, plain `@dataclass` | 128.6 MiB | 97.2 MiB | 118.3 | 19.1 MiB | 291 ms |
| `list[Posting]`, `slots=True` | 96.9 MiB | 64.3 MiB | 78.3 | 14.2 MiB | 2012 ms |
| `array('I')` pairs (doc_ids, tfs) | 62.6 MiB | 30.7 MiB | 37.3 | 10.0 MiB | 47 ms |

**Where the bytes went.** All three rows share 5.6 MiB (6.8 bytes per posting) for the 78,218 term
strings (3.7 MiB) and the dict table that maps them (1.8 MiB). On top of that, per posting: a **plain**
dataclass instance is **96 bytes** (measured: a 48-byte object plus 48 bytes of instance-attribute
storage — its `__dict__`), plus an 8-byte pointer in the list and ~7 bytes of list headers and
over-allocation; **`slots=True`** removes the attribute storage and puts the three fields in fixed
slots — **56 bytes** (16-byte header, 3 × 8-byte field pointers, GC links) — exactly the 40-byte
difference between rows 1 and 2. The **arrays** store `doc_id` and `tf` as raw 4-byte integers — **8
bytes per posting, no objects** — and the remaining ~22 bytes per posting in that row are per-*term*
overhead: two 80-byte `array` headers and a 64-byte tuple for each of the 78,218 terms, plus
over-allocated buffers. Most terms occur in a single document, so that overhead is never amortized.
The ints themselves cost nothing extra in the object rows: small ints are cached and each document's
`doc_id` object is shared by all its postings.

The surprise is the last column: the slotted class is smaller **and 7× slower to unpickle**. A frozen
dataclass with slots cannot be restored by filling a `__dict__`, so `dataclasses` generates a Python-level
`__setstate__` (`_dataclass_setstate`), called once per object — 861,841 Python function calls. The plain
class is restored by a C-level `__dict__` update, and arrays are pickled as one raw byte buffer each.
Fewer objects beats smaller objects; no objects beats both — which is why the binary format stores
arrays and why Lab 8 turns them into NumPy.

**What I learned.** A Python object is never "just its fields": the header, the attribute dict, the
pointer to it and the GC links are paid per instance, and at 861,841 instances they dominate. `slots`
fixes the dict, arrays fix the objects. And the hashing contract is what makes `frozen=True` the safe
way to get value objects that can live in sets and as keys.

---

## Lab 3 — Ranking and the object model

**Built:**

- **`Index`** (`findex.index`) is a read-only `collections.abc.Mapping[str, list[Posting]]`: implementing
  `__getitem__`, `__iter__` and `__len__` gives `in`, `.get`, `.keys/.items`, `==` for free, and
  `__contains__` is overridden only because a direct dict lookup beats `Mapping`'s try/except.
  `repr(index)` is `Index(terms=78_218, docs=1_297, positions=True)`; `num_docs` is a property,
  `avg_doc_length` a `cached_property` (BM25 needs it for every posting: computed once per index, not once
  per posting); `df()`, `doc_length()`, `title()`, `text()`. `with open_index(path) as ix:` (a
  `@contextmanager`) releases the memory map even when the body raises — a test raises inside it and
  checks the map is gone. The binary index now also stores every document's text, zlib-compressed (31 MB
  of text → 9.4 MB), so snippets need no corpus on disk.
- **Scoring** (`findex.rank`): a `Scorer` `typing.Protocol` (`score(term, posting, index) -> float`) and
  two independent implementations, `TfIdf` and `BM25(k1=1.5, b=0.75)` — frozen dataclasses with
  `__call__`, so they are hashable (they are part of the cache key) and usable as functions. No base class.
  `Searcher.search(query, scorer, k)` accumulates per-document scores in a `defaultdict(float)` and keeps
  the top `k` with `heapq.nlargest` — O(n log k) instead of sorting every match. Results are
  `SearchResult` dataclasses with `order=True` whose first field is `score`, so `sorted(results)` ranks
  them; `ResultList` is a `Sequence` whose slices stay `ResultList`s and keep `total`.
- **The query language** (`findex.query`): a recursive-descent parser (`or_expr → and_expr → not_expr →
  atom`) building `Term`, `Phrase`, `And`, `Or`, `Not` nodes — frozen dataclasses, so tests compare
  **trees**: `parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))`. Nodes overload `&`, `|`,
  `~` (`Term("a") & ~Term("b")`), and each has `evaluate(index) -> frozenset[int]`. Phrases are checked
  with Lab 2's two-pointer `intersect` on positions shifted by their offset in the phrase; a hyphenated
  word (`free-threaded`) becomes a phrase automatically — the tokenizer split it, positions glue it back.
  Bad syntax raises `QueryError` with the offset (`missing ')' for '(' at offset 0`).
- **Decorators and caching:** `@timed` (with `functools.wraps`) logs `build_index`, `load` and
  `Searcher.search` at DEBUG; the query → ranked-ids step and the snippets are `functools.lru_cache`d
  *per `Searcher` instance* (a cache on the method itself would be shared by every instance and keep
  every index alive). `explain()` is a `functools.singledispatch` function with one implementation per
  node type, and `--explain` prints each result's score term by term, like Elasticsearch's `_explain`.
- **Snippets** (`findex.snippets`): re-tokenize the stored text with offsets (`token_spans`, the same
  tokens as `tokenize`, but on text that is NFC-normalized and *not* case-folded, so offsets stay valid
  for display), slide a ±80-character window over the hits and keep the one covering the most distinct
  query terms, then highlight (`**term**` in the terminal, `<mark>` for HTML later).

```text
$ uv run python -m findex.search index.bin '"event loop" NOT asyncio' -k 3 --explain
9 matching documents for '"event loop" NOT asyncio' (bm25, top 3)
parsed as: ("event loop" AND NOT asyncio)
  1.   6.500  Tkinter dialogs  (pydocs/library/dialog.txt)
         … Render dialog and start **event** **loop**. ok_event(**event**) Exit dialog returning current selection.  …
           event            tf=17    df=216   ->  4.262
           loop             tf=1     df=245   ->  2.238
           doc length 1,380 tokens (average 3,202)
  2.   6.127  tkinter — Python interface to Tcl/Tk  (pydocs/library/tkinter.txt)
         … like "grid" controls where in the user interface they are placed. **event** **loop** Tkinter reacts to user input, …
           event            tf=82    df=216   ->  3.899
           loop             tf=14    df=245   ->  2.228
           doc length 33,536 tokens (average 3,202)
  ...
```

The cache hit, from the timing log (`--repeat 2 -v`):

```text
DEBUG findex.rank: ranked 'unicode normalization': 17 matches, top 2
DEBUG findex.rank: query cache: CacheInfo(hits=0, misses=1, maxsize=256, currsize=1)
DEBUG findex.rank: Searcher.search took 8.27 ms
DEBUG findex.rank: query cache: CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
DEBUG findex.rank: Searcher.search took 0.06 ms
```

### The three sanity checks, on this corpus

`uv run python scripts/sanity_ranking.py index.bin` (all three are also unit tests in `tests/test_rank.py`):

1. **A rare term outranks a common one.** `python OR walrus`: `python` is in 1,207 of 1,297 documents,
   `walrus` in 16. All four top results contain `walrus` (PEP 622, 3136, 635, 634); IDF makes `python`
   almost worthless.
2. **BM25 saturates term frequency.** The BM25 contribution of `walrus` in an average-length document:

   | tf | 1 | 2 | 3 | 5 | 10 | 19 | 20 | 21 |
   |---|---|---|---|---|---|---|---|---|
   | BM25 | 4.366 | 6.237 | 7.276 | 8.395 | 9.490 | 10.115 | 10.152 | 10.186 |
   | TF-IDF | 4.395 | 7.442 | 9.224 | 11.469 | 14.516 | 17.337 | 17.562 | 17.777 |

   The 2nd occurrence adds +1.87; the 21st adds +0.034 — 55× less, bounded by `idf · (k1 + 1)`. TF-IDF's
   `1 + ln tf` grows forever (+0.21 for the 21st).
3. **A short document with one match beats a long one with one match.** The term `mmdf` occurs exactly
   once in *Internet Data Handling* (228 tokens) and once in the *Changelog* (342,806 tokens): BM25 gives
   **9.732 vs 0.116**; TF-IDF gives both **5.782** — it cannot tell them apart.

### Evaluation: precision@5

Ten queries I know the answers to, with relevance labelled by hand in `eval/queries.json` (relevant =
the document is primarily *about* the topic; the changelog and the table of contents are never relevant;
I pooled the top 10 of both scorers and added known documents missing from the pool).
`uv run python scripts/evaluate.py index.bin`:

| Query | relevant | BM25 P@5 | TF-IDF P@5 |
|---|---|---|---|
| `global interpreter lock` | 7 | 0.6 | 0.4 |
| `structural pattern matching` | 7 | 1.0 | 1.0 |
| `generic type hints` | 10 | 0.8 | 0.4 |
| `virtual environment` | 7 | 1.0 | 0.8 |
| `f-string formatting` | 4 | 0.2 | 0.2 |
| `async generator` | 4 | 0.6 | 0.4 |
| `dataclass frozen slots` | 3 | 0.4 | 0.4 |
| `context manager with statement` | 6 | 0.8 | 0.6 |
| `unicode normalization` | 5 | 0.8 | 0.6 |
| `reference counting garbage collection` | 8 | 1.0 | 0.8 |
| **mean** | | **0.72** | **0.56** |

BM25 wins or ties on every query. The main TF-IDF failure is visible in the pools: the 342,806-token
changelog is TF-IDF's **#1 result for 7 of the 10 queries** — it mentions everything, and nothing in
TF-IDF penalizes length. The worst query for both is `f-string formatting`: the tokenizer splits
`f-string` into `f` + `string` and the query becomes the phrase `"f string"`, but the documents mostly
say `f-strings` — and without stemming `strings` ≠ `string`. PEP 701 (*Syntactic formalization of
f-strings*) is not in BM25's top 5. That is a tokenizer limitation, not a ranking one; Lab 8's semantic
search is the other fix. `dataclass frozen slots` stops at 0.4 for a similar reason: implicit AND drops
PEP 557 (*Data Classes*), which only ever writes `__slots__` — and Lab 1's rule 6 keeps identifiers whole,
so the token is `__slots__`, never `slots`. Every tokenizer rule is a trade-off that shows up in ranking.

**What I learned.** `len()`, `in`, `[]`, `with`, `sorted()` are protocols, and implementing three
methods of `Mapping` bought a dozen behaviors. Decorators are just `f = timed(f)`; `wraps` keeps the
identity of what they wrap; `lru_cache` needs hashable arguments, which is one more reason the scorers
are frozen dataclasses. And measuring ranking — even with ten queries — showed a problem (length
normalization) that no unit test would have.

---

## Lab 4 — Make it a real tool

**Built:** the `findex` command (`findex.cli`, typer + rich), installed by `[project.scripts] findex =
"findex.cli:app"`:

```text
$ findex --help
 Usage: findex [OPTIONS] COMMAND [ARGS]...

 findex - full-text search over a corpus you choose.

╭─ Options ────────────────────────────────────────────────────────────────╮
│ --verbose             -v      <int>  -v: info, -vv: debug (timings), to  │
│                                      stderr                              │
│                                      [default: 0]                        │
│ --version                            Show the version.                   │
│ --install-completion                 Install completion for the current  │
│                                      shell.                              │
│ --show-completion                    Show completion for the current     │
│                                      shell, to copy it or customize the  │
│                                      installation.                       │
│ --help                               Show this message and exit.         │
╰──────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────╮
│ index   Build an inverted index from a corpus and save it.               │
│ search  Ranked search with highlighted snippets.                         │
│ stats   Corpus and index statistics (the Lab 1 numbers, from the index). │
╰──────────────────────────────────────────────────────────────────────────╯

$ findex search index.bin "TaskGroup" -k 1 --json
{"rank": 1, "doc_id": 935, "score": 11.1514, "title": "Command-line introspection tools", "path": "pydocs/library/asyncio-tools.txt", "snippet": "… Finished: {track}\") async def album(name, tracks): async with asyncio.TaskGroup() as tg: for track in tracks: tg.create_task(play(track), …", "highlights": ["TaskGroup"]}

$ findex search index.bin "(event OR"; echo "exit $?"
error: bad query: query ends where a term was expected
exit 2
```

- `index` shows a rich progress counter on stderr; `search` prints a rich table with highlighted snippets,
  or one JSON object per line with `--json`; `stats` prints the Lab 1 numbers from the index (collection
  and document frequencies come from the binary header without decoding any postings).
- **stdout is data, stderr is everything else.** Logging is configured once, in the CLI callback
  (`-v` → INFO, `-vv` → DEBUG with the `@timed` lines); library modules only call
  `logging.getLogger(__name__)`. Expected failures (missing index, bad query, old index format) print one
  line and exit 1 or 2 — no tracebacks. A test asserts `stdout == ""` for `findex index`.

### Typed: pyright strict, zero errors

`[tool.pyright] typeCheckingMode = "strict"` over `src/`, `tests/` and `scripts/`: **0 errors**. Most of the
code was annotated from Lab 1; strict mode found 12 issues, all real:

- `@contextmanager` on a function annotated `-> Iterator[Index]` is deprecated in typeshed: the decorator
  needs a `Generator` (it calls `.throw()` into it to deliver the `with` body's exception).
- `frozenset()` without a type argument is `frozenset[Unknown]`; `functools._CacheInfo` is private (the
  `Searcher` now returns its own `CacheStats` `NamedTuple`); tests passed plain `str` where a `Literal`
  (`Engine`, `Format`) was required.
- PEP 695 is used where it pays: `def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]` keeps the exact
  signature of decorated functions, and `type Format = Literal["binary", "pickle"]`. `Scorer` is a
  `typing.Protocol`: `TfIdf` and `BM25` satisfy it structurally, without inheriting anything.
- One `# type: ignore` remains, in a test, on purpose: it assigns to a frozen dataclass to prove it raises.
- One wrinkle: typer cannot read a PEP 695 `type` alias, so the CLI options use `StrEnum`s instead.

### Tested: 77 tests, 7 properties, 94% coverage

`uv run pytest` — **77 tests** (`-m "not slow"` skips the 2 slow ones during development), structured as
the lab asks: a parametrized tokenizer table, a `conftest.py` fixture corpus with every input format (text
tree, Markdown, PEP header, JSON-lines), index build, merges, parser trees, the three ranking sanity checks
as tests, save/load round trips, the CLI through `typer.testing.CliRunner` (exit codes, `--json` parsing,
stdout/stderr separation), and the old `python -m` entry points.

**Seven Hypothesis properties** (`tests/test_properties.py`): tokenizing is idempotent; tokens are never
empty or padded; `token_spans` (snippets) yields exactly the tokens `tokenize` (indexing) does; the
merges equal the `set` oracle and stay sorted; `load(save(index)) == index` for random corpora with and
without positions; and `parse(explain(tree)) == tree` for random query trees.

**The bug Hypothesis found.** The third property failed within a second, on a one-character input:
`'ͅ'` (COMBINING GREEK YPOGEGRAMMENI). It is a combining mark — not a word character — but
`casefold()` turns it into the letter `ι`. Lab 1's tokenizer case-folded the whole text *before* finding
words, so it produced a token `ι` from nothing, while the snippet code (which finds words first, to keep
offsets) produced none: the index and the snippets disagreed about what a word is. The fix changed rule 1
of the policy — find words in the NFC text, *then* case-fold each word — with an ASCII fast path
(`str.isascii()` is an O(1) flag check, and for ASCII `lower() == casefold()`). On this corpus the tokens
are identical (4,152,788, same vocabulary), and the lazy `stats` peak dropped from 41.2 to 16.6 MiB,
because the 2.4 MB changelog is no longer copied whole three times.

**Coverage:** `uv run pytest --cov` — **94%** of statements and branches (1,115 statements, branch
coverage on). Not covered, and why: the `findex/__main__.py` shim (it only calls the app), the
`PackageNotFoundError` fallback for a source tree that was never installed, the AND-fallback for phrase
queries on an index built without positions, the error paths for corrupted or old-format index files, and
a few defensive branches in the snippet window and the query parser. I would rather test the error paths
users hit (bad query, missing index) than chase the last 6%.

### Packaged, CI'd, released

- `pyproject.toml` (PEP 621): metadata, MIT licence, runtime deps (`typer`, `rich`), a `dev` dependency
  group, `[project.scripts]`, and the ruff / pyright / pytest / coverage configuration; `uv.lock` is
  committed. `src/` layout: the tests import the installed package, never the working directory.
- **CI** (`.github/workflows/ci.yml`) on every push and PR: ruff check + format, pyright strict, pytest
  with coverage on **Python 3.14 and the free-threaded 3.14t** (ready for Lab 5), and a job that builds the
  wheel and installs it with `uv tool install`.
- **Release v0.4.0** with the wheel attached:
  [github.com/batpepe/findex/releases/tag/v0.4.0](https://github.com/batpepe/findex/releases/tag/v0.4.0).
- **Install-from-release verified in a fresh container** (nothing from this machine inside it):

  ```text
  $ docker run --rm python:3.14-slim sh -c 'pip install -q uv && \
      uv tool install https://github.com/batpepe/findex/releases/download/v0.4.0/findex-0.4.0-py3-none-any.whl && \
      PATH="$HOME/.local/bin:$PATH" findex --version'
  findex 0.4.0
  ```

**What I learned.** Type hints cost nothing at runtime and catch real bugs before running anything — but
they also have consumers I had not thought of (typer and dataclasses read them; typer cannot read a PEP 695
alias). Example tests check what I thought of; one Hypothesis property found a Unicode case I would never
have written down, and it was a genuine disagreement between two halves of the program. And "shipped" means
a stranger's clean machine can run it from one command — which I only know because I tried it.

---

## Lab 5 — Concurrency and the GIL

**Built:** `findex.parallel` — map/reduce indexing.

- **Map:** `build_partial(paths, root)` is a module-level (so: picklable) function that indexes a
  contiguous chunk of corpus *files* and returns a `PartialIndex` made only of `array('I')` buffers
  (`doc_ids`, `tfs`, concatenated positions), compressed texts and short lists — Lab 2's compact
  representation, now justified twice: it is small in memory *and* it pickles as raw bytes. Workers receive
  paths, never documents.
- **Reduce:** `merge(partials)` concatenates every term's arrays in chunk order. Each partial numbers its
  documents from 0 and the merge shifts them by the documents in earlier chunks, so the final ids are
  exactly those of the serial walk. (Ranges can't be fixed up front: a JSON-lines file holds an unknown
  number of documents.) Contiguous chunks of the sorted path list keep every merged postings list sorted.
- **Chunking by bytes, not by count** (`chunk_paths`): the changelog is 2.4 MB, the median document 15 KB.
  Parallel runs use 4 chunks per worker so a slow chunk doesn't leave the other workers idle.
- **One flag, three executors:** `findex index data/ --executor {serial,threads,processes} --workers N`. The
  same `build_partial`/`merge` run in a loop, a `ThreadPoolExecutor` or a `ProcessPoolExecutor`. Results
  are collected in submission order; a worker's exception re-raises in the parent with the worker's
  traceback attached, and the pool is shut down with `cancel_futures=True` — no hang, no waiting for chunks
  nobody will merge (tested with an unreadable file, for threads and processes).
- **Byte-identical:** a test builds the fixture corpus with serial, threads (1 and 3 workers) and processes
  and compares the saved files byte for byte with Lab 4's `build_index`; on the real corpus `cmp` agrees.
- **spawn-safe:** macOS starts workers with `spawn` — a fresh interpreter that re-imports the main module.
  `python -m findex` therefore guards `app()` with `if __name__ == "__main__"`, and the console-script entry
  point is guarded by construction. (I hit the failure mode for real: a benchmark piped into `python -` from
  stdin crashed every worker with `FileNotFoundError: '<stdin>'` — a spawned child cannot re-import a script
  that has no file.)
- **Default:** `processes` with at most 8 workers on a normal build — it wins below — and `threads` when
  `sys._is_gil_enabled()` is false.

### The benchmark

Machine: Apple M4 (10 cores = 4 performance + 6 efficiency), 16 GB, macOS 27.0.1, on power; Python 3.14.5
(GIL) and 3.14.6 free-threaded (`uv python install 3.14t`, separate venv). Task: build the full index
(1,297 documents, positions + compressed texts, no save). Every configuration ran 3 times, each in a fresh
child process; medians, with min–max for wall time. CPU = user + system of the parent **and** its worker
processes; RSS = peak of the parent; merge = the reduce step alone.
`uv run python scripts/bench_parallel.py data/ --ft .venv-ft/bin/python`:

| Executor | Workers | Wall (s, median, min–max) | CPU (s) | Peak RSS | Merge (s) | Speedup |
|---|---|---|---|---|---|---|
| serial | 1 | 2.08 (2.08–2.10) | 2.14 | 136 MiB | 0.01 | 1.00× |
| threads | 1 | 2.18 (2.18–2.20) | 2.24 | 168 MiB | 0.09 | 0.95× |
| threads | 2 | 1.86 (1.83–1.86) | 2.38 | 186 MiB | 0.08 | 1.12× |
| threads | 4 | 1.93 (1.91–1.99) | 2.57 | 209 MiB | 0.11 | 1.08× |
| threads | 8 | 2.07 (2.07–2.07) | 2.74 | 236 MiB | 0.14 | 1.01× |
| threads | 10 | 2.14 (2.13–2.17) | 2.76 | 244 MiB | 0.14 | 0.97× |
| processes | 1 | 2.59 (2.58–2.60) | 2.73 | 204 MiB | 0.16 | 0.80× |
| processes | 2 | 1.62 (1.62–1.63) | 3.06 | 219 MiB | 0.14 | 1.28× |
| processes | 4 | 1.30 (1.30–1.31) | 3.49 | 210 MiB | 0.21 | 1.60× |
| processes | 8 | **1.15** (1.14–1.16) | 5.14 | 236 MiB | 0.17 | **1.81×** |
| processes | 10 | 1.22 (1.22–1.22) | 5.67 | 247 MiB | 0.21 | 1.70× |
| threads, 3.14t (no GIL) | 1 | 2.08 (2.04–2.12) | 2.31 | 177 MiB | 0.06 | 1.00× |
| threads, 3.14t (no GIL) | 2 | 1.22 (1.21–1.23) | 2.42 | 220 MiB | 0.09 | 1.70× |
| threads, 3.14t (no GIL) | 4 | **0.91** (0.90–0.91) | 2.60 | 239 MiB | 0.11 | **2.29×** |
| threads, 3.14t (no GIL) | 8 | 1.07 (1.07–1.08) | 5.62 | 300 MiB | 0.14 | 1.94× |
| threads, 3.14t (no GIL) | 10 | 1.17 (1.17–1.18) | 7.37 | 319 MiB | 0.13 | 1.77× |

![speedup vs workers](docs/speedup.png)

### What the numbers say

**Threads under the GIL did not scale, because indexing is CPU-bound Python.** Only the thread holding
the GIL executes bytecode; tokenizing (a regex scan, a `casefold`, a dict update per token) never waits
for anything, so it never gives the GIL up except when the interpreter forces a switch every 5 ms. Eight
threads take turns on one core's worth of work, and the CPU column grows (2.14 → 2.76 s) from the switching
and contention. The only overlap — the 1.12× at two threads — is the work done in C *with the GIL
released*: `zlib.compress` of each document's text and the file reads.

**Processes scaled — and cost.** Each process has its own interpreter and its own GIL, so tokenizing really
runs on 8 cores: 1.81× at 8 workers. The price is visible in three columns. *CPU* rose from 2.14 to 5.14 s:
eight interpreters starting and importing `findex` (0.14 s of wall time, more of CPU), pickling, and work
on the efficiency cores, which need more cycles for the same instructions. *RSS* rose by 100 MiB in the
parent alone, which receives every partial. And **the pickling tax**: the 32 partials are 47.4 MiB of
pickles — 0.31 s to dump in the workers, **0.19 s to load in the parent, sequentially**
(`scripts/analyze_parallel.py`). Shipping arrays instead of `Posting` objects is what keeps this small:
Lab 2 measured slotted dataclasses at 7× slower to unpickle.

**Amdahl's law, in my own timings.** With 8 processes the strictly sequential part is pool start-up (0.14 s)
+ unpickling (0.19 s) + merge (0.17 s) ≈ 0.5 s of the 2.08 s serial run, i.e. ~24%: no number of cores can
beat 1 / 0.24 ≈ 4.2×. And one *document* cannot be split: the chunk holding the 2.4 MB changelog takes
0.34 s on its own, so the map phase can't end before it — together that caps this corpus at ~2.4×, and
the measured 1.81× is mostly that ceiling plus efficiency cores past 4 workers. Going from 8 to 10 workers
made it *slower* (1.22 s): the last two are E-cores, and more chunks mean more to pickle and merge.

**Free-threaded 3.14t: the same threaded code, now parallel.** With the GIL disabled, threads execute
bytecode simultaneously and share memory — no start-up, no pickling, no copy of partials — so 4 threads
reach **2.29×** (0.91 s), better than any process count. The single-thread overhead of the no-GIL build is
negligible here (2.08 s vs 2.08 s). Past 4 workers it degrades and the CPU column doubles (2.60 → 5.62 s):
the threads land on efficiency cores, and they contend on shared objects (every thread increments the
reference counts of the same compiled regex, module globals and interned strings, which the free-threaded
runtime must do atomically).

**The race the GIL was hiding.** My indexer was correct on 3.14t at the first run because it shares
nothing: each worker fills its own `PartialIndex`, and only finished partials cross threads. To see what
would have happened otherwise I wrote the naive version — four threads counting the corpus' tokens into
*one shared* `Counter` (`scripts/race_demo.py`, 4,152,788 tokens expected):

| Variant | GIL build (3.14.5) | Free-threaded (3.14.6t) |
|---|---|---|
| `shared[token] += 1` on one Counter | 4,152,788 — ok (0.36 s) | **4,108,690 — lost 44,098** (0.69 s) |
| `shared.update(tokens)` on one Counter | 4,152,788 — ok (0.17 s) | **3,810,800 — lost 341,988** (0.35 s) |
| one Counter per thread, merged after `join` | 4,152,788 — ok (0.20 s) | 4,152,788 — ok (**0.11 s**) |
| `+=` under a `threading.Lock` | 4,152,788 — ok (0.51 s) | 4,152,788 — ok (1.02 s) |

`counter[k] += 1` is a read, an add and a write — three bytecodes on 3.14 (`BINARY_OP 26 ([])`,
`BINARY_OP 13 (+=)`, `STORE_SUBSCR`); two threads that read the same old value both write `old + 1`, and one increment
vanishes. Under the GIL a switch inside that window is rare enough that every run came out right, which
means the code was *wrong but lucky*. Without the GIL it lost 1% of the updates — and `Counter.update`,
whose C loop does the same get-then-set per element, lost 8%. The fix is not the lock (correct, and the
slowest of all: every token now serializes on it) but **not sharing**: per-thread counters merged at the end,
which is also the fastest variant — the same design `build_partial`/`merge` uses.

**The counter-example: I/O-bound work, where plain GIL threads do scale.** Fetching 40 pages of the
library reference from docs.python.org (`scripts/bench_io_threads.py`, after an untimed warm-up pass):

| Threads | Pages | Wall (s) | Speedup |
|---|---|---|---|
| 1 | 40 (2.7 MiB) | 10.49 | 1.0× |
| 4 | 40 (2.7 MiB) | 2.65 | 4.0× |
| 16 | 40 (2.7 MiB) | 0.85 | 12.3× |

A thread waiting on a socket has released the GIL, so 16 threads overlap 16 waits on one core. Same
interpreter, same GIL, opposite result — the question is never "threads or processes" but "what is my
program waiting for". Thousands of such waits, with no thread per request, is Lab 6.

**What I learned.** The GIL protects the interpreter, not my invariants: code can be correct by luck under
it. Parallel speedup is bounded by everything that stays sequential — start-up, data transfer, the merge,
and the largest indivisible task — and I can only see those by measuring them separately. The cheapest
synchronization is none: design the data so workers never share mutable state.

---

## Lab 6 — The async crawler

**Built:** `findex.crawler` and `findex crawl`, one thread, one event loop:

```text
seeds -> [frontier: per-host queues] -> N workers (TaskGroup) -> [pages: asyncio.Queue] -> async for page -> JSON lines
```

- **`fetch`** (`crawler/fetch.py`): `httpx.AsyncClient.get` under `asyncio.timeout(10)`, which covers the
  whole request including the body; up to 3 retries for timeouts, connection errors, 429 and 5xx, with
  `min(cap, 0.5 · 2ⁿ) + uniform(0, 1)` backoff; a `Retry-After` header (seconds or HTTP date) replaces the
  computed delay when it is longer; no retry for other 4xx; a redirect loop is reported, not retried. It
  never raises for network or HTTP trouble — failures are data in the `FetchResult` (status, bytes,
  elapsed, attempts, error).
- **`RobotsCache`** (`crawler/robots.py`): one `robots.txt` request per host even when 20 workers ask at
  once (the first one creates a task, the others `await asyncio.shield` the same task); 404 → allow all,
  401/403 or 5xx → disallow all (until we know better); `Crawl-delay` is honoured if it exceeds ours.
- **`crawl()`** (`crawler/crawl.py`): an async generator. `concurrency` workers in an `asyncio.TaskGroup`;
  a global semaphore, a per-host semaphore (`--per-host`) and a per-host minimum delay between request
  starts (`--delay`); URL canonicalization (case, default port, fragment, `./` and `../`, `index.html`,
  sorted query) and a `seen` set, so nothing is fetched twice (the redirect *target* is added too); a scope
  allowlist (URL prefixes: by default each seed's directory, so no foreign hosts, no other versions or
  translations); `selectolax` (lexbor) extracts title, main text and links **in a thread**
  (`asyncio.to_thread`); a clean stop at `--max-pages` (`Queue.shutdown`-style close of the frontier, then
  the busy workers are cancelled at their current `await`). The generator never `yield`s inside the task
  group — the group runs in its own task and hands pages over through a queue. Yielding from inside a task
  group or a timeout scope lets the consumer's cancellation land in the wrong task: that is PEP 789, which
  happens to be in this corpus.
- **`findex crawl`** streams one JSON line per page as it arrives (the write goes through
  `asyncio.to_thread` too), shows a `rich` live line (pages, pages/s, in flight, queued, errors), writes a
  per-URL `crawl.log` (status, bytes, time, attempts, final URL) and has `--debug` (asyncio debug mode,
  warn on any step blocking the loop > 50 ms). Indexing stays outside: `findex index data/` reads the JSON
  lines like any other corpus file.

**Ethics.** I crawled only documentation sites whose `robots.txt` allows everything —
packaging.python.org, pip.pypa.io, devguide.python.org and typing.python.org — all hosted on Read the
Docs, whose [crawler policy](https://docs.readthedocs.com/platform/stable/automated-access.html) asks for
fewer than 4 requests per second and a user agent with a contact: mine is `findex-crawler/0.6 (student
search-engine project; +https://github.com/batpepe/findex)`, at most 2 requests in flight per host, and
request starts 1.2 s apart per host — 4 hosts × 1/1.2 s = **3.3 requests/s in total**. The pages are
copyrighted by their projects: they are indexed for search, not redistributed (`data/` is not in git), and
the web UI links back to the source.

```text
$ findex crawl https://packaging.python.org/en/latest/ https://devguide.python.org/ \
    https://typing.python.org/en/latest/ https://pip.pypa.io/en/stable/ \
    --max-pages 500 --concurrency 10 --per-host 2 --delay 1.2 --out data/crawl/python-sites.jsonl --debug
crawled 267 pages (269 fetched, 2 errors, 0 disallowed by robots.txt) in 268.9 s (1.0 pages/s)
-> data/crawl/python-sites.jsonl; per-URL log: crawl.log
```

The four sites have 267 reachable pages in scope (85 + 82 + 60 + 40); the crawl stopped on its own when the
frontier was empty. Median fetch 175 ms, p95 335 ms; 9 redirects followed (missing trailing slashes); 2
404s, not retried. `--debug` reported **no step that blocked the loop for 50 ms**.

### The benchmark — and the bug it found

Same seeds, `--max-pages 200`, concurrency 1 / 5 / 20, each run in a fresh process
(`uv run python scripts/bench_crawl.py real`). First version — one FIFO `asyncio.Queue` as the frontier:

| Concurrency | Pages | Wall (s) | Pages/s | Errors | Peak RSS |
|---|---|---|---|---|---|
| 1 | 200 | 232.3 | 0.9 | 0 | 58 MiB |
| 5 | 200 | 223.7 | 0.9 | 0 | 61 MiB |
| 20 | 200 | 173.5 | 1.2 | 0 | 66 MiB |

Five times the concurrency, the same speed. The cause was **head-of-line blocking**: links are discovered in
bursts — one page of the packaging guide yields 80 links to the same host — so the FIFO frontier handed
every worker a URL of *that* host, and they all queued behind its 1.2 s politeness delay while the other
three hosts sat idle. More workers only helped by luck (some held a URL of another host).

The fix is the classic crawler design (Mercator's "back queues"): **`HostFrontier`**, one FIFO per host plus
a heap of `(time this host may be contacted again, host)`. `get()` hands out a URL from whichever host is
ready first, and pushes that host's next turn `delay` seconds ahead. It keeps the `asyncio.Queue` interface
(`put_nowait`, `get`, `task_done`, `join`, `shutdown`), so the workers didn't change. A unit test feeds it
`a/1, a/2, a/3, b/1` and checks that `b/1` comes second, not fourth. Same benchmark, after:

| Concurrency | Pages | Wall (s) | Pages/s | Errors | Peak RSS |
|---|---|---|---|---|---|
| 1 | 200 | 64.1 | 3.1 | 0 | 74 MiB |
| 5 | 200 | 63.8 | 3.1 | 0 | 77 MiB |
| 20 | 200 | 63.8 | 3.1 | 0 | 78 MiB |

**2.7–3.6× faster, and now flat — on purpose.** 3.1 pages/s is the politeness ceiling: 4
hosts × one request start per 1.2 s = 3.3 requests/s. A single worker already reaches it, because a fetch
takes 175 ms and the next host is always ready; 20 workers can't go faster without breaking the per-host
limit. **That is what the per-host limits cost:** throughput is set by `hosts / delay`, not by concurrency.
I kept them because the alternative is a denial-of-service attack on volunteers' documentation hosting,
and a ban for my IP.

**What concurrency buys when the server allows it.** Against a local mirror of the HTML docs
(`scripts/get-mirror.sh` + `scripts/mirror_server.py`: docs.python.org's HTML archive, served with 150 ms of
simulated network latency, my server so no per-host limit), `--max-pages 300`:

| Concurrency | Pages | Wall (s) | Pages/s | Errors | Peak RSS |
|---|---|---|---|---|---|
| 1 (effectively sequential) | 300 | 51.5 | 5.8 | 1 | 162 MiB |
| 5 | 300 | 11.4 | 26.2 | 1 | 170 MiB |
| 20 | 300 | 3.8 | **78.9** | 0 | 86 MiB |

**13.6× with 20 coroutines, in one thread.** Each fetch spends ~150 ms waiting and well under 1 ms of CPU, so
one event loop overlaps 20 waits. Lab 5's indexer was the opposite workload — all CPU, no waiting — where
one thread can't overlap anything and threads only take turns on the GIL. Async wins when the program is
waiting; it does nothing for computing. (A thread per request would also have scaled here, but each thread
costs a stack and an OS scheduling slot; a coroutine costs a few KB, and there are no locks to get wrong
between `await`s.)

### Not blocking the loop

HTML parsing is CPU work: `extract()` takes **0.9 ms median, 7 ms p95 and 219 ms max** on the 595 pages of
the mirror (the largest index pages), and tokenizing a page's text 0.4 ms median / 101 ms max. As a
control I ran the concurrency-20 mirror crawl with `extract()` called directly in the coroutine: asyncio's
debug mode reported **2 steps blocking the loop for more than 50 ms** — during each, all 20 in-flight
requests stalled. With `asyncio.to_thread`: **0**. Had I indexed inline, the worst page would have frozen
the loop for ~100 ms more, every time: that is why the crawler only writes JSON lines and `findex index` runs
afterwards, in its own process pool (Lab 5).

### A failure, handled

Against the mirror started with `--fail-rate 0.05` (5% of first requests answer `503 Retry-After: 1`) and a
burst of 10 workers, two kinds of transient failure appeared — the 503s I injected and, unplanned,
connection refusals when the test server's listen backlog of 5 overflowed. `-v` shows the
retries, `crawl.log` that every one of them ended in a 200 (`attempts` is the column before the URL):

```text
INFO findex.crawler.fetch: retry 1/3 for http://127.0.0.1:8768/library/ in 0.8s (ConnectError: All connection attempts failed)
INFO findex.crawler.fetch: retry 1/3 for http://127.0.0.1:8768/howto/ in 0.6s (ConnectError: All connection attempts failed)
INFO findex.crawler.fetch: retry 1/3 for http://127.0.0.1:8768/download.html in 1.0s (HTTP 503)

2026-10-22 07:51:44,326 200    73961    904ms 2 http://127.0.0.1:8768/library/
2026-10-22 07:51:44,579 200    11404   1159ms 2 http://127.0.0.1:8768/download.html
2026-10-22 07:51:45,486 200    19322   2064ms 3 http://127.0.0.1:8768/howto/
```

32 URLs needed a retry, none was lost. `download.html` waited exactly the server's `Retry-After: 1`
(longer than the ~0.5–1.5 s backoff); `/howto/` needed three attempts. On the real sites the only failures
were two 404s — dead links — which were logged and, correctly, not retried.

**What I learned.** `await` is where a coroutine lets others run, so everything between two `await`s is
atomic — I never needed a lock — and anything slow between two `await`s freezes every other task. Structured
concurrency (TaskGroup, cancellation, shutdown) made "stop at 500 pages" a few lines. And the scheduling
policy matters more than the concurrency level: the same 20 workers went from 1.2 to 3.1 pages/s by changing
which URL they were given, not how many there were.

---

## Lab 7 — Search on the web

**Built:** `findex.web` — one FastAPI app with a JSON API and an HTML UI, served by uvicorn (`findex serve`
or the Docker image).

| Route | What |
|---|---|
| `GET /search?q=&k=&page=&scorer=` | `SearchResponse{query, scorer, total, page, k, pages, took_ms, results[]}`; each hit: `doc_id, title, path, url, score, snippet, highlights` |
| `GET /docs/{doc_id}` | the document (404 with `{"detail": "no document 999"}` if unknown) |
| `GET /stats` | documents, terms, tokens, index size, uptime, query-cache hits/misses |
| `GET /health` | `{"status": "ok", "documents": N}`, or **503** while no index is loaded |
| `GET /docs` | the OpenAPI browser FastAPI generates from the type hints |
| `GET /`, `GET /doc/{id}` | the UI: search box, BM25/TF-IDF toggle, highlighted snippets, pagination, document view |

- **Validation is the type hints.** `q: Annotated[str, Query(min_length=1, max_length=200)]`,
  `k: Annotated[int, Query(ge=1, le=100)]`, `scorer: Literal["bm25", "tfidf"]` — `?k=abc` or `?scorer=pagerank`
  is a 422 with a JSON body that says which field and why, without a line of validation code. A query with
  bad *syntax* is a 400 (`"bad query: missing ')' for '(' at offset 0"`); `page × k > 1000` is refused.
- **Pydantic only at the edge.** `SearchHit`, `SearchResponse`, `DocumentResponse`... are the wire format;
  inside, the Lab 2–3 dataclasses are unchanged. `SearchHit.from_result()` is the whole seam. Config is a
  `pydantic-settings` `Settings` read from the environment (`INDEX_PATH`, `LOG_LEVEL`, `LOG_JSON`,
  `CORS_ORIGINS`, `SEARCH_CACHE_SIZE`; a `.env` in development) — no paths in code.
- **The index is loaded once,** in the `lifespan` async context manager (Lab 3's `open_index`, async), and
  reaches the routes through `Depends(get_searcher)`. Tests swap it for the fixture corpus with
  `app.dependency_overrides[get_searcher] = ...` — no index file, no lifespan. If loading fails, the app
  still starts and `/health` says 503, so the platform keeps it out of rotation instead of crash-looping.
- **Every request gets an id** (`X-Request-ID`, generated or passed through) and one log line; an unexpected
  exception is logged with that id and the traceback and answered as a clean
  `500 {"detail": "internal server error", "request_id": "..."}`. Logs are JSON lines on stdout with
  `LOG_JSON=true` (the default in the image).
- **Licensing, checked:** the public image indexes only the Python documentation (PSF License) and the PEPs
  (public domain / CC0), which may be redistributed; for crawled pages the API and UI return a 600-character
  excerpt and the link to the source (`truncated: true`), never the whole page.
- **Tests:** 21 `TestClient` tests with the dependency overridden — success, pagination, eight 422s, the 400,
  404, `/health` 503 without an index, the lifespan loading from `INDEX_PATH`, the clean 500 with the request
  id, the OpenAPI paths, and the UI pages.

### `def` or `async def`? Measured

`/search` ranks with Python code — CPU work, ~20 ms for these queries. As `async def` it runs on the event
loop and blocks it; as a plain `def` FastAPI runs it in its thread pool. `scripts/bench_web.py` (each variant
its own uvicorn, query cache off) sends 20 different searches at once plus one `/health` probe 20 ms later:

| `/search` is | All 20 done (s) | Median search (ms) | `/health` probe waited (ms) |
|---|---|---|---|
| `async def` | 0.44 | 430 | 410 |
| `def` (thread pool) | 0.43 | 421 | 398 |
| `def` (thread pool), free-threaded 3.14t | **0.26** | **160** | **53** |

Under the GIL the two are nearly the same, and that is the lesson: `def` keeps the event loop *free*, but
the loop then has to win the GIL from 20 busy threads to answer `/health`, so the probe still waits ~400 ms.
Only on the free-threaded build does the thread pool actually run the searches in parallel (2.6× better
median, `/health` answered in 53 ms). **Decision: `def`** — it is the correct shape (a 200 ms query never
freezes static files and health checks behind it, and the same code scales on a no-GIL interpreter), and
CPU-bound throughput comes from processes (below).

### Load test

`oha`, 15 s, 50 concurrent clients cycling through 20 realistic queries, query cache off, keep-alive off (a
new connection per request, so with several worker processes an idle one accepts it), median of 3 runs
(`uv run python scripts/bench_web.py index.bin .venv-ft/bin/python`):

| Target | Route | Concurrency | RPS | p50 (ms) | p95 | p99 |
|---|---|---|---|---|---|---|
| local, 1 worker, `async def` | `/search` | 50 | 47 | 1088 | 1777 | 1826 |
| local, 1 worker, `def` | `/search` | 50 | 46 | 1058 | 1680 | 1717 |
| local, 4 workers, `def` | `/search` | 50 | **150** | **306** | 629 | 976 |
| local, 1 worker, `def`, 3.14t (no GIL) | `/search` | 50 | 76 | 558 | 1624 | 2956 |
| deployed: findex.batpepe.online, 1 pod (1 CPU), through Cloudflare, cached path | `/search` | 20 | 87 | 175 | 290 | 1190 |

- **1 worker: ~46 RPS either way** — one GIL, one core's worth of Python; 50 clients just queue, so p50 ≈ 50
  × 21 ms ≈ 1 s. The `async def` row has the slightly worse tail: the loop serves the queued requests strictly
  in turn and cannot even accept connections while ranking.
- **4 workers: 3.2× the throughput, a third of the latency** — four processes, four GILs, four cores (Lab 5
  again). And because the index is `mmap`ed, the four processes share one copy of it in the OS page cache.
- **Free-threaded: 1.6× with one process,** but the worst p99 — threads contend on shared objects (Lab 5's
  observation, now in a server).
- I first ran oha with keep-alive and got 93 RPS for 4 workers in one run and 52 in the next: with 50
  long-lived connections, whichever worker accepted most of them did most of the work. Measuring a server
  is measuring its connection handling too.
- **Deployed: 87 RPS at a 175 ms median, through Cloudflare** — one 15 s run from my Mac:
  `oha -z 15s -c 20 --disable-keepalive 'https://findex.batpepe.online/search?q=global+interpreter+lock'`.
  One query with the production query cache on, so this is the cached path, not ranking. With keep-alive off
  every request is a new TCP + TLS handshake with the Cloudflare edge: dial-up averaged 145 ms of the 229 ms
  mean, and the slowest request spent 2.10 s of its 2.18 s connecting. Most of the latency is the connection,
  not the pod.

### What the GIL was hiding, part 2

The free-threaded row first came out as a wall of 500s: `ValueError: operation forbidden on released
memoryview object`. Lab 2's `MappedPostings` kept **one** `memoryview` of the memory-mapped index and every
request sliced it. Under the GIL that is safe; on 3.14t, slicing one shared memoryview from many threads is
not — a 20-line repro (16 threads slicing one view) fails in all 16 threads on 3.14t and never with the GIL,
while a fresh `memoryview(mmap)` per call, or a copied slice, never fails. Each call now takes its own view
(free: a view is a small header, the data is not copied), and a regression test searches one mapped index
from 16 threads — CI runs it on 3.14t. The same lesson as Lab 5's `Counter`, one level down: shared state
was correct only because the GIL serialized it.

### Docker

A multi-stage `Dockerfile` following the uv guide: the **builder** (`ghcr.io/astral-sh/uv:0.12.1-python3.14`)
installs the locked dependencies, then the project, into `/app/.venv` (dependency layer cached until
`uv.lock` changes); the **index** stage downloads the corpus and builds `index.bin` in the image; the
**runtime** is `python:3.14-slim` with security updates applied, only the venv and the index, a non-root user
(uid 10001), a `HEALTHCHECK` on `/health`, and uvicorn as PID 1 in exec form so `SIGTERM` reaches it:

```text
$ docker build -t findex . && docker run -d -p 8000:8000 --name findex findex
$ curl -s localhost:8000/health
{"status":"ok","documents":1297}
$ docker stop findex && docker logs findex | tail -3
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [1]
```

323 MB (Debian slim + Python 32 MB + venv 48 MB + index 34 MB). `docker run -v $PWD/index.bin:/data/index.bin
-e INDEX_PATH=/data/index.bin` serves any other index.

### Deployed: findex.batpepe.online

```text
git push -> image.yml: build -> Trivy (CRITICAL/HIGH gate) -> smoke test -> ghcr.io/batpepe/findex:<commit sha>
          -> homelab repo: k8s-infrastructure/apps/findex pins that sha -> ArgoCD syncs it to K3s
user -> Cloudflare edge (TLS) -> Tunnel (outbound-only, no open port at home) -> Traefik -> findex pod
```

- **The image** is published by `.github/workflows/image.yml` in this repo, in the order that keeps a bad
  image out of the registry: build, then the Trivy scan (fails on fixable CRITICAL/HIGH CVEs - the reason the
  Dockerfile runs `apt-get upgrade`), then a smoke test against the running container, and only then the
  push. Tags are the commit SHA (immutable, what the cluster pins) and the version for releases. The gate
  earned its place on its first run: 5 HIGH CVEs, none in findex or its dependencies - all in the libraries
  pip vendors (urllib3 2.7.0, msgpack 1.1.2, setuptools 70.3.0's `pkg_resources`) inside the base image's pip.
  Nothing installs packages at runtime, so the runtime stage now uninstalls pip instead of waiting for a new
  base image: less in the image is less to patch.
- **The cluster** is my homelab K3s node, managed from
  [batpepe/devops-homelab-k3s-hybrid-cloud](https://github.com/batpepe/devops-homelab-k3s-hybrid-cloud) by
  ArgoCD (app-of-apps, `prune` + `selfHeal`). `k8s-infrastructure/apps/findex/findex.yaml` is a Deployment,
  a Service and an Ingress for `findex.batpepe.online`; the hostname is added to the Terraform-managed DNS
  allowlist of the Cloudflare Tunnel. The pod runs as uid 10001 with a read-only root filesystem, no Linux
  capabilities and the `RuntimeDefault` seccomp profile; it asks for 384 MiB and is capped at 768 MiB and one
  CPU. Configuration is environment variables only - the same image runs in CI, with `docker run` and here.
- **Health check wired, on purpose asymmetric:** the *readiness* probe is `GET /health`, so the pod gets no
  traffic until the index is loaded (and Traefik stops sending traffic if it ever reports 503). The
  *liveness* probe only checks that the port accepts connections: a pod whose index failed to load answers
  503 by design (the Lab 7 lifespan), and restarting it would just repeat the failure in a loop. The
  blackbox exporter probes `https://findex.batpepe.online/health` every minute from inside the cluster, so
  an outage reaches Prometheus and the Telegram alert, and the portal at batpepe.online shows a live dot.
- **Logs** are JSON lines on stdout (`LOG_JSON=true`), one per request with its id:
  `kubectl -n apps logs deploy/findex -f`, or Grafana/Loki.

**What I learned.** ASGI is the reason one process can serve thousands of waiting connections, and FastAPI
puts most of a service's contract in type hints: validation, the 422s, the OpenAPI page. But choosing `def` or
`async def` is not a style choice - it decides whether CPU work blocks the event loop - and under the GIL
even the thread pool runs one search at a time, which is why the 4-process row beats everything and why the
free-threaded build exposed a shared-`memoryview` race the GIL had hidden. Deploying added the operational
half: a health endpoint is a contract with the orchestrator (ready vs alive are different questions), and
"the same image everywhere, configured by the environment" is what made the CI smoke test, my laptop and
the cluster run identical bytes.

---

## Lab 8 - Fast, then smart

**Built:** a profile-driven speed-up of the hot paths, `findex.vector` (the NumPy engine: Boolean masks,
vectorized BM25/TF-IDF, `np.argpartition` top-k), a snippet scanner that runs in C, `pytest-benchmark` in CI,
and `findex.semantic`: chunk embeddings with all-MiniLM-L6-v2, cosine search as one matrix-vector product,
and hybrid search with reciprocal rank fusion - in the CLI (`--mode`), the API (`?mode=`) and the UI.

**Where these numbers come from.** A cloud VM - Intel Xeon @ 2.1 GHz, 2 vCPUs, 8 GB, Linux, Python 3.14.8,
NumPy 2.5 with OpenBLAS - on the corpus rebuilt from the same sources: the docs' reStructuredText converted
to text (docs.python.org was not reachable from that machine) plus all PEPs, 1,301 documents, 4.29 M tokens.
It is close to the Mac's corpus (`python` is in 1,207 documents on both, `walrus` in 16, BM25's precision@5
is 0.74 vs 0.72), but milliseconds are not comparable with Labs 1-7; the before/after ratios are. Every
table says which script reproduces it.

### M1 - Profile first

`findex search index.bin "global interpreter lock"` (one shot) under `python -m cProfile -s cumtime`:
0.45 s, of which **imports 0.19 s**, loading the index 0.085 s (0.075 s of it parsing the JSON header),
**snippets 0.134 s**, and ranking - the part I would have optimized first - **6 ms**. The steady state is
clearer still. `scripts/profile_search.py` runs the same query in a loop with the cache off, under py-spy
(sampling, no distortion) - [before](docs/flame-search-before.svg), [after](docs/flame-search-after.svg):

| Where a new 3-term query spent its time (py-spy, before) | Share |
|---|---|
| `make_snippet` -> `token_spans`: re-tokenizing each result document in full | 61% |
| `make_snippet` -> `display_text`: NFC + apostrophes over each whole document | 17% |
| `make_snippet` itself (the token loop, the window) | 11% |
| ranking: parse, `frozenset`s, `decode_postings`, `BM25.score` per posting | 4% |

Scalene splits each line into Python time and native time, and that split changed my hypothesis. The two
biggest lines, both in `token_spans`, were *native*: `normalize("NFC", word.casefold())` (34% of CPU, all of it
in C) and the `yield` with `match.start()/end()` (34%: 21% Python, 13% C). Native, but 2.5 million tiny C calls
issued by a Python loop, one per token of documents up to 530,390 tokens long, to find ten highlights. Making
the Python faster would not help; making far fewer calls would. The index build ([before](docs/flame-index-before.svg),
[after](docs/flame-index-after.svg)) had the same pattern: per-word normalization 16%, `zlib.compress` 15%,
the per-token `yield`/`append` loop about 25%, `str.translate` 7%.

The hotspots, written down before changing anything, and what happened:

| # | Hotspot | Hypothesis | Change | Outcome |
|---|---|---|---|---|
| 1 | Snippets re-tokenize every result document | the work grows with the document, not with the hits; a C-level scan for just the query terms is enough | `hit_spans`: one regex (the terms joined by `\|`) over a case-folded copy, then only the candidates go through the tokenizer's rules | a new 3-term query: 31.2 -> 7.5 ms |
| 2 | `display_text` used `str.translate` with a dict | translate looks every character up in the dict; the four apostrophes are rare | four `str.replace` calls (a vectorized search each) | 56 ms -> 1.4 ms on the changelog |
| 3 | Ranking: a `Posting` object and a `score()` call per posting | interpreter overhead per posting, growing with document frequency | the NumPy engine (M2) | `python`: 2.72 -> 0.08 ms |
| 4 | Tokenizer: `normalize(casefold(w))` for every word of a non-ASCII document | most words in such documents are ASCII, where `lower()` is the same thing | per-word ASCII fast path | serial build 4.67 -> 3.69 s, byte-identical index |
| 5 | CLI start-up imports the crawler (httpx, asyncio) and `importlib.metadata` | a search never crawls | imports inside the commands, PEP 562 lazy `__version__` | one-shot `findex search`: 337 -> 275 ms |
| - | `zlib.compress` at level 6: 15% of the build | level 1 is ~3× faster | measured, *kept* level 6 | 0.79 -> 0.27 s, but texts 9.5 -> 11.0 MiB; the build is offline, the file ships in the image and sits in the page cache |
| - | JSON header parse: 75 ms of a one-shot search | a binary term table would load in ~10 ms | not done: it is a file-format change | the next step for the CLI |

`hit_spans` must agree with the tokenizer exactly, or the snippets would highlight different words than
the index matched. It checks every regex candidate against the tokenizer's own rules - is this a token start,
what is the whole token, how does it normalize - and falls back to the old scan when case folding changes the
length of the text (`ß` -> `ss`). A Hypothesis property compares it with the old path on generated text full
of awkward cases (`ß`, `ΣΟΦΟΣ`, combining marks, `don't`, `__init__`); it held for 20,000 examples, and on all
1,301 documents × 10 term sets it returns the same hits.

End to end, a query nobody asked before (query cache off, snippets on; `scripts/bench_search.py`, run once with
`PYTHONPATH` pointing at a `lab-07` worktree and once on `lab-08`):

| Query | Lab 7 | Lab 8 | Faster |
|---|---|---|---|
| `python` | 15.34 ms | 3.11 ms | 4.9× |
| `walrus` | 33.77 ms | 4.59 ms | 7.4× |
| `global interpreter lock` | 31.22 ms | 7.48 ms | 4.2× |
| `the` | 50.88 ms | 15.87 ms | 3.2× |
| `"event loop" NOT asyncio` | 38.70 ms | 8.71 ms | 4.4× |
| `unicode normalization` | 21.16 ms | 11.49 ms | 1.8× |

After the change the snippets are still 94% of a new query, but now it is native time: the regex scan 28%,
`zlib.decompress` 22%, the case-folded copy 10%, the NFC check 7% (Scalene). What is left is proportional to
the document size; the next step is algorithmic - decompress incrementally and stop at the first window that
covers every query term - not more NumPy.

**Benchmarks in CI.** `tests/test_benchmarks.py` benchmarks `Searcher.search` (both engines, four query
shapes) and `build_index` on a generated Zipf-distributed corpus with `pytest-benchmark`; the CI job uploads
the timings as an artifact. Absolute timings vary between runners, so the test that can fail CI is a ratio
measured in the same run: the NumPy engine must stay at least 5× faster than the Python one on the most common
term. If a Python loop sneaks back into the hot path, that test fails.

### M2 - Vectorize the scorer

A term's postings become two `int32` arrays that **view the memory-mapped file**: `np.frombuffer(mmap,
dtype=np.int32, count=df, offset=...)` - no copy, no objects (the file stores `uint32`; every id and count is
below 2^31). Document lengths are one `int32[N]` array. The Boolean query becomes masks: a term is
`mask[doc_ids] = True`, `AND`/`OR`/`NOT` are `&`, `|`, `~`, and a phrase intersects `(doc << 32) | (position - i)`
keys of its terms with `np.intersect1d` - Lab 3's two-pointer merge on positions, in C. Scoring one term for
every matching document is one expression, written operation for operation like Lab 3's scalar code:

```python
length_norm = 1 - b + b * index.doc_len_array[doc_ids] / index.avg_doc_length  # fancy indexing
scores[doc_ids] += idf * tf * (k1 + 1) / (tf + k1 * length_norm)  # ufuncs; ids are unique
```

then `np.argpartition` picks the top k in O(n). `uv run python scripts/bench_numpy.py index.bin` - ranking only,
cache off, no snippets, both engines checked to return the same list first:

| Query | Docs matched | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |
|---|---|---|---|---|
| common term (`python`) | 1,207 | 2.722 | 0.080 | 33.9× |
| rare term (`walrus`) | 16 | 0.043 | 0.054 | 0.8× |
| 3-term query (`global interpreter lock`) | 73 | 1.181 | 0.112 | 10.5× |
| most common term (`the`) | 1,296 | 13.505 | 0.087 | 155.6× |
| phrase of common words (`"in the"`) | 1,169 | 38.277 | 4.832 | 7.9× |

- **Common terms: 34-156×.** Lab 3 builds 1,207 `Posting` objects and calls `score()` 1,207 times - each call a
  frame, attribute lookups, boxed floats. NumPy runs a dozen C loops over 1,207 numbers. Its cost hardly moves
  from `python` to `the` (0.080 -> 0.087 ms): it is mostly the fixed price of a dozen NumPy calls, while the
  Python engine's cost grows with every posting.
- **The rare term is slower (0.8×).** Sixteen postings cost Lab 3 about 40 µs. The NumPy engine pays its fixed
  overhead anyway - three `zeros(1301)` arrays, a dozen ufunc dispatches of 1-3 µs each, `flatnonzero`,
  `argpartition` - about 50 µs. Vectorization amortizes a per-call cost over many elements; with 16 elements
  there is nothing to amortize. A cut-over (Python below ~50 postings) would win those microseconds back; I left
  the engine uniform, because 0.05 ms is not where a query's time goes.
- **Three terms: 10×.** The AND keeps 73 documents, but all three postings lists (hundreds of entries each) are
  read and masked; the Python engine walked them posting by posting.
- **Phrase: 8×.** Every position of `in` and `the` becomes an `int64` key (about 280,000) and
  `np.intersect1d` sorts them: O(n log n) in C instead of O(n) in Python. Still the slowest query here.

**Identical rankings, tested.** `tests/test_vector.py` runs both engines on the fixture corpus - 11 queries
(phrases, `NOT`, `OR`, an unknown word) × 2 scorers × 3 values of k - and asserts the same documents, in the
same order, with **equal** floats (`==`, not "approximately"); the masks must equal Lab 3's `frozenset`s; and a
Hypothesis property repeats both checks on random corpora and random query trees. Bit-identical scores are a
design decision, not luck: the arithmetic runs in the same order, the IDF is the same `math.log` float, `log(tf)`
for TF-IDF comes from a table filled with `math.log` (`np.log` may round the last bit differently), and both
engines break ties the same way - higher score, then lower doc id. That last part needed care:
`np.argpartition` picks *arbitrary* members of a tie at the k-th place, so the engine takes the tied documents
in doc-id order; the Lab 3 engine got the same rule as its `heapq.nlargest` key.

**Free-threaded.** Lab 7's regression test - 16 threads searching one memory-mapped index - now runs for both
engines and passes on 3.14t: `np.frombuffer` asks the `mmap` for its own buffer on every call, like the
per-call `memoryview` fix. `onnxruntime` and `tokenizers` load without re-enabling the GIL.

### M3 - Semantic and hybrid search

```text
findex embed index.bin
  1,301 documents -> 32,367 chunks of 160 words (a new chunk every 128 words: 32 words of overlap)
  -> all-MiniLM-L6-v2 (ONNX Runtime): 384 floats per chunk, L2-normalized
  -> index.vectors.npy (32,367 × 384 float32 = 47.4 MiB, memory-mapped) + index.chunks.npy (doc, start, end)
embedded 32,367 chunks of 1,301 documents with all-MiniLM-L6-v2 (384 dims) in 2271.3 s (14 chunks/s)
```

That run used ONNX Runtime's default thread count, which is the number of *physical* cores - one on this
VM. `findex embed` now asks for every usable CPU (`os.process_cpu_count()`): 26.1 instead of 14.4 chunks/s on
both vCPUs (a 384-chunk sample), so about 20 minutes here. A query is embedded in 8 ms on one core.

- **The model without PyTorch.** sentence-transformers runs all-MiniLM-L6-v2 as tokenizer -> BERT -> mean
  pooling -> normalization. findex runs the model's own ONNX export with ONNX Runtime and the `tokenizers`
  library, and does the pooling itself in NumPy: `(hidden * mask[:, :, None]).sum(1) / mask.sum(1)`, padding
  excluded, then `x / ||x||`. That is 75 MB of dependencies (onnxruntime 63, tokenizers 12) instead of
  PyTorch's several hundred, which matters in the Docker image. The download is pinned to a model revision and
  checked against SHA-256 hashes (Hugging Face publishes the model file's); it happens once, in the Docker build or on the first
  `findex embed`, never at query time. Both are an optional extra (`findex[semantic]`), so keyword search
  installs anywhere, including the free-threaded build on macOS, where onnxruntime has no wheel.
- **Chunks, best chunk wins.** A 40-page document has one topic per section; its mean vector is mush. A document's
  semantic score is its best chunk's cosine, and that chunk is what the UI shows when the document shares no word
  with the question. The chunks are stored grouped by document, so "best chunk per document" is one
  `np.maximum.reduceat` over the segments.
- **Cosine is a dot product.** All rows are unit length, so `E @ q` gives all 32,367 similarities at once:
  **1.8 ms** for the matrix-vector product, 1.9 ms with the per-document maximum. Brute force is exact, and at this
  size there is nothing to win with an approximate index; HNSW starts to pay off around a million vectors.
- **Hybrid = RRF.** BM25 scores are unbounded and cosines live in [-1, 1], so adding them would need a calibration
  that changes per query. Reciprocal rank fusion uses only ranks: `score(d) = Σ 1/(60 + rank)` over the top 100 of
  each list. A document high in both lists wins; a document only one list found still gets in.
- **Where.** `findex search --mode keyword|semantic|hybrid`, `GET /search?mode=...` (400 with an explanation if the
  server has no embeddings), a Keyword / Meaning / Hybrid selector in the UI, `"semantic": true` in `/health`. The
  Docker image builds the embeddings at build time (`--build-arg EMBED=0` gives a keyword-only image in minutes;
  CI builds that one on every push, `image.yml` the full one). The query operators are stripped before a query is
  embedded (`NOT` is a keyword-side idea; in hybrid mode the keyword half still honours it).
- **Tested offline, checked against the original.** A `HashingEmbedder` (feature hashing of the tokens - lexical,
  not semantic) exercises the whole pipeline in the tests and in CI without a model download; a 2 KB ONNX model
  with the real model's interface (generated by `tests/data/make_tiny_minilm.py`) checks the ONNX code path itself,
  including that a batch padded to its longest text gives the same vectors as embedding each text alone. With the
  real model, my tokenizer + ONNX + mean-pooling pipeline reproduces the similarities printed on the model card to
  four decimals ("That is a happy person" vs "...happy dog" 0.6946, "...very happy person" 0.9429, "Today is a
  sunny day" 0.2569), so it computes what sentence-transformers computes.

### Evaluation: keyword vs semantic vs hybrid, 15 queries

The Lab 3 set (10 hand-labelled queries) plus **five paraphrases** - questions that describe a topic in other
words than the documents use, labelled by the same rule (`eval/queries.json`, `"paraphrase": true`):

| Paraphrase | What it is about | Relevant |
|---|---|---|
| *find out how long a piece of code takes to run* | `timeit`, the profilers (`profile`, `profiling` and its tracing and sampling pages), the perf HOWTO, PEP 418/564 clocks | 8 |
| *use every processor core from one program* | `multiprocessing`, `concurrent.futures`, `concurrent.interpreters`, free threading, PEP 371/554/684/703/734 | 9 |
| *turn an object into bytes and back again* | `pickle`, `marshal`, `copyreg`, pickle PEPs 307/3154/574 | 6 |
| *write messages about what my program is doing to a file* | `logging` and its HOWTO, cookbook, handlers, config, PEP 282 | 6 |
| *remember the results of slow function calls* | `functools` (`cache`, `lru_cache`) | 1 |

Keyword search, as expected, finds almost nothing for them: their words are common, implicit AND requires all
of them, and the documents say *profile*, *serialize*, *memoize* instead. The keyword column of the original ten
reproduces Lab 3 (0.74 here, 0.72 on the Mac).

Labels are made the Lab 3 way, and for all 15 queries the pool now holds the top 10 of all three modes: a
document only semantic search finds is judged, not silently counted as a miss. That added 10 relevant documents
to the original ten (PEP 653 on pattern-matching semantics, PEP 346 on `with`, newer typing PEPs, ...) under the
same rule: primarily about the topic, or the reference section that defines it (as the data model's
context-manager section already counted in Lab 3); a What's New page only when the topic is its headline (3.10
and pattern matching); glossary entries and passing mentions never.

`findex embed index.bin && uv run python scripts/evaluate.py index.bin --modes`:

| Query | relevant | keyword P@5 | semantic P@5 | hybrid P@5 |
|---|---|---|---|---|
| `global interpreter lock` | 7 | 0.6 | 0.8 | 1.0 |
| `structural pattern matching` | 8 | 1.0 | 1.0 | 1.0 |
| `generic type hints` | 15 | 1.0 | 0.8 | 0.8 |
| `virtual environment` | 8 | 1.0 | 0.8 | 1.0 |
| `f-string formatting` | 4 | 0.2 | 0.4 | 0.4 |
| `async generator` | 4 | 0.6 | 0.4 | 0.6 |
| `dataclass frozen slots` | 3 | 0.4 | 0.6 | 0.4 |
| `context manager with statement` | 8 | 0.8 | 0.8 | 1.0 |
| `unicode normalization` | 5 | 0.8 | 0.4 | 1.0 |
| `reference counting garbage collection` | 9 | 1.0 | 0.6 | 1.0 |
| *find out how long a piece of code takes to run* (paraphrase) | 8 | 0.0 | 0.6 | 0.2 |
| *use every processor core from one program* (paraphrase) | 9 | 0.0 | 0.8 | 0.2 |
| *turn an object into bytes and back again* (paraphrase) | 6 | 0.0 | 0.0 | 0.0 |
| *write messages about what my program is doing to a file* (paraphrase) | 6 | 0.2 | 0.4 | 0.2 |
| *remember the results of slow function calls* (paraphrase) | 1 | 0.0 | 0.2 | 0.0 |
| mean, original 10 | | 0.74 | 0.66 | **0.82** |
| mean, 5 paraphrases | | 0.04 | **0.40** | 0.12 |
| **mean, all 15** | | 0.51 | 0.57 | **0.59** |

What the table says:

- **On the paraphrases, keyword search fails as predicted (0.04) and semantic search does not (0.40).** For *find
  out how long a piece of code takes to run* it returns `timeit` and `profile` first; for *use every processor
  core*, after What's New in 3.12, PEP 703, `concurrent.interpreters`, PEP 684 and PEP 371. Keyword search finds documents that happen to
  contain all of *find, out, how, long, piece, code, takes, run* - PEP 238, `optparse` - because without stop words
  and with an implicit AND, a sentence is a strange query.
- **On the original queries, semantic alone is worse than BM25 (0.66 vs 0.74).** Those queries are made of exact
  terms, and the model keeps the general subject but lets the specific word go: for `unicode normalization` it
  returns What's New in 3.15, the Expressions chapter and PEP 100 - Unicode in general (0.4 vs 0.8); for
  `reference counting garbage collection`, PEP 454 (`tracemalloc`) and the data model - memory in general
  (0.6 vs 1.0). One rare word in a 160-word chunk moves its vector only a little; for BM25 it is the whole score.
  The model wins where the documents use other words than the query: `f-string formatting` (0.2 -> 0.4) and
  `dataclass frozen slots` (0.4 -> 0.6), the two queries the Lab 3 tokenizer lost, and `global interpreter lock`
  (0.6 -> 0.8).
- **Hybrid is the best on the original ten (0.82)**: RRF keeps BM25's exact hits and adds what the model found -
  `global interpreter lock` 0.6 -> 1.0 with PEP 311 (semantic #1, keyword #7). It also rewards agreement: the
  Unicode HOWTO is #8 for keywords and #6 for meaning, and #3 after fusion, which takes `unicode normalization`
  to 1.0.
- **But hybrid loses most of semantic's gain on the paraphrases (0.40 -> 0.12)**, and that is the most useful line
  in the table. RRF trusts both lists equally, and for a paraphrase the keyword list is confidently wrong: eight to
  twenty-six documents that contain every common word of the sentence, ranked at the top. RRF cannot tell a good
  list from a bad one by ranks alone. Fixes worth trying next: weight the fusion per query (trust the keyword side
  less when it matched few documents, or when most query terms have low IDF), or give natural-language queries an
  OR keyword side.
- **One paraphrase fails everywhere**: *turn an object into bytes and back again*. The model reads "bytes" as the
  type and returns PEP 358 (*The "bytes" Object*) and the bytes C API, not serialization; my paraphrase leaned on a
  word that has a more literal meaning in this corpus.

**What I learned.** Profile before optimizing: I would have vectorized the scorer, and the scorer was 4% of a
query; the snippets were 93%. Scalene's Python/native split was the most useful number, and it says something
subtle - "native" time can still be the interpreter's fault when it is millions of tiny C calls issued by a Python
loop; the cure is fewer calls, not faster Python. NumPy wins by amortizing a fixed per-call cost over many
elements, so it lost on the 16-posting query. And bit-for-bit equality with the old engine was something I had to
design for - operation order, the same `math.log`, an explicit tie rule - which is exactly what made "the ranking
did not change" a test instead of a hope. On the semantic side, the lesson is how little machinery is needed -
cosine similarity over normalized vectors is one matrix-vector product - and that evaluation is what keeps it
honest: hybrid search was the best mode on the queries I wrote first, yet kept less than a third of the
semantic gain on the paraphrases it was supposed to rescue (0.12 vs 0.40) - invisible without the paraphrase
rows. So was the pooling: judged only from keyword results, the original ten scored semantic search 0.60
instead of 0.66, because the relevant documents only it found had never been labelled.

## Limitations and next steps

- **No stemming, identifiers kept whole.** `strings` does not match `string`, and `slots` does not match
  `__slots__` - that is why `f-string formatting` and `dataclass frozen slots` stay low for keyword search (Lab 3).
  Semantic search partly covers it; a stemmer would change the index format and the phrase rules.
- **Snippets cost grows with document length.** A cold query still decompresses and normalizes each result's full
  text. Next: incremental decompression that stops at the first window covering all query terms.
- **Start-up of a one-shot CLI search** is dominated by parsing the JSON term table (75 ms for 95,669 terms). Next:
  a binary term table (sorted terms + NumPy offset arrays) in format version 3.
- **The index is immutable.** New documents mean a rebuild (about a second with processes); embeddings are rebuilt
  too. An incremental segment-and-merge design, like Lucene's, would be the next architecture step.
- **The semantic model is English and truncates at 256 word pieces**; long chunks lose their tail. The corpus is
  English, so this is fine here; a Ukrainian corpus would need a multilingual model.
- **Hybrid fusion is unweighted.** RRF gives a confidently wrong keyword list the same say as the model (hybrid 0.12
  vs semantic 0.40 on paraphrases). Next: per-query weights, or an OR keyword side for natural-language queries.
- **One replica on one homelab node.** The ThinkPad is a single point of failure; Cloudflare in front absorbs the
  public side, not a dead node.
- **Measurements across two machines.** Labs 1-7 on the M4, Lab 8 on a cloud VM: ratios compare, absolute
  milliseconds do not.
