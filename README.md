# findex

[![CI](https://github.com/batpepe/findex/actions/workflows/ci.yml/badge.svg)](https://github.com/batpepe/findex/actions/workflows/ci.yml)

A full-text search engine over the **Python documentation and every PEP**, built from scratch in Python
over the eight labs of the *Python — Build a Search Engine* course. Each lab adds one language feature
and one piece of the engine; every lab is tagged (`lab-01`, `lab-02`, …) so the history shows it growing.

| Lab | Feature | What it added | Tag |
|---|---|---|---|
| 1 | Iterators & generators | streaming corpus, tokenizer, stats in constant memory | `lab-01` |
| 2 | Dicts, hashing, memory | inverted index, Boolean search, two on-disk formats | `lab-02` |
| 3 | The data model, decorators | `Index` object, BM25/TF-IDF, query language, snippets, precision@5 | `lab-03` |
| 4 | Typing, testing, packaging | `findex` CLI (typer + rich), pyright strict, 77 tests, CI, v0.4.0 wheel | `lab-04` |

## Quick start

```bash
# install the command (Python 3.14+); or from a clone: uv tool install .
uv tool install https://github.com/batpepe/findex/releases/download/v0.4.0/findex-0.4.0-py3-none-any.whl

git clone https://github.com/batpepe/findex && cd findex
./scripts/get-corpus.sh                      # ~31 MB into data/ (git-ignored)
findex index data/ --out index.bin           # positions + texts, binary format
findex search index.bin '"event loop" NOT asyncio'
findex search index.bin 'TaskGroup' --json | jq .title
findex stats index.bin
```

Development: `uv sync`, then `uv run pytest`, `uv run ruff check .`, `uv run pyright`.
The lab-by-lab entry points still work: `uv run python -m findex.stats data/`, `python -m findex.index`,
`python -m findex.search`.

## The corpus

| Source | Where it comes from | Documents | Size on disk |
|---|---|---|---|
| Python 3.15 documentation, plain-text build | [docs.python.org/3/download.html](https://docs.python.org/3/download.html) (`python-3.15-docs-text.zip`) | 552 | 16 MB |
| All PEPs (reStructuredText sources) | [github.com/python/peps](https://github.com/python/peps), `peps/pep-*.rst` (public domain / CC0) | 745 | 15 MB |
| **Total** | `./scripts/get-corpus.sh` | **1,297** | **31 MB** |

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
