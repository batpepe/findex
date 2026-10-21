# findex

A full-text search engine over the **Python documentation and every PEP**, built from scratch in Python
over the eight labs of the *Python — Build a Search Engine* course. Each lab adds one language feature
and one piece of the engine; every lab is tagged (`lab-01`, `lab-02`, …) so the history shows it growing.

| Lab | Feature | What it added | Tag |
|---|---|---|---|
| 1 | Iterators & generators | streaming corpus, tokenizer, stats in constant memory | `lab-01` |
| 2 | Dicts, hashing, memory | inverted index, Boolean search, two on-disk formats | `lab-02` |

## Quick start

```bash
git clone https://github.com/batpepe/findex && cd findex
uv sync                                   # Python 3.14 + dev tools from uv.lock
./scripts/get-corpus.sh                   # ~31 MB into data/ (git-ignored)
uv run python -m findex.stats data/       # corpus statistics, lazily
uv run python -m findex.index data/ --out index.bin          # build + save the inverted index
uv run python -m findex.search index.bin "asyncio TaskGroup" # Boolean search
```

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
