# User Guide

This guide covers the core `DuckMem` Python API. Every snippet below is kept in
sync with [duckmem/core.py](../duckmem/core.py) and
[duckmem/models.py](../duckmem/models.py) — please file an issue if you find
drift. Runnable versions of each snippet live under [examples/](../examples/).

## Overview

DuckMem provides four main capabilities:

1. **Document ingestion** - Store text items with metadata, automatically chunked and embedded.
2. **Hybrid search** - Combine BM25 lexical and HNSW vector search with RRF fusion.
3. **Knowledge graph** - Track entities, current state, and relation history.
4. **RAG Q&A** - Ask questions and get LLM-synthesized answers from your content.

## Basic usage

### Creating a DuckMem instance

```python
from duckmem import DuckMem, Settings

mem = DuckMem("knowledge.duckdb")

settings = Settings(
    db_path="knowledge.duckdb",
    embed_model="openai/text-embedding-3-small",
    embed_dim=1536,
    chunk_max_chars=800,
)
mem = DuckMem(settings=settings)

mem.close()
```

### Context manager (recommended)

```python
from duckmem import DuckMem

with DuckMem("knowledge.duckdb") as mem:
    mem.add("Some text content", title="My Document")
    results = mem.search("text")
```

The database is closed automatically on exit.

## Document ingestion

### Adding items

An **item** is a document. DuckMem chunks it and computes embeddings during `add()`.
The call returns the new item id (a string).

```python
item_id = mem.add("This is my document content.", title="My Document")

item_id = mem.add(
    "Meeting notes from the Q3 planning session...",
    title="Q3 Planning Notes",
    uri="meetings/2024-01-15.txt",
    namespace="work",
    label="meeting",
    metadata={"date": "2024-01-15", "attendees": ["Alice", "Bob"]},
)

print(f"Added item: {item_id}")
```

Parameter reference:

| Kwarg | Type | Meaning |
|---|---|---|
| `title` | `str \| None` | Human-readable title. |
| `uri` | `str \| None` | Source locator (e.g. path, URL). |
| `namespace` | `str` | Category bucket. Default `"default"`. |
| `label` | `str \| None` | Free-form sub-category. |
| `metadata` | `dict \| None` | Arbitrary JSON metadata. |
| `skip_embedding` | `bool` | Skip embedding computation (lexical-only). |

### Namespaces

Namespaces group items and filter searches.

```python
mem.add("Personal note...", namespace="personal")
mem.add("Work document...", namespace="work")

results = mem.search("meeting", namespace="work")
```

### Listing items

```python
items = mem.list()
items = mem.list(namespace="work")
items = mem.list(label="meeting")
items = mem.list(limit=10, offset=20)

for item in items:
    print(f"{item.id}: {item.title}")
```

### Getting an item

```python
item = mem.get("item_abc123")
if item:
    print(f"Title: {item.title}")
    print(f"Text:  {item.text[:200]}...")
```

Fields on `Item`: `id`, `text`, `title`, `uri`, `namespace`, `label`,
`metadata`, `created_at`, `checksum`.

## Search

DuckMem combines BM25 lexical search with HNSW vector similarity using
Reciprocal Rank Fusion (RRF).

```python
results = mem.search("machine learning transformers")

for r in results:
    print(f"{r.score:.3f}  {r.chunk.text[:120]}...")
    print(f"         from item: {r.item.title}")
```

### Search options

```python
results = mem.search("query", top_k=5)
results = mem.search("query", namespace="research")

results = mem.search("exact phrase", mode="lexical")
results = mem.search("similar meaning", mode="semantic")
results = mem.search("both", mode="hybrid")

results = mem.search("q", start_ts=1_700_000_000_000, end_ts=1_710_000_000_000)
```

### Reading search results

Each `SearchResult` has:

- `score` - fused RRF score.
- `lexical_score`, `semantic_score` - component scores (may be `None` if the
  corresponding mode was skipped).
- `chunk` - the matching `Chunk` (`id`, `item_id`, `seq`, `text`).
- `item` - the parent `Item`.

## RAG (ask)

`ask()` runs a hybrid search, then uses the configured LLM to synthesize an
answer. It is async.

```python
import asyncio
from duckmem import DuckMem

async def main():
    with DuckMem("knowledge.duckdb") as mem:
        answer = await mem.ask("What are the key concepts in transformers?")

        print(answer.answer)
        print(f"confidence: {answer.confidence}")
        print(f"based on {len(answer.sources)} chunks: {answer.sources}")

asyncio.run(main())
```

`Answer` fields: `answer` (str), `confidence` (float), `sources` (list of chunk
ids), `context` (raw concatenated context fed to the LLM).

### RAG options

```python
answer = await mem.ask("question", top_k=10)
answer = await mem.ask("question", namespace="research")
answer = await mem.ask("question", model="openai/gpt-4o")
```

## Knowledge graph

### Entities and relations

```python
relation_id = mem.add_relation("Python", "is_a", "Programming Language")

relation_id = mem.add_relation(
    "Transformers", "use", "attention",
    item_id="item_abc123",
)
```

`add_relation(subject, predicate, obj, *, item_id=None)` returns the new
relation id.

### Entity state

The graph keeps a latest-wins view of each entity's properties.

```python
state = mem.state("Python")
print(state.entity)       # "Python"
print(state.properties)   # {"is_a": "Programming Language"}
```

### Relation history

```python
for rel in mem.history("Python"):
    print(f"{rel.subject} --[{rel.predicate}]--> {rel.object}")

for rel in mem.history("Python", predicate="used_by"):
    ...
```

`history()` returns `list[Relation]` ordered oldest-first.

### Graph traversal

```python
result = mem.traverse("Python", max_hops=2)
result = mem.traverse("Python", link="is_a", max_hops=3)

print(result.entities)  # list[str] - entity names reachable
for path in result.paths:
    hops = " -> ".join(f"[{r.predicate}] {r.object}" for r in path)
    print(f"Python -> {hops}")
```

### Relation extraction

Extract relations from an ingested item using the configured LLM:

```python
relation_ids = await mem.extract(item_id)
relation_ids = await mem.extract(item_id, model="openai/gpt-4o-mini")
```

Returns the list of newly-created relation ids.

## Sessions

Sessions record a timeline of DuckMem operations. Once a session is active,
calls to `add`, `search`, `ask`, `add_relation`, and `extract` auto-log events
with their params.

```python
session_id = mem.session_start(name="Research Session")

mem.add("Transformers use attention.", title="note")
results = mem.search("attention")
answer = await mem.ask("What do transformers use?")

mem.session_end()

for sess in mem.session_list():
    print(sess.id, sess.name, sess.started_at)

for event in mem.session_replay(session_id):
    print(event.timestamp, event.kind, event.params)
```

`SessionEvent` fields: `id`, `session_id`, `kind`, `params`, `result_summary`,
`timestamp`. `session_end()` always ends the currently-active session — it
takes no arguments.

## Maintenance

### Database statistics

```python
stats = mem.stats()
print(f"items:     {stats.items}")
print(f"chunks:    {stats.chunks}")
print(f"entities:  {stats.entities}")
print(f"relations: {stats.relations}")
print(f"sessions:  {stats.sessions}")
print(f"size:      {stats.file_size_bytes / 1024 / 1024:.2f} MiB")
```

### Verify integrity

```python
result = mem.verify(deep=True)

if result.errors:
    print("Issues found:")
    for err in result.errors:
        print(f"  - {err}")
else:
    print("OK")

if result.checksum_ok is False:
    print("Checksum mismatch detected.")
```

`VerifyResult` fields: `items`, `chunks`, `relations`, `entities`,
`checksum_ok` (when `deep=True`), `errors`.

### Doctor (opt-in maintenance)

```python
results = mem.doctor(vacuum=True, rebuild_fts=True, rebuild_vec=False)
for op, ok in results.items():
    print(f"{op}: {'OK' if ok else 'FAILED'}")
```

`doctor()` returns `dict[str, bool]` keyed by operation.

### Encryption

`lock()` / `unlock()` are module-level helpers that copy a file through a
password-derived Fernet key. They do **not** encrypt an open database in place.

```python
from duckmem import DuckMem
from duckmem.core import lock, unlock

lock("knowledge.duckdb", "knowledge.duckdb.enc", password="my-secret")

unlock("knowledge.duckdb.enc", "knowledge.duckdb", password="my-secret")

with DuckMem("knowledge.duckdb") as mem:
    ...
```

The CLI equivalents are `duckmem lock-db` and `duckmem unlock-db`.

## Best practices

### Chunking strategy

Set `chunk_strategy` in `Settings` or the `DUCKMEM_CHUNK_STRATEGY` env var:

- `markdown` (default) - Preserves heading structure; best for docs.
- `sentence` - Respects sentence boundaries; best for prose.
- `fixed` - Fixed-size character windows with overlap; predictable size.

### Namespace organization

```python
mem.add(content, namespace="project/backend")
mem.add(content, namespace="project/frontend")
mem.add(content, namespace="research/papers")
mem.add(content, namespace="personal/notes")
```

### Metadata

```python
mem.add(
    content,
    metadata={
        "author": "Jane Doe",
        "date": "2024-01-15",
        "tags": ["python", "machine-learning"],
        "version": "1.0",
    },
)
```

### Error handling

```python
from duckmem import DuckMem

try:
    with DuckMem("knowledge.duckdb") as mem:
        mem.add("content")
except FileNotFoundError:
    print("Database path is invalid")
except PermissionError:
    print("Cannot write to database location")
```
