# REST API Reference

DuckMem ships a FastAPI service that mirrors the Python SDK. The authoritative
schema is the live OpenAPI document: once the server is running, open
`http://<host>:<port>/docs` (Swagger UI) or `/redoc`. This page is a quick
reference; any discrepancy should be resolved in favor of [duckmem/api.py](../duckmem/api.py)
and the Pydantic models in [duckmem/models.py](../duckmem/models.py).

## Starting the server

```bash
duckmem serve knowledge.duckdb

uv run uvicorn duckmem.api:app --reload

DUCKMEM_DB_PATH=knowledge.duckdb uv run uvicorn duckmem.api:app --host 0.0.0.0 --port 8080
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc:      `http://127.0.0.1:8000/redoc`
- OpenAPI:    `http://127.0.0.1:8000/openapi.json`

No authentication is built in. For production, front the service with a reverse
proxy that handles auth and TLS.

## Endpoint summary

| Method | Path | Description |
|---|---|---|
| GET  | `/health` | Liveness probe. |
| POST | `/items` | Ingest a text item. |
| GET  | `/items/{item_id}` | Get an item by id. |
| GET  | `/items` | List items (filterable). |
| GET  | `/search` | Hybrid / lexical / semantic search. |
| POST | `/ask` | RAG question answering. |
| POST | `/relations` | Add a knowledge-graph relation. |
| GET  | `/entities/{entity}/state` | Current properties for an entity. |
| GET  | `/entities/{entity}/history` | Relation history for an entity. |
| GET  | `/entities/{entity}/traverse` | Graph traversal from an entity. |
| POST | `/extract` | Extract relations from an item via LLM. |
| POST | `/sessions` | Start a recording session. |
| POST | `/sessions/{session_id}/end` | End the active session. |
| GET  | `/sessions` | List sessions. |
| GET  | `/sessions/{session_id}/events` | Replay session events. |
| GET  | `/stats` | Counts and file size. |
| GET  | `/verify` | Integrity check. |
| POST | `/doctor` | Maintenance operations. |

## Health

```http
GET /health
```

```json
{"status": "healthy"}
```

## Items

### Add item

```http
POST /items
Content-Type: application/json

{
  "text": "Transformers use attention mechanisms for sequence modeling.",
  "title": "ML Notes",
  "uri": "notes/ml.md",
  "namespace": "default",
  "label": null,
  "metadata": {"author": "Jane"},
  "skip_embedding": false
}
```

Response:

```json
{"item_id": "item_abc123"}
```

### Get item

```http
GET /items/{item_id}
```

Returns a full `Item` object (`id`, `text`, `title`, `uri`, `namespace`,
`label`, `metadata`, `created_at`, `checksum`).

### List items

```http
GET /items?namespace=default&label=work&limit=100&offset=0
```

Returns `list[Item]`. All query params are optional.

## Search

### Search

```http
GET /search?query=attention+mechanism&mode=hybrid&top_k=10
```

Query parameters:

| Param | Type | Default | Notes |
|---|---|---|---|
| `query` | str | required | |
| `mode` | `hybrid` \| `lexical` \| `semantic` | `hybrid` | |
| `top_k` | int (1-100) | `10` | |
| `namespace` | str? | none | Filter by namespace. |
| `start_ts` | int? | none | Min created_at (epoch ms). |
| `end_ts` | int? | none | Max created_at (epoch ms). |

Response is `list[SearchResult]`. Each entry has `score`, `lexical_score`,
`semantic_score`, `chunk`, and `item`.

### Ask (RAG)

```http
POST /ask
Content-Type: application/json

{
  "question": "What do transformers use for sequence modeling?",
  "top_k": 5,
  "namespace": null,
  "model": null
}
```

Response is an `Answer`:

```json
{
  "answer": "Transformers use attention mechanisms...",
  "confidence": 0.84,
  "sources": ["chunk_xyz1", "chunk_xyz2"],
  "context": "Transformers use attention..."
}
```

`sources` is a list of chunk ids. Resolve them through `/items/{id}` or by
re-running a targeted `/search` call if you need titles or excerpts.

## Knowledge graph

### Add relation

```http
POST /relations
Content-Type: application/json

{
  "subject": "Transformers",
  "predicate": "use",
  "object": "attention",
  "item_id": null
}
```

Response: `{"relation_id": "rel_def456"}`.

### Entity state

```http
GET /entities/{entity}/state
```

Response (`EntityState`):

```json
{
  "entity": "Transformers",
  "properties": {"use": "attention"}
}
```

### Entity history

```http
GET /entities/{entity}/history?predicate=use
```

Response: `list[Relation]` ordered oldest-first.

### Traverse

```http
GET /entities/{entity}/traverse?link=use&max_hops=3
```

Response (`TraversalResult`):

```json
{
  "entities": ["Transformers", "attention"],
  "paths": [
    [{"id": "rel_1", "subject": "Transformers", "predicate": "use", "object": "attention", ...}]
  ]
}
```

### Extract relations

```http
POST /extract
Content-Type: application/json

{"item_id": "item_abc123", "model": null}
```

Response: `{"relation_ids": ["rel_1", "rel_2"]}`.

## Sessions

```http
POST /sessions                      # {"name": "optional"}  -> {"session_id": "..."}
POST /sessions/{session_id}/end     # -> {"status": "ended"}
GET  /sessions                      # -> list[Session]
GET  /sessions/{session_id}/events  # -> list[SessionEvent]
```

`SessionEvent` fields: `id`, `session_id`, `kind`, `params`, `result_summary`,
`timestamp`.

## Maintenance

### Stats

```http
GET /stats
```

Response (`Stats`): `items`, `chunks`, `relations`, `entities`, `sessions`, `file_size_bytes`.

### Verify

```http
GET /verify?deep=false
```

Response (`VerifyResult`): `items`, `chunks`, `relations`, `entities`,
`checksum_ok` (only when `deep=true`), `errors`.

### Doctor

```http
POST /doctor
Content-Type: application/json

{
  "vacuum": true,
  "rebuild_fts": false,
  "rebuild_vec": false,
  "timeout_seconds": null
}
```

Response: `{"results": {"vacuum": true}}`.

## Errors

All errors return the standard FastAPI shape:

```json
{"detail": "Error message"}
```

Typical status codes: `400` (bad input), `404` (not found), `422` (validation),
`500` (server error).
