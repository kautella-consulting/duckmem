# CLI Reference

DuckMem ships a small Typer-based CLI for common operations. The authoritative
source is [duckmem/cli.py](../duckmem/cli.py); this page mirrors it.

## Usage

```bash
uv run duckmem --help

duckmem --help
```

Every command takes the database path as the first positional argument.

## Commands

### `duckmem create <db_path>`

Create a new, empty DuckMem database file.

```bash
duckmem create knowledge.duckdb
```

Fails if the file already exists.

### `duckmem stats <db_path>`

Show counts and file size.

```bash
duckmem stats knowledge.duckdb
duckmem stats knowledge.duckdb --json
```

**Options**

| Option | Description |
|--------|-------------|
| `--json` | Emit JSON instead of a human table. |

**Human output**

```
Items:      42
Chunks:     156
Relations:  45
Entities:   23
Sessions:   5
File size:  2,456,789 bytes
```

### `duckmem add <db_path>`

Add a single text item to the knowledge base.

```bash
duckmem add knowledge.duckdb --text "Transformers use attention." --title "ML Notes"
duckmem add knowledge.duckdb -t "Sprint summary" -n work
```

**Options**

| Option | Description |
|--------|-------------|
| `--text, -t` | Text to add (required). |
| `--title` | Optional item title. |
| `--namespace, -n` | Namespace (default: `default`). |

Prints the new item id. Reading text from a file is not currently a CLI flag;
use shell substitution: `duckmem add db.duckdb --text "$(cat doc.md)" --title doc`.

### `duckmem search <db_path>`

Search the knowledge base.

```bash
duckmem search knowledge.duckdb --query "attention mechanism"
duckmem search knowledge.duckdb -q "transformers" --top-k 5 --mode hybrid
duckmem search knowledge.duckdb -q "tokens" --json
```

**Options**

| Option | Description |
|--------|-------------|
| `--query, -q` | Search query (required). |
| `--mode, -m` | `hybrid` (default), `lexical`, or `semantic`. |
| `--top-k, -k` | Max results (default: 10). |
| `--json` | Emit JSON. |

### `duckmem verify <db_path>`

Validate the database structure and optionally recompute checksums.

```bash
duckmem verify knowledge.duckdb
duckmem verify knowledge.duckdb --deep --json
```

**Options**

| Option | Description |
|--------|-------------|
| `--deep` | Recompute and compare stored checksums. |
| `--json` | Emit JSON. |

**Output (healthy)**

```
Items:     42
Chunks:    156
Relations: 45
Entities:  23
Checksums: OK
```

Any integrity problems are printed to stderr as `Error: ...` lines.

### `duckmem doctor <db_path>`

Run optional maintenance operations. Each sub-op is opt-in.

```bash
duckmem doctor knowledge.duckdb --vacuum
duckmem doctor knowledge.duckdb --rebuild-fts --rebuild-vec
```

**Options**

| Option | Description |
|--------|-------------|
| `--vacuum` | Compact storage (DuckDB `VACUUM`). |
| `--rebuild-fts` | Drop and rebuild the BM25 / FTS index. |
| `--rebuild-vec` | Drop and rebuild the HNSW vector index. |

Each op prints `op: OK` or `op: FAILED`.

### `duckmem lock-db <src>` / `duckmem unlock-db <src>`

Encrypt / decrypt the database file at rest with a password-derived key.

```bash
duckmem lock-db knowledge.duckdb --out knowledge.duckdb.enc
duckmem unlock-db knowledge.duckdb.enc --out knowledge.duckdb
```

**Options**

| Option | Description |
|--------|-------------|
| `--out, -o` | Output file path (required). |

The password is read interactively and never echoed. `lock-db` prompts twice
for confirmation.

### `duckmem serve <db_path>`

Start the FastAPI server against a database file.

```bash
duckmem serve knowledge.duckdb
duckmem serve knowledge.duckdb --host 0.0.0.0 --port 8080
```

**Options**

| Option | Description |
|--------|-------------|
| `--host, -h` | Bind host (default: `127.0.0.1`). |
| `--port, -p` | Bind port (default: `8000`). |

Auto-reload is enabled; the server is intended for development. For production,
run `uv run uvicorn duckmem.api:app` behind a reverse proxy.

## Environment variables

The CLI respects every `DUCKMEM_*` setting. See [Configuration](configuration.md)
and [.env.example](../.env.example) for the full list.

```bash
export DUCKMEM_EMBED_MODEL="openai/text-embedding-3-small"
export DUCKMEM_EMBED_DIM=1536
export OPENAI_API_KEY="sk-..."

duckmem add knowledge.duckdb --text "Cloud-embedded content"
```
