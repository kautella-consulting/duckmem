"""Integration tests for DuckMem REST API via real HTTP server.

Spawns uvicorn as documented (DUCKMEM_DB_PATH=... uv run uvicorn duckmem.api:app),
waits for server ready, then exercises all endpoints via httpx.

Requires: Ollama running with qwen3-embedding:latest, gpt-oss:20b.
OpenAI variant requires: OPENAI_API_KEY (test skipped if not set).

Run with logging: uv run pytest tests/test_rest_api.py -v --log-cli-level=INFO
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import duckdb
import httpx
import pytest

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """Find an available port for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_rest_server_fixture(llm_model: str, scope: str = "function"):
    """Create rest_server fixture with given DUCKMEM_LLM_MODEL."""

    @pytest.fixture(scope=scope)
    def _fixture(project_root: Path, tmp_path_factory):
        db_path = tmp_path_factory.mktemp("rest_api") / "test.duckdb"
        port = find_free_port()
        env = os.environ.copy()
        env.update(
            {
                "DUCKMEM_DB_PATH": str(db_path),
                "DUCKMEM_EMBED_MODEL": "ollama/qwen3-embedding:latest",
                "DUCKMEM_EMBED_DIM": "4096",
                "DUCKMEM_LLM_MODEL": llm_model,
            }
        )
        proc = subprocess.Popen(
            [
                "uv",
                "run",
                "--project",
                str(project_root),
                "uvicorn",
                "duckmem.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        base_url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                r = httpx.get(f"{base_url}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.2)
        else:
            proc.terminate()
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"Server failed to start. stderr: {stderr}")
        yield base_url, db_path, proc
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=5)

    return _fixture


rest_server_ollama = _make_rest_server_fixture("ollama/gpt-oss:20b")
rest_server_openai = _make_rest_server_fixture("openai/gpt-4o-mini")
rest_server = _make_rest_server_fixture("ollama/gpt-oss:20b", scope="module")


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Project root directory (session-scoped for module-scoped rest_server)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def http_client(rest_server):
    """HTTP client for unit tests (uses module-scoped rest_server)."""
    base_url, _, _ = rest_server
    return httpx.Client(base_url=base_url, timeout=30.0)


# =============================================================================
# Shared Full Flow
# =============================================================================


def _run_full_rest_flow(base_url: str, db_path: Path, proc: subprocess.Popen) -> None:
    """Shared full REST flow: all endpoints, then direct DB verification."""
    item_ids: list[str] = []
    session_id_1: str = ""

    client = httpx.Client(base_url=base_url, timeout=60.0)
    try:
        # --- 1. Initial state ---
        logger.info("Step 1: Initial state - stats, verify")
        r = client.get("/stats")
        r.raise_for_status()
        stats = r.json()
        assert stats["items"] == 0
        assert stats["chunks"] == 0
        assert stats["relations"] == 0
        assert stats["entities"] == 0
        assert stats["sessions"] == 0

        r = client.get("/verify", params={"deep": True})
        r.raise_for_status()
        verify = r.json()
        assert verify["checksum_ok"] is True
        assert verify["errors"] == []

        # --- 2. Ingest 4 items ---
        logger.info("Step 2: Ingest 4 items (Python, Rust, DuckDB, Guido)")
        items_data = [
            {
                "text": (
                    "Python is a high-level, interpreted programming language known for "
                    "its clear syntax and readability. It was created by Guido van Rossum "
                    "and first released in 1991. Python supports multiple programming "
                    "paradigms including procedural, object-oriented, and functional "
                    "programming. It is widely used in data science, machine learning, "
                    "web development, and automation."
                ),
                "title": "Python Programming Language Overview",
                "uri": "https://python.org",
                "namespace": "languages",
                "label": "programming",
            },
            {
                "text": (
                    "Rust is a systems programming language focused on safety, speed, "
                    "and concurrency. It was created by Graydon Hoare and sponsored by "
                    "Mozilla. Rust prevents memory safety issues at compile time through "
                    "its ownership and borrowing system."
                ),
                "title": "Rust Programming Language Overview",
                "uri": "https://rust-lang.org",
                "namespace": "languages",
                "label": "programming",
            },
            {
                "text": (
                    "DuckDB is an in-process analytical database system designed for "
                    "OLAP workloads. It was developed by Mark Raasveldt and Hannes "
                    "Mühleisen at CWI Amsterdam. DuckDB is embedded (no server needed), "
                    "supports SQL, and integrates seamlessly with Python pandas."
                ),
                "title": "DuckDB Analytical Database",
                "uri": "https://duckdb.org",
                "namespace": "ml",
                "label": "ai",
            },
            {
                "text": (
                    "Guido van Rossum is a Dutch programmer best known as the creator "
                    "of the Python programming language. He worked at Google and later "
                    "joined Dropbox. He was Python's BDFL until 2018. Rejoined Microsoft in 2020."
                ),
                "title": "Guido van Rossum - Python Creator",
                "uri": "",
                "namespace": "default",
                "label": "people",
            },
        ]

        for i, item_data in enumerate(items_data):
            r = client.post("/items", json=item_data)
            r.raise_for_status()
            data = r.json()
            assert "item_id" in data
            item_ids.append(data["item_id"])
            logger.info(
                "POST /items[%d] %s -> item_id=%s",
                i + 1,
                item_data["title"],
                data["item_id"],
            )

        # --- 3. Fetch by ID ---
        logger.info("Step 3: Fetch by ID - GET /items/%s", item_ids[0])
        r = client.get(f"/items/{item_ids[0]}")
        r.raise_for_status()
        item = r.json()
        assert item["title"] == "Python Programming Language Overview"
        assert "checksum" in item
        assert "created_at" in item

        # --- 4. List with filters ---
        logger.info("Step 4: List with filters - namespace=languages, label=people")
        r = client.get("/items", params={"namespace": "languages"})
        r.raise_for_status()
        listed = r.json()
        assert len(listed) == 2
        titles = {i["title"] for i in listed}
        assert "Python Programming Language Overview" in titles
        assert "Rust Programming Language Overview" in titles

        r = client.get("/items", params={"label": "people", "limit": 5})
        r.raise_for_status()
        listed2 = r.json()
        assert len(listed2) == 1
        assert listed2[0]["title"] == "Guido van Rossum - Python Creator"

        # --- 5. Search (hybrid, lexical, semantic) ---
        logger.info("Step 5: Search - hybrid, lexical, semantic")
        r = client.get(
            "/search",
            params={
                "query": "memory safety compile time ownership",
                "mode": "hybrid",
                "top_k": 3,
            },
        )
        r.raise_for_status()
        hybrid = r.json()
        assert len(hybrid) >= 1
        assert "chunk" in hybrid[0]
        chunk_text = hybrid[0]["chunk"]["text"]
        assert "Rust" in chunk_text or "rust" in chunk_text.lower()

        r = client.get(
            "/search",
            params={"query": "data science machine learning", "mode": "lexical", "top_k": 3},
        )
        r.raise_for_status()
        lexical = r.json()
        assert len(lexical) >= 1

        r = client.get(
            "/search",
            params={
                "query": "columnar analytical in-process database",
                "mode": "semantic",
                "top_k": 3,
            },
        )
        r.raise_for_status()

        # --- 6. RAG (POST /ask) ---
        logger.info("Step 6: RAG (POST /ask) - assert on real LLM response")
        r = client.post("/ask", json={"question": "Who created Python?"})
        r.raise_for_status()
        answer_data = r.json()
        assert "answer" in answer_data
        assert "confidence" in answer_data
        assert "sources" in answer_data
        answer_text = answer_data["answer"].lower()
        assert "guido" in answer_text or "van rossum" in answer_text, (
            f"RAG answer should mention Python creator; got: {answer_data['answer'][:200]}"
        )
        assert len(answer_data["answer"].strip()) > 0

        # --- 7. Add relations ---
        logger.info("Step 7: Add relations")
        relations_data = [
            ("Guido van Rossum", "created", "Python", item_ids[3]),
            ("Guido van Rossum", "worked_at", "Google", ""),
            ("Guido van Rossum", "worked_at", "Microsoft", ""),
            ("Graydon Hoare", "created", "Rust", ""),
            ("Rust", "sponsored", "Mozilla", ""),
            ("Mark Raasveldt", "created", "DuckDB", ""),
            ("DuckDB", "integrates_with", "Python", ""),
        ]

        for subj, pred, obj, iid in relations_data:
            payload = {
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "item_id": iid if iid else None,
            }
            r = client.post("/relations", json=payload)
            r.raise_for_status()
            rel_data = r.json()
            assert "relation_id" in rel_data

        # --- 8. State (latest-wins) ---
        logger.info("Step 8: State - GET /entities/Guido van Rossum/state")
        entity_escaped = quote("Guido van Rossum", safe="")
        r = client.get(f"/entities/{entity_escaped}/state")
        r.raise_for_status()
        state = r.json()
        assert state["properties"]["worked_at"] == "Microsoft"
        assert state["properties"]["created"] == "Python"

        # --- 9. History ---
        logger.info("Step 9: History - full + predicate filter")
        r = client.get(f"/entities/{entity_escaped}/history")
        r.raise_for_status()
        history = r.json()
        assert len(history) == 3

        r = client.get(f"/entities/{entity_escaped}/history", params={"predicate": "worked_at"})
        r.raise_for_status()
        history_pred = r.json()
        assert len(history_pred) == 2

        # --- 10. Traverse ---
        logger.info("Step 10: Traverse - full + link filter")
        r = client.get(f"/entities/{entity_escaped}/traverse", params={"max_hops": 3})
        r.raise_for_status()
        traverse = r.json()
        assert "paths" in traverse
        assert "entities" in traverse
        assert "Guido van Rossum" in traverse["entities"]

        duckdb_escaped = quote("DuckDB", safe="")
        r = client.get(
            f"/entities/{duckdb_escaped}/traverse",
            params={"link": "integrates_with", "max_hops": 2},
        )
        r.raise_for_status()
        traverse_link = r.json()
        assert "DuckDB" in traverse_link["entities"]
        assert "Python" in traverse_link["entities"]

        # --- 11. Extract ---
        logger.info("Step 11: Extract - POST /extract")
        r = client.post("/extract", json={"item_id": item_ids[2]})
        r.raise_for_status()
        extract_data = r.json()
        assert "relation_ids" in extract_data
        extracted_ids = extract_data["relation_ids"]
        assert isinstance(extracted_ids, list)
        assert len(extracted_ids) >= 1, "DuckDB item should yield extractable relations"

        # --- 12. Sessions ---
        logger.info("Step 12: Sessions - start, add, add_relation, search, end, list, replay")
        r = client.post("/sessions", json={"name": "test-session-1"})
        r.raise_for_status()
        session_data = r.json()
        session_id_1 = session_data["session_id"]

        r = client.post(
            "/items",
            json={
                "text": "TypeScript is a strongly typed superset of JavaScript by Microsoft.",
                "title": "TypeScript Overview",
                "uri": "",
                "namespace": "session-test",
                "label": "",
            },
        )
        r.raise_for_status()

        r = client.post(
            "/relations",
            json={
                "subject": "Microsoft",
                "predicate": "created",
                "object": "TypeScript",
                "item_id": None,
            },
        )
        r.raise_for_status()

        r = client.post(f"/sessions/{session_id_1}/end")
        r.raise_for_status()

        r = client.post("/sessions", json={"name": "test-session-2"})
        r.raise_for_status()
        session2_data = r.json()
        session_id_2 = session2_data["session_id"]

        r = client.get(
            "/search",
            params={
                "query": "TypeScript JavaScript Microsoft",
                "mode": "lexical",
                "top_k": 10,
            },
        )
        r.raise_for_status()

        r = client.post(f"/sessions/{session_id_2}/end")
        r.raise_for_status()

        r = client.get("/sessions")
        r.raise_for_status()
        sessions = r.json()
        assert len(sessions) == 2

        r = client.get(f"/sessions/{session_id_1}/events")
        r.raise_for_status()
        replay = r.json()
        add_events = [e for e in replay if e["kind"] == "add"]
        assert len(add_events) >= 1

        # --- 13. Doctor ---
        logger.info("Step 13: Doctor - vacuum, rebuild_fts, rebuild_vec")
        r = client.post("/doctor", json={"vacuum": True, "rebuild_fts": True, "rebuild_vec": True})
        r.raise_for_status()
        doctor = r.json()
        assert doctor["results"]["vacuum"] is True
        assert doctor["results"]["rebuild_fts"] is True
        assert doctor["results"]["rebuild_vec"] is True

        # --- 13.5. Semantic search post-doctor ---
        logger.info("Step 13.5: Semantic search post-doctor")
        r = client.get(
            "/search",
            params={
                "query": "columnar analytical in-process database",
                "mode": "semantic",
                "top_k": 5,
            },
        )
        r.raise_for_status()
        semantic_post = r.json()
        assert len(semantic_post) >= 1, "Semantic search must return results after rebuild_vec"
        top_title = semantic_post[0]["item"]["title"]
        assert "DuckDB" in top_title or "duckdb" in top_title.lower()

        # --- 14. Final stats ---
        logger.info("Step 14: Final stats")
        r = client.get("/stats")
        r.raise_for_status()
        final_stats = r.json()
        assert final_stats["items"] == 5
        assert final_stats["chunks"] == 5
        assert final_stats["relations"] >= 9, "8 manual + >=1 from extract"
        assert final_stats["entities"] >= 10, "10+ entities after extraction"
        assert final_stats["sessions"] == 2

    finally:
        client.close()

    # --- Phase 3: Direct DB verification (server must be terminated) ---
    logger.info("Phase 3: Direct DB verification (terminating server, then connecting to DB)")
    proc.terminate()
    proc.wait(timeout=5)

    conn = duckdb.connect(str(db_path))
    try:
        items_count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        chunks_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        relations_count = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        entities_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        sessions_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

        assert items_count == 5
        assert chunks_count == 5
        assert relations_count >= 9, "8 manual + >=1 from extract"
        assert entities_count >= 10, "10+ entities after extraction"
        assert sessions_count == 2
        logger.info(
            "Direct DB: items=%d, chunks=%d, relations=%d, entities=%d, sessions=%d",
            items_count,
            chunks_count,
            relations_count,
            entities_count,
            sessions_count,
        )

        titles = conn.execute("SELECT title FROM items ORDER BY created_at").fetchall()
        titles_flat = [t[0] for t in titles]
        assert "Python Programming Language Overview" in titles_flat
        assert "TypeScript Overview" in titles_flat

        rels = conn.execute("SELECT subject, predicate, object FROM relations LIMIT 3").fetchall()
        assert len(rels) >= 3

        chunks_with_embedding = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        assert chunks_with_embedding == chunks_count, (
            f"Expected all {chunks_count} chunks to have embeddings, got {chunks_with_embedding}"
        )

        sample_embedding = conn.execute(
            "SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()[0]
        assert sample_embedding is not None
        embed_dim = len(sample_embedding)
        assert embed_dim == 4096, f"Expected embed_dim=4096, got {embed_dim}"
        logger.info("Direct DB embeddings: dimension=%d", embed_dim)
    finally:
        conn.close()


# =============================================================================
# Full Flow Tests (Ollama + OpenAI)
# =============================================================================


def test_rest_api_full_flow(rest_server_ollama) -> None:
    """Run full REST API flow with Ollama LLM."""
    base_url, db_path, proc = rest_server_ollama
    _run_full_rest_flow(base_url, db_path, proc)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_rest_api_full_flow_openai(rest_server_openai) -> None:
    """Run full REST API flow with OpenAI gpt-4o-mini (verifies LLM responses)."""
    base_url, db_path, proc = rest_server_openai
    _run_full_rest_flow(base_url, db_path, proc)


# =============================================================================
# Unit Tests (use module-scoped rest_server)
# =============================================================================


class TestHealth:
    """Tests for GET /health."""

    def test_health_returns_healthy(self, http_client: httpx.Client) -> None:
        """Health check returns status healthy."""
        r = http_client.get("/health")
        r.raise_for_status()
        data = r.json()
        assert data["status"] == "healthy"


class TestItems:
    """Tests for item endpoints."""

    def test_add_item(self, http_client: httpx.Client) -> None:
        """POST /items creates an item."""
        r = http_client.post(
            "/items",
            json={"text": "Test document content.", "title": "Test Doc", "namespace": "default"},
        )
        r.raise_for_status()
        data = r.json()
        assert "item_id" in data
        assert len(data["item_id"]) == 22

    def test_add_item_empty_text_returns_400(self, http_client: httpx.Client) -> None:
        """POST /items with empty text returns 400."""
        r = http_client.post("/items", json={"text": "", "title": "Empty"})
        assert r.status_code == 400

    def test_get_item_not_found_returns_404(self, http_client: httpx.Client) -> None:
        """GET /items/nonexistent returns 404."""
        r = http_client.get("/items/nonexistent_id_xyz")
        assert r.status_code == 404
        assert "Item not found" in r.json()["detail"]

    def test_list_items(self, http_client: httpx.Client) -> None:
        """GET /items returns list."""
        r = http_client.get("/items")
        r.raise_for_status()
        data = r.json()
        assert isinstance(data, list)


class TestSearch:
    """Tests for GET /search."""

    def test_search_returns_list(self, http_client: httpx.Client) -> None:
        """GET /search returns list of results."""
        r = http_client.get("/search", params={"query": "test", "mode": "lexical", "top_k": 5})
        r.raise_for_status()
        data = r.json()
        assert isinstance(data, list)


class TestRelations:
    """Tests for POST /relations."""

    def test_add_relation(self, http_client: httpx.Client) -> None:
        """POST /relations creates a relation."""
        r = http_client.post(
            "/relations",
            json={"subject": "Alice", "predicate": "works_at", "object": "Acme"},
        )
        r.raise_for_status()
        data = r.json()
        assert "relation_id" in data


class TestEntities:
    """Tests for entity endpoints."""

    def test_entity_state(self, http_client: httpx.Client) -> None:
        """GET /entities/{entity}/state returns properties."""
        # Add relation first
        http_client.post(
            "/relations",
            json={"subject": "Bob", "predicate": "role", "object": "Engineer"},
        )
        r = http_client.get("/entities/Bob/state")
        r.raise_for_status()
        data = r.json()
        assert "entity" in data
        assert "properties" in data
        assert data["properties"]["role"] == "Engineer"

    def test_entity_history(self, http_client: httpx.Client) -> None:
        """GET /entities/{entity}/history returns relations."""
        http_client.post(
            "/relations",
            json={"subject": "Charlie", "predicate": "knows", "object": "Alice"},
        )
        r = http_client.get("/entities/Charlie/history")
        r.raise_for_status()
        data = r.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_entity_traverse(self, http_client: httpx.Client) -> None:
        """GET /entities/{entity}/traverse returns paths and entities."""
        http_client.post(
            "/relations",
            json={"subject": "Diana", "predicate": "links_to", "object": "Eve"},
        )
        r = http_client.get("/entities/Diana/traverse", params={"max_hops": 2})
        r.raise_for_status()
        data = r.json()
        assert "paths" in data
        assert "entities" in data


class TestSessions:
    """Tests for session endpoints."""

    def test_start_session(self, http_client: httpx.Client) -> None:
        """POST /sessions starts a session."""
        r = http_client.post("/sessions", json={"name": "unit-test-session"})
        r.raise_for_status()
        data = r.json()
        assert "session_id" in data

    def test_list_sessions(self, http_client: httpx.Client) -> None:
        """GET /sessions returns list."""
        r = http_client.get("/sessions")
        r.raise_for_status()
        data = r.json()
        assert isinstance(data, list)


class TestMaintenance:
    """Tests for maintenance endpoints."""

    def test_stats(self, http_client: httpx.Client) -> None:
        """GET /stats returns statistics."""
        r = http_client.get("/stats")
        r.raise_for_status()
        data = r.json()
        assert "items" in data
        assert "chunks" in data
        assert "relations" in data
        assert "entities" in data
        assert "sessions" in data
        assert "file_size_bytes" in data

    def test_verify(self, http_client: httpx.Client) -> None:
        """GET /verify returns verification result."""
        r = http_client.get("/verify", params={"deep": False})
        r.raise_for_status()
        data = r.json()
        assert "items" in data
        assert "errors" in data

    def test_doctor(self, http_client: httpx.Client) -> None:
        """POST /doctor returns results."""
        r = http_client.post(
            "/doctor",
            json={"vacuum": False, "rebuild_fts": False, "rebuild_vec": False},
        )
        r.raise_for_status()
        data = r.json()
        assert "results" in data


class TestErrors:
    """Tests for error handling."""

    def test_post_items_invalid_json_returns_422(self, http_client: httpx.Client) -> None:
        """POST /items with invalid body returns 422."""
        r = http_client.post("/items", json={})  # missing required 'text'
        assert r.status_code == 422
