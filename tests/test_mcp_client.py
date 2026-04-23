"""Integration tests for DuckMem MCP server via FastMCP client.

Exercises all MCP tools through the client (Phase 1), then verifies
database state directly after server shutdown (Phase 3).

Requires: Ollama running with:
  - qwen3-embedding:latest (embeddings)
  - gpt-oss:20b (LLM for RAG and extraction; or set DUCKMEM_LLM_MODEL)

OpenAI variant requires: OPENAI_API_KEY set (uses gpt-4o-mini).

Run with logging visible: uv run pytest tests/test_mcp_client.py -v --log-cli-level=INFO
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import duckdb
import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

logger = logging.getLogger(__name__)


def _parse_json_result(result) -> dict | list:
    """Parse JSON from MCP tool result content."""
    text = result.content[0].text
    return json.loads(text)


@pytest.fixture
def project_root() -> Path:
    """Project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Temporary database path for MCP server."""
    return tmp_path / "test_mcp.duckdb"


@pytest.fixture
def mcp_client(project_root: Path, test_db_path: Path) -> Client:
    """FastMCP client configured to spawn DuckMem MCP server via StdioTransport."""
    env = os.environ.copy()
    env.update(
        {
            "DUCKMEM_DB_PATH": str(test_db_path),
            "DUCKMEM_EMBED_MODEL": "ollama/qwen3-embedding:latest",
            "DUCKMEM_EMBED_DIM": "4096",
            "DUCKMEM_LLM_MODEL": "ollama/gpt-oss:20b",
        }
    )

    transport = StdioTransport(
        command="uv",
        args=[
            "run",
            "--project",
            str(project_root),
            "python",
            "-m",
            "duckmem.mcp_server",
        ],
        env=env,
        cwd=str(project_root),
        keep_alive=False,
    )
    return Client(transport)


@pytest.fixture
def mcp_client_openai(project_root: Path, test_db_path: Path) -> Client:
    """FastMCP client with OpenAI LLM (gpt-4o-mini). Requires OPENAI_API_KEY."""
    env = os.environ.copy()
    env.update(
        {
            "DUCKMEM_DB_PATH": str(test_db_path),
            "DUCKMEM_EMBED_MODEL": "ollama/qwen3-embedding:latest",
            "DUCKMEM_EMBED_DIM": "4096",
            "DUCKMEM_LLM_MODEL": "openai/gpt-4o-mini",
        }
    )

    transport = StdioTransport(
        command="uv",
        args=[
            "run",
            "--project",
            str(project_root),
            "python",
            "-m",
            "duckmem.mcp_server",
        ],
        env=env,
        cwd=str(project_root),
        keep_alive=False,
    )
    return Client(transport)


async def _run_full_mcp_flow(client: Client, test_db_path: Path) -> None:
    """Shared full MCP flow: all tools, then direct DB verification."""
    item_ids: list[str] = []
    relation_ids: list[str] = []
    session_id_1: str = ""

    async with client:
        # --- 1. Initial state ---
        logger.info("Step 1: Initial state - memory_stats, memory_verify")
        stats_result = await client.call_tool("memory_stats", {})
        stats = _parse_json_result(stats_result)
        logger.info("memory_stats: %s", stats)
        assert stats["items"] == 0
        assert stats["chunks"] == 0
        assert stats["relations"] == 0
        assert stats["entities"] == 0
        assert stats["sessions"] == 0

        verify_result = await client.call_tool("memory_verify", {"deep": True})
        verify = _parse_json_result(verify_result)
        logger.info(
            "memory_verify: checksum_ok=%s, errors=%s",
            verify["checksum_ok"],
            verify["errors"],
        )
        assert verify["checksum_ok"] is True
        assert verify["errors"] == []

        # --- 2. Ingest 4 items ---
        logger.info("Step 2: Ingest 4 items (Python, Rust, DuckDB, Guido)")
        python_text = (
            "Python is a high-level, interpreted programming language known for "
            "its clear syntax and readability. It was created by Guido van Rossum "
            "and first released in 1991. Python supports multiple programming "
            "paradigms including procedural, object-oriented, and functional "
            "programming. It is widely used in data science, machine learning, "
            "web development, and automation."
        )
        rust_text = (
            "Rust is a systems programming language focused on safety, speed, "
            "and concurrency. It was created by Graydon Hoare and sponsored by "
            "Mozilla. Rust prevents memory safety issues at compile time through "
            "its ownership and borrowing system. It is commonly used for "
            "WebAssembly, embedded systems, and performance-critical applications."
        )
        duckdb_text = (
            "DuckDB is an in-process analytical database system designed for "
            "OLAP workloads. It was developed by Mark Raasveldt and Hannes "
            "Mühleisen at CWI Amsterdam. DuckDB is embedded (no server needed), "
            "supports SQL, and is optimized for columnar storage and vectorized "
            "query execution. It integrates seamlessly with Python pandas and "
            "Apache Arrow."
        )
        guido_text = (
            "Guido van Rossum is a Dutch programmer best known as the creator "
            "of the Python programming language. He worked at Google for several "
            "years and later joined Dropbox. He was Python's \"Benevolent Dictator "
            'For Life" (BDFL) until he stepped down in 2018. He rejoined '
            "Microsoft in 2020."
        )
        items_data = [
            {
                "text": python_text,
                "title": "Python Programming Language Overview",
                "uri": "https://python.org",
                "namespace": "languages",
                "label": "programming",
            },
            {
                "text": rust_text,
                "title": "Rust Programming Language Overview",
                "uri": "https://rust-lang.org",
                "namespace": "languages",
                "label": "programming",
            },
            {
                "text": duckdb_text,
                "title": "DuckDB Analytical Database",
                "uri": "https://duckdb.org",
                "namespace": "ml",
                "label": "ai",
            },
            {
                "text": guido_text,
                "title": "Guido van Rossum - Python Creator",
                "uri": "",
                "namespace": "default",
                "label": "people",
            },
        ]

        for i, item_data in enumerate(items_data):
            add_result = await client.call_tool("memory_add", item_data)
            data = _parse_json_result(add_result)
            assert "item_id" in data
            item_ids.append(data["item_id"])
            logger.info(
                "memory_add[%d] %s -> item_id=%s",
                i + 1,
                item_data["title"],
                data["item_id"],
            )

        # --- 3. Fetch by ID ---
        logger.info("Step 3: Fetch by ID - memory_get(%s)", item_ids[0])
        get_result = await client.call_tool("memory_get", {"item_id": item_ids[0]})
        item = _parse_json_result(get_result)
        assert "error" not in item
        assert item["title"] == "Python Programming Language Overview"
        assert "checksum" in item
        assert "created_at" in item
        logger.info(
            "memory_get: title=%s, checksum=%s",
            item["title"],
            item["checksum"][:16] + "...",
        )

        # --- 4. List with filters ---
        logger.info("Step 4: List with filters - namespace=languages, label=people")
        list_result = await client.call_tool("memory_list", {"namespace": "languages"})
        listed = _parse_json_result(list_result)
        assert len(listed) == 2
        titles = {i["title"] for i in listed}
        assert "Python Programming Language Overview" in titles
        assert "Rust Programming Language Overview" in titles

        list_result2 = await client.call_tool("memory_list", {"label": "people", "limit": 5})
        listed2 = _parse_json_result(list_result2)
        assert len(listed2) == 1
        assert listed2[0]["title"] == "Guido van Rossum - Python Creator"
        logger.info(
            "memory_list: namespace=languages -> %d items, label=people -> %d items",
            len(listed),
            len(listed2),
        )

        # --- 5. Search (hybrid, lexical, semantic) ---
        logger.info("Step 5: Search - hybrid, lexical, semantic")
        hybrid_result = await client.call_tool(
            "memory_search",
            {
                "query": "memory safety compile time ownership",
                "mode": "hybrid",
                "top_k": 3,
            },
        )
        hybrid = _parse_json_result(hybrid_result)
        assert len(hybrid) >= 1
        assert "chunk_id" in hybrid[0]
        assert "Rust" in hybrid[0]["text"] or "rust" in hybrid[0]["text"].lower()
        logger.info("memory_search hybrid: %d results, top=%s", len(hybrid), hybrid[0]["title"])

        lexical_result = await client.call_tool(
            "memory_search",
            {"query": "data science machine learning", "mode": "lexical", "top_k": 3},
        )
        lexical = _parse_json_result(lexical_result)
        assert len(lexical) >= 1
        logger.info("memory_search lexical: %d results", len(lexical))

        # Semantic may be empty pre-doctor
        semantic_result = await client.call_tool(
            "memory_search",
            {
                "query": "columnar analytical in-process database",
                "mode": "semantic",
                "top_k": 3,
            },
        )
        semantic = _parse_json_result(semantic_result)  # Just ensure no error
        logger.info("memory_search semantic: %d results (may be empty pre-doctor)", len(semantic))

        # --- 6. RAG (memory_ask) - real LLM response ---
        logger.info("Step 6: RAG (memory_ask) - assert on real LLM response")
        ask_result = await client.call_tool("memory_ask", {"question": "Who created Python?"})
        answer_data = _parse_json_result(ask_result)
        assert "answer" in answer_data
        assert "confidence" in answer_data
        assert "sources" in answer_data
        answer_text = answer_data["answer"].lower()
        assert "guido" in answer_text or "van rossum" in answer_text, (
            f"RAG answer should mention Python creator; got: {answer_data['answer'][:200]}"
        )
        assert len(answer_data["answer"].strip()) > 0
        logger.info(
            "memory_ask: answer=%s..., confidence=%s, sources=%d",
            answer_data["answer"][:80],
            answer_data["confidence"],
            len(answer_data["sources"]),
        )

        # --- 7. Add relations ---
        logger.info("Step 7: Add 7 relations + 1 in session (8 total)")
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
            add_rel_result = await client.call_tool(
                "memory_add_relation",
                {
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "item_id": iid if iid else "",
                },
            )
            rel_data = _parse_json_result(add_rel_result)
            assert "relation_id" in rel_data
            relation_ids.append(rel_data["relation_id"])
        logger.info("memory_add_relation: %d relations added", len(relation_ids))

        # --- 8. State (latest-wins) ---
        logger.info("Step 8: State (latest-wins) - memory_state(Guido van Rossum)")
        state_result = await client.call_tool("memory_state", {"entity": "Guido van Rossum"})
        state = _parse_json_result(state_result)
        assert state["properties"]["worked_at"] == "Microsoft"
        assert state["properties"]["created"] == "Python"
        logger.info("memory_state: %s", state["properties"])

        # --- 9. History ---
        logger.info("Step 9: History - memory_history (full + predicate filter)")
        history_result = await client.call_tool(
            "memory_history", {"entity": "Guido van Rossum"}
        )
        history = _parse_json_result(history_result)
        assert len(history) == 3

        history_pred_result = await client.call_tool(
            "memory_history",
            {"entity": "Guido van Rossum", "predicate": "worked_at"},
        )
        history_pred = _parse_json_result(history_pred_result)
        assert len(history_pred) == 2
        logger.info(
            "memory_history: full=%d relations, predicate=worked_at -> %d",
            len(history),
            len(history_pred),
        )

        # --- 10. Traverse ---
        logger.info("Step 10: Traverse - memory_traverse (full + link filter)")
        traverse_result = await client.call_tool(
            "memory_traverse", {"start": "Guido van Rossum", "max_hops": 3}
        )
        traverse = _parse_json_result(traverse_result)
        assert "paths" in traverse
        assert "entities" in traverse
        assert "Guido van Rossum" in traverse["entities"]

        traverse_link_result = await client.call_tool(
            "memory_traverse",
            {"start": "DuckDB", "link": "integrates_with", "max_hops": 2},
        )
        traverse_link = _parse_json_result(traverse_link_result)
        assert "DuckDB" in traverse_link["entities"]
        assert "Python" in traverse_link["entities"]
        logger.info(
            "memory_traverse: %d paths, entities=%s",
            len(traverse["paths"]),
            traverse["entities"],
        )

        # --- 11. Extract - real LLM extraction ---
        logger.info("Step 11: Extract (memory_extract) - assert on real LLM extraction")
        extract_result = await client.call_tool("memory_extract", {"item_id": item_ids[2]})
        extract_data = _parse_json_result(extract_result)
        assert "error" not in extract_data, f"memory_extract should succeed; got: {extract_data}"
        assert "relation_ids" in extract_data
        assert "count" in extract_data
        extracted_count = extract_data["count"]
        extracted_ids = extract_data["relation_ids"]
        assert isinstance(extracted_ids, list)
        assert len(extracted_ids) == extracted_count
        # DuckDB item mentions Mark Raasveldt, Hannes Mühleisen, CWI, etc. - expect relations
        assert extracted_count >= 1, (
            f"DuckDB item should yield extractable relations; got count={extracted_count}"
        )
        relation_ids.extend(extracted_ids)
        logger.info(
            "memory_extract: extracted %d relations from DuckDB item",
            extracted_count,
        )

        # --- 12. Sessions ---
        logger.info("Step 12: Sessions - start, add, add_relation, search, end, list, replay")
        session_start_result = await client.call_tool(
            "memory_session_start", {"name": "test-session-1"}
        )
        session_data = _parse_json_result(session_start_result)
        session_id_1 = session_data["session_id"]

        typescript_text = (
            "TypeScript is a strongly typed superset of JavaScript developed by "
            "Microsoft. It adds optional static typing and compiles to plain "
            "JavaScript."
        )
        add_result = await client.call_tool(
            "memory_add",
            {
                "text": typescript_text,
                "title": "TypeScript Overview",
                "uri": "",
                "namespace": "session-test",
                "label": "",
            },
        )
        typescript_item_id = _parse_json_result(add_result)["item_id"]

        await client.call_tool(
            "memory_add_relation",
            {
                "subject": "Microsoft",
                "predicate": "created",
                "object": "TypeScript",
                "item_id": "",
            },
        )

        await client.call_tool(
            "memory_extract",
            {"item_id": typescript_item_id},
        )

        await client.call_tool("memory_session_end", {})

        await client.call_tool("memory_session_start", {"name": "test-session-2"})

        await client.call_tool(
            "memory_search",
            {
                "query": "TypeScript JavaScript Microsoft",
                "mode": "lexical",
                "top_k": 10,
            },
        )

        await client.call_tool("memory_session_end", {})

        session_list_result = await client.call_tool("memory_session_list", {})
        sessions = _parse_json_result(session_list_result)
        assert len(sessions) == 2

        replay_result = await client.call_tool(
            "memory_session_replay", {"session_id": session_id_1}
        )
        replay = _parse_json_result(replay_result)
        add_events = [e for e in replay if e["kind"] == "add"]
        add_relation_events = [e for e in replay if e["kind"] == "add_relation"]
        extract_events = [e for e in replay if e["kind"] == "extract"]
        assert len(add_events) >= 1
        assert "add_relation" in [e["kind"] for e in replay]
        assert len(add_relation_events) >= 1
        assert "extract" in [e["kind"] for e in replay]
        assert len(extract_events) >= 1
        logger.info(
            "memory_session: %d sessions, replay has %d events (add_relation, extract captured)",
            len(sessions),
            len(replay),
        )

        # --- 13. Doctor ---
        logger.info("Step 13: Doctor - vacuum, rebuild_fts, rebuild_vec")
        doctor_result = await client.call_tool(
            "memory_doctor",
            {"vacuum": True, "rebuild_fts": True, "rebuild_vec": True},
        )
        doctor = _parse_json_result(doctor_result)
        assert doctor.get("vacuum") is True
        assert doctor.get("rebuild_fts") is True
        assert doctor.get("rebuild_vec") is True
        logger.info("memory_doctor: %s", doctor)

        # --- 13.5. Semantic search post-doctor (verify embeddings + HNSW index) ---
        logger.info("Step 13.5: Semantic search post-doctor - verify embeddings work")
        semantic_post_result = await client.call_tool(
            "memory_search",
            {
                "query": "columnar analytical in-process database",
                "mode": "semantic",
                "top_k": 5,
            },
        )
        semantic_post = _parse_json_result(semantic_post_result)
        assert len(semantic_post) >= 1, "Semantic search must return results after rebuild_vec"
        top_title = semantic_post[0]["title"]
        assert "DuckDB" in top_title or "duckdb" in top_title.lower()
        logger.info(
            "memory_search semantic (post-doctor): %d results, top=%s",
            len(semantic_post),
            semantic_post[0]["title"],
        )

        # --- 14. Final stats ---
        logger.info("Step 14: Final stats - memory_stats")
        final_stats_result = await client.call_tool("memory_stats", {})
        final_stats = _parse_json_result(final_stats_result)
        assert final_stats["items"] == 5
        assert final_stats["chunks"] == 5
        assert final_stats["relations"] >= 9, "8 manual + >=1 from memory_extract"
        assert final_stats["entities"] >= 10, "10+ entities after extraction"
        assert final_stats["sessions"] == 2
        logger.info("memory_stats final: %s", final_stats)

    # --- Phase 3: Direct DB verification (server has terminated) ---
    logger.info("Phase 3: Direct DB verification (server terminated, lock released)")
    conn = duckdb.connect(str(test_db_path))
    try:
        items_count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        chunks_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        relations_count = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        entities_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        sessions_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

        assert items_count == 5
        assert chunks_count == 5
        assert relations_count >= 9, "8 manual + >=1 from memory_extract"
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

        # Spot-check item titles
        titles = conn.execute("SELECT title FROM items ORDER BY created_at").fetchall()
        titles_flat = [t[0] for t in titles]
        assert "Python Programming Language Overview" in titles_flat
        assert "TypeScript Overview" in titles_flat
        logger.info("Direct DB spot-check: titles OK, relations OK")

        # Spot-check relations
        rels = conn.execute("SELECT subject, predicate, object FROM relations LIMIT 3").fetchall()
        assert len(rels) >= 3

        # Verify embeddings: all chunks must have non-null embedding vectors
        chunks_with_embedding = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        assert chunks_with_embedding == chunks_count, (
            f"Expected all {chunks_count} chunks to have embeddings, got {chunks_with_embedding}"
        )
        logger.info("Direct DB embeddings: all %d chunks have non-null embeddings", chunks_count)

        # Verify embedding dimension (4096 for qwen3-embedding)
        sample_embedding = conn.execute(
            "SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()[0]
        assert sample_embedding is not None
        embed_dim = len(sample_embedding)
        assert embed_dim == 4096, f"Expected embed_dim=4096, got {embed_dim}"
        logger.info("Direct DB embeddings: dimension=%d", embed_dim)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_mcp_client_full_flow(
    mcp_client: Client,
    test_db_path: Path,
) -> None:
    """Run full MCP tool flow with Ollama LLM."""
    await _run_full_mcp_flow(mcp_client, test_db_path)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
async def test_mcp_client_full_flow_openai(
    mcp_client_openai: Client,
    test_db_path: Path,
) -> None:
    """Run full MCP tool flow with OpenAI gpt-4o-mini (verifies LLM responses)."""
    await _run_full_mcp_flow(mcp_client_openai, test_db_path)
