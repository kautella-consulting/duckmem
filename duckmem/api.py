"""FastAPI REST API for DuckMem.

Provides HTTP endpoints for all DuckMem operations including item
management, search, RAG Q&A, knowledge graph, and maintenance.
"""

from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from duckmem.config import get_settings
from duckmem.core import DuckMem
from duckmem.models import (
    Answer,
    EntityState,
    Item,
    Relation,
    SearchResult,
    Session,
    SessionEvent,
    Stats,
    TraversalResult,
    VerifyResult,
)

# =============================================================================
# Request/Response Models
# =============================================================================


class AddItemRequest(BaseModel):
    """Request to add an item."""

    text: str = Field(description="Text content to ingest")
    title: str | None = Field(default=None, description="Optional title")
    uri: str | None = Field(default=None, description="Optional URI reference")
    namespace: str = Field(default="default", description="Categorization namespace")
    label: str | None = Field(default=None, description="Optional label")
    metadata: dict | None = Field(default=None, description="Optional metadata")
    skip_embedding: bool = Field(default=False, description="Skip embedding computation")


class AddItemResponse(BaseModel):
    """Response after adding an item."""

    item_id: str = Field(description="Generated item ID")


class AskRequest(BaseModel):
    """Request for RAG Q&A."""

    question: str = Field(description="Question to answer")
    model: str | None = Field(default=None, description="LLM model override")
    top_k: int = Field(default=5, ge=1, le=50, description="Context chunks to retrieve")
    namespace: str | None = Field(default=None, description="Filter by namespace")


class AddRelationRequest(BaseModel):
    """Request to add a relation."""

    subject: str = Field(description="Subject entity")
    predicate: str = Field(description="Relationship type")
    object: str = Field(description="Object entity")
    item_id: str | None = Field(default=None, description="Source item ID")


class AddRelationResponse(BaseModel):
    """Response after adding a relation."""

    relation_id: str = Field(description="Generated relation ID")


class ExtractRequest(BaseModel):
    """Request to extract relations from an item."""

    item_id: str = Field(description="Item to extract from")
    model: str | None = Field(default=None, description="LLM model override")


class ExtractResponse(BaseModel):
    """Response after extracting relations."""

    relation_ids: list[str] = Field(description="Created relation IDs")


class SessionStartRequest(BaseModel):
    """Request to start a session."""

    name: str | None = Field(default=None, description="Session name")


class SessionStartResponse(BaseModel):
    """Response after starting a session."""

    session_id: str = Field(description="Session ID")


class DoctorRequest(BaseModel):
    """Request for maintenance operations."""

    vacuum: bool = Field(default=False, description="Compact storage")
    rebuild_fts: bool = Field(default=False, description="Rebuild FTS index")
    rebuild_vec: bool = Field(default=False, description="Rebuild vector index")
    timeout_seconds: float | None = Field(
        default=None,
        description="Max seconds for all ops; None = no limit",
    )


class DoctorResponse(BaseModel):
    """Response from maintenance operations."""

    results: dict[str, bool] = Field(description="Operation results")


# =============================================================================
# Application Setup
# =============================================================================


_duckmem: DuckMem | None = None


def get_duckmem() -> DuckMem:
    """Get the DuckMem instance."""
    if _duckmem is None:
        raise HTTPException(status_code=500, detail="DuckMem not initialized")
    return _duckmem


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _duckmem
    settings = get_settings()
    _duckmem = DuckMem(settings=settings)
    yield
    if _duckmem:
        _duckmem.close()
        _duckmem = None


app = FastAPI(
    title="DuckMem",
    description="Personal Knowledge Memory API - Document ingestion, hybrid search, RAG, and knowledge graph",
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Item Endpoints
# =============================================================================


@app.post("/items", response_model=AddItemResponse, tags=["Items"])
async def add_item(
    request: AddItemRequest,
    mem: Annotated[DuckMem, Depends(get_duckmem)],
) -> AddItemResponse:
    """Add an item to the knowledge base.

    Ingests text by chunking, computing embeddings, and storing in the database.
    """
    try:
        item_id = mem.add(
            request.text,
            title=request.title,
            uri=request.uri,
            namespace=request.namespace,
            label=request.label,
            metadata=request.metadata,
            skip_embedding=request.skip_embedding,
        )
        return AddItemResponse(item_id=item_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/items/{item_id}", response_model=Item, tags=["Items"])
async def get_item(
    item_id: str,
    mem: Annotated[DuckMem, Depends(get_duckmem)],
) -> Item:
    """Get an item by ID."""
    item = mem.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.get("/items", response_model=list[Item], tags=["Items"])
async def list_items(
    mem: Annotated[DuckMem, Depends(get_duckmem)],
    namespace: str | None = Query(default=None, description="Filter by namespace"),
    label: str | None = Query(default=None, description="Filter by label"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max items"),
    offset: int = Query(default=0, ge=0, description="Items to skip"),
) -> list[Item]:
    """List items with optional filtering."""
    return mem.list(namespace=namespace, label=label, limit=limit, offset=offset)


# =============================================================================
# Search Endpoints
# =============================================================================


@app.get("/search", response_model=list[SearchResult], tags=["Search"])
async def search_items(
    mem: Annotated[DuckMem, Depends(get_duckmem)],
    query: str = Query(description="Search query"),
    mode: Literal["hybrid", "lexical", "semantic"] = Query(
        default="hybrid", description="Search mode"
    ),
    top_k: int = Query(default=10, ge=1, le=100, description="Max results"),
    namespace: str | None = Query(default=None, description="Filter by namespace"),
    start_ts: int | None = Query(default=None, description="Filter by min timestamp"),
    end_ts: int | None = Query(default=None, description="Filter by max timestamp"),
) -> list[SearchResult]:
    """Search for items using hybrid BM25 + vector search."""
    return mem.search(
        query,
        mode=mode,
        top_k=top_k,
        namespace=namespace,
        start_ts=start_ts,
        end_ts=end_ts,
    )


@app.post("/ask", response_model=Answer, tags=["Search"])
async def ask_question(
    request: AskRequest,
    mem: Annotated[DuckMem, Depends(get_duckmem)],
) -> Answer:
    """Ask a question using RAG (retrieval-augmented generation)."""
    return await mem.ask(
        request.question,
        model=request.model,
        top_k=request.top_k,
        namespace=request.namespace,
    )


# =============================================================================
# Knowledge Graph Endpoints
# =============================================================================


@app.post("/relations", response_model=AddRelationResponse, tags=["Knowledge Graph"])
async def add_relation(
    request: AddRelationRequest,
    mem: Annotated[DuckMem, Depends(get_duckmem)],
) -> AddRelationResponse:
    """Add a relation to the knowledge graph."""
    relation_id = mem.add_relation(
        request.subject,
        request.predicate,
        request.object,
        item_id=request.item_id,
    )
    return AddRelationResponse(relation_id=relation_id)


@app.get("/entities/{entity}/state", response_model=EntityState, tags=["Knowledge Graph"])
async def get_entity_state(
    entity: str,
    mem: Annotated[DuckMem, Depends(get_duckmem)],
) -> EntityState:
    """Get current state of an entity (latest-wins properties)."""
    return mem.state(entity)


@app.get("/entities/{entity}/history", response_model=list[Relation], tags=["Knowledge Graph"])
async def get_entity_history(
    entity: str,
    mem: Annotated[DuckMem, Depends(get_duckmem)],
    predicate: str | None = Query(default=None, description="Filter by predicate"),
) -> list[Relation]:
    """Get relation history for an entity."""
    return mem.history(entity, predicate)


@app.get("/entities/{entity}/traverse", response_model=TraversalResult, tags=["Knowledge Graph"])
async def traverse_from_entity(
    entity: str,
    mem: Annotated[DuckMem, Depends(get_duckmem)],
    link: str | None = Query(default=None, description="Filter by predicate"),
    max_hops: int = Query(default=3, ge=1, le=10, description="Max traversal depth"),
) -> TraversalResult:
    """Traverse the knowledge graph from an entity."""
    return mem.traverse(entity, link=link, max_hops=max_hops)


@app.post("/extract", response_model=ExtractResponse, tags=["Knowledge Graph"])
async def extract_relations(
    request: ExtractRequest,
    mem: Annotated[DuckMem, Depends(get_duckmem)],
) -> ExtractResponse:
    """Extract relations from an item using LLM."""
    try:
        relation_ids = await mem.extract(request.item_id, model=request.model)
        return ExtractResponse(relation_ids=relation_ids)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# =============================================================================
# Session Endpoints
# =============================================================================


@app.post("/sessions", response_model=SessionStartResponse, tags=["Sessions"])
async def start_session(
    request: SessionStartRequest,
    mem: Annotated[DuckMem, Depends(get_duckmem)],
) -> SessionStartResponse:
    """Start a recording session."""
    session_id = mem.session_start(request.name)
    return SessionStartResponse(session_id=session_id)


@app.post("/sessions/{session_id}/end", tags=["Sessions"])
async def end_session(
    session_id: str,
    mem: Annotated[DuckMem, Depends(get_duckmem)],
) -> dict[str, str]:
    """End a recording session."""
    # Note: This ends whatever session is active, session_id is for API clarity
    mem.session_end()
    return {"status": "ended"}


@app.get("/sessions", response_model=list[Session], tags=["Sessions"])
async def list_sessions(
    mem: Annotated[DuckMem, Depends(get_duckmem)],
) -> list[Session]:
    """List all sessions."""
    return mem.session_list()


@app.get("/sessions/{session_id}/events", response_model=list[SessionEvent], tags=["Sessions"])
async def replay_session(
    session_id: str,
    mem: Annotated[DuckMem, Depends(get_duckmem)],
) -> list[SessionEvent]:
    """Replay events from a session."""
    return mem.session_replay(session_id)


# =============================================================================
# Maintenance Endpoints
# =============================================================================


@app.get("/stats", response_model=Stats, tags=["Maintenance"])
async def get_stats(
    mem: Annotated[DuckMem, Depends(get_duckmem)],
) -> Stats:
    """Get database statistics."""
    return mem.stats()


@app.get("/verify", response_model=VerifyResult, tags=["Maintenance"])
async def verify_database(
    mem: Annotated[DuckMem, Depends(get_duckmem)],
    deep: bool = Query(default=False, description="Verify checksums"),
) -> VerifyResult:
    """Verify database integrity."""
    return mem.verify(deep=deep)


@app.post("/doctor", response_model=DoctorResponse, tags=["Maintenance"])
async def run_doctor(
    request: DoctorRequest,
    mem: Annotated[DuckMem, Depends(get_duckmem)],
) -> DoctorResponse:
    """Run maintenance operations."""
    results = mem.doctor(
        vacuum=request.vacuum,
        rebuild_fts=request.rebuild_fts,
        rebuild_vec=request.rebuild_vec,
        timeout_seconds=request.timeout_seconds,
    )
    return DoctorResponse(results=results)


# =============================================================================
# Health Check
# =============================================================================


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
