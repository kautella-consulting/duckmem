# DuckMem Codebase Architecture

## 1. Module Dependency Diagram

```mermaid
flowchart TB
    subgraph EntryPoints["Entry Points"]
        CLI[cli.py<br/>Typer CLI]
        API[api.py<br/>FastAPI REST]
        MCP[mcp_server.py<br/>FastMCP]
        SDK["__init__.py<br/>Python SDK"]
    end

    subgraph Core["Core Layer"]
        DuckMem[core.py<br/>DuckMem + functional API]
    end

    subgraph Services["Services"]
        Inference[inference.py<br/>LiteLLM embeddings]
        Agents[agents.py<br/>PydanticAI RAG + extraction]
        Chunkers[ingestion/chunkers.py<br/>Text chunking]
    end

    subgraph Data["Data Layer"]
        Schema[schema.py<br/>DDL, indexes]
        Models[models.py<br/>Pydantic models]
        Config[config.py<br/>Settings]
        Utils[utils.py<br/>UID, timestamps]
    end

    subgraph Storage["Storage"]
        DuckDB[(DuckDB<br/>single-file)]
    end

    CLI --> DuckMem
    API --> DuckMem
    MCP --> DuckMem
    SDK --> DuckMem

    DuckMem --> Inference
    DuckMem --> Agents
    DuckMem --> Chunkers
    DuckMem --> Schema
    DuckMem --> Models
    DuckMem --> Config
    DuckMem --> Utils

    Schema --> DuckDB
    Inference --> Models
    Agents --> Models
    Chunkers --> Models
    Config --> DuckMem
```

## 2. Data Flow Overview

```mermaid
flowchart LR
    subgraph Input["Input"]
        Text[Raw Text]
        Query[Search Query]
        Question[RAG Question]
    end

    subgraph Ingestion["Ingestion Pipeline"]
        Chunk[chunk_text]
        Embed1[embed]
        Insert[(INSERT)]
    end

    subgraph Search["Search Pipeline"]
        BM25[BM25 FTS]
        Vector[HNSW Vector]
        RRF[RRF Fusion]
    end

    subgraph Output["Output"]
        Results[SearchResult]
        Answer[Answer + sources]
    end

    Text --> Chunk --> Embed1 --> Insert
    Query --> BM25
    Query --> Vector
    BM25 --> RRF
    Vector --> RRF
    RRF --> Results
    Question --> Search
    Results --> Answer
```

## 3. Ingestion Flow (Detail)

```mermaid
sequenceDiagram
    participant User
    participant DuckMem
    participant Chunkers
    participant Inference
    participant Schema
    participant DuckDB

    User->>DuckMem: add(text, title)
    DuckMem->>Chunkers: chunk_text(text, config)
    Chunkers-->>DuckMem: TextChunk list
    DuckMem->>Inference: embed(chunks)
    Inference-->>DuckMem: embedding vectors
    DuckMem->>DuckDB: INSERT items, chunks
    DuckMem->>Schema: init_fts_index()
    Schema->>DuckDB: Rebuild BM25 index
    DuckMem-->>User: item_id
```

## 4. Hybrid Search Flow

```mermaid
flowchart TB
    Query[Query String] --> Parallel{Parallel Search}
    Parallel --> Lexical[_search_lexical<br/>BM25 FTS]
    Parallel --> Semantic[_search_semantic<br/>embed_single + HNSW]

    Lexical --> RRF[_reciprocal_rank_fusion]
    Semantic --> RRF

    RRF --> Results["SearchResult[]<br/>chunks + items + scores"]
```

## 5. RAG & Knowledge Graph Flow

```mermaid
flowchart TB
    subgraph RAG["RAG Q&A"]
        Q[Question] --> Search[search]
        Search --> TopK[Top-K chunks]
        TopK --> Agent[PydanticAI Agent]
        Agent --> LLM[LiteLLM]
        LLM --> Answer[Answer + sources]
    end

    subgraph KG["Knowledge Graph"]
        Item[Item/Text] --> Extract[extract_relations]
        Extract --> LLM2[PydanticAI extraction]
        LLM2 --> Triples["ExtractedRelation[]"]
        Triples --> AddRel[add_relation]
        AddRel --> Entities[(entities)]
        AddRel --> Relations[(relations)]
        Relations --> Traverse[traverse_graph]
    end
```

## 6. Database Schema

```mermaid
erDiagram
    items ||--o{ chunks : contains
    items {
        string id PK
        string text
        string title
        string checksum
        bigint created_ms
    }

    chunks {
        string id PK
        string item_id FK
        int idx
        string text
        vector embedding
        bigint created_ms
    }

    entities ||--o{ relations : subject
    entities ||--o{ relations : object
    entities {
        string id PK
        string name
        jsonb state
    }

    relations {
        string id PK
        string subject_id FK
        string predicate
        string object_id FK
        string item_id FK
    }

    sessions ||--o{ session_events : has
    sessions {
        string id PK
        bigint started_ms
    }

    session_events {
        string id PK
        string session_id FK
        string event_type
        jsonb payload
    }
```

## 7. Component Overview

```mermaid
flowchart TB
    subgraph Interfaces["Interfaces"]
        direction TB
        REST[REST API<br/>/items, /search, /ask]
        MCPTools[MCP Tools<br/>duckmem_add, duckmem_search]
        CLIcmds[CLI Commands<br/>create, add, search, stats]
    end

    subgraph DuckMemCore["DuckMem Core"]
        direction TB
        Add[add / add_item]
        Search[search]
        Ask[ask - RAG]
        Extract[extract - KG]
        Traverse[traverse_graph]
        Stats[stats, doctor, verify]
    end

    subgraph Extensions["Extensions"]
        FTS[FTS extension<br/>BM25]
        VSS[VSS extension<br/>HNSW]
        DuckPGQ[DuckPGQ<br/>graph queries]
    end

    Interfaces --> DuckMemCore
    DuckMemCore --> Extensions
```

---

*Generated from DuckMem codebase analysis. View in any Mermaid-compatible renderer (GitHub, VS Code, Mermaid Live Editor).*
