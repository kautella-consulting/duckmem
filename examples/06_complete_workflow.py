#!/usr/bin/env python3
"""DuckMem end-to-end workflow example.

Demonstrates:
- Configuring DuckMem via `Settings`
- Ingesting documents with metadata and namespaces
- Building a small knowledge graph
- Hybrid search
- Session tracking (auto-logged events)
- Optional RAG Q&A (skipped if no LLM is configured)
- Verification, statistics, and maintenance
"""

import asyncio

from duckmem import DuckMem, Settings


async def main() -> None:
    print("=" * 60)
    print("DuckMem Complete Workflow Example")
    print("=" * 60)

    settings = Settings(
        db_path="complete_workflow.duckdb",
        chunk_max_chars=500,
        chunk_overlap=50,
    )

    with DuckMem(settings=settings) as mem:
        # Phase 1: Document ingestion
        print("\n" + "=" * 40)
        print("Phase 1: Document Ingestion")
        print("=" * 40)

        research_docs = [
            {
                "content": """
                Transformers are a neural network architecture that uses
                self-attention mechanisms. They were introduced in the paper
                "Attention Is All You Need" by Vaswani et al. in 2017. The
                key innovation is the ability to process sequences in parallel,
                unlike recurrent networks.
                """,
                "title": "Transformer Architecture",
                "namespace": "research/papers",
                "metadata": {"year": 2017, "authors": ["Vaswani"], "type": "foundational"},
            },
            {
                "content": """
                BERT (Bidirectional Encoder Representations from Transformers)
                was developed by Google in 2018. It uses masked language modeling
                for pre-training, where random tokens are masked and the model
                learns to predict them from bidirectional context.
                """,
                "title": "BERT Paper Summary",
                "namespace": "research/papers",
                "metadata": {"year": 2018, "organization": "Google", "type": "model"},
            },
            {
                "content": """
                GPT-4 is a multimodal model by OpenAI that can process both
                text and images. It demonstrates emergent capabilities and
                achieves human-level performance on various benchmarks.
                The model uses RLHF for alignment.
                """,
                "title": "GPT-4 Technical Overview",
                "namespace": "research/papers",
                "metadata": {"year": 2023, "organization": "OpenAI", "type": "model"},
            },
        ]

        project_docs = [
            {
                "content": """
                Project Athena uses transformer models for document classification.
                We fine-tuned BERT on our internal corpus and achieved 94% accuracy.
                The model is deployed on AWS with an average latency of 50ms.
                """,
                "title": "Project Athena Results",
                "namespace": "work/projects",
                "metadata": {"project": "athena", "status": "deployed"},
            },
            {
                "content": """
                Meeting: Discussed migrating from BERT to a smaller model.
                Action items: benchmark DistilBERT, evaluate latency requirements,
                prepare cost analysis for next review.
                """,
                "title": "Project Athena Planning Meeting",
                "namespace": "work/meetings",
                "metadata": {"date": "2024-01-20", "project": "athena"},
            },
        ]

        all_docs = research_docs + project_docs

        for doc in all_docs:
            mem.add(
                doc["content"].strip(),
                title=doc["title"],
                namespace=doc["namespace"],
                metadata=doc["metadata"],
            )
            print(f"   Added: [{doc['namespace']}] {doc['title']}")

        print(f"\n   Total: {len(all_docs)} documents ingested")

        # Phase 2: Knowledge graph
        print("\n" + "=" * 40)
        print("Phase 2: Building Knowledge Graph")
        print("=" * 40)

        relations = [
            ("Transformers", "introduced_by", "Vaswani et al."),
            ("Transformers", "year", "2017"),
            ("Transformers", "uses", "Self-Attention"),
            ("BERT", "is_a", "Transformer"),
            ("BERT", "developed_by", "Google"),
            ("BERT", "uses", "Masked Language Modeling"),
            ("GPT-4", "is_a", "Transformer"),
            ("GPT-4", "developed_by", "OpenAI"),
            ("GPT-4", "capability", "Multimodal"),
            ("Project Athena", "uses", "BERT"),
            ("Project Athena", "task", "Document Classification"),
        ]

        for subj, pred, obj in relations:
            mem.add_relation(subj, pred, obj)
            print(f"   Added: {subj} --[{pred}]--> {obj}")

        print(f"\n   Total: {len(relations)} relations added")

        print("\n   Exploring entity 'BERT':")
        state = mem.state("BERT")
        for predicate, value in state.properties.items():
            print(f"   - {predicate} = {value}")

        print("\n   Graph traversal from 'Transformers' (max_hops=2):")
        result = mem.traverse("Transformers", max_hops=2)
        print(f"   Reachable entities: {len(result.entities)}")
        for name in result.entities:
            print(f"   - {name}")

        # Phase 3: Search
        print("\n" + "=" * 40)
        print("Phase 3: Search & Discovery")
        print("=" * 40)

        print("\n   Hybrid search: 'attention mechanism'")
        for r in mem.search("attention mechanism", top_k=3):
            print(f"   [{r.score:.3f}] {r.item.title}")

        print("\n   Search in 'work/projects': 'model'")
        for r in mem.search("model", namespace="work/projects", top_k=3):
            print(f"   [{r.score:.3f}] {r.item.title}")

        # Phase 4: Sessions
        print("\n" + "=" * 40)
        print("Phase 4: Session Tracking")
        print("=" * 40)

        session_id = mem.session_start(name="Research Analysis")
        print(f"   Started session: id={session_id}")

        mem.search("transformer models")
        mem.add_relation("BERT", "differs_from", "GPT-4")
        mem.search("project athena model")

        mem.session_end()
        print("   Session ended")

        print("\n   Session replay:")
        for e in mem.session_replay(session_id):
            print(f"   - kind={e.kind} params={e.params}")

        # Phase 5: RAG (optional)
        print("\n" + "=" * 40)
        print("Phase 5: RAG Q&A")
        print("=" * 40)

        try:
            print("\n   Q: What model does Project Athena use and why?")
            answer = await mem.ask(
                "What model does Project Athena use and why?",
                top_k=3,
            )
            print(f"   A: {answer.answer}")
            print(f"   Confidence: {answer.confidence:.2f}")
            print(f"   Source chunk ids: {answer.sources}")
        except Exception as e:
            print(f"   Skipped (LLM not available): {e}")

        # Phase 6: Maintenance
        print("\n" + "=" * 40)
        print("Phase 6: Maintenance & Statistics")
        print("=" * 40)

        print("\n   Verifying database integrity (deep)...")
        verify = mem.verify(deep=True)
        if verify.errors:
            for err in verify.errors:
                print(f"   - error: {err}")
        else:
            print("   No errors.")
        if verify.checksum_ok is False:
            print("   Checksum mismatch detected.")

        stats = mem.stats()
        print("\n   Database statistics:")
        print(f"   - Items:     {stats.items}")
        print(f"   - Chunks:    {stats.chunks}")
        print(f"   - Entities:  {stats.entities}")
        print(f"   - Relations: {stats.relations}")
        print(f"   - Sessions:  {stats.sessions}")
        print(f"   - File size: {stats.file_size_bytes / 1024:.1f} KB")

        print("\n   Running doctor (vacuum)...")
        results = mem.doctor(vacuum=True)
        for op, ok in results.items():
            print(f"   {op}: {'OK' if ok else 'FAILED'}")

    print("\n" + "=" * 60)
    print("Complete workflow example finished!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
