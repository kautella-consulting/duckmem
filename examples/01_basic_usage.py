#!/usr/bin/env python3
"""DuckMem basic usage example.

Demonstrates:
- Creating a knowledge base
- Adding items
- Hybrid / lexical / semantic search
- Stats and integrity verification
"""

from duckmem import DuckMem


def main() -> None:
    print("=" * 50)
    print("DuckMem Basic Usage Example")
    print("=" * 50)

    with DuckMem("basic_example.duckdb") as mem:
        print("\n1. Adding documents...")

        item_id_1 = mem.add(
            """
            Machine learning is a subset of artificial intelligence that
            enables systems to automatically learn and improve from experience.
            It focuses on developing algorithms that can access data and use
            it to learn for themselves.
            """,
            title="Machine Learning Introduction",
            namespace="tech",
        )
        print(f"   Added: Machine Learning Introduction (id={item_id_1})")

        item_id_2 = mem.add(
            """
            Deep learning is a type of machine learning based on artificial
            neural networks. It uses multiple layers to progressively extract
            higher-level features from raw input. Deep learning has achieved
            remarkable success in image recognition, natural language processing,
            and speech recognition.
            """,
            title="Deep Learning Overview",
            namespace="tech",
        )
        print(f"   Added: Deep Learning Overview (id={item_id_2})")

        item_id_3 = mem.add(
            """
            Natural language processing (NLP) is a field of AI that gives
            machines the ability to read, understand, and derive meaning
            from human languages. Modern NLP systems use transformer
            architectures and large pre-trained models.
            """,
            title="NLP Basics",
            namespace="tech",
        )
        print(f"   Added: NLP Basics (id={item_id_3})")

        print("\n2. Searching for 'neural networks'...")
        results = mem.search("neural networks", top_k=3)

        for i, r in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   Score: {r.score:.3f}")
            print(f"   Title: {r.item.title}")
            print(f"   Text:  {r.chunk.text[:100].strip()}...")

        print("\n3. Comparing search modes...")
        query = "artificial intelligence algorithms"

        for mode in ("hybrid", "lexical", "semantic"):
            print(f"\n   {mode} search for '{query}':")
            for r in mem.search(query, mode=mode, top_k=2):
                print(f"   - [{r.score:.3f}] {r.item.title}")

        print("\n4. Database statistics")
        stats = mem.stats()
        print(f"   Items:     {stats.items}")
        print(f"   Chunks:    {stats.chunks}")
        print(f"   Entities:  {stats.entities}")
        print(f"   Relations: {stats.relations}")
        print(f"   DB size:   {stats.file_size_bytes / 1024:.1f} KB")

        print("\n5. Verifying database integrity...")
        verify_result = mem.verify()
        if verify_result.errors:
            print("   Issues found:")
            for err in verify_result.errors:
                print(f"   - {err}")
        else:
            print("   Database integrity OK.")

    print("\n" + "=" * 50)
    print("Example complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
