#!/usr/bin/env python3
"""DuckMem document-management example.

Demonstrates:
- Adding items with metadata
- Organizing with namespaces
- Listing, filtering, and pagination
- Retrieving a specific item by id
"""

from duckmem import DuckMem


def main() -> None:
    print("=" * 50)
    print("DuckMem Document Management Example")
    print("=" * 50)

    with DuckMem("documents_example.duckdb") as mem:
        print("\n1. Adding documents to namespaces...")

        mem.add(
            """
            Transformers revolutionized NLP by introducing self-attention
            mechanisms. The architecture was introduced in "Attention Is All
            You Need" in 2017.
            """,
            title="Transformer Architecture",
            namespace="research/ml",
            metadata={"year": 2017, "topic": "architecture"},
        )

        mem.add(
            """
            BERT (Bidirectional Encoder Representations from Transformers)
            is a language model developed by Google. It uses masked language
            modeling for pre-training.
            """,
            title="BERT Overview",
            namespace="research/ml",
            metadata={"year": 2018, "organization": "Google"},
        )

        mem.add(
            """
            Project Alpha uses machine learning for customer segmentation.
            The model achieved 95% accuracy on test data.
            """,
            title="Project Alpha Results",
            namespace="work/projects",
            metadata={"project": "alpha", "status": "complete"},
        )

        mem.add(
            """
            Meeting notes: Discussed Q1 roadmap. Key priorities include
            improving model latency and expanding to new markets.
            """,
            title="Q1 Planning Meeting",
            namespace="work/meetings",
            metadata={"date": "2024-01-15", "attendees": ["Alice", "Bob"]},
        )

        mem.add(
            """
            TIL: You can use list comprehensions with multiple for clauses
            in Python. This is really useful for flattening nested lists.
            """,
            title="Python Tips",
            namespace="personal/notes",
            metadata={"tags": ["python", "til"]},
        )

        print("   Added 5 documents across 4 namespaces")

        print("\n2. Listing all items...")
        all_items = mem.list()
        for item in all_items:
            print(f"   - [{item.namespace}] {item.title}")

        print("\n3. Items in 'research/ml' namespace:")
        for item in mem.list(namespace="research/ml"):
            print(f"   - {item.title}")
            print(f"     Metadata: {item.metadata}")

        print("\n4. Searching 'model' in different namespaces...")

        print("\n   All namespaces:")
        for r in mem.search("model", top_k=3):
            print(f"   - [{r.item.namespace}] {r.item.title}")

        print("\n   Only 'work/projects' namespace:")
        for r in mem.search("model", namespace="work/projects", top_k=3):
            print(f"   - [{r.item.namespace}] {r.item.title}")

        print("\n5. Retrieving a specific item by id...")
        if all_items:
            item_id = all_items[0].id
            item = mem.get(item_id)
            if item is not None:
                print(f"   ID:         {item.id}")
                print(f"   Title:      {item.title}")
                print(f"   Namespace:  {item.namespace}")
                print(f"   URI:        {item.uri}")
                print(f"   Preview:    {item.text[:100].strip()}...")

        print("\n6. Pagination...")
        page1 = mem.list(limit=2, offset=0)
        page2 = mem.list(limit=2, offset=2)

        print(f"   Page 1 ({len(page1)} items):")
        for item in page1:
            print(f"   - {item.title}")

        print(f"\n   Page 2 ({len(page2)} items):")
        for item in page2:
            print(f"   - {item.title}")

        stats = mem.stats()
        print("\n7. Final statistics:")
        print(f"   Total items:  {stats.items}")
        print(f"   Total chunks: {stats.chunks}")

    print("\n" + "=" * 50)
    print("Example complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
