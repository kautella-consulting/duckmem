#!/usr/bin/env python3
"""DuckMem knowledge-graph example.

Demonstrates:
- Adding Subject-Predicate-Object relations
- Reading entity state (latest-wins properties)
- Relation history for an entity
- Graph traversal, with and without predicate filtering
"""

from duckmem import DuckMem


def main() -> None:
    print("=" * 50)
    print("DuckMem Knowledge Graph Example")
    print("=" * 50)

    with DuckMem("knowledge_graph_example.duckdb") as mem:
        print("\n1. Building knowledge graph...")

        relations = [
            ("Python", "is_a", "Programming Language"),
            ("JavaScript", "is_a", "Programming Language"),
            ("Rust", "is_a", "Programming Language"),
            ("TypeScript", "is_a", "Programming Language"),
            ("TypeScript", "superset_of", "JavaScript"),
            ("Django", "written_in", "Python"),
            ("FastAPI", "written_in", "Python"),
            ("Flask", "written_in", "Python"),
            ("React", "written_in", "JavaScript"),
            ("Next.js", "written_in", "JavaScript"),
            ("Python", "created_by", "Guido van Rossum"),
            ("Rust", "developed_by", "Mozilla"),
            ("TypeScript", "developed_by", "Microsoft"),
            ("React", "developed_by", "Meta"),
            ("Python", "used_for", "Machine Learning"),
            ("Python", "used_for", "Web Development"),
            ("Python", "used_for", "Data Science"),
            ("JavaScript", "used_for", "Web Development"),
            ("Rust", "used_for", "Systems Programming"),
            ("FastAPI", "uses", "Pydantic"),
            ("FastAPI", "uses", "Starlette"),
            ("Django", "includes", "ORM"),
        ]

        for subj, pred, obj in relations:
            mem.add_relation(subj, pred, obj)

        print(f"   Added {len(relations)} relations")

        print("\n2. Entity state (latest-wins properties):")
        for entity_name in ("Python", "FastAPI", "JavaScript"):
            state = mem.state(entity_name)
            print(f"\n   {state.entity}:")
            for predicate, value in state.properties.items():
                print(f"     {predicate} = {value}")

        print("\n3. Relation history for 'Python':")
        for rel in mem.history("Python"):
            print(f"   {rel.subject} --[{rel.predicate}]--> {rel.object}")

        print("\n4. Graph traversal from 'Python' (max_hops=2)...")
        result = mem.traverse("Python", max_hops=2)

        print(f"\n   Reachable entities ({len(result.entities)}):")
        for name in result.entities:
            print(f"   - {name}")

        print(f"\n   Paths ({len(result.paths)}):")
        for path in result.paths[:10]:
            hops = " -> ".join(f"[{r.predicate}] {r.object}" for r in path)
            print(f"   Python -> {hops}")

        print("\n5. Filtered traversal: follow only 'used_for' from 'Python':")
        result = mem.traverse("Python", link="used_for", max_hops=1)
        for path in result.paths:
            for rel in path:
                print(f"   - {rel.object}")

        print("\n6. Deeper traversal from 'Machine Learning' (max_hops=3)...")
        result = mem.traverse("Machine Learning", max_hops=3)
        print("   Reachable entities:")
        for name in result.entities:
            print(f"   - {name}")

        stats = mem.stats()
        print("\n7. Knowledge-graph statistics:")
        print(f"   Entities:  {stats.entities}")
        print(f"   Relations: {stats.relations}")

    print("\n" + "=" * 50)
    print("Example complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
