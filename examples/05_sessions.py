#!/usr/bin/env python3
"""DuckMem sessions example.

Demonstrates:
- Starting and ending a session
- Letting `add`, `search`, `add_relation` auto-log session events
- Listing and replaying sessions

While a session is active, calls to `mem.add`, `mem.search`, `mem.ask`,
`mem.add_relation`, and `mem.extract` automatically emit a `SessionEvent`.
There is no public `session_log_event` method; `session_end()` always ends
the currently-active session.
"""

from datetime import datetime

from duckmem import DuckMem


def main() -> None:
    print("=" * 50)
    print("DuckMem Sessions Example")
    print("=" * 50)

    with DuckMem("sessions_example.duckdb") as mem:
        print("\n1. Setting up knowledge base...")

        mem.add("Transformers use self-attention for sequence modeling.", title="Transformers")
        mem.add("BERT is a bidirectional transformer model.", title="BERT")
        mem.add("GPT models are autoregressive language models.", title="GPT")
        print("   Added 3 documents")

        print("\n2. Starting research session...")
        session_id = mem.session_start(name="ML Research Session")
        print(f"   Session id: {session_id}")

        print("\n3. Performing searches and edits (auto-logged)...")

        results = mem.search("attention mechanisms")
        print(f"   search 'attention mechanisms' -> {len(results)} results")

        results = mem.search("transformer architecture")
        print(f"   search 'transformer architecture' -> {len(results)} results")

        note_id = mem.add(
            "Key insight: Transformers process sequences in parallel unlike RNNs.",
            title="Research Note",
            namespace="notes",
        )
        print(f"   add note id={note_id}")

        mem.add_relation("Transformers", "enable", "parallel processing")
        print("   add_relation Transformers -[enable]-> parallel processing")

        print("\n4. Ending session...")
        mem.session_end()
        print("   Session ended")

        print("\n5. Replaying session events...")
        print(f"\n   Session: ML Research Session ({session_id})")
        print("   " + "-" * 40)

        for event in mem.session_replay(session_id):
            ts = datetime.fromtimestamp(event.timestamp / 1000)
            print(f"   [{ts:%H:%M:%S}] kind={event.kind}")
            if event.params:
                print(f"              params={event.params}")
            if event.result_summary:
                print(f"              result={event.result_summary}")

        print("\n6. Creating another session...")
        session2_id = mem.session_start(name="Quick Review")
        mem.search("BERT")
        mem.session_end()
        print(f"   Created and ended 'Quick Review' (id={session2_id})")

        print("\n7. Listing all sessions...")
        for s in mem.session_list():
            started = datetime.fromtimestamp(s.started_at / 1000)
            status = "active" if s.ended_at is None else "completed"
            print(f"\n   {s.name or '(unnamed)'}")
            print(f"      id:       {s.id}")
            print(f"      started:  {started}")
            print(f"      status:   {status}")

        stats = mem.stats()
        print("\n8. Statistics:")
        print(f"   Sessions: {stats.sessions}")
        print(f"   Items:    {stats.items}")

    print("\n" + "=" * 50)
    print("Example complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
