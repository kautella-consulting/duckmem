#!/usr/bin/env python3
"""DuckMem RAG Q&A example.

Demonstrates:
- Ingesting a small knowledge base
- Asking questions with `mem.ask(...)`
- Inspecting answer, confidence, and source chunk ids

Requires a configured LLM (local via Ollama or cloud via an API key).
"""

import asyncio

from duckmem import DuckMem


async def main() -> None:
    print("=" * 50)
    print("DuckMem RAG Q&A Example")
    print("=" * 50)

    with DuckMem("rag_example.duckdb") as mem:
        print("\n1. Building knowledge base...")

        documents = [
            (
                """
                Transformers are a neural network architecture introduced in 2017.
                The key innovation is the self-attention mechanism, which allows
                the model to weigh the importance of different parts of the input
                when producing each part of the output. Unlike RNNs, transformers
                can process all positions in parallel, making them much faster to
                train on modern hardware.
                """,
                "Transformer Architecture",
            ),
            (
                """
                BERT (Bidirectional Encoder Representations from Transformers) is
                a language model developed by Google in 2018. It uses masked
                language modeling, where random words are masked and the model
                learns to predict them from context. BERT processes text
                bidirectionally, considering both left and right context
                simultaneously.
                """,
                "BERT Model",
            ),
            (
                """
                GPT (Generative Pre-trained Transformer) is a family of language
                models by OpenAI. GPT models are trained using next-token
                prediction, learning to predict the next word given all previous
                words. GPT-3 has 175 billion parameters, and GPT-4 is multimodal,
                capable of processing both text and images.
                """,
                "GPT Models",
            ),
            (
                """
                The attention mechanism allows neural networks to focus on
                relevant parts of the input when producing output. Self-attention
                computes attention weights between all positions in a sequence.
                Multi-head attention runs multiple attention operations in
                parallel with different learned projections, capturing different
                types of relationships in the data.
                """,
                "Attention Mechanisms",
            ),
            (
                """
                Transfer learning in NLP involves pre-training a model on a large
                corpus of text, then fine-tuning it on a specific task. This
                approach was popularized by models like BERT and GPT. Pre-training
                allows models to learn general language understanding, which
                transfers well to downstream tasks like classification and
                question answering.
                """,
                "Transfer Learning in NLP",
            ),
        ]

        for content, title in documents:
            mem.add(content.strip(), title=title, namespace="ml_knowledge")
            print(f"   Added: {title}")

        print(f"\n   Total: {len(documents)} documents added")

        print("\n2. Asking questions...")

        questions = [
            "What is the key innovation in transformers?",
            "How does BERT process text differently from GPT?",
            "What is the purpose of multi-head attention?",
        ]

        for q in questions:
            print(f"\n   Q: {q}")
            print("   " + "-" * 40)

            try:
                answer = await mem.ask(q, top_k=3, namespace="ml_knowledge")

                print(f"   A: {answer.answer}")
                print(f"\n   Confidence:    {answer.confidence:.2f}")
                print(f"   Source chunks: {len(answer.sources)}")
                for chunk_id in answer.sources:
                    print(f"     - {chunk_id}")
            except Exception as e:
                print(f"   Error: {e}")
                print("   (Make sure Ollama is running or an API key is configured.)")

        print("\n3. Resolving sources back to items...")

        try:
            answer = await mem.ask(
                "Explain the difference between BERT and GPT training approaches",
                top_k=5,
            )
            print(f"\n   Answer: {answer.answer}")
            print("\n   Source chunks (by id):")
            for chunk_id in answer.sources:
                print(f"     - {chunk_id}")

            print("\n   (To get chunk text + parent item, run a targeted search:)")
            for r in mem.search("BERT GPT training", top_k=3):
                print(f"     [{r.score:.3f}] {r.item.title}: {r.chunk.text[:80]}...")
        except Exception as e:
            print(f"   Error: {e}")

        stats = mem.stats()
        print("\n4. Statistics:")
        print(f"   Items:  {stats.items}")
        print(f"   Chunks: {stats.chunks}")

    print("\n" + "=" * 50)
    print("Example complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
