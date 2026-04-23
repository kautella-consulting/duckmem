"""PydanticAI agents for RAG and relation extraction.

Provides structured LLM interactions using PydanticAI agents with
typed outputs. Used for answering questions and extracting knowledge
graph relations from text. Uses pydantic-ai-litellm for unified
LiteLLM model strings (e.g. ollama/gpt-oss:20b, openai/gpt-4o-mini).
"""

import contextlib
import json
import re

from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel

from duckmem.models import Answer, ExtractedRelation


def _model_from_string(model: str) -> LiteLLMModel:
    """Create LiteLLMModel from LiteLLM model string (e.g. ollama/gpt-oss:20b)."""
    return LiteLLMModel(model_name=model)


RAG_SYSTEM_PROMPT = """You answer questions based only on the provided context.

Reply with a JSON object having exactly these keys:
- "answer": (string) The answer to the question
- "confidence": (number 0.0-1.0) How well the context supports your answer
- "source_indices": (array of integers) 0-indexed indices of context chunks used, e.g. [0, 1]

Instructions:
1. Answer using ONLY information from the context
2. If context is insufficient, say so in the answer
3. Be concise but complete
4. Output ONLY valid JSON, no markdown or extra text

Do not make up information or use external knowledge."""


EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting factual relations from text.

Reply with a JSON array of objects. Each object has: "subject", "predicate", "object".
Example: [{"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"}]

Focus on:
- People and roles (e.g., "Alice works_at Acme Corp")
- Organizations and properties (e.g., "Acme Corp located_in New York")
- Concepts (e.g., "DuckDB integrates_with Python")

Guidelines:
- Use lowercase_with_underscores for predicates
- Keep entity names as in the text
- Only extract explicitly stated facts
- Output ONLY valid JSON array, no markdown or extra text
- Return [] if no clear relations"""


def create_rag_agent(model: str = "ollama/gpt-oss:20b") -> Agent[None, str]:
    r"""Create a RAG agent for answering questions.

    Creates a PydanticAI agent configured for retrieval-augmented generation.
    The agent takes context and a question, returning a JSON string that is
    parsed in run_rag_query.

    Args:
        model: LiteLLM model string (e.g. ollama/gpt-oss:20b, openai/gpt-4o-mini).

    Returns:
        Configured Agent instance.

    Example:
        >>> agent = create_rag_agent("ollama/gpt-oss:20b")
        >>> result = await agent.run("Context: X\n\nQuestion: Y")
    """
    return Agent(
        _model_from_string(model),
        output_type=str,
        system_prompt=RAG_SYSTEM_PROMPT,
        retries=3,
    )


def create_extraction_agent(model: str = "ollama/gpt-oss:20b") -> Agent[None, str]:
    """Create an agent for extracting relations from text.

    Creates a PydanticAI agent that extracts subject-predicate-object
    triples from text. Returns JSON string parsed in extract_relations.

    Args:
        model: LiteLLM model string (e.g. ollama/gpt-oss:20b, openai/gpt-4o-mini).

    Returns:
        Configured Agent instance.
    """
    return Agent(
        _model_from_string(model),
        output_type=str,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        retries=3,
    )


async def run_rag_query(
    question: str,
    context_chunks: list[str],
    *,
    model: str = "ollama/gpt-oss:20b",
    chunk_ids: list[str] | None = None,
) -> Answer:
    """Run a RAG query and return a structured answer.

    Combines context chunks into a prompt, sends to the LLM, and
    returns a structured Answer with sources.

    Args:
        question: The question to answer.
        context_chunks: List of relevant text chunks for context.
        model: LiteLLM model string (e.g. ollama/gpt-oss:20b, openai/gpt-4o-mini).
        chunk_ids: Optional list of chunk IDs corresponding to context_chunks.

    Returns:
        Answer object with answer text, confidence, and source IDs.

    Example:
        >>> answer = await run_rag_query(
        ...     "What do transformers use?",
        ...     ["Transformers use attention mechanisms.", "Attention is key."],
        ...     chunk_ids=["chunk1", "chunk2"],
        ... )
        >>> answer.answer
        'Transformers use attention mechanisms.'
    """
    if not context_chunks:
        return Answer(
            answer="No context available to answer the question.",
            confidence=0.0,
            sources=[],
            context=None,
        )

    # Format context with indices
    formatted_context = "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks))

    prompt = f"""Context:
{formatted_context}

Question: {question}"""

    agent = create_rag_agent(model)
    result = await agent.run(prompt)
    raw = result.output.strip()

    # Parse JSON from raw output (may be wrapped in markdown code blocks)
    answer_text = ""
    confidence_val = 0.5
    source_indices: list[int] = []

    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        # Try to find JSON object in the response
        brace = raw.find("{")
        if brace >= 0:
            depth = 0
            end = brace
            for i, c in enumerate(raw[brace:], brace):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            raw = raw[brace : end + 1]

    try:
        parsed = json.loads(raw)
        answer_text = str(parsed.get("answer", "")).strip() or "No answer provided."
        conf = parsed.get("confidence", 0.5)
        if isinstance(conf, (int, float)):
            confidence_val = float(max(0.0, min(1.0, conf)))
        elif isinstance(conf, str):
            with contextlib.suppress(ValueError):
                confidence_val = float(max(0.0, min(1.0, float(conf))))
        idx = parsed.get("source_indices", [])
        if isinstance(idx, list):
            for x in idx:
                if isinstance(x, int):
                    source_indices.append(x)
                elif isinstance(x, str):
                    with contextlib.suppress(ValueError):
                        source_indices.append(int(x))
                elif isinstance(x, float):
                    source_indices.append(int(x))
    except Exception:
        answer_text = raw if raw else "No answer provided."

    sources = []
    if chunk_ids and source_indices:
        sources = [chunk_ids[i] for i in source_indices if 0 <= i < len(chunk_ids)]

    return Answer(
        answer=answer_text,
        confidence=confidence_val,
        sources=sources,
        context=formatted_context,
    )


async def extract_relations(
    text: str,
    *,
    model: str = "ollama/gpt-oss:20b",
) -> list[ExtractedRelation]:
    """Extract relations from text using LLM.

    Sends text to the extraction agent and returns structured
    subject-predicate-object relations.

    Args:
        text: Text to extract relations from.
        model: LiteLLM model string (e.g. ollama/gpt-oss:20b, openai/gpt-4o-mini).

    Returns:
        List of ExtractedRelation objects.

    Example:
        >>> relations = await extract_relations(
        ...     "Alice is a software engineer at Acme Corp."
        ... )
        >>> relations[0].subject
        'Alice'
        >>> relations[0].predicate
        'works_at'
    """
    if not text.strip():
        return []

    agent = create_extraction_agent(model)
    result = await agent.run(text)
    raw = result.output.strip()

    # Parse JSON array from raw output
    relations: list[ExtractedRelation] = []
    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        bracket = raw.find("[")
        if bracket >= 0:
            depth = 0
            end = bracket
            for i, c in enumerate(raw[bracket:], bracket):
                if c in "[{":
                    depth += 1
                elif c in "]}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            raw = raw[bracket : end + 1]

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    subj = str(item.get("subject", "")).strip()
                    pred = str(item.get("predicate", "")).strip()
                    obj = str(item.get("object", "")).strip()
                    if subj and pred and obj:
                        relations.append(
                            ExtractedRelation(subject=subj, predicate=pred, object=obj)
                        )
    except Exception:
        pass

    return relations
