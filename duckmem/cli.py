"""Command-line interface for DuckMem.

Provides CLI commands for all DuckMem operations.
Run with: duckmem <command> [args]
"""

import json
from pathlib import Path
from typing import Annotated

import typer

from duckmem.config import Settings
from duckmem.core import DuckMem, lock, unlock

app = typer.Typer(
    name="duckmem",
    help="DuckMem - Personal Knowledge Memory System",
    add_completion=False,
)


def get_mem(db_path: str) -> DuckMem:
    """Get a DuckMem instance for the given database."""
    settings = Settings(db_path=db_path)
    return DuckMem(settings=settings)


@app.command()
def create(
    db_path: Annotated[str, typer.Argument(help="Path to database file")],
):
    """Create a new DuckMem database."""
    if Path(db_path).exists():
        typer.echo(f"Database already exists: {db_path}", err=True)
        raise typer.Exit(1)

    with get_mem(db_path) as mem:
        typer.echo(f"Created database: {db_path}")


@app.command()
def stats(
    db_path: Annotated[str, typer.Argument(help="Path to database file")],
    json_output: Annotated[bool, typer.Option("--json", help="JSON output")] = False,
):
    """Show database statistics."""
    with get_mem(db_path) as mem:
        result = mem.stats()
        if json_output:
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(f"Items:      {result.items}")
            typer.echo(f"Chunks:     {result.chunks}")
            typer.echo(f"Relations:  {result.relations}")
            typer.echo(f"Entities:   {result.entities}")
            typer.echo(f"Sessions:   {result.sessions}")
            typer.echo(f"File size:  {result.file_size_bytes:,} bytes")


@app.command()
def add(
    db_path: Annotated[str, typer.Argument(help="Path to database file")],
    text: Annotated[str, typer.Option("--text", "-t", help="Text to add")],
    title: Annotated[str | None, typer.Option("--title", help="Item title")] = None,
    namespace: Annotated[str, typer.Option("--namespace", "-n", help="Namespace")] = "default",
):
    """Add an item to the knowledge base."""
    with get_mem(db_path) as mem:
        item_id = mem.add(text, title=title, namespace=namespace)
        typer.echo(f"Added item: {item_id}")


@app.command()
def search(
    db_path: Annotated[str, typer.Argument(help="Path to database file")],
    query: Annotated[str, typer.Option("--query", "-q", help="Search query")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="Search mode")] = "hybrid",
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Max results")] = 10,
    json_output: Annotated[bool, typer.Option("--json", help="JSON output")] = False,
):
    """Search the knowledge base."""
    with get_mem(db_path) as mem:
        results = mem.search(query, mode=mode, top_k=top_k)  # type: ignore

        if json_output:
            output = [r.model_dump() for r in results]
            typer.echo(json.dumps(output, indent=2, default=str))
        else:
            for i, r in enumerate(results, 1):
                typer.echo(f"\n[{i}] Score: {r.score:.4f}")
                if r.item.title:
                    typer.echo(f"    Title: {r.item.title}")
                typer.echo(f"    {r.chunk.text[:200]}...")


@app.command()
def verify(
    db_path: Annotated[str, typer.Argument(help="Path to database file")],
    deep: Annotated[bool, typer.Option("--deep", help="Verify checksums")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="JSON output")] = False,
):
    """Verify database integrity."""
    with get_mem(db_path) as mem:
        result = mem.verify(deep=deep)

        if json_output:
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(f"Items:     {result.items}")
            typer.echo(f"Chunks:    {result.chunks}")
            typer.echo(f"Relations: {result.relations}")
            typer.echo(f"Entities:  {result.entities}")
            if result.checksum_ok is not None:
                typer.echo(f"Checksums: {'OK' if result.checksum_ok else 'FAILED'}")
            if result.errors:
                for err in result.errors:
                    typer.echo(f"Error: {err}", err=True)


@app.command()
def doctor(
    db_path: Annotated[str, typer.Argument(help="Path to database file")],
    vacuum: Annotated[bool, typer.Option("--vacuum", help="Compact storage")] = False,
    rebuild_fts: Annotated[bool, typer.Option("--rebuild-fts", help="Rebuild FTS index")] = False,
    rebuild_vec: Annotated[
        bool, typer.Option("--rebuild-vec", help="Rebuild vector index")
    ] = False,
):
    """Run maintenance operations."""
    with get_mem(db_path) as mem:
        results = mem.doctor(
            vacuum=vacuum,
            rebuild_fts=rebuild_fts,
            rebuild_vec=rebuild_vec,
        )
        for op, success in results.items():
            status = "OK" if success else "FAILED"
            typer.echo(f"{op}: {status}")


@app.command()
def lock_db(
    src_path: Annotated[str, typer.Argument(help="Source database path")],
    dst_path: Annotated[str, typer.Option("--out", "-o", help="Output encrypted path")],
):
    """Encrypt a database file."""
    password = typer.prompt("Password", hide_input=True)
    password_confirm = typer.prompt("Confirm password", hide_input=True)

    if password != password_confirm:
        typer.echo("Passwords do not match", err=True)
        raise typer.Exit(1)

    lock(src_path, dst_path, password)
    typer.echo(f"Encrypted: {dst_path}")


@app.command()
def unlock_db(
    src_path: Annotated[str, typer.Argument(help="Encrypted file path")],
    dst_path: Annotated[str, typer.Option("--out", "-o", help="Output database path")],
):
    """Decrypt an encrypted database file."""
    password = typer.prompt("Password", hide_input=True)

    try:
        unlock(src_path, dst_path, password)
        typer.echo(f"Decrypted: {dst_path}")
    except Exception as e:
        typer.echo(f"Decryption failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def serve(
    db_path: Annotated[str, typer.Argument(help="Path to database file")],
    host: Annotated[str, typer.Option("--host", "-h", help="Server host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 8000,
):
    """Start the FastAPI server."""
    import os

    import uvicorn

    os.environ["DUCKMEM_DB_PATH"] = db_path
    uvicorn.run("duckmem.api:app", host=host, port=port, reload=True)


def main():
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
