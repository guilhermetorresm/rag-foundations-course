"""
3_retrieval.py — Busca Vetorial e Re-ranking
=============================================
Bloco 3 do Pipeline RAG: dado uma query do usuário, converte-a
em vetor e busca os chunks mais semanticamente similares no ChromaDB.

Conceito-chave ensinado aqui:
    A busca vetorial encontra documentos por SIGNIFICADO, não por
    palavras-chave.

Fluxo:
    Query (texto) → Embedding (Ollama) → ChromaDB Query → Top-K Chunks
"""

from dataclasses import dataclass
from pathlib import Path
import chromadb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import RetrievalConfig, retrieval_cfg, chroma_cfg
from embeddings import get_single_embedding

console = Console()

# ─────────────────────────────────────────────
# Tipos de Dados
# ─────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    Representa um chunk recuperado da busca vetorial.
    """
    id: str
    text: str
    source: str
    page: int
    distance: float
    score: float

# ─────────────────────────────────────────────
# Busca Vetorial Principal
# ─────────────────────────────────────────────
from sentence_transformers import CrossEncoder

# ... (RetrievedChunk remains same)

def search(
    query: str,
    collection: chromadb.Collection,
    where: dict | None = None,  # Adicionado filtro de metadata
    cfg: RetrievalConfig = retrieval_cfg,
) -> list[RetrievedChunk]:
    """
    Busca os chunks mais relevantes com suporte a filtros e re-ranking.
    """
    # 1. Vetorizar a query via Ollama
    query_embedding = get_single_embedding(query)

    # 2. Buscar no ChromaDB (pede mais se for re-rankear para ter margem)
    k_initial = cfg.top_k * 5 if cfg.rerank else cfg.top_k
    n_results = min(k_initial, collection.count())

    if n_results == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # 3. Parsear resultados
    chunks: list[RetrievedChunk] = []
    # ... (parsing logic)

    for chunk_id, doc, meta, dist in zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]):
        chunks.append(
            RetrievedChunk(
                id=chunk_id,
                text=doc,
                source=meta.get("source", "desconhecido"),
                page=int(meta.get("page", 0)),
                distance=dist,
                score=1.0 - dist,
            )
        )

    # 4. Re-ranking (Opcional)
    if cfg.rerank and chunks:
        console.print("[dim]🔄 Aplicando Re-ranking (Cross-Encoder)...[/dim]")
        encoder = CrossEncoder(cfg.rerank_model)
        pairs = [(query, c.text) for c in chunks]
        scores = encoder.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        chunks = sorted(chunks, key=lambda x: x.score, reverse=True)

    return chunks[:cfg.top_k]

# ─────────────────────────────────────────────
# Formatação de Resultados
# ─────────────────────────────────────────────

def format_context(chunks: list[RetrievedChunk], max_chars: int = 3000) -> str:
    """
    Formata os chunks recuperados em um bloco de contexto para o LLM.
    """
    context_parts: list[str] = []
    total_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        source_name = Path(chunk.source).name
        header = f"[Trecho {i} | Fonte: {source_name} | Pág. {chunk.page}]"
        block = f"{header}\n{chunk.text}"

        if total_chars + len(block) > max_chars:
            break

        context_parts.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(context_parts)

# ─────────────────────────────────────────────
# Execução Standalone
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from importlib import import_module
    indexing = import_module("2_indexing")

    client = indexing.get_chroma_client()
    collection = indexing.get_or_create_collection(client)

    query = "Quais são os direitos dos funcionários?"
    console.print(Panel.fit(
        f"[bold cyan]BLOCO 3: Retrieval[/bold cyan]\n"
        f"Query: [italic]\"{query}\"[/italic]",
        border_style="cyan",
    ))

    chunks = search(query, collection)

    if not chunks:
        console.print("[yellow]Nenhum resultado encontrado.[/yellow]")
    else:
        table = Table(
            title=f"Top {len(chunks)} Resultados",
            show_lines=True,
        )
        table.add_column("#", justify="center", style="bold")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Fonte")
        table.add_column("Pág.", justify="center")
        table.add_column("Prévia", style="dim")

        for i, chunk in enumerate(chunks, start=1):
            table.add_row(
                str(i),
                f"{chunk.score:.3f}",
                Path(chunk.source).name,
                str(chunk.page),
                chunk.text[:80].replace("\n", " ") + "...",
            )

        console.print(table)
        console.print("\n[bold]Contexto formatado para o LLM:[/bold]")
        console.print(Panel(format_context(chunks), border_style="dim"))
