"""
3_retrieval.py — Busca Vetorial e Re-ranking
=============================================
Bloco 3 do Pipeline RAG: dado uma query do usuário, converte-a
em vetor e busca os chunks mais semanticamente similares no ChromaDB.

Conceito-chave ensinado aqui:
    A busca vetorial encontra documentos por SIGNIFICADO, não por
    palavras-chave. Uma query como "férias" pode encontrar chunks
    que falam sobre "descanso remunerado" ou "licença".

Fluxo:
    Query (texto) → Embedding → ChromaDB Query → Top-K Chunks → Re-rank
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import RetrievalConfig, retrieval_cfg, chroma_cfg

console = Console()


# ─────────────────────────────────────────────
# Tipos de Dados
# ─────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    Representa um chunk recuperado da busca vetorial.

    Attributes:
        id:         Identificador único do chunk no ChromaDB.
        text:       Conteúdo textual do chunk.
        source:     Arquivo de origem do chunk.
        page:       Número da página de origem.
        distance:   Distância vetorial (menor = mais similar para L2).
        score:      Score de similaridade normalizado [0, 1].
        rerank_score: Score após re-ranking com CrossEncoder (opcional).
    """
    id: str
    text: str
    source: str
    page: int
    distance: float
    score: float
    rerank_score: float | None = None


# ─────────────────────────────────────────────
# Busca Vetorial Principal
# ─────────────────────────────────────────────

def search(
    query: str,
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    cfg: RetrievalConfig = retrieval_cfg,
) -> list[RetrievedChunk]:
    """
    Busca os chunks mais relevantes para uma query no ChromaDB.

    Processo:
        1. Converte a query em vetor usando o modelo de embeddings
        2. Executa busca ANN (Approximate Nearest Neighbors) no ChromaDB
        3. Filtra por score mínimo de similaridade
        4. Retorna os Top-K resultados

    Args:
        query:       Pergunta ou texto de busca do usuário.
        collection:  Coleção ChromaDB para buscar.
        embed_model: Modelo de embeddings (deve ser o mesmo usado na indexação!).
        cfg:         Configurações de retrieval (top_k, threshold).

    Returns:
        Lista de RetrievedChunk ordenados por relevância (mais relevante primeiro).

    Note:
        A métrica "cosine" retorna distâncias entre 0 (idêntico) e 2 (oposto).
        Convertemos para score: score = 1 - (distance / 2), resultando em [0, 1].
    """
    # 1. Vetorizar a query
    query_embedding = embed_model.encode(query).tolist()

    # 2. Buscar no ChromaDB (pede mais que top_k para ter margem no filtro)
    n_results = min(cfg.top_k * 2, collection.count())
    if n_results == 0:
        console.print("[yellow]⚠ Coleção vazia. Execute a indexação primeiro.[/yellow]")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # 3. Parsear e filtrar resultados
    chunks: list[RetrievedChunk] = []

    ids       = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        # Converte distância cosine para score [0, 1]
        score = 1.0 - (dist / 2.0)

        if score >= cfg.similarity_threshold:
            chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=doc,
                    source=meta.get("source", "desconhecido"),
                    page=int(meta.get("page", 0)),
                    distance=dist,
                    score=score,
                )
            )

    # Limita ao top_k após filtro
    return chunks[: cfg.top_k]


# ─────────────────────────────────────────────
# Re-ranking com Cross-Encoder
# ─────────────────────────────────────────────

def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[RetrievedChunk]:
    """
    Re-ordena os chunks usando um Cross-Encoder para maior precisão.

    Por que re-rankar?
        O Bi-Encoder (embeddings) é rápido mas impreciso: compara vetores
        independentemente. O Cross-Encoder é mais lento mas avalia a
        relevância considerando QUERY e DOCUMENTO juntos.

    Estratégia Two-Stage:
        1. Bi-Encoder: busca rápida nos milhões de docs (recall)
        2. Cross-Encoder: reordena os Top-K com maior precisão (precision)

    Args:
        query:      Query original do usuário.
        chunks:     Chunks recuperados pelo Bi-Encoder.
        model_name: Nome do Cross-Encoder no HuggingFace Hub.

    Returns:
        Chunks reordenados por score do Cross-Encoder (maior = mais relevante).
    """
    if not chunks:
        return []

    console.print(f"[dim]🔄 Re-ranking com Cross-Encoder...[/dim]")

    cross_encoder = CrossEncoder(model_name)

    # Cria pares (query, documento) para o cross-encoder avaliar
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = cross_encoder.predict(pairs)

    # Atribui scores e reordena
    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = float(score)

    return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)


# ─────────────────────────────────────────────
# Formatação de Resultados
# ─────────────────────────────────────────────

def format_context(chunks: list[RetrievedChunk], max_chars: int = 3000) -> str:
    """
    Formata os chunks recuperados em um bloco de contexto para o LLM.

    O contexto é formatado de forma estruturada para facilitar
    a compreensão do LLM e incluir metadados de fonte.

    Args:
        chunks:    Chunks a incluir no contexto.
        max_chars: Limite de caracteres total do contexto.

    Returns:
        String formatada com o contexto para o prompt RAG.
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
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from importlib import import_module
    indexing = import_module("2_indexing")

    client = indexing.get_chroma_client()
    collection = indexing.get_or_create_collection(client)
    embed_model = indexing.load_embedding_model()

    query = "Quais são os direitos dos funcionários?"
    console.print(Panel.fit(
        f"[bold cyan]BLOCO 3: Retrieval[/bold cyan]\n"
        f"Query: [italic]\"{query}\"[/italic]",
        border_style="cyan",
    ))

    chunks = search(query, collection, embed_model)

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
