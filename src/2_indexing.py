"""
2_indexing.py — Embeddings e Indexação no ChromaDB
====================================================
Bloco 2 do Pipeline RAG: converte chunks de texto em vetores
numéricos (embeddings) e os armazena no ChromaDB para busca
posterior por similaridade.

Conceito-chave ensinado aqui:
    Embeddings são representações matemáticas de significado.
    Textos semanticamente similares ficam "próximos" no espaço
    vetorial, permitindo busca por relevância, não por palavras-chave.

Fluxo:
    Chunks → Ollama (nomic-embed-text) → Vetores → ChromaDB
"""

from pathlib import Path
import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import ChromaConfig, chroma_cfg, DATA_DIR
from embeddings import get_embeddings

# Alias para facilitar import correto
try:
    from ingestion import Chunk
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from importlib import import_module
    _m = import_module("1_ingestion")
    Chunk = _m.Chunk

console = Console()


# ─────────────────────────────────────────────
# Cliente ChromaDB
# ─────────────────────────────────────────────

def get_chroma_client(cfg: ChromaConfig = chroma_cfg) -> chromadb.PersistentClient:
    """
    Cria e retorna um cliente ChromaDB com persistência em disco.
    """
    Path(cfg.persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=cfg.persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client


def get_or_create_collection(
    client: chromadb.PersistentClient,
    cfg: ChromaConfig = chroma_cfg,
) -> chromadb.Collection:
    """
    Obtém ou cria uma coleção no ChromaDB.
    """
    collection = client.get_or_create_collection(
        name=cfg.collection_name,
        metadata={"hnsw:space": cfg.distance_metric},
    )
    return collection


# ─────────────────────────────────────────────
# Geração de Embeddings e Indexação
# ─────────────────────────────────────────────

def index_chunks(
    chunks: list[Chunk],
    collection: chromadb.Collection,
    batch_size: int = 50,
) -> int:
    """
    Indexa uma lista de chunks no ChromaDB usando Ollama para embeddings.
    """
    if not chunks:
        console.print("[yellow]⚠ Nenhum chunk para indexar.[/yellow]")
        return 0

    console.print(f"[cyan]⚙ Gerando embeddings e inserindo no ChromaDB ({len(chunks)} chunks)...[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexando...", total=len(chunks))

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_texts = [c.text for c in batch]
            
            # Gera embeddings via Ollama
            batch_embeddings = get_embeddings(batch_texts)

            collection.upsert(
                ids=[c.id for c in batch],
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=[
                    {
                        "source": c.source,
                        "page": c.page,
                        "chunk_idx": c.chunk_idx,
                        "doc_hash": c.doc_hash,
                        "status": "vigente",  # Padrão para Slide 16
                        "version": "v1.0"      # Controle de versão (Slide 8)
                    }
                    for c in batch
                ],
            )
            progress.advance(task, len(batch))

    total = collection.count()
    console.print(f"[green]✓ Indexação concluída! Total na coleção: {total}[/green]")
    return total


# ─────────────────────────────────────────────
# Execução Standalone
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from importlib import import_module
    ingestion = import_module("1_ingestion")
    ingest_directory = ingestion.ingest_directory
    from rich.panel import Panel

    console.print(Panel.fit(
        "[bold cyan]BLOCO 2: Embeddings e Indexação[/bold cyan]\n"
        "Stack: Ollama (nomic-embed-text) + ChromaDB",
        border_style="cyan",
    ))

    chunks = ingest_directory(DATA_DIR)
    if not chunks:
        raise SystemExit(1)

    client = get_chroma_client()
    collection = get_or_create_collection(client)
    
    index_chunks(chunks, collection)
