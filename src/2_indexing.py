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
    Chunks → SentenceTransformer → Vetores → ChromaDB
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import ChromaConfig, chroma_cfg
from ingestion import Chunk  # type: ignore[import-untyped]

# Alias para facilitar import correto no contexto do curso
try:
    from ingestion import Chunk
except ModuleNotFoundError:
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

    O ChromaDB persiste os dados automaticamente no diretório
    especificado, sobrevivendo a reinicializações do processo.

    Args:
        cfg: Configurações do ChromaDB.

    Returns:
        Cliente ChromaDB configurado.
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

    Uma coleção é análoga a uma tabela: contém documentos
    organizados com seus vetores e metadados.

    Args:
        client: Cliente ChromaDB conectado.
        cfg:    Configurações da coleção (nome e métrica).

    Returns:
        Coleção ChromaDB pronta para uso.
    """
    collection = client.get_or_create_collection(
        name=cfg.collection_name,
        metadata={"hnsw:space": cfg.distance_metric},
    )
    return collection


# ─────────────────────────────────────────────
# Modelo de Embeddings
# ─────────────────────────────────────────────

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Carrega o modelo de embeddings via sentence-transformers.

    Modelo padrão: all-MiniLM-L6-v2
        - Tamanho: ~80MB
        - Dimensões: 384
        - Velocidade: muito rápida para uso local
        - Qualidade: excelente para português e inglês

    Args:
        model_name: Nome do modelo no HuggingFace Hub.

    Returns:
        Modelo de embeddings carregado em memória.
    """
    console.print(f"[cyan]🤖 Carregando modelo de embeddings:[/cyan] {model_name}")
    model = SentenceTransformer(model_name)
    console.print(
        f"   [green]✓[/green] Dimensão dos vetores: "
        f"[bold]{model.get_sentence_embedding_dimension()}[/bold]"
    )
    return model


# ─────────────────────────────────────────────
# Geração de Embeddings
# ─────────────────────────────────────────────

def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Converte uma lista de textos em vetores de embeddings.

    Processa em batches para eficiência de memória.
    Mostra progresso via rich.

    Args:
        texts:      Lista de textos para vetorizar.
        model:      Modelo de embeddings carregado.
        batch_size: Número de textos por batch de processamento.

    Returns:
        Lista de vetores (cada vetor é uma lista de floats).

    Note:
        O tamanho de cada vetor depende do modelo usado.
        Para all-MiniLM-L6-v2: 384 dimensões.
    """
    console.print(f"[cyan]⚙ Gerando embeddings para {len(texts)} chunks...[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Vetorizando...", total=len(texts))

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = model.encode(batch, convert_to_numpy=True)
            all_embeddings.extend(embeddings.tolist())
            progress.advance(task, len(batch))

    return all_embeddings


# ─────────────────────────────────────────────
# Indexação no ChromaDB
# ─────────────────────────────────────────────

def index_chunks(
    chunks: list[Chunk],
    collection: chromadb.Collection,
    model: SentenceTransformer,
    batch_size: int = 100,
) -> int:
    """
    Indexa uma lista de chunks no ChromaDB.

    Para cada chunk:
        1. Gera embedding do texto
        2. Prepara metadados (source, page, chunk_idx)
        3. Insere no ChromaDB via upsert (cria ou atualiza)

    O upsert evita duplicatas: se um chunk com mesmo ID já
    existir, ele é substituído — permitindo re-indexação segura.

    Args:
        chunks:     Chunks a indexar.
        collection: Coleção ChromaDB de destino.
        model:      Modelo de embeddings.
        batch_size: Tamanho do batch para inserção no ChromaDB.

    Returns:
        Número total de chunks indexados com sucesso.

    Raises:
        chromadb.errors.ChromaError: Em caso de erro de indexação.
    """
    if not chunks:
        console.print("[yellow]⚠ Nenhum chunk para indexar.[/yellow]")
        return 0

    texts = [chunk.text for chunk in chunks]
    embeddings = generate_embeddings(texts, model)

    console.print(f"[cyan]💾 Inserindo {len(chunks)} chunks no ChromaDB...[/cyan]")

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        collection.upsert(
            ids=[chunk.id for chunk in batch_chunks],
            embeddings=batch_embeddings,
            documents=[chunk.text for chunk in batch_chunks],
            metadatas=[
                {
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_idx": chunk.chunk_idx,
                }
                for chunk in batch_chunks
            ],
        )

    total = collection.count()
    console.print(
        f"[green]✓ Indexação concluída![/green] "
        f"Total na coleção: [bold]{total}[/bold] documentos"
    )
    return total


# ─────────────────────────────────────────────
# Execução Standalone
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from importlib import import_module
    ingestion = import_module("1_ingestion")
    ingest_directory = ingestion.ingest_directory

    from config import DATA_DIR
    from rich.panel import Panel

    console.print(Panel.fit(
        "[bold cyan]BLOCO 2: Embeddings e Indexação[/bold cyan]\n"
        "Convertendo documentos em vetores e armazenando no ChromaDB",
        border_style="cyan",
    ))

    # 1. Ingerir documentos
    chunks = ingest_directory(DATA_DIR)

    if not chunks:
        console.print("[red]Nenhum documento encontrado em data/. Abortando.[/red]")
        raise SystemExit(1)

    # 2. Conectar ao ChromaDB
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # 3. Carregar modelo de embeddings
    model = load_embedding_model()

    # 4. Indexar
    total = index_chunks(chunks, collection, model)

    console.print(f"\n[bold green]✓ Pipeline de indexação completo![/bold green]")
    console.print(f"  📦 Chunks processados: {len(chunks)}")
    console.print(f"  🗄️  Total na coleção:   {total}")
