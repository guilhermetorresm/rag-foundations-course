"""
1_ingestion.py — Ingestão e Chunking de Documentos
====================================================
Bloco 1 do Pipeline RAG: carrega documentos brutos do disco,
limpa o texto e divide em chunks sobrepostos para indexação.
"""

import re
import unicodedata
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pypdf
from rich.console import Console

from config import ChunkingConfig, chunking_cfg

console = Console()


# ─────────────────────────────────────────────
# Tipos de Dados
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    """
    Representa um fragmento de texto extraído de um documento.
    """
    id: str
    text: str
    source: str
    page: int
    chunk_idx: int
    doc_hash: str = ""  # Para governança e versionamento


# ─────────────────────────────────────────────
# Limpeza de Texto
# ─────────────────────────────────────────────

def clean_text(raw_text: str) -> str:
    """
    Normaliza e limpa texto bruto
    """
    # 1. Normalização Unicode
    text = unicodedata.normalize("NFKC", raw_text)

    # 2. Remove soft-hyphens (\u00ad) - invisíveis mas tóxicos
    text = text.replace("\u00ad", "")

    # 3. Junta palavras quebradas por hífen no fim da linha
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # 4. Colapsa espaços e tabs excessivos
    text = re.sub(r"[ \t]+", " ", text)
    
    # 5. Normaliza quebras de linha
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def compute_hash(file_path: Path) -> str:
    """Gera um hash SHA-256 do arquivo para controle de versão."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


# ─────────────────────────────────────────────
# Leitores de Documento
# ─────────────────────────────────────────────

def load_pdf(path: Path) -> list[tuple[str, int]]:
    """
    Carrega um PDF e retorna o texto por página.
    """
    pages: list[tuple[str, int]] = []

    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = clean_text(raw)
            if cleaned:
                pages.append((cleaned, page_num))

    return pages


def load_text(path: Path) -> list[tuple[str, int]]:
    """
    Carrega um arquivo de texto plano (.txt ou .md).
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    return [(clean_text(raw), 1)]


def load_document(path: Path) -> list[tuple[str, int]]:
    """
    Roteador de carregamento.
    """
    suffix = path.suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".md":  load_text,
    }

    if suffix not in loaders:
        raise ValueError(f"Formato '{suffix}' não suportado.")

    return loaders[suffix](path)


# ─────────────────────────────────────────────
# Chunking com Sobreposição (Sliding Window)
# ─────────────────────────────────────────────

def split_into_chunks(
    text: str,
    source: str,
    page: int,
    doc_chunk_offset: int = 0,
    cfg: ChunkingConfig = chunking_cfg,
) -> list[Chunk]:
    """
    Divide um texto em chunks com sobreposição usando janela deslizante.
    """
    chunks: list[Chunk] = []
    step = cfg.chunk_size - cfg.chunk_overlap
    source_name = Path(source).name
    start = 0
    chunk_idx = doc_chunk_offset
    
    while start < len(text):
        end = start + cfg.chunk_size
        chunk_text = text[start:end].strip()

        if len(chunk_text) >= cfg.min_chunk_size:
            chunk_id = f"{source_name}_{chunk_idx}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    source=source,
                    page=page,
                    chunk_idx=chunk_idx,
                )
            )
            chunk_idx += 1
        start += step
    return chunks


# ─────────────────────────────────────────────
# Ponto de Entrada Principal
# ─────────────────────────────────────────────

def ingest_directory(directory: Path, cfg: ChunkingConfig = chunking_cfg) -> list[Chunk]:
    """
    Processa todos os documentos de um diretório e retorna todos os chunks.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {directory}")

    all_chunks: list[Chunk] = []
    supported = {".pdf", ".txt", ".md"}
    files = [f for f in directory.iterdir() if f.suffix.lower() in supported]

    if not files:
        console.print(f"[yellow]⚠ Nenhum documento encontrado em {directory}[/yellow]")
        return []

    for file_path in files:
        console.print(f"[cyan]📄 Processando:[/cyan] {file_path.name}")
        try:
            doc_hash = compute_hash(file_path)
            pages = load_document(file_path)
            doc_offset = len(all_chunks)

            for page_text, page_num in pages:
                page_chunks = split_into_chunks(
                    text=page_text,
                    source=str(file_path),
                    page=page_num,
                    doc_chunk_offset=doc_offset + len(all_chunks),
                    cfg=cfg,
                )
                for c in page_chunks:
                    c.doc_hash = doc_hash
                all_chunks.extend(page_chunks)

            console.print(f"   [green]✓[/green] {len(all_chunks)} chunks totais até agora")

        except Exception as e:
            console.print(f"   [red]✗ Erro ao processar {file_path.name}: {e}[/red]")

    return all_chunks


if __name__ == "__main__":
    from rich.table import Table
    data_dir = Path(__file__).parent.parent / "data"

    console.print("\n[bold cyan]═══ BLOCO 1: Ingestão e Chunking ═══[/bold cyan]\n")
    chunks = ingest_directory(data_dir)

    if chunks:
        table = Table(title=f"Chunks Gerados ({len(chunks)} total)", show_lines=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Fonte", style="yellow")
        table.add_column("Pág.", justify="center")
        table.add_column("Tamanho", justify="right")
        table.add_column("Prévia", style="dim")

        for chunk in chunks[:10]:
            table.add_row(
                chunk.id,
                Path(chunk.source).name,
                str(chunk.page),
                f"{len(chunk.text)} chars",
                chunk.text[:60].replace("\n", " ") + "...",
            )

        console.print(table)
