"""
1_ingestion.py — Ingestão e Chunking de Documentos
====================================================
Bloco 1 do Pipeline RAG: carrega documentos brutos do disco,
limpa o texto e divide em chunks sobrepostos para indexação.

Formatos suportados:
    • PDF  → via pypdf
    • TXT  → leitura direta
    • MD   → leitura direta

Conceito-chave ensinado aqui:
    Chunking com sobreposição (overlap) garante que o contexto
    não seja perdido nas bordas de cada janela de texto.
"""

import re
import unicodedata
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

    Attributes:
        id:        Identificador único no formato "nome_arquivo_idx".
        text:      Conteúdo textual do chunk.
        source:    Caminho do arquivo de origem.
        page:      Número da página (relevante para PDFs).
        chunk_idx: Índice sequencial do chunk dentro do documento.
    """
    id: str
    text: str
    source: str
    page: int
    chunk_idx: int


# ─────────────────────────────────────────────
# Limpeza de Texto
# ─────────────────────────────────────────────

def clean_text(raw_text: str) -> str:
    """
    Normaliza e limpa texto bruto extraído de documentos.

    Pipeline de limpeza:
        1. Normalização Unicode → converte caracteres especiais para forma canônica
        2. Remove quebras de linha duplas → preserva apenas parágrafos
        3. Remove espaços excessivos
        4. Remove caracteres de controle (exceto newline)

    Args:
        raw_text: Texto bruto extraído do documento.

    Returns:
        Texto limpo e normalizado.

    Example:
        >>> clean_text("  Olá\\n\\n\\n  Mundo!  ")
        'Olá\\n\\nMundo!'
    """
    # 1. Normalização Unicode (converte ligatures, diacríticos, etc.)
    text = unicodedata.normalize("NFKC", raw_text)

    # 2. Substitui múltiplas quebras de linha por no máximo duas
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 3. Remove espaços e tabs excessivos em cada linha
    text = "\n".join(line.strip() for line in text.splitlines())

    # 4. Remove caracteres de controle (exceto \n e \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text.strip()


# ─────────────────────────────────────────────
# Leitores de Documento
# ─────────────────────────────────────────────

def load_pdf(path: Path) -> list[tuple[str, int]]:
    """
    Carrega um PDF e retorna o texto por página.

    Args:
        path: Caminho para o arquivo PDF.

    Returns:
        Lista de tuplas (texto_da_página, número_da_página).

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        pypdf.errors.PdfReadError: Se o PDF estiver corrompido.
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

    Args:
        path: Caminho para o arquivo de texto.

    Returns:
        Lista com uma única tupla (texto_completo, página=1).
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    return [(clean_text(raw), 1)]


def load_document(path: Path) -> list[tuple[str, int]]:
    """
    Roteador de carregamento: detecta o tipo do arquivo e chama o leitor correto.

    Args:
        path: Caminho para o documento.

    Returns:
        Lista de tuplas (texto, número_da_página).

    Raises:
        ValueError: Se o formato do arquivo não for suportado.
    """
    suffix = path.suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".md":  load_text,
    }

    if suffix not in loaders:
        raise ValueError(
            f"Formato '{suffix}' não suportado. "
            f"Formatos aceitos: {list(loaders.keys())}"
        )

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

    Algoritmo:
        - Avança pelo texto em passos de (chunk_size - chunk_overlap)
        - Cada chunk tem tamanho máximo de chunk_size caracteres
        - A sobreposição garante continuidade entre chunks adjacentes

    Args:
        text:             Texto a ser dividido.
        source:           Nome/caminho do arquivo de origem.
        page:             Número da página de origem.
        doc_chunk_offset: Offset para numerar chunks sequencialmente entre páginas.
        cfg:              Configurações de chunking (tamanho, overlap, mínimo).

    Returns:
        Lista de objetos Chunk.

    Example:
        >>> chunks = split_into_chunks("texto longo...", "doc.pdf", page=1)
        >>> print(chunks[0].id)  # "doc.pdf_0"
    """
    chunks: list[Chunk] = []
    step = cfg.chunk_size - cfg.chunk_overlap
    source_name = Path(source).name

    start = 0
    chunk_idx = doc_chunk_offset

    while start < len(text):
        end = start + cfg.chunk_size
        chunk_text = text[start:end].strip()

        # Descarta chunks muito pequenos (provavelmente ruído)
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

    Args:
        directory: Diretório contendo os documentos a processar.
        cfg:       Configurações de chunking.

    Returns:
        Lista consolidada de todos os chunks de todos os documentos.

    Raises:
        FileNotFoundError: Se o diretório não existir.
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
                all_chunks.extend(page_chunks)

            console.print(
                f"   [green]✓[/green] {len(all_chunks)} chunks totais até agora"
            )

        except Exception as e:
            console.print(f"   [red]✗ Erro ao processar {file_path.name}: {e}[/red]")

    return all_chunks


# ─────────────────────────────────────────────
# Execução Standalone (para testar e demonstrar)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
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

        for chunk in chunks[:10]:  # mostra os primeiros 10
            table.add_row(
                chunk.id,
                Path(chunk.source).name,
                str(chunk.page),
                f"{len(chunk.text)} chars",
                chunk.text[:60].replace("\n", " ") + "...",
            )

        console.print(table)

        if len(chunks) > 10:
            console.print(f"[dim]... e mais {len(chunks) - 10} chunks.[/dim]")
