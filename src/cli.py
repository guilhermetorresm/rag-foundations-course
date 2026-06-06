"""
cli.py — Interface de Terminal Interativa (Estilo Claude Code)
==============================================================
Frontend de terminal elegante para o pipeline RAG, usando:
    • questionary → menus interativos com setas do teclado
    • rich        → painéis, tabelas, spinners e renderização Markdown

Como executar:
    uv run python src/cli.py

Menu principal:
    1. Ingerir Documentos   → processa data/ e indexa no ChromaDB
    2. Fazer Pergunta (RAG) → busca + geração com Ollama
    3. Listar Documentos    → mostra o que está no ChromaDB
    4. Configurações        → exibe configuração atual
    5. Sair
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from importlib import import_module

import questionary
from questionary import Style as QStyle

from rich import box
from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule
from rich.prompt import Prompt

from config import DATA_DIR, chroma_cfg, ollama_cfg, chunking_cfg, retrieval_cfg

# Imports dinâmicos para módulos com nomes numéricos
_ingestion  = import_module("1_ingestion")
_indexing   = import_module("2_indexing")
_retrieval  = import_module("3_retrieval")
_generation = import_module("4_generation")

console = Console()

# ─────────────────────────────────────────────
# Estilo Visual do Questionary
# ─────────────────────────────────────────────

QUESTIONARY_STYLE = QStyle([
    ("qmark",        "fg:#61afef bold"),      # ? azul
    ("question",     "fg:#e5c07b bold"),      # pergunta amarela
    ("answer",       "fg:#98c379 bold"),      # resposta verde
    ("pointer",      "fg:#c678dd bold"),      # ▶ roxo
    ("highlighted",  "fg:#c678dd bold"),      # item selecionado roxo
    ("selected",     "fg:#98c379"),           # selecionado verde
    ("separator",    "fg:#5c6370"),           # separador cinza
    ("instruction",  "fg:#5c6370 italic"),    # instrução cinza
    ("text",         "fg:#abb2bf"),           # texto normal
    ("disabled",     "fg:#5c6370 italic"),    # desabilitado cinza
])

# ─────────────────────────────────────────────
# Banner de Boas-Vindas
# ─────────────────────────────────────────────

BANNER = """
██████╗  █████╗  ██████╗     ██████╗ ██████╗  ██╗   ██╗██████╗ ███████╗███████╗
██╔══██╗██╔══██╗██╔════╝     ██╔════╝██╔═══██╗██║   ██║██╔══██╗██╔════╝██╔════╝
██████╔╝███████║██║  ███╗    ██║     ██║   ██║██║   ██║██████╔╝███████╗█████╗  
██╔══██╗██╔══██║██║   ██║    ██║     ██║   ██║██║   ██║██╔══██╗╚════██║██╔══╝  
██║  ██║██║  ██║╚██████╔╝    ╚██████╗╚██████╔╝╚██████╔╝██║  ██║███████║███████╗
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝      ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝
"""

def print_banner() -> None:
    """Exibe o banner ASCII da aplicação com gradiente de cores."""
    console.print()
    console.print(Text(BANNER, style="bold cyan"), justify="center")
    console.print(
        Align.center(
            Text("📚 RAG Foundations Course — Interface Interativa", style="bold white")
        )
    )
    console.print(
        Align.center(
            Text(
                f"Modelo: {ollama_cfg.model}  |  Coleção: {chroma_cfg.collection_name}",
                style="dim"
            )
        )
    )
    console.print()


# ─────────────────────────────────────────────
# Estado Global Compartilhado
# ─────────────────────────────────────────────

_app_state: dict = {}

def get_app_state() -> dict:
    """Inicializa o estado da aplicação com conexões aos serviços (lazy)."""
    if not _app_state:
        with console.status("[bold cyan]🔌 Conectando aos serviços...[/bold cyan]"):
            _app_state["client"]     = _indexing.get_chroma_client()
            _app_state["collection"] = _indexing.get_or_create_collection(
                _app_state["client"]
            )
    return _app_state


# ─────────────────────────────────────────────
# Ação 1: Ingerir Documentos
# ─────────────────────────────────────────────

def action_ingest() -> None:
    """
    Fluxo de ingestão de documentos do diretório data/.

    Passos:
        1. Lista os arquivos encontrados
        2. Confirma com o usuário
        3. Processa chunks + embeddings + indexação no ChromaDB
        4. Exibe resumo final
    """
    console.print(Rule("[bold cyan]📥 Ingestão de Documentos[/bold cyan]"))
    console.print()

    # Lista arquivos disponíveis
    supported = {".pdf", ".txt", ".md"}
    files = [f for f in DATA_DIR.iterdir() if f.suffix.lower() in supported]

    if not files:
        console.print(
            Panel(
                f"[yellow]Nenhum documento encontrado em:[/yellow]\n"
                f"[bold]{DATA_DIR}[/bold]\n\n"
                "Coloque arquivos .pdf, .txt ou .md nessa pasta e tente novamente.",
                title="⚠ Pasta Vazia",
                border_style="yellow",
            )
        )
        return

    # Mostra tabela de arquivos
    table = Table(title="Documentos Encontrados", box=box.ROUNDED, border_style="cyan")
    table.add_column("Arquivo", style="cyan")
    table.add_column("Tipo", justify="center")
    table.add_column("Tamanho", justify="right")

    for f in files:
        size = f.stat().st_size
        size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} bytes"
        table.add_row(f.name, f.suffix.upper(), size_str)

    console.print(table)
    console.print()

    confirmed = questionary.confirm(
        f"Indexar {len(files)} arquivo(s) no ChromaDB?",
        default=True,
        style=QUESTIONARY_STYLE,
    ).ask()

    if not confirmed:
        console.print("[dim]Operação cancelada.[/dim]")
        return

    state = get_app_state()

    # Ingestão
    console.print()
    chunks = _ingestion.ingest_directory(DATA_DIR)

    if not chunks:
        console.print("[red]Nenhum chunk gerado. Verifique os documentos.[/red]")
        return

    # Indexação
    total = _indexing.index_chunks(
        chunks=chunks,
        collection=state["collection"],
    )

    # Resumo final
    console.print()
    console.print(Panel(
        f"[green bold]✓ Ingestão concluída com sucesso![/green bold]\n\n"
        f"  📄 Arquivos processados: [bold]{len(files)}[/bold]\n"
        f"  🔀 Chunks gerados:       [bold]{len(chunks)}[/bold]\n"
        f"  🗄️  Total na coleção:     [bold]{total}[/bold]",
        title="✅ Resumo da Ingestão",
        border_style="green",
    ))


# ─────────────────────────────────────────────
# Ação 2: Fazer Pergunta ao RAG
# ─────────────────────────────────────────────

def action_ask() -> None:
    """
    Fluxo completo de pergunta ao sistema RAG.

    Passos:
        1. Verifica se há documentos indexados
        2. Solicita pergunta ao usuário
        3. Executa retrieval (busca vetorial)
        4. Exibe fontes encontradas
        5. Gera resposta com Ollama (com spinner)
        6. Renderiza resposta em Markdown
    """
    console.print(Rule("[bold cyan]🤖 Fazer Pergunta ao RAG[/bold cyan]"))
    console.print()

    state = get_app_state()
    total_docs = state["collection"].count()

    if total_docs == 0:
        console.print(Panel(
            "[yellow]Nenhum documento indexado ainda.[/yellow]\n"
            "Use a opção [bold]'1. Ingerir Documentos'[/bold] primeiro.",
            title="⚠ Coleção Vazia",
            border_style="yellow",
        ))
        return

    console.print(
        f"[dim]💾 {total_docs} chunks disponíveis | "
        f"Modelo: {ollama_cfg.model}[/dim]\n"
    )

    # Verificar Ollama
    if not _generation.check_ollama_available():
        console.print(Panel(
            "[red bold]Ollama não está disponível![/red bold]\n\n"
            "Inicie o Ollama em outro terminal:\n"
            "[cyan bold]  ollama serve[/cyan bold]\n\n"
            "E baixe o modelo se necessário:\n"
            f"[cyan bold]  ollama pull {ollama_cfg.model}[/cyan bold]",
            title="❌ Ollama Offline",
            border_style="red",
        ))
        return

    # Loop de perguntas
    while True:
        query = questionary.text(
            "Sua pergunta (ou 'sair' para voltar ao menu):",
            style=QUESTIONARY_STYLE,
            validate=lambda t: len(t.strip()) >= 3 or "Pergunta muito curta.",
        ).ask()

        if not query or query.strip().lower() in {"sair", "exit", "quit", "q"}:
            break

        query = query.strip()
        console.print()

        # ── Retrieval ──
        with console.status("[cyan]🔍 Buscando documentos relevantes...[/cyan]"):
            chunks = _retrieval.search(
                query=query,
                collection=state["collection"],
            )

        if not chunks:
            console.print(Panel(
                "[yellow]Não encontrei trechos relevantes para essa pergunta.[/yellow]\n"
                "Tente reformular ou verifique se os documentos corretos foram indexados.",
                border_style="yellow",
            ))
            console.print()
            continue

        # ── Exibe fontes ──
        sources_table = Table(
            title=f"📎 {len(chunks)} fonte(s) recuperada(s)",
            box=box.SIMPLE_HEAD,
            border_style="dim",
        )
        sources_table.add_column("#", justify="center", style="dim", width=3)
        sources_table.add_column("Score", justify="right", style="green", width=7)
        sources_table.add_column("Fonte", style="cyan")
        sources_table.add_column("Pág.", justify="center", width=5)
        sources_table.add_column("Prévia", style="dim")

        for i, chunk in enumerate(chunks, 1):
            sources_table.add_row(
                str(i),
                f"{chunk.score:.2f}",
                Path(chunk.source).name,
                str(chunk.page),
                chunk.text[:70].replace("\n", " ") + "…",
            )

        console.print(sources_table)
        console.print()

        # ── Geração ──
        context = _retrieval.format_context(chunks)

        with console.status(
            f"[bold cyan]🤖 Gerando resposta com {ollama_cfg.model}...[/bold cyan]"
        ):
            try:
                rag_response = _generation.generate_answer(query, context)
            except ConnectionError as e:
                console.print(f"[red]{e}[/red]")
                continue

        # ── Exibe resposta ──
        console.print(Panel(
            Markdown(rag_response.answer),
            title=f"[bold green]💬 Resposta[/bold green]   [dim]({rag_response.model})[/dim]",
            border_style="green",
            padding=(1, 2),
        ))
        console.print()

        # Continuar?
        continuar = questionary.confirm(
            "Fazer outra pergunta?",
            default=True,
            style=QUESTIONARY_STYLE,
        ).ask()
        if not continuar:
            break
        console.print()


# ─────────────────────────────────────────────
# Ação 3: Listar Documentos Indexados
# ─────────────────────────────────────────────

def action_list_documents() -> None:
    """Exibe um resumo dos documentos atualmente indexados no ChromaDB."""
    console.print(Rule("[bold cyan]📚 Documentos Indexados[/bold cyan]"))
    console.print()

    state = get_app_state()
    total = state["collection"].count()

    if total == 0:
        console.print("[yellow]Nenhum documento indexado. Use '1. Ingerir Documentos'.[/yellow]")
        return

    results = state["collection"].get(include=["metadatas"], limit=5000)
    metadatas = results.get("metadatas") or []

    # Agrupa por fonte
    sources: dict[str, list[int]] = {}
    for meta in metadatas:
        src = Path(meta.get("source", "?")).name
        page = int(meta.get("page", 0))
        sources.setdefault(src, []).append(page)

    table = Table(
        title=f"Coleção: [bold]{chroma_cfg.collection_name}[/bold]  ({total} chunks)",
        box=box.ROUNDED,
        border_style="cyan",
    )
    table.add_column("Arquivo", style="cyan")
    table.add_column("Chunks", justify="right", style="yellow")
    table.add_column("Páginas", style="dim")

    for src, pages in sorted(sources.items()):
        unique_pages = sorted(set(pages))
        table.add_row(
            src,
            str(len(pages)),
            ", ".join(map(str, unique_pages[:10]))
            + (" ..." if len(unique_pages) > 10 else ""),
        )

    console.print(table)


# ─────────────────────────────────────────────
# Ação 4: Configurações
# ─────────────────────────────────────────────

def action_show_config() -> None:
    """Exibe as configurações atuais do sistema em painéis organizados."""
    console.print(Rule("[bold cyan]⚙ Configurações do Sistema[/bold cyan]"))
    console.print()

    ollama_panel = Panel(
        f"[bold]URL Base:[/bold]    {ollama_cfg.base_url}\n"
        f"[bold]Modelo LLM:[/bold]  {ollama_cfg.model}\n"
        f"[bold]Temperatura:[/bold] {ollama_cfg.temperature}\n"
        f"[bold]Contexto:[/bold]    {ollama_cfg.num_ctx} tokens",
        title="🤖 Ollama",
        border_style="cyan",
    )

    chroma_panel = Panel(
        f"[bold]Coleção:[/bold]    {chroma_cfg.collection_name}\n"
        f"[bold]Diretório:[/bold]  {chroma_cfg.persist_dir}\n"
        f"[bold]Métrica:[/bold]    {chroma_cfg.distance_metric}",
        title="🗄️ ChromaDB",
        border_style="magenta",
    )

    chunking_panel = Panel(
        f"[bold]Chunk size:[/bold]    {chunking_cfg.chunk_size} chars\n"
        f"[bold]Overlap:[/bold]       {chunking_cfg.chunk_overlap} chars\n"
        f"[bold]Mínimo:[/bold]        {chunking_cfg.min_chunk_size} chars",
        title="✂️ Chunking",
        border_style="yellow",
    )

    retrieval_panel = Panel(
        f"[bold]Top-K:[/bold]         {retrieval_cfg.top_k} chunks\n"
        f"[bold]Threshold:[/bold]     {retrieval_cfg.similarity_threshold:.0%}",
        title="🔍 Retrieval",
        border_style="green",
    )

    console.print(Columns([ollama_panel, chroma_panel]))
    console.print(Columns([chunking_panel, retrieval_panel]))

    # Status live do Ollama
    console.print()
    ollama_ok = _generation.check_ollama_available()
    status_icon = "[green]✓ Online[/green]" if ollama_ok else "[red]✗ Offline[/red]"
    console.print(f"  Status do Ollama: {status_icon}")


# ─────────────────────────────────────────────
# Menu Principal
# ─────────────────────────────────────────────

MENU_CHOICES = [
    questionary.Choice("📥  1. Ingerir Documentos",   value="ingest"),
    questionary.Choice("🤖  2. Fazer Pergunta (RAG)",  value="ask"),
    questionary.Choice("📚  3. Listar Documentos",     value="list"),
    questionary.Choice("⚙   4. Configurações",         value="config"),
    questionary.Separator(),
    questionary.Choice("🚪  5. Sair",                  value="exit"),
]

ACTION_MAP = {
    "ingest": action_ingest,
    "ask":    action_ask,
    "list":   action_list_documents,
    "config": action_show_config,
}


def main() -> None:
    """
    Ponto de entrada principal da CLI.

    Exibe o banner, inicializa os serviços e entra no loop
    do menu interativo até o usuário escolher sair.
    """
    # Limpa o terminal para uma apresentação limpa
    os.system("cls" if os.name == "nt" else "clear")
    print_banner()

    while True:
        console.print()

        choice = questionary.select(
            "O que deseja fazer?",
            choices=MENU_CHOICES,
            style=QUESTIONARY_STYLE,
            use_shortcuts=False,
            use_arrow_keys=True,
        ).ask()

        if choice is None or choice == "exit":
            console.print()
            console.print(Panel.fit(
                "[bold cyan]Até logo! 👋[/bold cyan]\n"
                "[dim]RAG Foundations Course[/dim]",
                border_style="cyan",
            ))
            console.print()
            break

        console.print()

        if action := ACTION_MAP.get(choice):
            try:
                action()
            except KeyboardInterrupt:
                console.print("\n[dim]Operação interrompida.[/dim]")
            except Exception as e:
                console.print(Panel(
                    f"[red bold]Erro inesperado:[/red bold] {e}\n"
                    f"[dim]{type(e).__name__}[/dim]",
                    title="❌ Erro",
                    border_style="red",
                ))

        console.print()
        console.print(Rule(style="dim"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[dim]Até logo![/dim]\n")
