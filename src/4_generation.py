"""
4_generation.py — Geração de Resposta com Ollama (RAG)
=======================================================
Bloco 4 do Pipeline RAG: monta um prompt estruturado com o
contexto recuperado e envia ao LLM local via Ollama para
gerar uma resposta fundamentada nos documentos.

Conceito-chave ensinado aqui:
    O prompt RAG tem três seções críticas:
        1. System Prompt  → Define papel e restrições do LLM
        2. Contexto       → Trechos relevantes buscados no ChromaDB
        3. Pergunta       → A query original do usuário

    A instrução "responda APENAS com base nos trechos" é o
    que transforma um LLM genérico num especialista nos seus docs.

Fluxo:
    Query + Chunks → Prompt Estruturado → Ollama → Resposta
"""

from dataclasses import dataclass
from pathlib import Path

import ollama
from rich.console import Console

from config import OllamaConfig, ollama_cfg

console = Console()


# ─────────────────────────────────────────────
# Tipos de Dados
# ─────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Representa uma resposta completa do pipeline RAG.

    Attributes:
        answer:         Resposta gerada pelo LLM.
        query:          Pergunta original do usuário.
        model:          Nome do modelo LLM utilizado.
        context_used:   Contexto enviado ao LLM (para debug/auditoria).
        prompt_tokens:  Tokens usados no prompt (se disponível).
        total_tokens:   Tokens totais usados na geração.
    """
    answer: str
    query: str
    model: str
    context_used: str
    prompt_tokens: int = 0
    total_tokens: int = 0


# ─────────────────────────────────────────────
# Construção do Prompt RAG
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Você é um assistente especializado que responde perguntas com base
EXCLUSIVAMENTE nos trechos de documentos fornecidos abaixo.

Regras que você DEVE seguir:
1. Responda SOMENTE com informações presentes nos trechos fornecidos.
2. Se a resposta não estiver nos trechos, diga claramente: "Não encontrei essa informação nos documentos disponíveis."
3. Cite a fonte (nome do arquivo e página) ao final de cada informação relevante.
4. Use formatação Markdown para clareza (listas, negrito, etc.).
5. Seja objetivo e preciso. Não invente informações."""


def build_rag_prompt(query: str, context: str) -> list[dict[str, str]]:
    """
    Constrói a lista de mensagens no formato de chat para o Ollama.

    A estrutura segue o padrão "instruction-following" dos LLMs modernos:
        - role: "system" → instrução de comportamento
        - role: "user"   → contexto + pergunta

    Args:
        query:   Pergunta do usuário.
        context: Contexto formatado com os chunks recuperados.

    Returns:
        Lista de mensagens no formato esperado pela API do Ollama.

    Example:
        >>> msgs = build_rag_prompt("O que é férias?", "Trecho 1: ...")
        >>> print(msgs[0]["role"])  # "system"
    """
    user_message = f"""TRECHOS DOS DOCUMENTOS:
{context}

---

PERGUNTA DO USUÁRIO:
{query}

Por favor, responda a pergunta acima com base nos trechos fornecidos."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]


# ─────────────────────────────────────────────
# Chamada ao Ollama (Síncrona)
# ─────────────────────────────────────────────

def generate_answer(
    query: str,
    context: str,
    cfg: OllamaConfig = ollama_cfg,
) -> RAGResponse:
    """
    Envia o prompt RAG ao Ollama e retorna a resposta completa.

    A função usa o modo síncrono (não-streaming) do Ollama.
    Para streaming em tempo real, veja generate_answer_stream().

    Args:
        query:   Pergunta do usuário.
        context: Contexto formatado com os chunks recuperados.
        cfg:     Configurações do Ollama (modelo, temperatura, etc.).

    Returns:
        RAGResponse com a resposta gerada e metadados.

    Raises:
        ollama.ResponseError: Se o modelo não estiver disponível.
        ConnectionError: Se o Ollama não estiver rodando.
    """
    messages = build_rag_prompt(query, context)

    try:
        response = ollama.chat(
            model=cfg.model,
            messages=messages,
            options={
                "temperature": cfg.temperature,
                "num_ctx": cfg.num_ctx,
            },
        )

        answer = response.message.content or ""
        usage = response.usage or {}

        return RAGResponse(
            answer=answer,
            query=query,
            model=cfg.model,
            context_used=context,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )

    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            raise ConnectionError(
                f"Não foi possível conectar ao Ollama em {cfg.base_url}.\n"
                "Certifique-se de que o Ollama está rodando com: ollama serve"
            ) from e
        raise


def generate_answer_stream(
    query: str,
    context: str,
    cfg: OllamaConfig = ollama_cfg,
):
    """
    Versão streaming do gerador: retorna tokens à medida que são produzidos.

    Use esta versão para experiências de usuário mais responsivas,
    onde o texto aparece progressivamente (como o ChatGPT).

    Args:
        query:   Pergunta do usuário.
        context: Contexto formatado com os chunks recuperados.
        cfg:     Configurações do Ollama.

    Yields:
        Strings com tokens gerados um a um.

    Example:
        >>> for token in generate_answer_stream(query, context):
        ...     print(token, end="", flush=True)
    """
    messages = build_rag_prompt(query, context)

    stream = ollama.chat(
        model=cfg.model,
        messages=messages,
        stream=True,
        options={
            "temperature": cfg.temperature,
            "num_ctx": cfg.num_ctx,
        },
    )

    for chunk in stream:
        token = chunk.message.content
        if token:
            yield token


# ─────────────────────────────────────────────
# Verificação de Disponibilidade do Ollama
# ─────────────────────────────────────────────

def check_ollama_available(cfg: OllamaConfig = ollama_cfg) -> bool:
    """
    Verifica se o servidor Ollama está acessível e o modelo está disponível.

    Args:
        cfg: Configurações do Ollama.

    Returns:
        True se disponível, False caso contrário.
    """
    try:
        models = ollama.list()
        available_models = [m.model for m in models.models]
        model_base = cfg.model.split(":")[0]  # remove tag como ":latest"
        return any(model_base in m for m in available_models)
    except Exception:
        return False


# ─────────────────────────────────────────────
# Execução Standalone
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from importlib import import_module
    from rich.panel import Panel
    from rich.markdown import Markdown

    indexing = import_module("2_indexing")
    retrieval = import_module("3_retrieval")

    # Verificar Ollama
    if not check_ollama_available():
        console.print(
            Panel(
                "[red bold]Ollama não está disponível![/red bold]\n\n"
                "Para iniciar o Ollama, execute em outro terminal:\n"
                "[cyan]ollama serve[/cyan]\n\n"
                "Para baixar o modelo padrão:\n"
                f"[cyan]ollama pull {ollama_cfg.model}[/cyan]",
                title="⚠ Erro de Conexão",
                border_style="red",
            )
        )
        raise SystemExit(1)

    # Pipeline completo
    client = indexing.get_chroma_client()
    collection = indexing.get_or_create_collection(client)
    embed_model = indexing.load_embedding_model()

    query = "Quais são as regras de férias dos funcionários?"

    console.print(Panel.fit(
        f"[bold cyan]BLOCO 4: Geração RAG[/bold cyan]\n"
        f"Query: [italic]\"{query}\"[/italic]",
        border_style="cyan",
    ))

    chunks = retrieval.search(query, collection, embed_model)
    context = retrieval.format_context(chunks)

    console.print(f"[dim]Contexto: {len(context)} chars | Chunks: {len(chunks)}[/dim]\n")

    with console.status("[bold cyan]🤖 Gerando resposta com Ollama...[/bold cyan]"):
        rag_response = generate_answer(query, context)

    console.print(Panel(
        Markdown(rag_response.answer),
        title=f"[bold green]Resposta[/bold green] — Modelo: {rag_response.model}",
        border_style="green",
    ))
