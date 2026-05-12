"""
4_generation.py — Geração de Resposta com Ollama (RAG)
=======================================================
Bloco 4 do Pipeline RAG: monta um prompt estruturado com o
contexto recuperado e envia ao LLM local via Ollama.
"""

from dataclasses import dataclass
from pathlib import Path
import ollama
from rich.console import Console
from config import OllamaConfig, ollama_cfg, gen_cfg

console = Console()


# ─────────────────────────────────────────────
# Tipos de Dados
# ─────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Representa uma resposta completa do pipeline RAG.
    """
    answer: str
    query: str
    model: str
    context_used: str


# ─────────────────────────────────────────────
# Construção do Prompt RAG
# ─────────────────────────────────────────────

def build_rag_prompt(query: str, context: str) -> list[dict[str, str]]:
    """
    Constrói o prompt RAG conforme o 'Contrato' (Slide 4 e 5 do Bloco 3).
    """
    return [
        {"role": "system", "content": gen_cfg.default_system_prompt},
        {"role": "user",   "content": f"CONTEXTO:\n{context}\n\nPERGUNTA: {query}"},
    ]


def format_context_with_budget(chunks: list, max_chars: int = gen_cfg.max_context_chars) -> str:
    """
    Formata contexto respeitando o Token Budget via len() (Slide 8 do Bloco 3).
    """
    parts = []
    current_chars = 0
    
    for i, c in enumerate(chunks, start=1):
        source_name = Path(c.source).name
        header = f"({i}) Fonte: {source_name} | Pág: {c.page}\n"
        block = header + c.text
        
        if current_chars + len(block) > max_chars:
            break
            
        parts.append(block)
        current_chars += len(block)
        
    return "\n\n".join(parts)


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

        return RAGResponse(
            answer=answer,
            query=query,
            model=cfg.model,
            context_used=context,
        )

    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            raise ConnectionError(
                f"Não foi possível conectar ao Ollama em {cfg.base_url}.\n"
                "Certifique-se de que o Ollama está rodando com: ollama serve"
            ) from e
        raise


def check_ollama_available(cfg: OllamaConfig = ollama_cfg) -> bool:
    """
    Verifica se o servidor Ollama está acessível.
    """
    try:
        ollama.list()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    from rich.panel import Panel
    from rich.markdown import Markdown

    # Pipeline simplificado para teste
    query = "Quais são as regras de férias?"
    context = "(1) Fonte: teste.txt | Pág: 1\nTodo funcionário tem 30 dias de férias."
    
    if check_ollama_available():
        res = generate_answer(query, context)
        console.print(Panel(Markdown(res.answer), title="Resposta Teste"))
    else:
        console.print("[red]Ollama offline[/red]")
