"""
embeddings.py — Utilitário de Embeddings via Ollama
==================================================
Centraliza a geração de vetores usando a API do Ollama.
Embeddings são gerados localmente.
"""

import ollama
from rich.console import Console
from config import ollama_cfg

console = Console()

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Gera embeddings para uma lista de textos usando o Ollama.
    """
    embeddings = []
    for text in texts:
        try:
            response = ollama.embeddings(model=ollama_cfg.embed_model, prompt=text)
            embeddings.append(response["embedding"])
        except Exception as e:
            console.print(f"[red]Erro ao gerar embedding via Ollama: {e}[/red]")
            raise
    return embeddings

def get_single_embedding(text: str) -> list[float]:
    """
    Gera embedding para um único texto.
    """
    return get_embeddings([text])[0]
