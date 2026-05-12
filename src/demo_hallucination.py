"""
demo_hallucination.py — Demonstração do Problema e Solução (Bloco 1)
===================================================================
Este script replica o exemplo das páginas 2 e 8 dos slides:
O caso da "Resolução nº 12/2025" que não existe.

Demonstra:
1. LLM Puro alucinando um prazo que não existe.
2. RAG fundamentando a resposta (ou admitindo falta de dados).
"""

import ollama
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from config import ollama_cfg
from importlib import import_module
retrieval = import_module("3_retrieval")
indexing = import_module("2_indexing")
generation = import_module("4_generation")

console = Console()

QUERY = "Qual o prazo para recurso administrativo segundo a Resolução nº 12/2025 da nossa instituição?"

def ask_pure_llm(query: str):
    console.print(Panel(f"[bold red]Desafio:[/bold red] LLM Puro sem documentos\nQuery: {query}", border_style="red"))
    
    response = ollama.chat(
        model=ollama_cfg.model,
        messages=[{"role": "user", "content": query}],
        options={"temperature": 0.7} # Aumenta chance de "criatividade"/alucinação
    )
    return response.message.content

def ask_rag(query: str):
    console.print(Panel(f"[bold green]Solução:[/bold green] RAG com fundamentação\nQuery: {query}", border_style="green"))
    
    # 1. Retrieval
    client = indexing.get_chroma_client()
    collection = indexing.get_or_create_collection(client)
    
    chunks = retrieval.search(query, collection)
    context = retrieval.format_context(chunks)
    
    # 2. Generation
    response = generation.generate_answer(query, context)
    return response.answer

if __name__ == "__main__":
    console.print("\n[bold cyan]═══ DEMONSTRAÇÃO: ALUCINAÇÃO VS RAG ═══[/bold cyan]\n")
    
    # Parte 1: O Problema (Alucinação)
    with console.status("[bold red]Consultando LLM puro (sem contexto)...[/bold red]"):
        hallucination = ask_pure_llm(QUERY)
    
    console.print(Panel(hallucination, title="Resposta do LLM Puro", border_style="red"))
    console.print("[dim italic]Nota: Repare como o modelo inventa prazos (ex: 15 dias) com total confiança.[/dim italic]\n")
    
    # Parte 2: A Solução (RAG)
    with console.status("[bold green]Consultando RAG (buscando evidências)...[/bold green]"):
        rag_solution = ask_rag(QUERY)
        
    console.print(Panel(rag_solution, title="Resposta com RAG", border_style="green"))
    console.print("[dim italic]Nota: Aqui o modelo deve dizer que NÃO encontrou a Resolução 12/2025 nos documentos oficiais.[/dim italic]\n")
