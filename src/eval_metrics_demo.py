"""
eval_metrics_demo.py — Demonstração de Métricas de Retrieval (Recall@K e MRR)
=============================================================================
Calcula e exibe as métricas de relevância usando um dataset de teste fictício.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Definição das Métricas de Relevância
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(expected_ids: set, ret_ids: list, k: int) -> float:
    """
    Recall@K: O chunk esperado apareceu nos top-K resultados?
    Retorna 1.0 se houver interseção entre os IDs esperados e os K primeiros retornados,
    caso contrário retorna 0.0.
    """
    return 1.0 if any(x in expected_ids for x in ret_ids[:k]) else 0.0


def mrr(expected_ids: set, ret_ids: list, k: int) -> float:
    """
    MRR (Mean Reciprocal Rank): Quão alto na lista o chunk correto apareceu?
    Retorna 1/rank do primeiro chunk esperado encontrado nos top-K, ou 0.0 se não encontrar.
    """
    for rank, cid in enumerate(ret_ids[:k], start=1):
        if cid in expected_ids:
            return 1.0 / rank  # rank 1 = 1.0, rank 2 = 0.5, rank 3 = 0.33, etc.
    return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dados de Teste (Golden Dataset vs Resultados do Retrieval)
# ─────────────────────────────────────────────────────────────────────────────

# Golden Dataset (Perguntas e os Chunks reais que deveriam ser recuperados)
golden_dataset = [
    {
        "query": "Qual o prazo de recurso?",
        "expected_ids": {"reg:p14:03"}
    },
    {
        "query": "Como solicitar reembolso de curso?",
        "expected_ids": {"reg:p83:13"}
    },
    {
        "query": "Quem tem direito ao plano de saúde?",
        "expected_ids": {"reg:p53:08"}
    },
    {
        "query": "Qual a senha padrão do Wi-Fi?",
        "expected_ids": {"reg:p20:01"}
    }
]

# Resultados simulados retornados pelo banco vetorial (retrieved IDs) para K=5
simulated_retrieval = {
    "Qual o prazo de recurso?": [
        "reg:p10:02", 
        "reg:p14:03",  # Encontrado na posição 2 (Rank 2)
        "reg:p14:04", 
        "reg:p12:01", 
        "reg:p15:01"
    ],
    "Como solicitar reembolso de curso?": [
        "reg:p83:13",  # Encontrado na posição 1 (Rank 1)
        "reg:p83:14", 
        "reg:p85:01", 
        "reg:p10:05", 
        "reg:p12:02"
    ],
    "Quem tem direito ao plano de saúde?": [
        "reg:p40:01", 
        "reg:p42:02",  
        "reg:p45:03", 
        "reg:p48:01", 
        "reg:p53:08"   # Encontrado na posição 5 (Rank 5)
    ],
    "Qual a senha padrão do Wi-Fi?": [
        "reg:p30:02", 
        "reg:p31:01", 
        "reg:p35:03", 
        "reg:p36:01", 
        "reg:p38:02"   # Não foi encontrado nos top-5 (Erro de retrieval)
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Execução e Exibição de Resultados
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(k: int = 5):
    console.print(Panel.fit(
        f"[bold cyan]Avaliação de Retrieval — Métricas Objetivas (K={k})[/bold cyan]\n"
        "Compara as previsões contra o Golden Dataset.",
        border_style="cyan"
    ))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Pergunta", style="dim", width=38)
    table.add_column("Esperado", justify="center")
    table.add_column("Top Chunks Retornados", width=25)
    table.add_column("Posição (Rank)", justify="center", style="yellow")
    table.add_column(f"Recall@{k}", justify="right", style="green")
    table.add_column(f"MRR@{k}", justify="right", style="green")

    total_recall = 0.0
    total_mrr = 0.0

    for item in golden_dataset:
        query = item["query"]
        expected = item["expected_ids"]
        retrieved = simulated_retrieval[query]

        # Calcular métricas
        rec = recall_at_k(expected, retrieved, k)
        mrr_val = mrr(expected, retrieved, k)

        total_recall += rec
        total_mrr += mrr_val

        # Identificar onde o item esperado apareceu
        rank_str = "N/A"
        for idx, cid in enumerate(retrieved[:k], start=1):
            if cid in expected:
                rank_str = f"#{idx}"
                break

        table.add_row(
            query,
            ", ".join(expected),
            str(retrieved[:k]),
            rank_str,
            f"{rec:.2f}",
            f"{mrr_val:.2f}"
        )

    console.print(table)
    console.print()

    # Médias Globais
    num_queries = len(golden_dataset)
    avg_recall = total_recall / num_queries
    avg_mrr = total_mrr / num_queries

    # Níveis de Qualidade (conforme tabela do slide)
    def get_status(val: float, minima: float, boa: float) -> str:
        if val >= boa:
            return "[bold green]Excelente[/bold green]"
        elif val >= minima:
            return "[bold yellow]Bom[/bold yellow]"
        return "[bold red]Abaixo do Mínimo[/bold red]"

    recall_status = get_status(avg_recall, 0.70, 0.85)
    mrr_status = get_status(avg_mrr, 0.50, 0.70)

    console.print(Panel(
        f"[bold]Resultados Médios (Dataset com {num_queries} perguntas):[/bold]\n\n"
        f"  • [bold]Mean Recall@{k}:[/bold] {avg_recall:.2%} — Status: {recall_status}\n"
        f"  • [bold]Mean MRR@{k}:[/bold]    {avg_mrr:.2f} — Status: {mrr_status}\n\n"
        "[dim]Métricas de Referência (Slide):\n"
        "  Recall@5: Mínima > 0.70 | Boa > 0.85 | Excelente > 0.95\n"
        "  MRR@5:    Mínima > 0.50 | Boa > 0.70 | Excelente > 0.85[/dim]",
        title="📊 Resumo Final",
        border_style="green"
    ))

if __name__ == "__main__":
    run_evaluation(k=5)
