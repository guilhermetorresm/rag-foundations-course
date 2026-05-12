"""
api.py — Servidor FastAPI para o RAG
=====================================
Expõe o pipeline RAG como uma REST API, permitindo
integração com frontends, outros serviços ou clientes HTTP.

Endpoints:
    GET  /health        → Status do servidor e Ollama
    GET  /documents     → Lista documentos indexados
    POST /ask           → Faz uma pergunta ao RAG
    POST /ingest        → Inicia ingestão de novos documentos
    DELETE /collection  → Limpa a coleção do ChromaDB

Execução:
    uv run uvicorn api:app --reload --port 8000
    # ou: python api.py
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from importlib import import_module

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import DATA_DIR, chroma_cfg, ollama_cfg

# Imports dinâmicos para evitar conflito com nomes numéricos
_indexing  = import_module("2_indexing")
_retrieval = import_module("3_retrieval")
_generation = import_module("4_generation")
_ingestion = import_module("1_ingestion")


# ─────────────────────────────────────────────
# Inicialização da Aplicação
# ─────────────────────────────────────────────

app = FastAPI(
    title="RAG Foundations Course API",
    description="API REST para o pipeline de Retrieval-Augmented Generation do curso.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global da aplicação (compartilhado entre requests)
_state: dict[str, Any] = {}


def get_state() -> dict[str, Any]:
    """Inicializa e retorna o estado da aplicação (lazy loading)."""
    if not _state:
        _state["client"] = _indexing.get_chroma_client()
        _state["collection"] = _indexing.get_or_create_collection(_state["client"])
        _state["embed_model"] = _indexing.load_embedding_model()
    return _state


# ─────────────────────────────────────────────
# Schemas Pydantic (Request / Response)
# ─────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """Schema para a requisição de pergunta ao RAG."""
    query: str = Field(..., min_length=3, max_length=1000, description="Pergunta do usuário")
    top_k: int = Field(default=5, ge=1, le=20, description="Número de chunks a recuperar")
    use_rerank: bool = Field(default=False, description="Ativar re-ranking com Cross-Encoder")


class QuestionResponse(BaseModel):
    """Schema para a resposta do pipeline RAG."""
    answer: str
    query: str
    model: str
    sources: list[dict[str, Any]]
    chunks_retrieved: int


class HealthResponse(BaseModel):
    """Schema para o health check."""
    status: str
    collection_count: int
    ollama_available: bool
    model: str


class IngestResponse(BaseModel):
    """Schema para a resposta da ingestão de documentos."""
    message: str
    chunks_indexed: int


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check() -> HealthResponse:
    """
    Verifica o status do servidor, ChromaDB e Ollama.

    Retorna:
        - Status geral do sistema
        - Quantidade de documentos indexados
        - Disponibilidade do Ollama e modelo configurado
    """
    state = get_state()
    return HealthResponse(
        status="ok",
        collection_count=state["collection"].count(),
        ollama_available=_generation.check_ollama_available(),
        model=ollama_cfg.model,
    )


@app.get("/documents", tags=["Documentos"])
async def list_documents() -> dict[str, Any]:
    """
    Lista todos os documentos indexados na coleção ChromaDB.

    Retorna metadados de cada chunk (fonte, página, etc.)
    sem o conteúdo completo para não sobrecarregar a resposta.
    """
    state = get_state()
    collection = state["collection"]

    total = collection.count()
    if total == 0:
        return {"total": 0, "documents": [], "sources": []}

    results = collection.get(include=["metadatas"], limit=1000)
    metadatas = results.get("metadatas") or []

    # Agrupa por fonte para resumo
    sources: dict[str, set[int]] = {}
    for meta in metadatas:
        src = Path(meta.get("source", "desconhecido")).name
        page = int(meta.get("page", 0))
        sources.setdefault(src, set()).add(page)

    return {
        "total": total,
        "sources": [
            {"file": src, "pages": sorted(pages)}
            for src, pages in sources.items()
        ],
    }


@app.post("/ask", response_model=QuestionResponse, tags=["RAG"])
async def ask(request: QuestionRequest) -> QuestionResponse:
    """
    Executa o pipeline RAG completo para uma pergunta.

    Pipeline:
        1. Busca vetorial (Top-K chunks relevantes)
        2. Re-ranking opcional com Cross-Encoder
        3. Geração de resposta com Ollama
        4. Retorna resposta + fontes

    Args:
        request: Pergunta e parâmetros opcionais.
    """
    state = get_state()

    if state["collection"].count() == 0:
        raise HTTPException(
            status_code=422,
            detail="Nenhum documento indexado. Execute /ingest primeiro.",
        )

    from config import RetrievalConfig
    cfg = RetrievalConfig(top_k=request.top_k)

    # 1. Retrieval
    chunks = _retrieval.search(
        query=request.query,
        collection=state["collection"],
        cfg=cfg,
    )

    # 2. Guard Clause: Contexto vazio (Slide 6)
    if not chunks:
        return QuestionResponse(
            answer="Informação não encontrada nos documentos oficiais.",
            query=request.query,
            model=ollama_cfg.model,
            sources=[],
            chunks_retrieved=0
        )

    # 3. Formatar contexto com Token Budget (Slide 8)
    context = _generation.format_context_with_budget(chunks)

    # 4. Gerar resposta
    try:
        rag_response = _generation.generate_answer(request.query, context)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # 5. Montar resposta com Fontes (Slide 9)
    sources = [
        {
            "source": Path(chunk.source).name,
            "page": chunk.page,
            "score": round(chunk.score, 3),
        }
        for chunk in chunks
    ]

    return QuestionResponse(
        answer=rag_response.answer,
        query=request.query,
        model=rag_response.model,
        sources=sources,
        chunks_retrieved=len(chunks),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Documentos"])
async def ingest_documents(background_tasks: BackgroundTasks) -> IngestResponse:
    """
    Inicia a ingestão de todos os documentos do diretório data/.

    Processa PDFs e TXTs, gera embeddings e indexa no ChromaDB.
    A operação é executada em background para não bloquear a API.
    """
    state = get_state()

    chunks = _ingestion.ingest_directory(DATA_DIR)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"Nenhum documento encontrado em {DATA_DIR}",
        )

    total = _indexing.index_chunks(
        chunks=chunks,
        collection=state["collection"],
    )

    return IngestResponse(
        message=f"Ingestão concluída com sucesso!",
        chunks_indexed=total,
    )


@app.delete("/collection", tags=["Sistema"])
async def clear_collection() -> dict[str, str]:
    """
    Remove todos os documentos da coleção ChromaDB.

    ⚠️ Operação destrutiva! Use com cuidado.
    """
    state = get_state()
    collection_name = chroma_cfg.collection_name

    state["client"].delete_collection(collection_name)
    state["collection"] = _indexing.get_or_create_collection(state["client"])

    return {"message": f"Coleção '{collection_name}' foi limpa com sucesso."}


# ─────────────────────────────────────────────
# Execução Standalone
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
