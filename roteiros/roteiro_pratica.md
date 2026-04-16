# Roteiro — Prática Hands-On: Pipeline RAG Completo
## Curso: RAG Foundations (4 horas)
### Duração: ~2h30 (Blocos 2, 3 e 4 combinados)

---

## 🎯 O que vamos construir

Um sistema RAG funcional que:
- Ingere PDFs e TXTs reais
- Armazena vetores no ChromaDB
- Busca por similaridade semântica
- Gera respostas com Llama3.2 via Ollama
- Tem uma CLI elegante com rich + questionary
- Tem uma API REST com FastAPI

---

## Bloco 2 — Ingestão e Chunking (45 min)

### Conceito: Por que chunking?

> **[FALAR]** Um LLM tem um limite de tokens no contexto.
> O Llama3.2 suporta ~128k tokens, mas enviar um PDF inteiro é
> ineficiente e dilui a relevância. Por isso dividimos em pedaços
> (chunks) e só enviamos os mais relevantes pra cada pergunta.

**Experimento ao vivo:**
```bash
# Rodar o script de ingestão standalone
uv run python src/1_ingestion.py
```

**Pontos para comentar no código:**

1. **`clean_text()`**: Normalização Unicode — por que isso importa?
   ```python
   # "café" em NFD = 'cafe\u0301' (5 chars)
   # "café" em NFC = 'caf\xe9' (4 chars)  ← queremos isso
   ```

2. **`split_into_chunks()` — Sliding Window**:
   ```
   chunk_size = 512, overlap = 64
   
   [  chunk 1 (512)  ]
                 [  chunk 2 (512)  ]
                               [  chunk 3  ]
         ←─64─→               ←─ overlap ─→
   ```
   O overlap garante que uma informação no "meio" de dois chunks
   não seja perdida.

3. **`@dataclass`**: `Chunk` é imutável por padrão — boa prática.

### Exercício (10 min)
> Adicione suporte a arquivos `.docx` na função `load_document()`.
> Dica: use a biblioteca `python-docx`.

---

## Bloco 3 — Embeddings e Indexação (40 min)

### Conceito: O que é um embedding?

> **[FALAR]** Um embedding transforma texto em coordenadas num espaço
> de 384 dimensões (modelo all-MiniLM-L6-v2). Textos com significados
> próximos ficam próximos nesse espaço — como pontos num mapa.

**Demonstração visual:**
```python
# Rodar interativamente
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

v1 = model.encode("o cachorro late")
v2 = model.encode("o cão faz barulho")
v3 = model.encode("a bolsa de valores caiu")

from numpy import dot
from numpy.linalg import norm

def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))

print(cosine(v1, v2))  # ~0.85 — muito similares!
print(cosine(v1, v3))  # ~0.20 — bem diferentes
```

**Rodando a indexação:**
```bash
uv run python src/2_indexing.py
```

**Pontos para comentar:**
1. **`upsert` vs `insert`**: re-indexação segura (idempotente)
2. **Batches no modelo**: por que processar em lotes?
3. **Persistência do ChromaDB**: onde ficam os dados no disco?

---

## Bloco 4 — Retrieval e Geração (45 min)

### Retrieval: Bi-Encoder vs Cross-Encoder

```
              Velocidade ←──────────────────→ Precisão
              
  Keyword     Bi-Encoder          Cross-Encoder
  Search      (Embeddings)        (Re-ranking)
     │              │                   │
    TF-IDF       Rápido,            Lento, mas
   BM25++       bom recall         alta precisão
                                   
Strategy:    Bi-Encoder (Top-20) → Cross-Encoder (Top-5)
```

### Live Coding: Retrieval

```bash
uv run python src/3_retrieval.py
```

Perguntar à turma: "Como o score de 0.85 é calculado?"
- Mostrar a fórmula: `score = 1 - (cosine_distance / 2)`
- Por que dividir por 2? Distância cosine ∈ [0, 2]

### Live Coding: Geração

```bash
uv run python src/4_generation.py
```

**Anatomia do Prompt RAG:**
```
[SYSTEM]  → Regras do assistente
              "Responda SOMENTE com base nos trechos..."

[USER]    → Contexto recuperado + Pergunta
              "[Trecho 1 | Fonte: doc.pdf | Pág. 3]
               Artigo 5: Os funcionários têm direito...
               
               PERGUNTA: Quantos dias de férias tenho?"
```

> **[PERGUNTA]** O que acontece se eu mudar o system prompt para
> "Você pode usar seu conhecimento geral além dos documentos"?
> Como isso afeta o comportamento do modelo?

---

## Bloco 5 — Demo Completa: CLI + API (30 min)

### A CLI

```bash
# Rodar a CLI interativa
uv run python src/cli.py
```

Mostrar:
1. Menu com setas do questionary
2. Ingestão com progress bar
3. Pergunta com spinner do rich
4. Resposta em Markdown renderizado

### A API FastAPI

```bash
# Iniciar o servidor
uv run uvicorn src.api:app --reload --port 8000
```

Acessar `http://localhost:8000/docs` e demonstrar:
1. `GET /health` — status do sistema
2. `POST /ingest` — ingerir documentos
3. `POST /ask` — fazer pergunta

```bash
# Testar via curl
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Quantos dias de férias tenho direito?", "top_k": 3}'
```

---

## 🏁 Encerramento (10 min)

### O que construímos hoje:

| Componente | Arquivo | Tecnologia |
|------------|---------|-----------|
| Ingestão | `1_ingestion.py` | pypdf, re, unicodedata |
| Indexação | `2_indexing.py` | sentence-transformers, ChromaDB |
| Retrieval | `3_retrieval.py` | ChromaDB, CrossEncoder |
| Geração | `4_generation.py` | Ollama (Llama3.2) |
| API | `api.py` | FastAPI |
| CLI | `cli.py` | rich, questionary |

### Próximos Passos (para depois do curso):

1. **Avaliação de RAG**: RAGAS framework para medir qualidade
2. **Hybrid Search**: combinar busca vetorial + BM25
3. **Metadata Filtering**: filtrar por data, autor, tipo de doc
4. **Streaming**: respostas em tempo real na API
5. **Multi-modal**: RAG com imagens e PDFs scaneados (OCR)

---

## 📚 Recursos para Continuar

- [LangChain Docs](https://python.langchain.com) — framework para RAG avançado
- [LlamaIndex](https://llamaindex.ai) — alternativa ao LangChain
- [RAGAS](https://ragas.io) — avaliação de sistemas RAG
- [Qdrant](https://qdrant.tech) — banco vetorial alternativo ao ChromaDB
- [Paper: RAG Survey (2024)](https://arxiv.org/abs/2312.10997) — estado da arte

---

*Obrigado pela participação! 🚀*
