# RAG Foundations Course

Repositório do curso prático de RAG (Retrieval-Augmented Generation).

## 🗂 Estrutura do Projeto

```
rag-foundations-course/
├── pyproject.toml              # Gerenciado pelo uv
├── .venv/                      # Ambiente virtual (gerenciado pelo uv)
├── roteiros/
│   ├── roteiro_bloco_1.md      # Script do Bloco 1: Setup + Arquitetura
│   └── roteiro_pratica.md      # Script da prática hands-on (Blocos 2-5)
├── slides/                     # Slides do curso (adicione aqui)
├── assets/                     # Imagens e recursos visuais
├── data/                       # Documentos para ingestão (PDFs, TXTs)
│   └── regimento_interno.txt   # Exemplo: política interna fictícia
└── src/
    ├── config.py               # Configurações centrais (Ollama, ChromaDB)
    ├── 1_ingestion.py          # Bloco 1: Limpeza de texto e chunking
    ├── 2_indexing.py           # Bloco 2: Embeddings e inserção no ChromaDB
    ├── 3_retrieval.py          # Bloco 3: Busca vetorial Top-K e re-rank
    ├── 4_generation.py         # Bloco 4: Prompt RAG e chamada ao Ollama
    ├── api.py                  # Servidor FastAPI (REST API)
    └── cli.py                  # Interface de terminal interativa
```

## 🛠 Pré-requisitos

- **Python** 3.14+
- **uv** (gerenciador de pacotes) — [instalar](https://docs.astral.sh/uv/)
- **Ollama** — [instalar](https://ollama.ai)

## 🚀 Início Rápido

### 1. Instalar dependências

```bash
uv sync
```

### 2. Configurar o Ollama

```bash
# Baixar o modelo (primeira vez)
ollama pull llama3.2

# Iniciar o servidor Ollama (manter rodando em outro terminal)
ollama serve
```

### 3. Adicionar documentos

Coloque seus arquivos `.pdf`, `.txt` ou `.md` na pasta `data/`.

Um documento de exemplo já está incluído: `data/regimento_interno.txt`.

### 4. Usar a CLI Interativa

```bash
uv run python src/cli.py
```

O menu vai guiá-lo por:
- 📥 Ingerir documentos
- 🤖 Fazer perguntas ao RAG
- 📚 Listar documentos indexados
- ⚙ Ver configurações

### 5. Usar a API REST (opcional)

```bash
uv run uvicorn src.api:app --reload --port 8000
```

Acesse a documentação interativa em: **http://localhost:8000/docs**

## 🔧 Configuração

Edite `src/config.py` para personalizar:

```python
# Modelo do Ollama
model: str = "llama3.2"           # Troque por "llama3.1:8b", "mistral", etc.

# Parâmetros de chunking
chunk_size: int = 512              # Caracteres por chunk
chunk_overlap: int = 64            # Sobreposição entre chunks

# Parâmetros de retrieval
top_k: int = 5                     # Chunks recuperados por query
similarity_threshold: float = 0.3  # Score mínimo (0.0 a 1.0)
```

## 🚀 Demonstrações do Bloco 1

Para acompanhar os exemplos dos slides, utilize os scripts abaixo:

1. **Demonstração de Alucinação vs RAG**:
   Veja na prática o LLM inventando informações vs o RAG sendo honesto.
   ```bash
   uv run python src/demo_hallucination.py
   ```

2. **Teste de Embeddings Locais (Ollama)**:
   Verifique se o seu stack do Ollama está configurado corretamente.
   ```bash
   uv run python src/embeddings.py
   ```

3. **Pipeline Completo (Modo Didático)**:
   Siga a sequência 1 a 4 nos arquivos `src/`.

## 📖 Executar Cada Bloco Individualmente

```bash
uv run python src/1_ingestion.py   # Bloco 1: ingestão e chunking
uv run python src/2_indexing.py    # Bloco 2: embeddings e indexação
uv run python src/3_retrieval.py   # Bloco 3: busca vetorial
uv run python src/4_generation.py  # Bloco 4: geração com Ollama
```

## 🏗 Tech Stack

| Componente | Tecnologia | Por quê? |
|------------|-----------|----------|
| Linguagem | Python 3.14+ | Type hints modernos, performance |
| Gerenciador | uv (Astral) | Resolução rápida, lockfile, venvs |
| LLM Local | Ollama + Llama3.2 | Privacidade, sem custo de API |
| Banco Vetorial | ChromaDB | Simples, persistente, sem servidor |
| Embeddings | sentence-transformers | Rápido, roda local, multilíngue |
| API | FastAPI + uvicorn | Async, docs automáticas, Pydantic |
| CLI | rich + questionary | Terminal bonito e interativo |
| PDF | pypdf | Leve, sem dependências pesadas |

## 📝 Licença

MIT — use livremente para aprendizado e projetos comerciais.
