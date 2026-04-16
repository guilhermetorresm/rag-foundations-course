# Roteiro — Bloco 1: Fundamentos e Configuração do Ambiente
## Curso: RAG Foundations (4 horas)
### Duração deste bloco: ~45 minutos

---

## 🎯 Objetivos de Aprendizagem

Ao final deste bloco, o aluno será capaz de:
1. Explicar o que é RAG e por que ele resolve alucinações em LLMs
2. Configurar o ambiente de desenvolvimento com `uv` e Python 3.14+
3. Rodar o Ollama localmente e validar a conexão
4. Compreender o pipeline RAG de ponta a ponta

---

## ⏱ Agenda do Bloco 1 (45 min)

| Tempo | Tópico |
|-------|--------|
| 00:00 - 05:00 | Apresentação e contexto: Por que RAG? |
| 05:00 - 15:00 | Setup do ambiente: uv, Python, Ollama |
| 15:00 - 30:00 | Arquitetura RAG: Diagrama e componentes |
| 30:00 - 45:00 | Live coding: config.py + execução do Ollama |

---

## 📝 Script Detalhado

### Abertura (0-5 min)

> **[FALAR]** Bom dia/tarde! Bem-vindos ao RAG Foundations Course.
> Hoje vamos construir um sistema RAG completo, do zero, em 4 horas.
> Por que RAG? Alguém aqui já tentou perguntar a um ChatGPT sobre
> documentos internos da empresa e recebeu uma resposta inventada?
> Isso se chama alucinação — e RAG é a solução industrial para esse problema.

**[SLIDE: O Problema das Alucinações]**
- LLMs como Llama3 treinam com dados até uma data de corte
- Não conhecem sua documentação interna
- Quando não sabem, inventam (com muita confiança!)

**[SLIDE: A Solução — RAG]**
- **R**etrieval: busca documentos relevantes
- **A**ugmented: aumenta o prompt com esses documentos
- **G**eneration: LLM gera resposta fundamentada nos docs

### Setup do Ambiente (5-15 min)

```bash
# 1. Instalar uv (gerenciador moderno de Python)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
curl -LsSf https://astral.sh/uv/install.sh | sh                # Mac/Linux

# 2. Criar o projeto
uv init rag-foundations-course
cd rag-foundations-course

# 3. Instalar dependências
uv add fastapi uvicorn chromadb ollama pypdf rich questionary sentence-transformers

# 4. Instalar e iniciar Ollama
# Baixe em: https://ollama.ai
ollama pull llama3.2
ollama serve  # em outro terminal
```

> **[FALAR]** O `uv` é o substituto moderno do pip + virtualenv.
> Ele resolve dependências em milissegundos e gerencia versões do Python.
> É o que projetos sérios de Python usam em 2025.

### Arquitetura RAG (15-30 min)

**[DIAGRAMA NO SLIDE]**

```
┌─────────────┐     ┌─────────────┐     ┌────────────────┐
│  Documentos │────▶│  Embeddings │────▶│   ChromaDB     │
│  (PDF, TXT) │     │  (vetores)  │     │  (banco vetor) │
└─────────────┘     └─────────────┘     └────────────────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌─────────────┐     ┌────────────────┐
│   Usuário   │────▶│    Query    │────▶│  Busca Top-K   │
│  (pergunta) │     │  (vetoriza) │     │  (similaridade)│
└─────────────┘     └─────────────┘     └────────────────┘
                                                  │
                                    ┌─────────────▼──────┐
                                    │       Prompt RAG    │
                                    │  [System + Context  │
                                    │   + Pergunta]       │
                                    └─────────────────────┘
                                                  │
                                    ┌─────────────▼──────┐
                                    │      Ollama         │
                                    │  (Llama3.2 local)  │
                                    └─────────────────────┘
                                                  │
                                    ┌─────────────▼──────┐
                                    │      Resposta       │
                                    │  (fundamentada)     │
                                    └─────────────────────┘
```

**[PERGUNTAR À TURMA]**
- Qual componente é o "cérebro"? (Ollama/LLM)
- Qual é o "arquivo"? (ChromaDB)
- O que é chunking? Por que precisamos disso?

### Live Coding: config.py (30-45 min)

> **[COMPARTILHAR TELA — ABRIR `src/config.py`]**

Pontos-chave para comentar:
1. **`@dataclass(frozen=True)`**: imutável por design → evita bugs de estado
2. **Type hints**: documentação que o Python verifica em tempo de execução
3. **Instâncias padrão**: `ollama_cfg`, `chroma_cfg` → padrão de singleton simples

```python
# Mostrar ao vivo:
from config import ollama_cfg
print(ollama_cfg.base_url)  # http://localhost:11434
print(ollama_cfg.model)     # llama3.2
```

---

## ❓ Q&A e Transição (5 min finais)

Perguntas para engajar:
1. "Por que usar ChromaDB e não um banco SQL com índice de texto?"
2. "O Ollama roda na GPU? E se eu não tiver GPU?"
3. "Posso usar a API da OpenAI em vez do Ollama?"

**[TRANSIÇÃO]** No Bloco 2, vamos mergulhar na ingestão: como transformar
PDFs bagunçados em chunks limpos e vetores prontos para busca.

---

## 📚 Recursos Adicionais

- [Documentação do uv](https://docs.astral.sh/uv/)
- [Ollama Models](https://ollama.ai/library)
- [ChromaDB Docs](https://docs.trychroma.com)
- [Sentence Transformers](https://www.sbert.net)
