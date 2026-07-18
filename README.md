# Neo

> A private AI assistant that remembers, understands and evolves.

Neo é um assistente pessoal local focado em privacidade, memória persistente e inteligência para desenvolvimento de software.

Construído com FastAPI, Ollama e ChromaDB, o projeto evolui continuamente para se tornar um agente capaz de compreender contexto, código-fonte e histórico de conversas sem depender de serviços em nuvem.

O objetivo do projeto é criar um agente capaz de manter contexto, armazenar memórias relevantes e evoluir progressivamente para se tornar um assistente pessoal completo.

---

## Funcionalidades

### 💬 Conversação

- Chat em tempo real
- Streaming de respostas
- Histórico de conversas
- Renomeação automática
- Deep Mode

### 🧠 Memória

- Memória persistente em JSON
- Memória vetorial
- Recuperação contextual (RAG)
- Busca semântica
- Filtro de relevância
- Proteção contra Prompt Injection

### 💻 Inteligência para Repositórios

- Repository Reader
- Repository Indexer
- Chunking inteligente
- Embeddings de código
- Busca híbrida (vetorial + literal)
- Ranking de resultados

### 🎙 Interface

- Glassmorphism
- Empty State
- Sidebar
- Reconhecimento de voz
- Auto Resize
- Markdown

### Voz

- Speech Recognition
- Ditado direto para o campo de mensagem

---

Usuário
    │
    ▼
Frontend (HTML / JS)
    │
    ▼
FastAPI
    │
    ├───────────────┐
    ▼               ▼
Conversation     Repository
 History          Search
    │               │
    ▼               ▼
Vector Memory   Repository Indexer
    │               │
    └──────┬────────┘
           ▼
        ChromaDB
           │
           ▼
         Ollama

---

Neo Memory/
│
├── app.py
├── vector_memory.py
├── repo_reader.py
├── repo_indexer.py
├── memory.json
├── requirements.txt
│
├── static/
├── templates/
├── conversations/
├── chroma_db/
│
├── tests/
└── README.md

---

## Tecnologias

- Python 3
- FastAPI
- Ollama
- Gemma 3 / Gemma 4
- ChromaDB
- HTML
- CSS
- JavaScript
- Markdown
- Fetch API

---

## Roadmap

### ✅ Sprint 1

- Chat
- Histórico
- Memória JSON

### ✅ Sprint 2

- ChromaDB
- Memória Vetorial
- RAG

### ✅ Sprint 3

- Streaming
- Deep Mode
- Prompt Injection

### ✅ Sprint 4

- Repository Reader
- Repository Indexer
- Hybrid Search

### 🚧 Sprint 5

- Repository Reasoner
- Repository Chat
- Repository Explain
- Repository Refactor

### 🔮 Futuro

- GitHub Integration
- Local File Editing
- Multi-Agent
- Skills
- Calendário
- Email
- N8N

---

## Status

Versão Atual

Neo V1.1

### Principais recursos

✅ Streaming

✅ Deep Mode

✅ Memória Vetorial

✅ Repository Intelligence Engine V1

🚧 Repository Reasoner

🚧 GitHub Intelligence

🚧 Multi-Agent