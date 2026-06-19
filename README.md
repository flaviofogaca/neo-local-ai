# Neo

Neo é um assistente pessoal local desenvolvido com FastAPI, Ollama e ChromaDB.

O objetivo do projeto é criar um agente capaz de manter contexto, armazenar memórias relevantes e evoluir progressivamente para se tornar um assistente pessoal completo.

---

## Funcionalidades

### Conversação

- Chat em tempo real
- Streaming de respostas
- Histórico de conversas
- Renomeação automática de chats
- Interface moderna inspirada em aplicações de IA

### Memória

- Memória persistente em JSON
- Memória vetorial com ChromaDB
- Recuperação contextual (RAG)
- Busca inteligente de contexto
- Filtro de relevância para evitar memórias incorretas

### Interface

- Glassmorphism
- Sidebar de conversas
- Empty State
- Scrollbar personalizada
- Avatares
- Animações suaves

### Voz

- Speech Recognition
- Ditado direto para o campo de mensagem

---

## Arquitetura

```
Neo
│
├── FastAPI
├── Ollama
│   └── gemma3:12b
│
├── ChromaDB
│   └── Memória vetorial
│
├── JSON Memory
│   └── Perfil persistente
│
└── Frontend
    ├── HTML
    ├── CSS
    └── JavaScript
```

---

## Estrutura do Projeto

```text
Neo Memory/
│
├── app.py
├── vector_memory.py
├── memory.json
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── tests/
│
├── conversations/
│
├── chroma_db/
│
└── legacy/
```

---

## Tecnologias

- Python
- FastAPI
- Ollama
- Gemma 3 12B
- ChromaDB
- HTML
- CSS
- JavaScript

---

## Roadmap V2

### Inteligência

- [ ] Deep Mode (Gemma 26B)
- [ ] Consciência temporal
- [ ] Memória episódica
- [ ] Skills dinâmicas

### Desenvolvimento

- [ ] Leitura de repositórios GitHub
- [ ] Leitura de código local
- [ ] Sugestão automática de melhorias
- [ ] Edição assistida de arquivos

### Integrações

- [ ] N8N
- [ ] Monitoramento de e-mails
- [ ] Calendário
- [ ] Notificações

---

## Status

Neo V1.0 — Stable Release