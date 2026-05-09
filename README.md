# 🧠 Neo Local AI

Neo é um assistente pessoal com IA rodando localmente via Ollama, com memória persistente, histórico de conversas, streaming de respostas e interface web própria.

O projeto foi desenvolvido com foco em aprendizado prático de arquitetura de aplicações com IA, integração frontend/backend e gerenciamento de contexto conversacional.

---

# 🚀 Features

✅ Interface web própria  
✅ Integração com Ollama  
✅ Streaming de respostas em tempo real  
✅ Memória persistente (`memory.json`)  
✅ Histórico contextual de conversa  
✅ Salvamento automático de conversas  
✅ Sidebar com gerenciamento de chats  
✅ Renomear conversas  
✅ Apagar conversas  
✅ Markdown nas respostas  
✅ Conversas persistidas em JSON  

---

# 🛠️ Stack utilizada

## Backend
- Python
- FastAPI
- Uvicorn

## Frontend
- HTML
- CSS
- JavaScript

## IA Local
- Ollama
- Gemma 4 26B

---

# 📂 Estrutura do projeto

```bash
Neo Memory/
│
├── conversations/        # Conversas salvas
├── templates/
│   └── index.html
│
├── app.py                # Backend FastAPI
├── memory.json           # Memória persistente
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Instalação

## 1. Clone o repositório

```bash
git clone https://github.com/SEU-USUARIO/neo-local-ai.git
```

---

## 2. Entre na pasta

```bash
cd neo-local-ai
```

---

## 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 4. Instale o Ollama

https://ollama.com/

---

## 5. Baixe um modelo

Exemplo:

```bash
ollama pull gemma4:26b
```

---

## 6. Rode o projeto

```bash
python -m uvicorn app:app --reload
```

---

# 🌐 Acesso

Após iniciar:

```txt
http://127.0.0.1:8000
```

---

# 🧠 Como funciona a memória

O Neo utiliza dois tipos de memória:

## 🔵 Memória persistente
Arquivo:

```txt
memory.json
```

Responsável por:
- nome do usuário
- objetivos
- skills
- preferências
- informações permanentes

---

## 🟣 Memória contextual
Histórico recente da conversa enviado junto ao prompt.

Responsável por:
- continuidade da conversa
- contexto atual
- referências recentes

---

# 💾 Conversas

As conversas são salvas automaticamente na pasta:

```txt
conversations/
```

Cada conversa é persistida em JSON.

---

# 🔥 Roadmap

## Próximas funcionalidades

- [ ] Auto-save contínuo
- [ ] Conversa ativa persistente
- [ ] Upload de arquivos
- [ ] Busca semântica
- [ ] Embeddings
- [ ] Multi-model support
- [ ] Voz/TTS
- [ ] Tema estilo ChatGPT
- [ ] Sistema de plugins/tools

---

# 🎯 Objetivo do projeto

O Neo foi criado como projeto de estudo e experimentação prática de:
- aplicações com IA local
- arquitetura fullstack
- engenharia de prompts
- gerenciamento de memória contextual
- UX para assistentes conversacionais

---

# 📜 Licença

Projeto para fins educacionais e experimentação pessoal.