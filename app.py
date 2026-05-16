import json
import requests
import webbrowser
import threading
import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:26b"

app = FastAPI()

conversation_history = []
active_conversation_file = None

CONVERSATIONS_DIR = "conversations"
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)


def load_memory():
    with open("memory.json", "r", encoding="utf-8") as file:
        return json.load(file)


def save_memory(new_data):
    memory = load_memory()
    memory.update(new_data)

    with open("memory.json", "w", encoding="utf-8") as file:
        json.dump(memory, file, ensure_ascii=False, indent=2)


def create_conversation_file():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"conversation_{timestamp}.json"
    filepath = os.path.join(CONVERSATIONS_DIR, filename)

    data = {"created_at": timestamp, "messages": []}

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return filename


def save_active_conversation():
    global active_conversation_file

    if not active_conversation_file:
        active_conversation_file = create_conversation_file()

    filepath = os.path.join(CONVERSATIONS_DIR, active_conversation_file)

    data = {
        "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "messages": conversation_history,
    }

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return active_conversation_file


def build_history_text(history=None):
    history = history or []
    recent_history = history[-10:]

    history_text = ""

    for item in recent_history:
        role = item.get("role") if isinstance(item, dict) else item.role
        content = item.get("content") if isinstance(item, dict) else item.content

        speaker = "Você" if role == "user" else "Neo"
        history_text += f"{speaker}: {content}\n"

    return history_text


def build_prompt(user_message, history=None):
    memory = load_memory()
    history_text = build_history_text(history)

    system_prompt = f"""
Você é o Neo, assistente pessoal do Flávio.

Use estas informações como memória persistente interna:
{json.dumps(memory, ensure_ascii=False, indent=2)}

Regras de uso da memória:
- Use a memória apenas quando ela for relevante para responder ao pedido do usuário.
- Não mencione informações da memória sem necessidade.
- Se o usuário apenas cumprimentar, cumprimente de volta de forma breve e natural.
- Não faça resumo do perfil, skills, projetos, localização ou objetivos a menos que o usuário peça.
- Se houver conflito entre memória persistente e conversa recente, priorize a conversa recente.

Regras de resposta:
- Responda em português do Brasil.
- Seja direto, parceiro e prático.
- Não use a memória como apresentação automática.
- Se o usuário fizer uma saudação simples, responda apenas a saudação de forma natural.

Se o usuário pedir para salvar algo, responda APENAS em JSON no formato:

{{
  "action": "save_memory",
  "data": {{
    "campo": "valor"
  }}
}}

Se for conversa normal, responda normalmente.

Nunca misture texto com JSON.
"""

    return (
        system_prompt
        + "\n\nCONVERSA RECENTE, use como contexto principal quando necessário:\n"
        + history_text
        + "\n\nMENSAGEM ATUAL DO USUÁRIO:\n"
        + user_message
        + "\n\nResponda mantendo continuidade com a conversa recente. "
        + "Se houver conflito entre memória fixa e conversa recente, priorize a conversa recente.\n\nNeo:"
    )


def handle_memory_action(response_text):
    try:
        parsed = json.loads(response_text)

        if parsed.get("action") == "save_memory":
            save_memory(parsed["data"])
            return "🧠 Memória salva com sucesso!"

    except json.JSONDecodeError:
        pass

    return response_text


def ask_ollama(user_message, history=None):
    prompt = build_prompt(user_message, history)

    payload = {"model": MODEL, "prompt": prompt, "stream": False}

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    response_text = response.json()["response"]

    return handle_memory_action(response_text)


def generate_conversation_title(first_message):

    prompt = f"""
Crie um título curto para uma conversa.

Regras:
- máximo 5 palavras
- sem aspas
- sem ponto final
- responda apenas o título

Mensagem:
{first_message}
"""

    payload = {"model": MODEL, "prompt": prompt, "stream": False}

    try:

        response = requests.post(OLLAMA_URL, json=payload)

        response.raise_for_status()

        title = response.json()["response"].strip()

        invalid_chars = ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]

        for char in invalid_chars:
            title = title.replace(char, "")

        return title[:60]

    except:
        return None


def auto_rename_conversation(first_message):

    global active_conversation_file

    if not active_conversation_file:
        return

    if not active_conversation_file.startswith("conversation_"):
        return

    title = generate_conversation_title(first_message)

    if not title:
        return

    new_filename = f"{title}.json"

    old_path = os.path.join(CONVERSATIONS_DIR, active_conversation_file)

    new_path = os.path.join(CONVERSATIONS_DIR, new_filename)

    counter = 1

    while os.path.exists(new_path):

        new_filename = f"{title}_{counter}.json"

        new_path = os.path.join(CONVERSATIONS_DIR, new_filename)

        counter += 1

    os.rename(old_path, new_path)

    active_conversation_file = new_filename


def stream_ollama(user_message, history=None):
    global conversation_history
    global active_conversation_file

    if not active_conversation_file:
        active_conversation_file = create_conversation_file()

    prompt = build_prompt(user_message, history)

    payload = {"model": MODEL, "prompt": prompt, "stream": True}

    full_response = ""

    with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))
            chunk = data.get("response", "")

            if chunk:
                full_response += chunk
                yield chunk

            if data.get("done"):
                final_response = handle_memory_action(full_response)

                if final_response != full_response:
                    yield final_response

                conversation_history.append({"role": "user", "content": user_message})

                conversation_history.append({"role": "neo", "content": final_response})

                while len(conversation_history) > 10:
                    conversation_history.pop(0)

                save_active_conversation()


                if len(conversation_history) == 2:
                    auto_rename_conversation(user_message)

                break


@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())


@app.post("/chat")
def chat(request: ChatRequest):
    global conversation_history
    global active_conversation_file

    if not active_conversation_file:
        active_conversation_file = create_conversation_file()

    response = ask_ollama(request.message, conversation_history)

    conversation_history.append({"role": "user", "content": request.message})

    conversation_history.append({"role": "neo", "content": response})

    conversation_history = conversation_history[-10:]
    save_active_conversation()

    return {"response": response, "active_conversation": active_conversation_file}


@app.post("/chat-stream")
def chat_stream(request: ChatRequest):
    return StreamingResponse(
        stream_ollama(request.message, conversation_history), media_type="text/plain"
    )


@app.get("/memory")
def memory():
    return load_memory()


@app.post("/new-chat")
def new_chat():
    global conversation_history
    global active_conversation_file

    conversation_history = []
    active_conversation_file = None

    return {"status": "ok"}


@app.post("/rename-conversation")
def rename_conversation(data: dict):
    global active_conversation_file

    old_name = data.get("old_name")
    new_name = data.get("new_name")

    if not old_name or not new_name:
        return {"status": "error"}

    old_path = os.path.join(CONVERSATIONS_DIR, old_name)

    if not new_name.endswith(".json"):
        new_name += ".json"

    new_path = os.path.join(CONVERSATIONS_DIR, new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)

        if active_conversation_file == old_name:
            active_conversation_file = new_name

    return {"status": "ok", "new_name": new_name}


@app.get("/conversations")
def get_conversations():
    files = os.listdir(CONVERSATIONS_DIR)
    files.sort(reverse=True)

    return {"conversations": files, "active_conversation": active_conversation_file}


@app.get("/conversations/{filename}")
def get_conversation(filename: str):
    global conversation_history
    global active_conversation_file

    filepath = os.path.join(CONVERSATIONS_DIR, filename)

    if not os.path.exists(filepath):
        return {"messages": []}

    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    conversation_history = data.get("messages", [])[-10:]
    active_conversation_file = filename

    return data


@app.delete("/delete-conversation/{filename}")
def delete_conversation(filename: str):
    global active_conversation_file
    global conversation_history

    filepath = os.path.join(CONVERSATIONS_DIR, filename)

    if os.path.exists(filepath):
        os.remove(filepath)

    if active_conversation_file == filename:
        active_conversation_file = None
        conversation_history = []

    return {"status": "ok"}


def open_browser():
    webbrowser.open("http://localhost:8000")


if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
