import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:26b"


def load_memory():
    with open("memory.json", "r", encoding="utf-8") as file:
        return json.load(file)


def save_memory(new_data):
    memory = load_memory()
    memory.update(new_data)

    with open("memory.json", "w", encoding="utf-8") as file:
        json.dump(memory, file, ensure_ascii=False, indent=2)


def ask_ollama(user_message):
    memory = load_memory()

    system_prompt = f"""
Você é o Neo, assistente pessoal do Flávio.

Use estas informações como memória persistente:
{json.dumps(memory, ensure_ascii=False, indent=2)}

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

    payload = {
        "model": MODEL,
        "prompt": system_prompt + "\n\nUsuário: " + user_message + "\nNeo:",
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    response_text = response.json()["response"]

    try:
        parsed = json.loads(response_text)

        if parsed.get("action") == "save_memory":
            save_memory(parsed["data"])
            print("\n🧠 Memória salva com sucesso!")
            return None

    except json.JSONDecodeError:
        pass

    return response_text


while True:
    user_input = input("\nVocê: ")

    if user_input.lower() in ["sair", "exit", "quit"]:
        break

    answer = ask_ollama(user_input)

    if answer:
        print(f"\nNeo: {answer}")