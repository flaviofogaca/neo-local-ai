let history = [];
let activeConversation = null;
let isGenerating = false;
let isRecording = false;
let recognition = null;

function formatConversationName(fileName) {
  return fileName.replace(".json", "");
}

function hideEmptyState() {
  const emptyState = document.getElementById("empty-state");

  if (emptyState) {
    emptyState.style.display = "none";
  }
}

function renderMessage(role, content) {
  hideEmptyState();

  const chat = document.getElementById("chat");
  const className = role === "user" ? "user" : "neo";
  const avatar = role === "user" ? "F" : "";

  chat.innerHTML += `
    <div class="message-row ${className} fade-in">
      ${role !== "user" ? `<div class="message-avatar neo-avatar"></div>` : ""}

      <div class="message-bubble">
        ${marked.parse(content)}
      </div>

      ${role === "user" ? `<div class="message-avatar user-avatar">${avatar}</div>` : ""}
    </div>
  `;
}

function showEmptyState() {
  document.getElementById("chat").innerHTML = `
    <div class="empty-state" id="empty-state">
      <div class="empty-card">
        <div class="empty-logo"></div>

        <div class="empty-title">
          Bem-vindo ao Neo
        </div>

        <div class="empty-subtitle">
          Seu assistente local com memória persistente,
          reconhecimento de voz e execução local via Ollama.
        </div>

        <div class="empty-actions">
          <button class="empty-action" onclick="newChat()">Nova Conversa</button>
          <button class="empty-action">Deep Mode</button>
          <button class="empty-action">Sobre o Neo</button>
        </div>
      </div>
    </div>
  `;
}

function scrollToBottom() {
  const chat = document.getElementById("chat");

  chat.scrollTo({
    top: chat.scrollHeight,
    behavior: "smooth"
  });
}

function addThinking() {
  hideEmptyState();

  const chat = document.getElementById("chat");
  const thinkingId = "thinking-" + Date.now();

  chat.innerHTML += `
    <div class="message-row neo" id="${thinkingId}">
      <div class="message-avatar neo-avatar"></div>

      <div class="message-bubble">
        <div class="typing-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>
  `;

  scrollToBottom();

  return thinkingId;
}

function replaceThinking(id, response) {
  const element = document.getElementById(id);

  if (element) {
    element.innerHTML = `
      <div class="message-avatar neo-avatar"></div>

      <div class="message-bubble">
        ${marked.parse(response)}
      </div>
    `;
  }
}

async function newChat() {
  history = [];
  activeConversation = null;
  showEmptyState();

  await fetch("/new-chat", {
    method: "POST"
  });

  await loadConversations();
}

async function loadConversations() {
  const response = await fetch("/conversations");
  const data = await response.json();

  const list = document.getElementById("conversation-list");
  list.innerHTML = "";

  data.conversations.forEach(file => {
    const encodedFile = encodeURIComponent(file);
    const activeClass = activeConversation === file ? "active" : "";

    list.innerHTML += `
      <div class="conversation-item ${activeClass}">
        <div class="conversation-name" onclick="loadConversation('${encodedFile}')">
          ${formatConversationName(file)}
        </div>

        <div class="conversation-actions">
          <button class="rename-btn" onclick="event.stopPropagation(); renameConversation('${file}')">
            ✏️
          </button>

          <button class="delete-btn" onclick="event.stopPropagation(); deleteConversation('${file}')">
            🗑️
          </button>
        </div>
      </div>
    `;
  });
}

async function renameConversation(oldName) {
  const newName = prompt("Novo nome da conversa:");

  if (!newName) {
    return;
  }

  await fetch("/rename-conversation", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      old_name: oldName,
      new_name: newName
    })
  });

  if (activeConversation === oldName) {
    activeConversation = newName.endsWith(".json") ? newName : `${newName}.json`;
  }

  await loadConversations();
}

async function loadConversation(filename) {
  activeConversation = decodeURIComponent(filename);

  await loadConversations();

  const response = await fetch(`/conversations/${filename}`);
  const data = await response.json();

  const chat = document.getElementById("chat");
  chat.innerHTML = "";
  history = [];

  data.messages.forEach(msg => {
    renderMessage(msg.role, msg.content);

    history.push({
      role: msg.role,
      content: msg.content
    });
  });

  if (history.length === 0) {
    showEmptyState();
  }

  scrollToBottom();
}

async function deleteConversation(filename) {
  const confirmed = confirm(
    `Deseja apagar:\n\n${formatConversationName(filename)}?`
  );

  if (!confirmed) return;

  await fetch(`/delete-conversation/${encodeURIComponent(filename)}`, {
    method: "DELETE"
  });

  if (activeConversation === filename) {
    activeConversation = null;
    history = [];
    showEmptyState();
  }

  await loadConversations();
}

async function sendMessage() {
  const input = document.getElementById("message");
  const message = input.value.trim();

  if (!message || isGenerating) return;

  isGenerating = true;

  const sendButton = document.querySelector(".send-btn");
  sendButton.disabled = true;
  sendButton.classList.add("disabled");
  sendButton.classList.add("generating");

  renderMessage("user", message);

  history.push({
    role: "user",
    content: message
  });

  input.value = "";
  input.style.height = "90px";

  const thinkingId = addThinking();
  let fullResponse = "";

  try {
    const response = await fetch("/chat-stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: message,
        history: history
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    while (true) {
      const { value, done } = await reader.read();

      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      fullResponse += chunk;

      replaceThinking(thinkingId, fullResponse);
      scrollToBottom();
    }

    history.push({
      role: "neo",
      content: fullResponse
    });

    history = history.slice(-10);

    await loadConversations();

  } catch (error) {
    replaceThinking(thinkingId, "Erro ao gerar resposta.");
  }

  isGenerating = false;

  sendButton.disabled = false;
  sendButton.classList.remove("disabled");
  sendButton.classList.remove("generating");

  scrollToBottom();
}

const textarea = document.getElementById("message");

textarea.addEventListener("keydown", function (event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

function autoResizeTextarea() {
  textarea.style.height = "auto";
  textarea.style.height = textarea.scrollHeight + "px";
}

textarea.addEventListener("input", autoResizeTextarea);

function setupSpeechRecognition() {
  const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

  const voiceBtn = document.querySelector(".voice-btn");

  if (!SpeechRecognition) {
    voiceBtn.disabled = true;
    voiceBtn.title = "Reconhecimento de voz não suportado neste navegador";
    voiceBtn.style.opacity = "0.4";
    return;
  }

  recognition = new SpeechRecognition();

  recognition.lang = "pt-BR";
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onresult = function (event) {
    const texto = event.results[0][0].transcript;

    textarea.value = texto;
    textarea.dispatchEvent(new Event("input"));
  };

  recognition.onend = function () {
    const voiceBtn = document.querySelector(".voice-btn");

    isRecording = false;
    voiceBtn.classList.remove("recording");
  };

  recognition.onerror = function () {
    const voiceBtn = document.querySelector(".voice-btn");

    isRecording = false;
    voiceBtn.classList.remove("recording");
  };
}

function startRecognition() {
  const voiceBtn = document.querySelector(".voice-btn");

  if (!recognition) {
    return;
  }

  if (isRecording) {
    recognition.stop();
    isRecording = false;
    voiceBtn.classList.remove("recording");
    return;
  }

  isRecording = true;
  voiceBtn.classList.add("recording");

  recognition.start();
}

setupSpeechRecognition();
loadConversations();
showEmptyState();