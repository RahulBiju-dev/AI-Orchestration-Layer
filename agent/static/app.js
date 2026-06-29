"use strict";

const DEFAULT_SETTINGS = {
  options: { temperature: 0.7, top_p: 0.9, top_k: 40, num_ctx: 8192 },
  verbose: false,
  wordwrap: true,
  system: "",
  history: true,
  format: "",
  think: true
};

const SLASH_COMMANDS = [
  { command: "/help", description: "Show available commands" },
  { command: "/clear", description: "Clear conversation history" },
  { command: "/save ", description: "Save current session with an optional name" },
  { command: "/load ", description: "Load a saved session by name or index" },
  { command: "/set parameter ", description: "Set a model parameter, e.g. temperature 0.7" },
  { command: "/set system \"\"", description: "Set the system prompt for this session" },
  { command: "/set history", description: "Enable conversation history" },
  { command: "/set nohistory", description: "Disable conversation history" },
  { command: "/set wordwrap", description: "Enable word wrapping" },
  { command: "/set nowordwrap", description: "Disable word wrapping" },
  { command: "/set format json", description: "Force JSON output from the model" },
  { command: "/set noformat", description: "Disable forced output format" },
  { command: "/set verbose", description: "Show generation stats after each response" },
  { command: "/set quiet", description: "Hide generation stats" },
  { command: "/set think", description: "Enable model thinking/reasoning" },
  { command: "/set nothink", description: "Disable model thinking" },
  { command: "/show parameters", description: "Show current session parameters" },
  { command: "/show system", description: "Show the active system prompt" },
  { command: "/show model", description: "Show model info" },
  { command: "/vault list", description: "List indexed vault collections" },
  { command: "/vault aliases", description: "List registered vault aliases" },
  { command: "/vault alias ", description: "Register a friendly alias for a collection" },
  { command: "/vault rename ", description: "Rename a vault collection" },
  { command: "/vault add ", description: "Add a file or folder to the searchable vault" },
  { command: "/vault search ", description: "Search the indexed vault" },
  { command: "/vault delete ", description: "Delete indexed vault chunks by source or path" },
  { command: "/quit", description: "Exit the agent" },
  { command: "/exit", description: "Exit the agent" },
  { command: "/q", description: "Exit the agent" }
];

const state = {
  history: [],
  settings: cloneSettings(DEFAULT_SETTINGS),
  savedSessions: [],
  activeSessionName: "New conversation",
  modelName: "selene",
  isGenerating: false,
  controller: null,
  stream: {
    assistantStack: null,
    assistantBubble: null,
    assistantText: "",
    thinkingBlock: null,
    thinkingText: "",
    lastToolBlock: null
  },
  slash: {
    open: false,
    selected: 0,
    matches: []
  }
};

const el = {};

document.addEventListener("DOMContentLoaded", () => {
  bindElements();
  bindEvents();
  loadState();
});

function bindElements() {
  el.messages = document.getElementById("messages");
  el.form = document.getElementById("composer-form");
  el.input = document.getElementById("chat-input");
  el.send = document.getElementById("send-btn");
  el.contextLabel = document.getElementById("context-label");
  el.contextFill = document.getElementById("context-fill");
  el.contextMeter = document.querySelector(".context-meter");
  el.sessionList = document.getElementById("session-list");
  el.connection = document.getElementById("connection-label");
  el.title = document.getElementById("chat-title");
  el.history = document.getElementById("setting-history");
  el.think = document.getElementById("setting-think");
  el.temperature = document.getElementById("setting-temperature");
  el.temperatureValue = document.getElementById("temperature-value");
  el.context = document.getElementById("setting-context");
  el.slashMenu = document.getElementById("slash-menu");
}

function bindEvents() {
  el.form?.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.isGenerating) stopGeneration();
    else sendMessage();
  });

  el.input?.addEventListener("input", () => {
    resizeComposer();
    updateComposerState();
    updateSlashMenu();
  });

  el.input?.addEventListener("keydown", (event) => {
    if (handleSlashKeydown(event)) return;
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (state.isGenerating) stopGeneration();
      else sendMessage();
    }
  });

  document.addEventListener("click", (event) => {
    if (!el.slashMenu?.contains(event.target) && event.target !== el.input) {
      closeSlashMenu();
    }
  });

  document.getElementById("new-chat-btn")?.addEventListener("click", clearConversation);
  document.getElementById("clear-btn")?.addEventListener("click", clearConversation);
  document.getElementById("save-btn")?.addEventListener("click", saveSession);

  el.history?.addEventListener("change", () => {
    state.settings.history = el.history.checked;
    persistSettings();
  });

  el.think?.addEventListener("change", () => {
    state.settings.think = el.think.checked;
    persistSettings();
  });

  el.temperature?.addEventListener("input", () => {
    const value = Number(el.temperature.value);
    state.settings.options = state.settings.options || {};
    state.settings.options.temperature = value;
    el.temperatureValue.textContent = value.toFixed(2);
    persistSettings();
  });

  el.context?.addEventListener("change", () => {
    state.settings.options = state.settings.options || {};
    state.settings.options.num_ctx = Number(el.context.value);
    persistSettings();
    updateContextMeter();
  });
}

async function loadState() {
  try {
    const response = await fetch("/api/settings");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    state.history = data.history || [];
    state.savedSessions = data.saved_sessions || [];
    state.settings = mergeSettings(data.settings || {});
    state.activeSessionName = data.active_session_name || "New conversation";
    state.modelName = data.model_name || "selene";

    el.connection.textContent = data.ollama_status === "Online" ? "Ollama online" : "Ollama offline";
    document.title = titleForSession(state.activeSessionName);
    el.title.textContent = cleanSessionName(state.activeSessionName);

    syncSettingsUI();
    renderSessions();
    renderMessages();
    updateComposerState();
  } catch (error) {
    el.connection.textContent = "Backend unavailable";
    toast("Could not reach Selene. Make sure the backend and Ollama are running.");
    renderWelcome();
  }
}

function mergeSettings(settings) {
  return {
    ...DEFAULT_SETTINGS,
    ...settings,
    options: {
      ...DEFAULT_SETTINGS.options,
      ...(settings.options || {})
    }
  };
}

function cloneSettings(settings) {
  return JSON.parse(JSON.stringify(settings));
}

function syncSettingsUI() {
  if (el.history) el.history.checked = state.settings.history !== false;
  if (el.think) el.think.checked = state.settings.think !== false;

  const temp = Number(state.settings.options?.temperature ?? 0.7);
  if (el.temperature) el.temperature.value = String(temp);
  if (el.temperatureValue) el.temperatureValue.textContent = temp.toFixed(2);

  const ctx = String(contextBudget());
  if (el.context) {
    const hasOption = [...el.context.options].some((option) => option.value === ctx);
    if (!hasOption) {
      const option = document.createElement("option");
      option.value = ctx;
      option.textContent = `${Math.round(Number(ctx) / 1024)}k`;
      el.context.appendChild(option);
    }
    el.context.value = ctx;
  }
}

async function persistSettings() {
  try {
    await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.settings)
    });
  } catch {
    toast("Settings could not be saved.");
  }
}

function renderSessions() {
  if (!el.sessionList) return;
  el.sessionList.innerHTML = "";

  if (!state.savedSessions.length) {
    const empty = document.createElement("div");
    empty.className = "empty-sessions";
    empty.textContent = "Saved chats will appear here.";
    el.sessionList.appendChild(empty);
    return;
  }

  state.savedSessions.forEach((name) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-item";
    button.textContent = cleanSessionName(name);
    button.title = name;
    button.addEventListener("click", () => loadSession(name));
    el.sessionList.appendChild(button);
  });
}

function renderMessages() {
  el.messages.innerHTML = "";
  const display = toDisplayMessages(state.history);

  if (!display.length) {
    renderWelcome();
    return;
  }

  display.forEach((message) => {
    if (message.role === "user") {
      appendUserMessage(message.content, false);
    } else {
      appendAssistantMessage(message, false);
    }
  });
  scrollToBottom(true);
}

function toDisplayMessages(history) {
  const display = [];
  for (let i = 0; i < history.length; i += 1) {
    const message = history[i];
    if (message.role === "system") continue;
    if (message.role === "user") {
      display.push({ role: "user", content: message.content || "" });
      continue;
    }
    if (message.role === "assistant") {
      const toolResults = [];
      let j = i + 1;
      while (history[j]?.role === "tool") {
        toolResults.push(history[j]);
        j += 1;
      }
      display.push({
        role: "assistant",
        content: message.content || "",
        thinking: message.thinking || "",
        toolCalls: message.tool_calls || [],
        toolResults
      });
      i = j - 1;
    }
  }
  return display;
}

function renderWelcome() {
  el.messages.innerHTML = `
    <div class="welcome">
      <h3>Selene</h3>
      <p>A clean local AI chat interface for thinking, tools, and answers in one dark workspace.</p>
      <div class="suggestions">
        <button class="suggestion" type="button" data-prompt="Summarize this project and identify the most important files.">
          Project summary <span>Understand the workspace</span>
        </button>
        <button class="suggestion" type="button" data-prompt="Search the web for the latest AI developer tooling news.">
          Web research <span>Use tools when needed</span>
        </button>
        <button class="suggestion" type="button" data-prompt="Help me debug a Python error step by step.">
          Debug with me <span>Reason through a problem</span>
        </button>
        <button class="suggestion" type="button" data-prompt="/help">
          Commands <span>Show slash commands</span>
        </button>
      </div>
    </div>
  `;

  el.messages.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
      el.input.value = button.dataset.prompt || "";
      resizeComposer();
      updateComposerState();
      el.input.focus();
    });
  });
}

function appendUserMessage(text, scroll = true) {
  el.messages.querySelector(".welcome")?.remove();
  const row = messageShell("user", "You");
  row.stack.appendChild(bubble(text, false));
  el.messages.appendChild(row.root);
  if (scroll) scrollToBottom(true);
}

function appendAssistantMessage(message, scroll = true) {
  el.messages.querySelector(".welcome")?.remove();
  const row = messageShell("assistant", "S");

  if (message.thinking) {
    row.stack.appendChild(detailBlock("Thinking", "reasoning", message.thinking, false));
  }

  message.toolCalls?.forEach((call, index) => {
    const fn = call.function || {};
    const result = message.toolResults?.[index]?.content || "";
    const args = formatJSON(fn.arguments || {});
    const body = `Arguments:\n${args}${result ? `\n\nResult:\n${result}` : ""}`;
    row.stack.appendChild(detailBlock(fn.name || "tool", "tool", body, false));
  });

  if (message.content) row.stack.appendChild(bubble(message.content, true));
  el.messages.appendChild(row.root);
  if (scroll) scrollToBottom(true);
}

function messageShell(role, avatarText) {
  const root = document.createElement("article");
  root.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = avatarText;

  const stack = document.createElement("div");
  stack.className = "message-stack";

  root.append(avatar, stack);
  return { root, stack };
}

function bubble(content, markdown) {
  const node = document.createElement("div");
  node.className = "bubble";
  node.innerHTML = markdown ? renderMarkdown(content) : escapeHTML(content).replace(/\n/g, "<br>");
  return node;
}

function detailBlock(title, label, content, running) {
  const block = document.createElement("section");
  block.className = `detail-block open${running ? " running" : ""}`;

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "block-toggle";
  toggle.innerHTML = `
    <span class="block-title">${escapeHTML(title)} <span class="pill">${escapeHTML(label)}</span></span>
    <span aria-hidden="true">+</span>
  `;

  const body = document.createElement("div");
  body.className = "block-body";
  body.textContent = content || (running ? "Running..." : "");

  toggle.addEventListener("click", () => block.classList.toggle("open"));
  block.append(toggle, body);
  return block;
}

async function sendMessage() {
  if (state.isGenerating) return;
  const text = el.input.value.trim();
  if (!text) return;

  closeSlashMenu();
  appendUserMessage(text);
  el.input.value = "";
  resizeComposer();
  updateComposerState();

  state.isGenerating = true;
  state.controller = new AbortController();
  resetStream();
  updateComposerState();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
      signal: state.controller.signal
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await readEventStream(response);
  } catch (error) {
    if (error.name !== "AbortError") {
      toast("Selene lost the response stream. Check Ollama and try again.");
      finishGeneration();
    }
  }
}

async function readEventStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try {
        handleStreamEvent(JSON.parse(line.slice(6)));
      } catch {
        // Ignore malformed SSE chunks.
      }
    }
  }
}

function handleStreamEvent(event) {
  switch (event.type) {
    case "status":
      appendStatus(event.message || "");
      break;
    case "thinking_start":
      ensureAssistantStack();
      state.stream.thinkingBlock = detailBlock("Thinking", "reasoning", "", true);
      state.stream.assistantStack.appendChild(state.stream.thinkingBlock);
      scrollToBottom();
      break;
    case "thinking_chunk":
      state.stream.thinkingText += event.text || "";
      setBlockBody(state.stream.thinkingBlock, state.stream.thinkingText);
      break;
    case "thinking_end":
      state.stream.thinkingBlock?.classList.remove("running");
      state.stream.thinkingBlock?.classList.remove("open");
      break;
    case "tool_start":
      ensureAssistantStack();
      state.stream.lastToolBlock = detailBlock(event.name || "tool", "tool", `Arguments:\n${formatJSON(event.arguments || {})}`, true);
      state.stream.assistantStack.appendChild(state.stream.lastToolBlock);
      scrollToBottom();
      break;
    case "tool_end":
      if (state.stream.lastToolBlock) {
        state.stream.lastToolBlock.classList.remove("running");
        setBlockBody(state.stream.lastToolBlock, `Result:\n${event.result || ""}`);
      }
      resetStream();
      break;
    case "content_chunk":
      ensureAssistantStack();
      if (!state.stream.assistantBubble) {
        state.stream.assistantBubble = document.createElement("div");
        state.stream.assistantBubble.className = "bubble";
        state.stream.assistantStack.appendChild(state.stream.assistantBubble);
      }
      state.stream.assistantText += event.text || event.content || "";
      state.stream.assistantBubble.innerHTML = renderMarkdown(state.stream.assistantText);
      scrollToBottom();
      break;
    case "token_usage":
      updateContextMeter(event.total, event.budget);
      break;
    case "done":
      if (event.history) state.history = event.history;
      finishGeneration();
      break;
  }
}

function ensureAssistantStack() {
  if (state.stream.assistantStack) return;
  const row = messageShell("assistant", "S");
  el.messages.querySelector(".welcome")?.remove();
  el.messages.appendChild(row.root);
  state.stream.assistantStack = row.stack;
}

function resetStream() {
  state.stream.assistantStack = null;
  state.stream.assistantBubble = null;
  state.stream.assistantText = "";
  state.stream.thinkingBlock = null;
  state.stream.thinkingText = "";
  state.stream.lastToolBlock = null;
}

function finishGeneration() {
  state.isGenerating = false;
  state.controller = null;
  resetStream();
  updateComposerState();
}

function stopGeneration() {
  state.controller?.abort();
  toast("Generation stopped.");
  finishGeneration();
}

function appendStatus(text) {
  const status = document.createElement("div");
  status.className = "status-line";
  status.textContent = text;
  el.messages.appendChild(status);
  scrollToBottom();
}

function setBlockBody(block, text) {
  const body = block?.querySelector(".block-body");
  if (body) body.textContent = text;
}

async function clearConversation() {
  if (state.isGenerating) stopGeneration();
  try {
    await fetch("/api/clear-session", { method: "POST" });
    state.history = [];
    state.activeSessionName = "New conversation";
    el.title.textContent = "New conversation";
    document.title = "Selene";
    renderWelcome();
    updateContextMeter();
  } catch {
    toast("Could not clear the conversation.");
  }
}

async function saveSession() {
  const name = window.prompt("Save this chat as:", cleanSessionName(state.activeSessionName));
  if (name === null) return;
  try {
    const response = await fetch("/api/save-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    toast("Session saved.");
    loadState();
  } catch {
    toast("Could not save this session.");
  }
}

async function loadSession(name) {
  try {
    const response = await fetch("/api/load-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await loadState();
    toast("Session loaded.");
  } catch {
    toast("Could not load that session.");
  }
}

function updateComposerState() {
  const hasText = Boolean(el.input?.value.trim());
  if (el.send) {
    el.send.disabled = !state.isGenerating && !hasText;
    el.send.textContent = state.isGenerating ? "Stop" : "Send";
    el.send.classList.toggle("stop", state.isGenerating);
  }
  updateContextMeter();
}

function updateSlashMenu() {
  if (!el.input || !el.slashMenu) return;

  const text = el.input.value;
  const cursor = el.input.selectionStart ?? text.length;
  const beforeCursor = text.slice(0, cursor);

  if (!beforeCursor.startsWith("/") || beforeCursor.includes("\n") || state.isGenerating) {
    closeSlashMenu();
    return;
  }

  const query = beforeCursor.toLowerCase();
  const matches = SLASH_COMMANDS
    .filter((item) => {
      const haystack = `${item.command} ${item.description}`.toLowerCase();
      return haystack.includes(query) || item.command.toLowerCase().startsWith(query);
    })
    .slice(0, 10);

  state.slash.open = true;
  state.slash.matches = matches;
  state.slash.selected = Math.min(state.slash.selected, Math.max(matches.length - 1, 0));
  renderSlashMenu();
}

function renderSlashMenu() {
  if (!el.slashMenu) return;
  el.slashMenu.innerHTML = "";
  el.slashMenu.classList.toggle("open", state.slash.open);

  if (!state.slash.open) return;

  if (!state.slash.matches.length) {
    const empty = document.createElement("div");
    empty.className = "slash-empty";
    empty.textContent = "No matching commands";
    el.slashMenu.appendChild(empty);
    return;
  }

  state.slash.matches.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `slash-item${index === state.slash.selected ? " active" : ""}`;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(index === state.slash.selected));
    button.innerHTML = `
      <span class="slash-command">${escapeHTML(item.command)}</span>
      <span class="slash-desc">${escapeHTML(item.description)}</span>
    `;
    button.addEventListener("mouseenter", () => {
      state.slash.selected = index;
      renderSlashMenu();
    });
    button.addEventListener("click", () => chooseSlashCommand(index));
    el.slashMenu.appendChild(button);
  });
}

function handleSlashKeydown(event) {
  if (!state.slash.open) return false;

  if (event.key === "ArrowDown") {
    event.preventDefault();
    moveSlashSelection(1);
    return true;
  }
  if (event.key === "ArrowUp") {
    event.preventDefault();
    moveSlashSelection(-1);
    return true;
  }
  if (event.key === "Tab") {
    event.preventDefault();
    chooseSlashCommand();
    return true;
  }
  if (event.key === "Enter" && !event.shiftKey && state.slash.matches.length) {
    event.preventDefault();
    chooseSlashCommand();
    return true;
  }
  if (event.key === "Escape") {
    event.preventDefault();
    closeSlashMenu();
    return true;
  }
  return false;
}

function moveSlashSelection(delta) {
  const count = state.slash.matches.length;
  if (!count) return;
  state.slash.selected = (state.slash.selected + delta + count) % count;
  renderSlashMenu();
}

function chooseSlashCommand(index = state.slash.selected) {
  const item = state.slash.matches[index];
  if (!item || !el.input) return;

  const suffix = item.command.endsWith(" ") || item.command.endsWith("\"\"") ? "" : " ";
  el.input.value = `${item.command}${suffix}`;

  if (item.command.endsWith("\"\"")) {
    const cursor = item.command.length - 1;
    el.input.setSelectionRange(cursor, cursor);
  } else {
    const cursor = el.input.value.length;
    el.input.setSelectionRange(cursor, cursor);
  }

  closeSlashMenu();
  resizeComposer();
  updateComposerState();
  el.input.focus();
}

function closeSlashMenu() {
  state.slash.open = false;
  state.slash.selected = 0;
  state.slash.matches = [];
  if (el.slashMenu) {
    el.slashMenu.classList.remove("open");
    el.slashMenu.innerHTML = "";
  }
}

function updateContextMeter(forcedUsed = null, forcedBudget = null) {
  const budget = forcedBudget || contextBudget();
  const used = forcedUsed ?? estimatedContextTokens();
  const pct = Math.min(100, Math.round((used / budget) * 100));

  if (el.contextLabel) el.contextLabel.textContent = `${used} / ${budget}`;
  if (el.contextFill) el.contextFill.style.width = `${pct}%`;
  if (el.contextMeter) {
    el.contextMeter.classList.toggle("warn", pct >= 75 && pct < 90);
    el.contextMeter.classList.toggle("hot", pct >= 90);
  }
}

function estimatedContextTokens() {
  const historyText = state.history
    .map((message) => `${message.content || ""}\n${message.thinking || ""}`)
    .join("\n");
  return estimateTokens(`${historyText}\n${el.input?.value || ""}`);
}

function estimateTokens(text) {
  return Math.max(0, Math.ceil(String(text || "").length / 4));
}

function contextBudget() {
  return Number(state.settings.options?.num_ctx || 8192);
}

function resizeComposer() {
  if (!el.input) return;
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, 220)}px`;
}

function scrollToBottom(force = false) {
  if (!el.messages) return;
  const nearBottom = el.messages.scrollHeight - el.messages.scrollTop - el.messages.clientHeight < 160;
  if (force || nearBottom) el.messages.scrollTop = el.messages.scrollHeight;
}

function renderMarkdown(text) {
  const escaped = escapeHTML(text || "");
  const withCode = escaped.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
  return withCode
    .split(/\n{2,}/)
    .map((chunk) => {
      if (chunk.startsWith("<pre>")) return chunk;
      return `<p>${chunk.replace(/\n/g, "<br>")}</p>`;
    })
    .join("");
}

function formatJSON(value) {
  if (typeof value === "string") {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  return JSON.stringify(value, null, 2);
}

function escapeHTML(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function cleanSessionName(name) {
  const base = String(name || "New conversation").replace(/\.json$/, "");
  return base
    .replace(/_\d{8}_\d{6}$/, "")
    .replace(/_/g, " ")
    .replace(/^Active Session$/, "New conversation");
}

function titleForSession(name) {
  const clean = cleanSessionName(name);
  return clean === "New conversation" ? "Selene" : `${clean} - Selene`;
}

function toast(message) {
  const node = document.createElement("div");
  node.className = "toast";
  node.textContent = message;
  document.getElementById("toast-region")?.appendChild(node);
  setTimeout(() => node.remove(), 3600);
}
