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
  followOutput: true,
  controller: null,
  generation: null,
  stream: {
    assistantStack: null,
    assistantBubble: null,
    assistantText: "",
    thinkingBlock: null,
    thinkingContent: null,
    thinkingText: "",
    lastToolBlock: null
  },
  slash: {
    open: false,
    selected: 0,
    matches: []
  },
  sky: {
    frame: null,
    resizeObserver: null,
    scene: null
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
  el.title = document.getElementById("chat-title");
  el.history = document.getElementById("setting-history");
  el.think = document.getElementById("setting-think");
  el.temperature = document.getElementById("setting-temperature");
  el.temperatureValue = document.getElementById("temperature-value");
  el.context = document.getElementById("setting-context");
  el.system = document.getElementById("setting-system");
  el.settingsPanel = document.getElementById("settings-panel");
  el.settingsBackdrop = document.getElementById("settings-backdrop");
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

  document.getElementById("new-chat-btn")?.addEventListener("click", newConversation);
  document.getElementById("settings-btn")?.addEventListener("click", openSettings);
  document.getElementById("settings-close")?.addEventListener("click", closeSettings);
  el.settingsBackdrop?.addEventListener("click", closeSettings);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && el.settingsPanel?.classList.contains("open")) closeSettings();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopWelcomeSky();
    else if (el.messages?.querySelector(".welcome")) startWelcomeSky();
  });
  window.addEventListener("resize", resizeComposer);

  el.messages?.addEventListener("wheel", (event) => {
    if (event.deltaY < 0) state.followOutput = false;
  }, { passive: true });
  el.messages?.addEventListener("touchmove", () => { state.followOutput = false; }, { passive: true });
  el.messages?.addEventListener("scroll", () => {
    if (distanceFromBottom() < 24) state.followOutput = true;
  }, { passive: true });
  el.messages?.addEventListener("click", (event) => {
    const button = event.target.closest(".code-copy-btn");
    if (button && el.messages.contains(button)) copyCodeBlock(button);
  });

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

  const persistSystemPrompt = debounce(persistSettings, 350);
  el.system?.addEventListener("input", () => {
    state.settings.system = el.system.value;
    updateContextMeter();
    persistSystemPrompt();
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

    document.title = titleForSession(state.activeSessionName);
    el.title.textContent = cleanSessionName(state.activeSessionName);

    syncSettingsUI();
    renderSessions();
    renderMessages();
    updateComposerState();
  } catch (error) {
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
  if (el.system) el.system.value = state.settings.system || "";

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

  if (!state.savedSessions.length) return;

  state.savedSessions.forEach((name) => {
    const item = document.createElement("div");
    item.className = `session-item${name === state.activeSessionName ? " active" : ""}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-open";
    button.textContent = cleanSessionName(name);
    button.title = name;
    button.addEventListener("click", () => loadSession(name));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "session-delete";
    remove.setAttribute("aria-label", `Delete ${cleanSessionName(name)}`);
    remove.title = "Delete chat";
    remove.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3m3 0-1 14H7L6 7m4 4v6m4-6v6"/></svg>`;
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteSession(name);
    });
    item.append(button, remove);
    el.sessionList.appendChild(item);
  });
}

function renderMessages() {
  stopWelcomeSky();
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
      let j = i + 1;
      while (history[j]?.role === "tool") {
        j += 1;
      }
      const entry = {
        role: "assistant",
        content: message.content || "",
        thoughtItems: [
          ...(message.thinking ? [{ type: "thinking", text: message.thinking }] : []),
          ...(message.tool_calls || []).map((call) => ({
            type: "tool",
            name: call.function?.name || "tool"
          }))
        ]
      };
      const previous = display[display.length - 1];
      if (previous?.role === "assistant" && !previous.content) {
        previous.content = entry.content;
        previous.thoughtItems.push(...entry.thoughtItems);
      } else {
        display.push(entry);
      }
      i = j - 1;
    }
  }
  return display;
}

function renderWelcome() {
  el.messages.innerHTML = `
    <div class="welcome">
      <canvas class="welcome-sky" aria-hidden="true"></canvas>
      <h3 data-text="Selene">Selene</h3>
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
  startWelcomeSky();
}

function startWelcomeSky() {
  stopWelcomeSky();
  const canvas = el.messages.querySelector(".welcome-sky");
  const context = canvas?.getContext("2d");
  if (!canvas || !context || document.hidden) return;

  const now = performance.now();
  const scene = {
    canvas,
    context,
    width: 0,
    height: 0,
    lastFrame: now,
    shootingStar: null,
    nextShootingStar: now + randomBetween(14000, 30000),
    stars: Array.from({ length: 28 }, () => newCanvasStar(now, true))
  };
  state.sky.scene = scene;

  const resize = () => {
    const bounds = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    scene.width = Math.max(1, bounds.width);
    scene.height = Math.max(1, bounds.height);
    canvas.width = Math.round(scene.width * ratio);
    canvas.height = Math.round(scene.height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
  };
  resize();
  state.sky.resizeObserver = new ResizeObserver(resize);
  state.sky.resizeObserver.observe(canvas);

  const drawFrame = (timestamp) => {
    if (!canvas.isConnected || document.hidden || state.sky.scene !== scene) return;
    const delta = Math.min(34, Math.max(0, timestamp - scene.lastFrame));
    scene.lastFrame = timestamp;
    drawWelcomeSky(scene, timestamp, delta);
    state.sky.frame = requestAnimationFrame(drawFrame);
  };
  state.sky.frame = requestAnimationFrame(drawFrame);
}

function stopWelcomeSky() {
  if (state.sky.frame !== null) cancelAnimationFrame(state.sky.frame);
  state.sky.resizeObserver?.disconnect();
  state.sky.frame = null;
  state.sky.resizeObserver = null;
  state.sky.scene = null;
}

function newCanvasStar(now, initial = false) {
  return {
    x: Math.random(),
    y: Math.random(),
    radius: randomBetween(.45, 1.25),
    brightness: randomBetween(.28, .72),
    born: now + (initial ? randomBetween(-5000, 1200) : randomBetween(350, 2600)),
    duration: randomBetween(4500, 10000)
  };
}

function drawWelcomeSky(scene, now, delta) {
  const { context, width, height, stars } = scene;
  context.clearRect(0, 0, width, height);

  stars.forEach((star, index) => {
    const progress = (now - star.born) / star.duration;
    if (progress >= 1) {
      stars[index] = newCanvasStar(now);
      return;
    }
    if (progress < 0) return;
    const opacity = Math.pow(Math.sin(Math.PI * progress), 1.7) * star.brightness;
    context.beginPath();
    context.arc(star.x * width, star.y * height, star.radius, 0, Math.PI * 2);
    context.fillStyle = `rgba(240, 244, 255, ${opacity})`;
    context.fill();
  });

  if (!scene.shootingStar && now >= scene.nextShootingStar) {
    const angle = randomBetween(14, 27) * Math.PI / 180;
    const speed = randomBetween(250, 350);
    scene.shootingStar = {
      x: randomBetween(width * .02, width * .52),
      y: randomBetween(height * .04, height * .45),
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      age: 0,
      duration: randomBetween(1.8, 2.5),
      trail: randomBetween(72, 118)
    };
  }

  const shot = scene.shootingStar;
  if (!shot) return;
  shot.age += delta / 1000;
  shot.x += shot.vx * delta / 1000;
  shot.y += shot.vy * delta / 1000;
  const speed = Math.hypot(shot.vx, shot.vy);
  const ux = shot.vx / speed;
  const uy = shot.vy / speed;
  const fadeIn = Math.min(1, shot.age / .24);
  const fadeOut = Math.min(1, Math.max(0, (shot.duration - shot.age) / .9));
  const opacity = Math.min(fadeIn, fadeOut) * .82;
  const tailX = shot.x - ux * shot.trail;
  const tailY = shot.y - uy * shot.trail;
  const streak = context.createLinearGradient(tailX, tailY, shot.x, shot.y);
  streak.addColorStop(0, "rgba(255,255,255,0)");
  streak.addColorStop(.55, `rgba(238,243,255,${opacity * .22})`);
  streak.addColorStop(1, `rgba(255,255,255,${opacity})`);
  context.beginPath();
  context.moveTo(tailX, tailY);
  context.lineTo(shot.x, shot.y);
  context.lineWidth = .9;
  context.strokeStyle = streak;
  context.stroke();
  context.beginPath();
  context.arc(shot.x, shot.y, 1.15, 0, Math.PI * 2);
  context.fillStyle = `rgba(255,255,255,${opacity})`;
  context.fill();

  if (shot.age >= shot.duration || shot.x > width + shot.trail || shot.y > height + shot.trail) {
    scene.shootingStar = null;
    scene.nextShootingStar = now + randomBetween(24000, 52000);
  }
}

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function appendUserMessage(text, scroll = true) {
  stopWelcomeSky();
  el.messages.querySelector(".welcome")?.remove();
  const row = messageShell("user", "You");
  row.stack.appendChild(bubble(text, false));
  el.messages.appendChild(row.root);
  if (scroll) {
    state.followOutput = true;
    scrollToBottom(true);
  }
}

function appendAssistantMessage(message, scroll = true) {
  el.messages.querySelector(".welcome")?.remove();
  const row = messageShell("assistant", "S");
  const thoughtItems = message.thoughtItems || [];

  if (thoughtItems.length) {
    const thinkingBlock = detailBlock("Thinking", "reasoning", "", false);
    const body = thinkingBlock.querySelector(".block-body");
    thoughtItems.forEach((item) => {
      if (item.type === "tool") {
        body?.appendChild(toolIndicator(item.name));
      } else if (item.text) {
        body?.appendChild(thinkingContent(item.text));
      }
    });
    row.stack.appendChild(thinkingBlock);
  }

  if (message.content) row.stack.appendChild(bubble(message.content, true));
  el.messages.appendChild(row.root);
  if (scroll) scrollToBottom(true);
}

function messageShell(role, avatarText) {
  const root = document.createElement("article");
  root.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  if (role === "assistant") {
    const image = document.createElement("img");
    image.src = "/avatar.png";
    image.alt = "Selene";
    avatar.appendChild(image);
  } else {
    avatar.textContent = avatarText;
  }

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
  block.className = `detail-block${running ? " running" : ""}`;

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "block-toggle";
  toggle.innerHTML = `
    <span class="block-title">${escapeHTML(title)} <span class="pill">${escapeHTML(label)}</span></span>
    <span aria-hidden="true">+</span>
  `;

  const body = document.createElement("div");
  body.className = "block-body";

  const blockContent = thinkingContent(content || (running ? "Running..." : ""));

  toggle.addEventListener("click", () => {
    const isOpen = block.classList.toggle("open");
    if (isOpen) scrollThinkingToBottom(block);
  });
  body.appendChild(blockContent);
  block.append(toggle, body);
  return block;
}

function thinkingContent(text = "") {
  const content = document.createElement("div");
  content.className = "block-content";
  content.textContent = text;
  return content;
}

function scrollThinkingToBottom(block) {
  if (!block?.classList.contains("open")) return;
  const body = block.querySelector(".block-body");
  if (!body) return;
  body.scrollTop = body.scrollHeight;
}

function toolIndicator(name, running = false) {
  const indicator = document.createElement("div");
  indicator.className = `tool-indicator${running ? " running" : ""}`;

  const label = document.createElement("span");
  label.className = "pill";
  label.textContent = "tool";

  const status = document.createElement("span");
  status.className = "tool-indicator-status";
  status.textContent = running ? `Using ${humanizeToolName(name)}…` : `Used ${humanizeToolName(name)}`;

  indicator.append(label, status);
  return indicator;
}

function humanizeToolName(name) {
  return String(name || "tool")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

async function sendMessage() {
  if (state.isGenerating) return;
  const text = el.input.value.trim();
  if (!text) return;

  closeSlashMenu();
  appendUserMessage(text);
  if (!text.startsWith("/")) state.history.push({ role: "user", content: text });
  el.input.value = "";
  resizeComposer();
  updateComposerState();

  state.isGenerating = true;
  state.controller = new AbortController();
  const generation = {
    controller: state.controller,
    sessionName: state.activeSessionName,
    detached: false
  };
  state.generation = generation;
  resetStream();
  updateComposerState();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_name: generation.sessionName }),
      signal: state.controller.signal
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await readEventStream(response, generation);
    if (state.generation === generation) finishGeneration(generation);
  } catch (error) {
    if (error.name !== "AbortError") {
      toast("Selene lost the response stream. Check Ollama and try again.");
      finishGeneration(generation);
    }
  }
}

async function readEventStream(response, generation) {
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
        handleStreamEvent(JSON.parse(line.slice(6)), generation);
      } catch {
        // Ignore malformed SSE chunks.
      }
    }
  }
}

function handleStreamEvent(event, generation) {
  const visible = state.generation === generation && !generation.detached;
  switch (event.type) {
    case "conversation_started":
      generation.sessionName = event.session_name || generation.sessionName;
      if (visible) {
        state.activeSessionName = generation.sessionName;
        el.title.textContent = cleanSessionName(state.activeSessionName);
        document.title = titleForSession(state.activeSessionName);
      }
      break;
    case "status":
      if (!visible) break;
      appendStatus(event.message || "");
      break;
    case "thinking_start":
      if (!visible) break;
      ensureAssistantStack();
      if (state.stream.assistantBubble) {
        state.stream.assistantBubble = null;
        state.stream.assistantText = "";
        state.stream.thinkingBlock = null;
        state.stream.thinkingContent = null;
        state.stream.thinkingText = "";
      }
      if (!state.stream.thinkingBlock) {
        state.stream.thinkingBlock = detailBlock("Thinking", "reasoning", "", true);
        state.stream.thinkingContent = state.stream.thinkingBlock.querySelector(".block-content");
        state.stream.assistantStack.appendChild(state.stream.thinkingBlock);
      } else {
        state.stream.thinkingContent = thinkingContent();
        state.stream.thinkingBlock.querySelector(".block-body")?.appendChild(state.stream.thinkingContent);
        state.stream.thinkingBlock.classList.add("running");
      }
      state.stream.thinkingText = "";
      scrollToBottom();
      break;
    case "thinking_chunk":
      if (!visible) break;
      state.stream.thinkingText += event.text || "";
      if (state.stream.thinkingContent) state.stream.thinkingContent.textContent = state.stream.thinkingText;
      scrollThinkingToBottom(state.stream.thinkingBlock);
      break;
    case "thinking_end":
      if (!visible) break;
      state.stream.thinkingBlock?.classList.remove("running");
      state.stream.thinkingBlock?.classList.remove("open");
      break;
    case "tool_start":
      if (!visible) break;
      ensureAssistantStack();
      if (!state.stream.thinkingBlock) {
        state.stream.thinkingBlock = detailBlock("Thinking", "reasoning", "", false);
        state.stream.thinkingContent = state.stream.thinkingBlock.querySelector(".block-content");
        state.stream.assistantStack.appendChild(state.stream.thinkingBlock);
      }
      state.stream.thinkingBlock.classList.add("running");
      state.stream.lastToolBlock = toolIndicator(event.name || "tool", true);
      state.stream.thinkingBlock.querySelector(".block-body")?.appendChild(state.stream.lastToolBlock);
      scrollThinkingToBottom(state.stream.thinkingBlock);
      scrollToBottom();
      break;
    case "tool_end":
      if (!visible) break;
      if (state.stream.lastToolBlock) {
        state.stream.lastToolBlock.classList.remove("running");
        const status = state.stream.lastToolBlock.querySelector(".tool-indicator-status");
        if (status) status.textContent = `Used ${humanizeToolName(event.name || "tool")}`;
      }
      state.stream.thinkingBlock?.classList.remove("running");
      state.stream.lastToolBlock = null;
      break;
    case "content_chunk":
      if (!visible) break;
      ensureAssistantStack();
      if (!state.stream.assistantBubble) {
        state.stream.thinkingBlock?.classList.remove("open", "running");
        state.stream.assistantBubble = document.createElement("div");
        state.stream.assistantBubble.className = "bubble";
        state.stream.assistantStack.appendChild(state.stream.assistantBubble);
        state.stream.thinkingBlock = null;
        state.stream.thinkingContent = null;
        state.stream.thinkingText = "";
        state.stream.lastToolBlock = null;
      }
      state.stream.assistantText += event.text || event.content || "";
      state.stream.assistantBubble.innerHTML = renderMarkdown(state.stream.assistantText);
      scrollToBottom();
      break;
    case "token_usage":
      if (!visible) break;
      updateContextMeter(event.total, event.budget);
      break;
    case "done":
      const viewedGeneratingConversation = state.activeSessionName === generation.sessionName;
      const finishedName = event.active_session_name || generation.sessionName;
      if (event.saved_sessions) state.savedSessions = event.saved_sessions;
      if (visible) {
        if (event.history) state.history = event.history;
        state.activeSessionName = finishedName;
        el.title.textContent = cleanSessionName(state.activeSessionName);
        document.title = titleForSession(state.activeSessionName);
        renderSessions();
        updateContextMeter();
        finishGeneration(generation);
      } else {
        finishGeneration(generation);
        if (viewedGeneratingConversation) {
          loadState();
        } else {
          renderSessions();
          toast(`Response ready in “${cleanSessionName(finishedName)}”.`);
        }
      }
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
  state.stream.thinkingContent = null;
  state.stream.thinkingText = "";
  state.stream.lastToolBlock = null;
}

function settleStreamActivity(interrupted = false) {
  // Streaming can end without a thinking_end/tool_end event when fetch is
  // aborted or the connection fails. Settle the DOM before resetStream drops
  // the only references to these still-visible elements.
  state.stream.thinkingBlock?.classList.remove("open", "running");
  if (state.stream.lastToolBlock) {
    state.stream.lastToolBlock.classList.remove("running");
    if (interrupted) {
      const status = state.stream.lastToolBlock.querySelector(".tool-indicator-status");
      if (status) status.textContent = "Tool use stopped";
    }
  }
}

function finishGeneration(generation = state.generation, { interrupted = false } = {}) {
  if (generation && state.generation !== generation) return;
  settleStreamActivity(interrupted);
  state.isGenerating = false;
  state.controller = null;
  state.generation = null;
  resetStream();
  updateComposerState();
}

function resetComposer({ clear = true } = {}) {
  closeSlashMenu();
  if (!el.input) return;
  el.input.disabled = false;
  el.input.readOnly = false;
  if (clear) el.input.value = "";
  resizeComposer();
  updateComposerState();
  requestAnimationFrame(() => el.input?.focus({ preventScroll: true }));
}

function stopGeneration({ refresh = true } = {}) {
  state.controller?.abort();
  toast("Generation stopped.");
  finishGeneration(state.generation, { interrupted: true });
  if (refresh) refreshSessions().catch(() => {});
}

function appendStatus(text) {
  const status = document.createElement("div");
  status.className = "status-line";
  status.textContent = text;
  el.messages.appendChild(status);
  scrollToBottom();
}

async function clearConversation() {
  if (state.isGenerating) stopGeneration({ refresh: false });
  try {
    await fetch("/api/clear-session", { method: "POST" });
    state.history = [];
    state.activeSessionName = "New conversation";
    el.title.textContent = "New conversation";
    document.title = "Selene";
    renderWelcome();
    updateContextMeter();
    resetComposer();
  } catch {
    toast("Could not clear the conversation.");
    resetComposer();
  }
}

async function newConversation() {
  // There is one composer and one stream controller. Leaving the old stream
  // detached keeps isGenerating=true and makes the fresh tab act locked.
  if (state.isGenerating || state.generation) stopGeneration({ refresh: false });
  try {
    const response = await fetch("/api/new-session", { method: "POST" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    state.history = [];
    state.activeSessionName = "New conversation";
    state.savedSessions = data.saved_sessions || state.savedSessions;
    el.title.textContent = "New conversation";
    document.title = "Selene";
    state.followOutput = true;
    resetStream();
    renderWelcome();
    updateContextMeter();
    renderSessions();
    resetComposer();
  } catch {
    toast("Could not start a new conversation.");
    resetComposer();
  }
}

async function deleteSession(name) {
  if (!window.confirm(`Delete “${cleanSessionName(name)}”? This cannot be undone.`)) return;
  const deletingActiveSession = name === state.activeSessionName;
  if (deletingActiveSession && (state.isGenerating || state.generation)) {
    stopGeneration({ refresh: false });
  }
  try {
    const response = await fetch("/api/delete-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    state.savedSessions = data.saved_sessions || state.savedSessions.filter((session) => session !== name);
    const backendStartedNewSession = ["Active Session", "New conversation"].includes(data.active_session_name);
    if (deletingActiveSession || backendStartedNewSession) {
      if (state.isGenerating || state.generation) stopGeneration({ refresh: false });
      state.history = [];
      state.activeSessionName = "New conversation";
      el.title.textContent = "New conversation";
      document.title = "Selene";
      state.followOutput = true;
      resetStream();
      renderWelcome();
      updateContextMeter();
      resetComposer();
    }
    renderSessions();
    if (!deletingActiveSession && !backendStartedNewSession && !state.isGenerating) {
      resetComposer({ clear: false });
    }
    toast("Chat deleted.");
  } catch {
    toast("Could not delete that chat.");
    if (deletingActiveSession) resetComposer();
    else if (!state.isGenerating) resetComposer({ clear: false });
  }
}

async function refreshSessions() {
  const response = await fetch("/api/settings");
  if (!response.ok) return;
  const data = await response.json();
  state.savedSessions = data.saved_sessions || [];
  state.activeSessionName = data.active_session_name || state.activeSessionName;
  renderSessions();
}

function openSettings() {
  el.settingsBackdrop.hidden = false;
  requestAnimationFrame(() => {
    el.settingsBackdrop.classList.add("open");
    el.settingsPanel.classList.add("open");
    el.settingsPanel.setAttribute("aria-hidden", "false");
  });
  document.getElementById("settings-close")?.focus();
}

function closeSettings() {
  el.settingsBackdrop?.classList.remove("open");
  el.settingsPanel?.classList.remove("open");
  el.settingsPanel?.setAttribute("aria-hidden", "true");
  setTimeout(() => { if (el.settingsBackdrop) el.settingsBackdrop.hidden = true; }, 180);
  document.getElementById("settings-btn")?.focus();
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
  if (state.isGenerating || state.generation) stopGeneration({ refresh: false });
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
    .filter((message) => message.role !== "system")
    .map((message) => [
    message.role || "",
    message.content || "",
    message.thinking || "",
    JSON.stringify(message.tool_calls || "")
  ].join("\n")).join("\n");
  const systemPrompt = state.settings.system ||
    (state.history.find((message) => message.role === "system")?.content || "");
  return estimateTokens(`${systemPrompt}\n${historyText}\n${el.input?.value || ""}`);
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
  const maxHeight = Number.parseFloat(getComputedStyle(el.input).maxHeight) || 220;
  el.input.style.height = `${Math.min(el.input.scrollHeight, maxHeight)}px`;
}

function scrollToBottom(force = false) {
  if (!el.messages) return;
  if (force || state.followOutput) el.messages.scrollTop = el.messages.scrollHeight;
}

function distanceFromBottom() {
  if (!el.messages) return 0;
  return el.messages.scrollHeight - el.messages.scrollTop - el.messages.clientHeight;
}

const LATEX_UNICODE_SYMBOLS = Object.freeze({
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ϵ",
  zeta: "ζ", eta: "η", theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ",
  lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", omicron: "ο", pi: "π", varpi: "ϖ",
  rho: "ρ", varrho: "ϱ", sigma: "σ", varsigma: "ς", tau: "τ", upsilon: "υ",
  phi: "φ", varphi: "ϕ", chi: "χ", psi: "ψ", omega: "ω",
  Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π",
  Sigma: "Σ", Upsilon: "Υ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",

  plusmn: "±", pm: "±", mp: "∓", times: "×", div: "÷", cdot: "·", ast: "∗",
  star: "⋆", circ: "∘", bullet: "•", sqrt: "√", sum: "∑", prod: "∏",
  coprod: "∐", int: "∫", iint: "∬", iiint: "∭", oint: "∮", partial: "∂",
  nabla: "∇", infinity: "∞", infty: "∞", hbar: "ℏ", ell: "ℓ", degree: "°",
  oplus: "⊕", ominus: "⊖", otimes: "⊗", oslash: "⊘", odot: "⊙",
  bigoplus: "⨁", bigotimes: "⨂", bigodot: "⨀", dagger: "†", ddagger: "‡",

  eq: "=", neq: "≠", ne: "≠", equiv: "≡", approx: "≈", sim: "∼", simeq: "≃",
  cong: "≅", propto: "∝", le: "≤", leq: "≤", ge: "≥", geq: "≥",
  ll: "≪", gg: "≫", prec: "≺", succ: "≻", preceq: "⪯", succeq: "⪰",
  lt: "&lt;", gt: "&gt;", parallel: "∥", nparallel: "∦", perp: "⊥", mid: "∣",
  asymp: "≍", doteq: "≐", models: "⊨", vdots: "⋮", ddots: "⋱", dots: "…",
  ldots: "…", cdots: "⋯",

  forall: "∀", exists: "∃", nexists: "∄", neg: "¬", lnot: "¬", land: "∧",
  wedge: "∧", lor: "∨", vee: "∨", therefore: "∴", because: "∵", top: "⊤", bot: "⊥",
  emptyset: "∅", varnothing: "∅", in: "∈", notin: "∉", ni: "∋", notni: "∌",
  subset: "⊂", subseteq: "⊆", nsubseteq: "⊈", supset: "⊃", supseteq: "⊇",
  nsupseteq: "⊉", cup: "∪", cap: "∩", uplus: "⊎", setminus: "∖",
  bigcup: "⋃", bigcap: "⋂", sqsubset: "⊏", sqsupset: "⊐", sqsubseteq: "⊑",
  sqsupseteq: "⊒", sqcup: "⊔", sqcap: "⊓",

  leftarrow: "←", gets: "←", rightarrow: "→", to: "→", leftrightarrow: "↔",
  Leftarrow: "⇐", Rightarrow: "⇒", implies: "⇒", Leftrightarrow: "⇔", iff: "⇔",
  mapsto: "↦", hookleftarrow: "↩", hookrightarrow: "↪", uparrow: "↑",
  downarrow: "↓", updownarrow: "↕", Uparrow: "⇑", Downarrow: "⇓",
  Updownarrow: "⇕", nearrow: "↗", searrow: "↘", swarrow: "↙", nwarrow: "↖",
  longleftarrow: "⟵", longrightarrow: "⟶", longleftrightarrow: "⟷",
  Longleftarrow: "⟸", Longrightarrow: "⟹", Longleftrightarrow: "⟺",
  leftharpoonup: "↼", leftharpoondown: "↽", rightharpoonup: "⇀",
  rightharpoondown: "⇁", rightleftharpoons: "⇌", rightsquigarrow: "⇝",

  angle: "∠", measuredangle: "∡", triangle: "△", square: "□", diamond: "◇",
  lozenge: "◊", checkmark: "✓", clubsuit: "♣", diamondsuit: "♦",
  heartsuit: "♥", spadesuit: "♠", aleph: "ℵ", beth: "ℶ", gimel: "ℷ",
  Re: "ℜ", Im: "ℑ", wp: "℘", prime: "′", backprime: "‵",
  copyright: "©", registered: "®", pounds: "£", euro: "€", yen: "¥",

  quad: " ", qquad: "  ", left: "", right: ""
});

function renderLatexSymbols(text) {
  let value = String(text || "");
  // Preserve the contents of common presentation commands without attempting
  // full TeX layout. Input has already been HTML-escaped by inlineMarkdown.
  for (let depth = 0; depth < 3; depth += 1) {
    value = value.replace(
      /\\(?:text|textrm|textsf|texttt|mathrm|mathbf|mathit|mathsf|mathtt|mathcal)\s*\{([^{}]*)\}/g,
      "$1"
    );
  }
  value = value.replace(/\\([A-Za-z]+)(?![A-Za-z])/g, (match, command) => (
    Object.prototype.hasOwnProperty.call(LATEX_UNICODE_SYMBOLS, command)
      ? LATEX_UNICODE_SYMBOLS[command]
      : match
  ));
  return value
    .replace(/\\([{}%$#_])/g, "$1")
    .replace(/\\&amp;/g, "&amp;")
    .replace(/\\[,;:]/g, " ")
    .replace(/\\!/g, "");
}

function splitMarkdownTableRow(line) {
  const value = String(line || "").trim();
  const cells = [];
  let cell = "";
  let inCode = false;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (character === "`" && value[index - 1] !== "\\") {
      inCode = !inCode;
      cell += character;
    } else if (character === "\\" && value[index + 1] === "|") {
      cell += "|";
      index += 1;
    } else if (character === "|" && !inCode) {
      cells.push(cell.trim());
      cell = "";
    } else {
      cell += character;
    }
  }
  cells.push(cell.trim());
  if (value.startsWith("|")) cells.shift();
  if (value.endsWith("|") && value[value.length - 2] !== "\\") cells.pop();
  return cells;
}

function tableAlignments(line) {
  const cells = splitMarkdownTableRow(line);
  if (!cells.length || !cells.every((cell) => /^:?-{3,}:?$/.test(cell))) return null;
  return cells.map((cell) => {
    if (cell.startsWith(":") && cell.endsWith(":")) return "center";
    if (cell.endsWith(":")) return "right";
    return "left";
  });
}

function renderTableCell(tag, content, alignment) {
  return `<${tag} class="align-${alignment}">${inlineMarkdown(content)}</${tag}>`;
}

function renderMarkdownTable(headers, alignments, rows) {
  const width = alignments.length;
  const normalizedHeaders = Array.from({ length: width }, (_, index) => headers[index] || "");
  const head = normalizedHeaders
    .map((cell, index) => renderTableCell("th", cell, alignments[index]))
    .join("");
  const body = rows.map((row) => {
    const cells = Array.from({ length: width }, (_, index) => row[index] || "");
    return `<tr>${cells.map((cell, index) => renderTableCell("td", cell, alignments[index])).join("")}</tr>`;
  }).join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead>${body ? `<tbody>${body}</tbody>` : ""}</table></div>`;
}

function renderListItem(content) {
  const task = String(content || "").match(/^\[([ xX])\]\s+(.+)$/);
  if (!task) return `<li>${inlineMarkdown(content)}</li>`;
  const checked = task[1].toLowerCase() === "x";
  return `<li class="task-list-item"><input type="checkbox" disabled${checked ? " checked" : ""} aria-label="${checked ? "Completed" : "Not completed"}"><span>${inlineMarkdown(task[2])}</span></li>`;
}

function renderCodeBlock(code, language = "") {
  const safeLanguage = escapeHTML(language || "");
  const languageClass = safeLanguage ? ` class="language-${safeLanguage}"` : "";
  return `<div class="code-block"><div class="code-toolbar"><span>${safeLanguage || "code"}</span><button class="code-copy-btn" type="button" aria-label="Copy code to clipboard">Copy</button></div><pre><code${languageClass}>${escapeHTML(code)}</code></pre></div>`;
}

async function copyCodeBlock(button) {
  const code = button.closest(".code-block")?.querySelector("code")?.textContent;
  if (code == null) return;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(code);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = code;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      textarea.remove();
      if (!copied) throw new Error("Clipboard command was rejected");
    }
    button.textContent = "Copied";
    button.classList.add("copied");
    setTimeout(() => {
      if (!button.isConnected) return;
      button.textContent = "Copy";
      button.classList.remove("copied");
    }, 1400);
  } catch {
    toast("Could not copy that code block.");
  }
}

function renderMarkdown(text) {
  const lines = String(text || "").replace(/\r/g, "").split("\n");
  const output = [];
  let paragraph = [];
  let listType = "";
  let inCode = false;
  let code = [];
  let language = "";

  const flushParagraph = () => {
    if (paragraph.length) output.push(`<p>${inlineMarkdown(paragraph.join("\n")).replace(/\n/g, "<br>")}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (listType) output.push(`</${listType}>`);
    listType = "";
  };

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    const fence = line.match(/^```\s*([\w+-]*)/);
    if (fence) {
      if (inCode) {
        output.push(renderCodeBlock(code.join("\n"), language));
        code = []; language = ""; inCode = false;
      } else {
        flushParagraph(); closeList(); inCode = true; language = fence[1] || "";
      }
      continue;
    }
    if (inCode) { code.push(line); continue; }
    if (!line.trim()) { flushParagraph(); closeList(); continue; }

    const alignments = lineIndex + 1 < lines.length ? tableAlignments(lines[lineIndex + 1]) : null;
    const headers = alignments && line.includes("|") ? splitMarkdownTableRow(line) : null;
    if (headers && headers.length === alignments.length) {
      flushParagraph(); closeList();
      const rows = [];
      lineIndex += 2;
      while (lineIndex < lines.length && lines[lineIndex].trim() && lines[lineIndex].includes("|")) {
        rows.push(splitMarkdownTableRow(lines[lineIndex]));
        lineIndex += 1;
      }
      lineIndex -= 1;
      output.push(renderMarkdownTable(headers, alignments, rows));
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) { flushParagraph(); closeList(); const level = heading[1].length; output.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`); continue; }
    if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) { flushParagraph(); closeList(); output.push("<hr>"); continue; }
    const quote = line.match(/^>\s?(.*)$/);
    if (quote) { flushParagraph(); closeList(); output.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`); continue; }
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      flushParagraph();
      const wanted = ordered ? "ol" : "ul";
      if (listType !== wanted) { closeList(); output.push(`<${wanted}>`); listType = wanted; }
      output.push(renderListItem((unordered || ordered)[1]));
      continue;
    }
    closeList(); paragraph.push(line);
  }
  if (inCode) output.push(renderCodeBlock(code.join("\n"), language));
  flushParagraph(); closeList();
  return output.join("");
}

function inlineMarkdown(text) {
  let value = escapeHTML(text);
  const codeSpans = [];
  value = value.replace(/`([^`]+)`/g, (_, code) => {
    codeSpans.push(`<code>${code}</code>`);
    return `\u0000CODE${codeSpans.length - 1}\u0000`;
  });
  value = renderLatexSymbols(value);
  value = value
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/~~([^~]+)~~/g, "<del>$1</del>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[^_])_([^_\n]+)_/g, "$1<em>$2</em>");
  return value.replace(/\u0000CODE(\d+)\u0000/g, (_, index) => codeSpans[Number(index)]);
}

function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
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
    .replace(/_\d{8}_\d{6}(?:_\d+)?$/, "")
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
