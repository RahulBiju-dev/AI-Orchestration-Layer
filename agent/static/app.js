"use strict";

const DEFAULT_SETTINGS = {
  // Explicit per-session overrides only. Effective values come from the
  // backend-selected hardware profile.
  options: {},
  runtime_profile: "manual",
  verbose: true,
  wordwrap: true,
  system: "",
  history: true,
  format: "",
  think: true,
  agent_mode: "normal",
  model_id: "local:default"
};

const AGENT_MODES = {
  normal: { label: "Fast", icon: "○" },
  ultra: { label: "Ultra thinking", icon: "✦" },
  "deep-research": { label: "Deep research", icon: "◎" }
};

const FALLBACK_MODEL_OPTIONS = {
  temperature: 0.25,
  top_p: 0.85,
  top_k: 40,
  repeat_penalty: 1.08,
  num_predict: 2048,
  num_batch: 128
};

function createRuntimeId(prefix) {
  const random = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}:${random}`;
}

function browserClientId() {
  const key = "selene-client-id";
  try {
    const existing = sessionStorage.getItem(key);
    if (existing) return existing;
    const created = createRuntimeId("tab");
    sessionStorage.setItem(key, created);
    return created;
  } catch {
    return createRuntimeId("tab");
  }
}

const CLIENT_ID = browserClientId();

function apiHeaders(json = false) {
  const headers = { "X-Selene-Client-ID": CLIENT_ID };
  if (json) headers["Content-Type"] = "application/json";
  return headers;
}

async function apiError(response, fallback) {
  try {
    const payload = await response.json();
    return new Error(String(payload.error || fallback));
  } catch {
    return new Error(fallback);
  }
}

const SLASH_COMMANDS = [
  { command: "/help", description: "Show available commands" },
  { command: "/theme", description: "Choose a place color theme" },
  { command: "/clear", description: "Clear conversation history" },
  { command: "/save ", description: "Save current session with an optional name" },
  { command: "/load ", description: "Load a saved session by name or index" },
  { command: "/set parameter ", description: "Set a model parameter, e.g. temperature 0.7" },
  { command: "/set profile ", description: "Select auto, low-vram, balanced, or manual" },
  { command: "/model ", description: "List or select a configured model" },
  { command: "/set model ", description: "Select a configured model" },
  { command: "/fast", description: "Switch this conversation to Fast mode" },
  { command: "/ultrathink", description: "Switch this conversation to Ultra Thinking mode" },
  { command: "/deepresearch", description: "Switch this conversation to Deep Research mode" },
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

const THEME_STORAGE_KEY = "selene-web-theme";
const PLACE_THEMES = [
  { id: "oslo", name: "Oslo", description: "Monochrome grey & white", background: "#101010", surface: "#171717", primary: "#cfcfcf", accent: "#e8e8e8" },
  { id: "tokyo", name: "Tokyo", description: "Futuristic hollow blue", background: "#070b14", surface: "#0c1220", primary: "#5ec8ff", accent: "#7dcfff" },
  { id: "rome", name: "Rome", description: "Royal gold & marble", background: "#120e18", surface: "#1a1424", primary: "#d4af37", accent: "#f0d78c" },
  { id: "amazon", name: "Amazon", description: "Deep rainforest", background: "#0a120c", surface: "#101a12", primary: "#6dbf6d", accent: "#a8d4a0" },
  { id: "cairo", name: "Cairo", description: "Sand & desert brown", background: "#14100c", surface: "#1c1610", primary: "#c4a574", accent: "#e0c898" },
  { id: "kyoto", name: "Kyoto", description: "Soft sakura dusk", background: "#141018", surface: "#1c1620", primary: "#d4a0c0", accent: "#ebbcba" },
  { id: "bergen", name: "Bergen", description: "Fjord frost", background: "#121820", surface: "#1a222c", primary: "#88c0d0", accent: "#a3d4e0" },
  { id: "marrakech", name: "Marrakech", description: "Terracotta sunset", background: "#140e0c", surface: "#1c1410", primary: "#e08860", accent: "#f0b090" },
  { id: "shanghai", name: "Shanghai", description: "Neon night market", background: "#0a0814", surface: "#12101c", primary: "#e060c0", accent: "#60d0f0" },
  { id: "reykjavik", name: "Reykjavik", description: "Aurora ice", background: "#080e12", surface: "#0e161c", primary: "#60e0b8", accent: "#a080e0" },
  { id: "venice", name: "Venice", description: "Lagoon teal & rose", background: "#0c1214", surface: "#141c1e", primary: "#50b0b8", accent: "#e0a0a8" },
  { id: "seoul", name: "Seoul", description: "Electric violet night", background: "#0c0a14", surface: "#14101e", primary: "#a070f0", accent: "#c0a0ff" },
  { id: "santorini", name: "Santorini", description: "Aegean blue & white", background: "#0a1018", surface: "#121a24", primary: "#60b0e8", accent: "#e8f0f8" },
  { id: "havana", name: "Havana", description: "Tropical coral & mint", background: "#120c0c", surface: "#1a1212", primary: "#e07070", accent: "#70d0b0" }
];
const PLACE_THEME_IDS = new Set(PLACE_THEMES.map((theme) => theme.id));

function storedTheme() {
  try {
    const stored = String(localStorage.getItem(THEME_STORAGE_KEY) || "").trim().toLowerCase();
    if (PLACE_THEME_IDS.has(stored)) return stored;
  } catch {
    // Storage can be unavailable in privacy-restricted browser contexts.
  }
  return "oslo";
}

const INITIAL_THEME = storedTheme();
document.documentElement.dataset.theme = INITIAL_THEME;

const state = {
  history: [],
  settings: cloneSettings(DEFAULT_SETTINGS),
  runtime: { effective_options: { ...FALLBACK_MODEL_OPTIONS } },
  savedSessions: [],
  activeSessionName: "New conversation",
  modelName: "selene",
  models: [
    { id: "local:default", display_name: "Gemma 4 E4B", provider: "Ollama (local)", capabilities: [] }
  ],
  theme: INITIAL_THEME,
  isGenerating: false,
  followOutput: true,
  controller: null,
  generation: null,
  generations: new Map(),
  viewVersion: 0,
  clientId: CLIENT_ID,
  ollama: { status: "", reason: "" },
  vaults: { data: null, loaded: false, loading: false, error: "" },
  stream: {
    assistantStack: null,
    assistantBubble: null,
    assistantText: "",
    thinkingBlock: null,
    thinkingContent: null,
    thinkingText: "",
    renderFrame: null,
    modeStatusLine: null,
    // Keyed `${round}:${id}` -> { row, startedAt, name }. tool_start ids are a
    // per-batch index that restarts every round, so a bare id collides.
    activeToolBlocks: new Map(),
    toolRound: 0
  },
  slash: {
    open: false,
    selected: 0,
    matches: []
  },
  promptRecall: {
    index: null
  },
  voice: {
    recognition: null,
    active: false,
    baseText: "",
    suppressError: false
  },
  sky: {
    frame: null,
    watchdog: null,
    resizeObserver: null,
    detachPointer: null,
    scene: null
  }
};
let settingsWriteChain = Promise.resolve();
let lastRuntimeWarning = "";
let themeCloseTimer = null;
let themeTriggerElement = null;
let startupProfilePromptShown = false;

const NEW_CONVERSATION_NAMES = new Set(["", "Active Session", "New conversation"]);

function isNewConversationName(name) {
  return NEW_CONVERSATION_NAMES.has(String(name || ""));
}

function generationForSession(name = state.activeSessionName) {
  const requested = String(name || "");
  for (const generation of state.generations.values()) {
    if (generation.sessionName === requested) return generation;
    if (
      isNewConversationName(requested)
      && isNewConversationName(generation.sessionName)
      && generation.viewVersion === state.viewVersion
    ) {
      return generation;
    }
  }
  return null;
}

function isGenerationVisible(generation) {
  return Boolean(generation && generationForSession(state.activeSessionName) === generation);
}

function syncActiveGeneration() {
  const generation = generationForSession();
  state.generation = generation;
  state.controller = generation?.controller || null;
  state.isGenerating = Boolean(generation);
  return generation;
}

function markCurrentGenerationBackgrounded() {
  const generation = generationForSession();
  if (generation) generation.wasBackgrounded = true;
}

function selectConversationView(name) {
  if (String(name || "") !== state.activeSessionName) {
    markCurrentGenerationBackgrounded();
    state.viewVersion += 1;
  }
  state.activeSessionName = name || "New conversation";
  syncActiveGeneration();
}

async function waitForPendingConversationIdentity() {
  const generation = generationForSession();
  if (generation && isNewConversationName(generation.sessionName)) {
    await generation.identityReady;
  }
}

function reportRuntimeWarnings(runtime) {
  const warnings = Array.isArray(runtime?.warnings) ? runtime.warnings : [];
  const signature = warnings.join("\n");
  if (signature && signature !== lastRuntimeWarning) toast(warnings[0]);
  lastRuntimeWarning = signature;
}

const el = {};

document.addEventListener("DOMContentLoaded", () => {
  bindElements();
  bindEvents();
  restoreVaultPanel();
  loadState();
});

// Restores the panel's remembered open state. Expanding is what triggers the
// (potentially slow) first /api/vaults call, never page load itself.
function restoreVaultPanel() {
  const section = el.vaultToggle?.closest(".vault-section");
  if (!section || !vaultPanelOpen()) return;
  section.classList.add("open");
  el.vaultToggle.setAttribute("aria-expanded", "true");
  loadVaults();
}

function bindElements() {
  el.messages = document.getElementById("messages");
  el.form = document.getElementById("composer-form");
  el.input = document.getElementById("chat-input");
  el.mic = document.getElementById("mic-btn");
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
  el.profileSetting = document.getElementById("setting-profile");
  el.profileBackdrop = document.getElementById("profile-backdrop");
  el.profileDialog = document.getElementById("profile-dialog");
  el.profileChoice = document.getElementById("profile-choice");
  el.profileDescription = document.getElementById("profile-choice-description");
  el.profileRuntimeSummary = document.getElementById("profile-runtime-summary");
  el.profileApply = document.getElementById("profile-apply");
  el.system = document.getElementById("setting-system");
  el.chatShell = document.querySelector(".chat-shell");
  el.settingsView = document.getElementById("settings-view");
  el.settingsBtn = document.getElementById("settings-btn");
  el.settingsBackdrop = document.getElementById("settings-backdrop");
  el.settingsClose = document.getElementById("settings-close");
  el.topbarLabel = document.getElementById("topbar-label");
  el.ollamaStatus = document.getElementById("ollama-status");
  el.ollamaDot = document.getElementById("ollama-dot");
  el.ollamaReason = document.getElementById("ollama-reason");
  el.vaultToggle = document.getElementById("vault-toggle");
  el.vaultList = document.getElementById("vault-list");
  el.vaultCount = document.getElementById("vault-count");
  el.slashMenu = document.getElementById("slash-menu");
  el.modePicker = document.getElementById("mode-picker");
  el.modeTrigger = document.getElementById("mode-trigger");
  el.modeMenu = document.getElementById("mode-menu");
  el.modeLabel = document.getElementById("mode-label");
  el.modeIcon = document.getElementById("mode-icon");
  el.modeClear = document.getElementById("mode-clear");
  el.modelPicker = document.getElementById("model-picker");
  el.modelTrigger = document.getElementById("model-trigger");
  el.modelMenu = document.getElementById("model-menu");
  el.modelLabel = document.getElementById("model-label");
  el.themeButton = document.getElementById("theme-btn");
  el.themeBackdrop = document.getElementById("theme-backdrop");
  el.themeDialog = document.getElementById("theme-dialog");
  el.themeOptions = document.getElementById("theme-options");
}

function bindEvents() {
  initializeVoiceInput();

  el.form?.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.isGenerating) stopGeneration();
    else sendMessage();
  });

  el.input?.addEventListener("input", () => {
    resetPromptRecall();
    resizeComposer();
    updateComposerState();
    updateSlashMenu();
  });

  el.input?.addEventListener("keydown", (event) => {
    if (handleSlashKeydown(event)) return;
    if (handlePromptHistoryKeydown(event)) return;
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
    if (!el.modePicker?.contains(event.target)) closeModeMenu();
    if (!el.modelPicker?.contains(event.target)) closeModelMenu();
  });

  el.modelTrigger?.addEventListener("click", (event) => {
    event.stopPropagation();
    if (state.isGenerating) return;
    if (el.modelMenu?.hidden) openModelMenu();
    else closeModelMenu();
  });
  el.modelTrigger?.addEventListener("keydown", handleModelTriggerKeydown);
  el.modelMenu?.addEventListener("keydown", handleModelMenuKeydown);
  el.modelMenu?.addEventListener("click", (event) => {
    const option = event.target.closest("[data-model-id]");
    if (option) selectModel(option.dataset.modelId);
  });

  el.modeTrigger?.addEventListener("click", (event) => {
    event.stopPropagation();
    if (state.isGenerating) return;
    if (el.modeMenu?.hidden) openModeMenu();
    else closeModeMenu();
  });
  el.modeTrigger?.addEventListener("keydown", handleModeTriggerKeydown);
  el.modeMenu?.addEventListener("keydown", handleModeMenuKeydown);
  el.modeMenu?.addEventListener("click", (event) => {
    const option = event.target.closest("[data-agent-mode]");
    if (option) setAgentMode(option.dataset.agentMode);
  });
  el.modeClear?.addEventListener("click", (event) => {
    event.stopPropagation();
    setAgentMode("normal");
  });

  document.getElementById("new-chat-btn")?.addEventListener("click", newConversation);
  document.getElementById("settings-btn")?.addEventListener("click", openSettings);
  el.settingsClose?.addEventListener("click", closeSettings);
  el.settingsBackdrop?.addEventListener("click", closeSettings);
  el.vaultToggle?.addEventListener("click", toggleVaultPanel);
  el.settingsView?.addEventListener("scroll", syncSettingsNav, { passive: true });
  el.themeButton?.addEventListener("click", () => openThemeDialog(el.themeButton));
  document.getElementById("theme-close")?.addEventListener("click", closeThemeDialog);
  el.themeBackdrop?.addEventListener("click", (event) => {
    if (event.target === el.themeBackdrop) closeThemeDialog();
  });
  el.profileChoice?.addEventListener("change", updateStartupProfileCopy);
  el.profileApply?.addEventListener("click", applyStartupProfile);
  document.addEventListener("keydown", (event) => {
    if (el.profileBackdrop?.classList.contains("open")) {
      handleStartupProfileKeydown(event);
      return;
    }
    if (el.themeBackdrop?.classList.contains("open")) {
      handleThemeDialogKeydown(event);
      return;
    }
    if (event.key === "Escape" && settingsOpen()) closeSettings();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopWelcomeSky();
      stopVoiceInput({ abort: true, silent: true });
    }
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
    const codeButton = event.target.closest(".code-copy-btn");
    if (codeButton && el.messages.contains(codeButton)) {
      copyCodeBlock(codeButton);
      return;
    }
    const messageButton = event.target.closest(".message-action");
    if (!messageButton || !el.messages.contains(messageButton)) return;
    if (messageButton.dataset.action === "edit") editPromptMessage(messageButton);
    if (messageButton.dataset.action === "copy") copyResponseMessage(messageButton);
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
    updateRangeFill(el.temperature);
    persistSettings();
  });

  el.context?.addEventListener("change", () => {
    state.settings.options = state.settings.options || {};
    state.settings.options.num_ctx = Number(el.context.value);
    persistSettings();
    updateContextMeter();
  });

  el.profileSetting?.addEventListener("change", () => {
    state.settings.runtime_profile = el.profileSetting.value;
    persistSettings();
  });

  const persistSystemPrompt = debounce(persistSettings, 350);
  el.system?.addEventListener("input", () => {
    state.settings.system = el.system.value;
    updateContextMeter();
    persistSystemPrompt();
  });
}

function initializeVoiceInput() {
  if (!el.mic) return;
  const Recognition = globalThis.SpeechRecognition || globalThis.webkitSpeechRecognition;
  if (typeof Recognition !== "function") {
    el.mic.disabled = true;
    el.mic.title = "Voice input is not supported by this browser";
    el.mic.setAttribute("aria-label", el.mic.title);
    return;
  }

  try {
    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.lang = navigator.language || "en-US";

    recognition.onstart = () => setVoiceInputActive(true);
    recognition.onresult = (event) => {
      if (!state.voice.active || !el.input) return;
      const transcript = Array.from(event.results, (result) => result[0]?.transcript || "")
        .join(" ")
        .trim();
      const base = state.voice.baseText.trimEnd();
      el.input.value = base && transcript ? `${base} ${transcript}` : (base || transcript);
      resizeComposer();
      updateComposerState();
      updateSlashMenu();
    };
    recognition.onerror = (event) => {
      if (state.voice.suppressError || event.error === "aborted") return;
      const messages = {
        "not-allowed": "Microphone access was denied. Allow it in your browser settings to use voice input.",
        "service-not-allowed": "Voice recognition is blocked by your browser settings.",
        "audio-capture": "No working microphone was found.",
        "no-speech": "I didn't hear anything. Try speaking a little closer to the microphone.",
        network: "Voice recognition is unavailable right now. You can still type your message."
      };
      toast(messages[event.error] || "Voice input stopped unexpectedly. You can still type your message.");
    };
    recognition.onend = () => {
      setVoiceInputActive(false);
      state.voice.suppressError = false;
    };

    state.voice.recognition = recognition;
    el.mic.addEventListener("click", toggleVoiceInput);
  } catch {
    el.mic.disabled = true;
    el.mic.title = "Voice input could not be initialized";
    el.mic.setAttribute("aria-label", el.mic.title);
  }
}

function setVoiceInputActive(active) {
  state.voice.active = active;
  if (!el.mic) return;
  el.mic.classList.toggle("recording", active);
  el.mic.setAttribute("aria-pressed", String(active));
  el.mic.setAttribute("aria-label", active ? "Stop voice input" : "Start voice input");
  el.mic.title = active ? "Listening... click to stop" : "Start voice input";
}

function toggleVoiceInput() {
  if (!state.voice.recognition || state.isGenerating) return;
  if (state.voice.active) {
    stopVoiceInput();
    return;
  }

  state.voice.baseText = el.input?.value || "";
  state.voice.suppressError = false;
  setVoiceInputActive(true);
  try {
    state.voice.recognition.start();
    el.input?.focus({ preventScroll: true });
  } catch {
    setVoiceInputActive(false);
    toast("Voice input is already busy. Wait a moment and try again.");
  }
}

function stopVoiceInput({ abort = false, silent = false } = {}) {
  if (!state.voice.recognition || !state.voice.active) return;
  state.voice.suppressError = silent;
  setVoiceInputActive(false);
  try {
    if (abort) state.voice.recognition.abort();
    else state.voice.recognition.stop();
  } catch {
    state.voice.suppressError = false;
  }
}

async function loadState() {
  try {
    const response = await fetch("/api/settings", { headers: apiHeaders() });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    state.history = data.history || [];
    state.savedSessions = data.saved_sessions || [];
    state.settings = mergeSettings(data.settings || {});
    state.runtime = data.runtime || state.runtime;
    reportRuntimeWarnings(state.runtime);
    selectConversationView(data.active_session_name || "New conversation");
    state.modelName = data.model_name || "selene";
    state.ollama = { status: data.ollama_status || "", reason: data.ollama_reason || "" };
    state.models = Array.isArray(data.models) && data.models.length
      ? data.models
      : state.models;

    document.title = titleForSession(state.activeSessionName);
    el.title.textContent = cleanSessionName(state.activeSessionName);

    syncSettingsUI();
    renderSessions();
    renderActiveConversation();
    updateComposerState();
  } catch (error) {
    toast("Could not reach the Selene backend. Start it and try again.");
    renderWelcome();
  }
}

function updateStartupProfileCopy() {
  const selected = el.profileChoice?.value || "manual";
  const descriptions = {
    manual: "Use the Modelfile and your explicit session settings without hardware-based overrides.",
    auto: "Inspect this device and choose a conservative hardware profile automatically.",
    "low-vram": "Use smaller context and batch defaults for GPUs with about 4 GiB of VRAM.",
    balanced: "Use larger context and batch defaults when the device has more headroom."
  };
  if (el.profileDescription) {
    el.profileDescription.textContent = descriptions[selected] || descriptions.manual;
  }
  if (el.profileRuntimeSummary) {
    const effective = state.runtime?.profile || state.settings.runtime_profile || "manual";
    const reason = state.runtime?.selection_reason || "Manual is the startup default.";
    el.profileRuntimeSummary.textContent = `Currently active: ${effective}. ${reason}`;
  }
}

function showStartupProfileDialog() {
  if (startupProfilePromptShown || !el.profileBackdrop || !el.profileDialog) return;
  startupProfilePromptShown = true;
  const selected = ["manual", "auto", "low-vram", "balanced"].includes(
    state.settings.runtime_profile
  ) ? state.settings.runtime_profile : "manual";
  if (el.profileChoice) el.profileChoice.value = selected;
  updateStartupProfileCopy();
  el.profileBackdrop.hidden = false;
  el.profileDialog.setAttribute("aria-hidden", "false");
  requestAnimationFrame(() => {
    el.profileBackdrop?.classList.add("open");
    el.profileChoice?.focus();
  });
}

async function applyStartupProfile() {
  const selected = el.profileChoice?.value || "manual";
  if (!el.profileBackdrop || !el.profileDialog) return;
  state.settings.runtime_profile = selected;
  if (el.profileApply) el.profileApply.disabled = true;
  await persistSettings();
  el.profileBackdrop.classList.remove("open");
  el.profileDialog.setAttribute("aria-hidden", "true");
  if (el.profileApply) el.profileApply.disabled = false;
  setTimeout(() => {
    if (el.profileBackdrop) el.profileBackdrop.hidden = true;
    el.input?.focus();
  }, 180);
}

function handleStartupProfileKeydown(event) {
  if (event.key === "Enter" && event.target !== el.profileChoice) {
    event.preventDefault();
    applyStartupProfile();
    return;
  }
  if (event.key !== "Tab") return;
  const focusable = [el.profileChoice, el.profileApply].filter(Boolean);
  if (!focusable.length) return;
  const current = focusable.indexOf(document.activeElement);
  const next = event.shiftKey
    ? (current <= 0 ? focusable.length - 1 : current - 1)
    : (current + 1) % focusable.length;
  event.preventDefault();
  focusable[next].focus();
}

function mergeSettings(settings) {
  return {
    ...DEFAULT_SETTINGS,
    ...settings,
    options: { ...(settings.options || {}) }
  };
}

function cloneSettings(settings) {
  return JSON.parse(JSON.stringify(settings));
}

function syncSettingsUI() {
  if (el.history) el.history.checked = state.settings.history !== false;
  if (el.think) el.think.checked = state.settings.think !== false;
  if (el.system) el.system.value = state.settings.system || "";
  if (el.profileSetting) {
    const profile = String(state.settings.runtime_profile || "manual");
    // Assigning an unknown value silently sets selectedIndex = -1, which
    // renders as a blank control.
    const known = [...el.profileSetting.options].some((option) => option.value === profile);
    el.profileSetting.value = known ? profile : "manual";
  }
  applyOllamaStatus();
  updateModelUI();
  updateModeUI();

  const temp = Number(
    state.settings.options?.temperature
      ?? state.runtime?.effective_options?.temperature
      ?? FALLBACK_MODEL_OPTIONS.temperature
  );
  if (el.temperature) {
    el.temperature.value = String(temp);
    updateRangeFill(el.temperature);
  }
  if (el.temperatureValue) el.temperatureValue.textContent = temp.toFixed(2);

  const budget = contextBudget();
  if (el.context) el.context.disabled = !budget;
  if (el.context && budget) {
    const ctx = String(budget);
    el.context.querySelector('option[value=""]')?.remove();
    const hasOption = [...el.context.options].some((option) => option.value === ctx);
    if (!hasOption) {
      const option = document.createElement("option");
      option.value = ctx;
      option.textContent = Number(ctx) >= 1_048_576
        ? `${Number(ctx) / 1_048_576}M`
        : `${Math.round(Number(ctx) / 1024)}k`;
      el.context.appendChild(option);
    }
    el.context.value = ctx;
  }
}

function updateModelUI() {
  if (!el.modelMenu || !el.modelLabel) return;
  let selected = state.settings.model_id || "local:default";
  
  // Verify selected is valid, otherwise use first available
  if (!state.models.some((m) => m.id === selected) && state.models.length > 0) {
    selected = state.models[0].id;
    state.settings.model_id = selected;
  }

  el.modelMenu.replaceChildren();
  state.models.forEach((model) => {
    const isSelected = model.id === selected;
    const btn = document.createElement("button");
    btn.className = "mode-option" + (isSelected ? " selected" : "");
    btn.type = "button";
    btn.setAttribute("role", "menuitemradio");
    btn.setAttribute("aria-checked", String(isSelected));
    btn.dataset.modelId = model.id;
    
    // Structure similar to mode-option but customized for models
    const icon = document.createElement("span");
    icon.className = "mode-option-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "⚙"; // generic icon for model
    if (model.id.includes("gpt")) icon.textContent = "✧";
    else if (model.id.includes("claude")) icon.textContent = "✺";
    else if (model.provider.includes("local")) icon.textContent = "○";

    const textWrap = document.createElement("span");
    const nameStr = document.createElement("strong");
    nameStr.textContent = model.display_name;
    const descStr = document.createElement("small");
    descStr.textContent = `${model.provider} · ${(model.capabilities || []).join(", ")}`;
    textWrap.appendChild(nameStr);
    textWrap.appendChild(descStr);

    const check = document.createElement("span");
    check.className = "mode-check";
    check.setAttribute("aria-hidden", "true");
    check.textContent = "✓";

    btn.appendChild(icon);
    btn.appendChild(textWrap);
    btn.appendChild(check);
    el.modelMenu.appendChild(btn);
  });

  const active = state.models.find((model) => model.id === selected);
  if (active) {
    el.modelLabel.textContent = active.display_name;
    el.modelTrigger.title = `${active.provider} · ${active.display_name}`;
    state.modelName = active.display_name;
  }
}

function openModelMenu(focus = false) {
  if (!el.modelMenu || !el.modelTrigger || state.isGenerating) return;
  el.modelMenu.hidden = false;
  el.modelTrigger.setAttribute("aria-expanded", "true");
  el.modelPicker.classList.add("active");
  requestAnimationFrame(() => el.modelMenu?.classList.add("open"));
  if (focus) {
    const selected = el.modelMenu.querySelector(".selected") || el.modelMenu.querySelector("[data-model-id]");
    selected?.focus();
  }
}

function closeModelMenu({ restoreFocus = false } = {}) {
  if (!el.modelMenu || el.modelMenu.hidden) return;
  el.modelMenu.classList.remove("open");
  el.modelMenu.hidden = true;
  el.modelTrigger?.setAttribute("aria-expanded", "false");
  el.modelPicker.classList.remove("active");
  if (restoreFocus) el.modelTrigger?.focus();
}

async function selectModel(modelId) {
  if (state.isGenerating) return;
  const previousModel = state.settings.model_id || "local:default";
  state.settings.model_id = modelId;
  updateModelUI();
  closeModelMenu({ restoreFocus: true });
  await persistSettings();
  if (previousModel !== "local:default" && modelId === "local:default") {
    showStartupProfileDialog();
  }
}

function handleModelTriggerKeydown(event) {
  if (event.key === "Escape" && !el.modelMenu?.hidden) {
    event.preventDefault();
    closeModelMenu({ restoreFocus: true });
    return;
  }
  if (!["Enter", " ", "ArrowDown", "ArrowUp"].includes(event.key)) return;
  event.preventDefault();
  openModelMenu(true);
}

function handleModelMenuKeydown(event) {
  const options = [...el.modelMenu.querySelectorAll("[data-model-id]")];
  if (!options.length) return;
  
  if (event.key === "Escape") {
    event.preventDefault();
    closeModelMenu({ restoreFocus: true });
    return;
  }
  
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const current = options.indexOf(document.activeElement);
    const next = event.key === "ArrowDown"
      ? (current + 1) % options.length
      : (current <= 0 ? options.length - 1 : current - 1);
    options[next].focus();
  }
}

function persistSettings() {
  const snapshot = cloneSettings(state.settings);
  settingsWriteChain = settingsWriteChain.then(async () => {
    try {
      const response = await fetch("/api/settings", {
        method: "POST",
        headers: apiHeaders(true),
        body: JSON.stringify(snapshot)
      });
      if (!response.ok) throw await apiError(response, "Settings could not be saved.");
      const data = await response.json();
      state.settings = mergeSettings(data.settings || state.settings);
      state.runtime = data.runtime || state.runtime;
      reportRuntimeWarnings(state.runtime);
      syncSettingsUI();
      if (Array.isArray(data.history)) {
        // A model switch re-primes the conversation server-side, so the
        // transcript on screen no longer matches what the model will read.
        state.history = data.history;
        renderMessages();
      }
      if (data.context_compaction?.compacted) {
        const summarized = Number(data.context_compaction.summarized_messages) || 0;
        toast(`Context compacted for the new model — ${summarized} earlier messages summarized.`);
      }
      // Refresh after the server resolves profile defaults and session overrides.
      updateContextMeter();
      return true;
    } catch (error) {
      toast(error.message || "Settings could not be saved.");
      return false;
    }
  });
  return settingsWriteChain;
}

function activeAgentMode() {
  return AGENT_MODES[state.settings.agent_mode] ? state.settings.agent_mode : "normal";
}

function updateModeUI() {
  const mode = activeAgentMode();
  const config = AGENT_MODES[mode];
  const active = mode !== "normal";
  if (el.modeLabel) el.modeLabel.textContent = config.label;
  if (el.modeIcon) el.modeIcon.textContent = config.icon;
  if (el.modePicker) {
    el.modePicker.dataset.mode = mode;
    el.modePicker.classList.toggle("active", active);
    el.modePicker.classList.toggle("running", active && state.isGenerating);
  }
  if (el.modeClear) el.modeClear.hidden = !active;
  el.modeMenu?.querySelectorAll("[data-agent-mode]").forEach((option) => {
    const selected = option.dataset.agentMode === mode;
    option.classList.toggle("selected", selected);
    option.setAttribute("aria-checked", String(selected));
  });
}

function openModeMenu(focus = false) {
  if (!el.modeMenu || !el.modeTrigger || state.isGenerating) return;
  el.modeMenu.hidden = false;
  el.modeTrigger.setAttribute("aria-expanded", "true");
  requestAnimationFrame(() => el.modeMenu?.classList.add("open"));
  if (focus) {
    const selected = el.modeMenu.querySelector(".selected") || el.modeMenu.querySelector("[data-agent-mode]");
    selected?.focus();
  }
}

function closeModeMenu({ restoreFocus = false } = {}) {
  if (!el.modeMenu || el.modeMenu.hidden) return;
  el.modeMenu.classList.remove("open");
  el.modeMenu.hidden = true;
  el.modeTrigger?.setAttribute("aria-expanded", "false");
  if (restoreFocus) el.modeTrigger?.focus();
}

function setAgentMode(mode) {
  if (!AGENT_MODES[mode] || state.isGenerating) return;
  state.settings.agent_mode = mode;
  updateModeUI();
  closeModeMenu({ restoreFocus: true });
  persistSettings();
}

function handleModeTriggerKeydown(event) {
  if (event.key === "Escape" && !el.modeMenu?.hidden) {
    event.preventDefault();
    closeModeMenu({ restoreFocus: true });
    return;
  }
  if (!["Enter", " ", "ArrowDown", "ArrowUp"].includes(event.key)) return;
  event.preventDefault();
  openModeMenu(true);
}

function handleModeMenuKeydown(event) {
  const options = [...el.modeMenu.querySelectorAll("[data-agent-mode]")];
  if (!options.length) return;
  if (event.key === "Escape") {
    event.preventDefault();
    closeModeMenu({ restoreFocus: true });
    return;
  }
  if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const current = Math.max(0, options.indexOf(document.activeElement));
  let next = current;
  if (event.key === "ArrowDown") next = (current + 1) % options.length;
  if (event.key === "ArrowUp") next = (current - 1 + options.length) % options.length;
  if (event.key === "Home") next = 0;
  if (event.key === "End") next = options.length - 1;
  options[next].focus();
}

function renderSessions() {
  if (!el.sessionList) return;
  el.sessionList.innerHTML = "";

  if (!state.savedSessions.length) return;

  state.savedSessions.forEach((name) => {
    const generation = generationForSession(name);
    const item = document.createElement("div");
    item.className = [
      "session-item",
      name === state.activeSessionName ? "active" : "",
      generation ? "running" : ""
    ].filter(Boolean).join(" ");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-open";
    button.textContent = cleanSessionName(name);
    button.title = generation ? `${name} · response running` : name;
    if (generation) {
      const running = document.createElement("span");
      running.className = "session-running";
      running.setAttribute("aria-label", "Response running");
      running.title = "Response running";
      button.appendChild(running);
    }
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

function renderActiveConversation() {
  const generation = syncActiveGeneration();
  if (!generation) {
    resetStream();
    renderMessages();
    return;
  }
  state.history = JSON.parse(JSON.stringify(generation.baseHistory || state.history));
  resetStream();
  renderMessages();
  for (const event of generation.events) {
    handleStreamEvent(event, generation, { record: false, forceVisible: true, replay: true });
  }
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
    } else if (message.role === "notice") {
      appendCompactionNotice(message, false);
    } else {
      appendAssistantMessage(message, false);
    }
  });
  scrollToBottom(true);
}

function appendCompactionNotice(message, scroll = true) {
  el.messages.querySelector(".welcome")?.remove();
  const row = messageShell("assistant", "S");
  row.root.classList.add("compaction-notice");
  row.stack.appendChild(
    detailBlock(message.label || "Context compacted", "memory", message.content, false)
  );
  el.messages.appendChild(row.root);
  if (scroll) scrollToBottom(true);
}

function toDisplayMessages(history) {
  const display = [];
  for (let i = 0; i < history.length; i += 1) {
    const message = history[i];
    if (message.role === "system") continue;
    if (message.role === "user") {
      display.push({ role: "user", content: displayText(message.content) });
      continue;
    }
    if (message.role === "assistant" && message.metadata?.compacted) {
      // Older turns collapsed into an extractive summary. Show it as a
      // collapsed memory block rather than a spoken assistant reply.
      const summarized = Number(message.metadata.source_messages) || 0;
      display.push({
        role: "notice",
        label: summarized
          ? `Context compacted — ${summarized} earlier messages summarized`
          : "Context compacted",
        content: displayText(message.content)
      });
      continue;
    }
    if (message.role === "assistant") {
      // Collect the tool results instead of skipping past them. as_tool_message
      // carries no tool_call_id, so pairing is positional — which is reliable
      // because web.py emits tool_results ordered by call index.
      const toolMessages = [];
      let j = i + 1;
      while (history[j]?.role === "tool") {
        toolMessages.push(history[j]);
        j += 1;
      }
      const entry = {
        role: "assistant",
        content: displayText(message.content),
        error: Boolean(message.error),
        thoughtItems: [
          ...(message.planning ? [{ type: "thinking", text: displayText(message.planning) }] : []),
          ...(message.thinking ? [{ type: "thinking", text: displayText(message.thinking) }] : []),
          ...(message.tool_calls || []).map((call, index) => ({
            type: "tool",
            name: call.function?.name || toolMessages[index]?.tool_name || "tool",
            args: call.function?.arguments,
            // Null when the turn was interrupted after the assistant message
            // was saved but before its tool message was.
            result: toolMessages[index]?.content ?? null,
            historical: true
          }))
        ]
      };
      const previous = display[display.length - 1];
      if (previous?.role === "assistant" && !previous.content) {
        previous.content = entry.content;
        previous.error ||= entry.error;
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
      <div class="welcome-core">
        <h3 data-text="Selene">Selene</h3>
        <div class="suggestions">
          <button class="suggestion" type="button" data-prompt="Summarize this project and identify the most important files.">
            <span class="suggestion-index">01</span><span class="suggestion-copy"><strong>Project summary</strong><small>Understand the workspace</small></span>
          </button>
          <button class="suggestion" type="button" data-prompt="Search the web for the latest AI developer tooling news.">
            <span class="suggestion-index">02</span><span class="suggestion-copy"><strong>Web research</strong><small>Use tools when needed</small></span>
          </button>
          <button class="suggestion" type="button" data-prompt="Help me debug a Python error step by step.">
            <span class="suggestion-index">03</span><span class="suggestion-copy"><strong>Debug with me</strong><small>Reason through a problem</small></span>
          </button>
          <button class="suggestion" type="button" data-prompt="/help">
            <span class="suggestion-index">04</span><span class="suggestion-copy"><strong>Commands</strong><small>Show slash commands</small></span>
          </button>
        </div>
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

// The blanket prefers-reduced-motion rule in CSS only silences transitions and
// keyframes. Anything driven from JS — the canvas below, scrollIntoView — has
// to consult the query itself.
function prefersReducedMotion() {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
}

// Star count follows the area the canvas actually occupies, so a narrow window
// does not end up with the same 36 stars crammed into a fraction of the space.
function welcomeStarCount(width, height) {
  return Math.round(Math.min(150, Math.max(40, (width * height) / 8000)));
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
    nextShootingStar: now + randomBetween(2000, 6000),
    pointerX: 0,
    pointerY: 0,
    targetX: 0,
    targetY: 0,
    stars: Array.from({ length: 80 }, () => newCanvasStar(now, true))
  };
  refreshWelcomeSkyPalette(scene);
  state.sky.scene = scene;

  const resize = () => {
    const bounds = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    scene.width = Math.max(1, bounds.width);
    scene.height = Math.max(1, bounds.height);
    canvas.width = Math.round(scene.width * ratio);
    canvas.height = Math.round(scene.height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);

    const target = welcomeStarCount(scene.width, scene.height);
    while (scene.stars.length < target) scene.stars.push(newCanvasStar(performance.now(), true));
    if (scene.stars.length > target) scene.stars.length = target;
  };
  resize();

  if (prefersReducedMotion()) {
    // Paint a single settled frame: every star at mid-brightness, no shooting
    // star, no animation loop and no watchdog scheduled at all.
    scene.nextShootingStar = Infinity;
    scene.stars.forEach((star) => { star.born = now - star.duration / 2; });
    scene.reducedMotion = true;
    drawWelcomeSky(scene, now, 0);
    state.sky.resizeObserver = new ResizeObserver(() => {
      resize();
      scene.stars.forEach((star) => { star.born = performance.now() - star.duration / 2; });
      drawWelcomeSky(scene, performance.now(), 0);
    });
    state.sky.resizeObserver.observe(canvas);
    return;
  }

  state.sky.resizeObserver = new ResizeObserver(resize);
  state.sky.resizeObserver.observe(canvas);

  // Parallax has been removed per user request.

  const drawFrame = (timestamp) => {
    if (!canvas.isConnected || state.sky.scene !== scene) return;
    if (document.hidden) {
      scene.lastFrame = timestamp;
      state.sky.frame = requestAnimationFrame(drawFrame);
      return;
    }
    const delta = Math.min(34, Math.max(0, timestamp - scene.lastFrame));
    scene.lastFrame = timestamp;
    drawWelcomeSky(scene, timestamp, delta);
    state.sky.frame = requestAnimationFrame(drawFrame);
  };
  state.sky.frame = requestAnimationFrame(drawFrame);

  // Visible tabs normally receive a continuous animation frame loop. This
  // watchdog restarts it after aggressive browser or power-saving throttling.
  state.sky.watchdog = window.setInterval(() => {
    if (document.hidden || !canvas.isConnected || state.sky.scene !== scene) return;
    const now = performance.now();
    if (now - scene.lastFrame < 1500) return;
    if (state.sky.frame !== null) cancelAnimationFrame(state.sky.frame);
    scene.lastFrame = now;
    state.sky.frame = requestAnimationFrame(drawFrame);
  }, 2000);
}

function stopWelcomeSky() {
  if (state.sky.frame !== null) cancelAnimationFrame(state.sky.frame);
  if (state.sky.watchdog !== null) clearInterval(state.sky.watchdog);
  state.sky.resizeObserver?.disconnect();
  state.sky.detachPointer?.();
  state.sky.frame = null;
  state.sky.watchdog = null;
  state.sky.resizeObserver = null;
  state.sky.detachPointer = null;
  state.sky.scene = null;
}

function newCanvasStar(now, initial = false) {
  return {
    x: Math.random(),
    y: Math.random(),
    radius: randomBetween(0.8, 1.8),
    brightness: randomBetween(.24, .62),
    depth: randomBetween(.35, 1),
    born: now + (initial ? randomBetween(-5000, 900) : randomBetween(350, 1800)),
    duration: randomBetween(5000, 10000)
  };
}

function cssVariableRgb(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const short = value.match(/^#([0-9a-f])([0-9a-f])([0-9a-f])$/i);
  if (short) return short.slice(1).map((part) => Number.parseInt(part + part, 16));
  const full = value.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
  if (full) return full.slice(1).map((part) => Number.parseInt(part, 16));
  return fallback;
}

function refreshWelcomeSkyPalette(scene) {
  if (!scene) return;
  scene.starRgb = cssVariableRgb("--accent", [232, 232, 232]);
  scene.shineRgb = cssVariableRgb("--text", [242, 242, 242]);
  scene.auroraRgb = cssVariableRgb("--primary", [207, 207, 207]);
  // A theme switch while reduced motion is on has no frame loop to repaint it.
  if (scene.reducedMotion) drawWelcomeSky(scene, performance.now(), 0);
}

// Two very low-alpha washes drifting on slow, mutually-prime periods so the
// backdrop never visibly repeats. Alpha stays under .05 — this should read as
// depth in the page, not as a gradient someone applied.
function drawWelcomeAurora(scene, now) {
  const { context, width, height } = scene;
  const primary = scene.auroraRgb || [207, 207, 207];
  const accent = scene.starRgb || [232, 232, 232];
  const seconds = now / 1000;
  const blobs = [
    {
      rgb: primary,
      x: (0.5 + Math.sin((seconds / 40) * Math.PI * 2) * 0.22) * width,
      y: (0.36 + Math.cos((seconds / 55) * Math.PI * 2) * 0.16) * height,
      radius: Math.max(width, height) * 0.62,
      alpha: 0.05
    },
    {
      rgb: accent,
      x: (0.5 + Math.cos((seconds / 55) * Math.PI * 2 + 1.6) * 0.26) * width,
      y: (0.64 + Math.sin((seconds / 40) * Math.PI * 2 + 0.8) * 0.14) * height,
      radius: Math.max(width, height) * 0.54,
      alpha: 0.038
    }
  ];

  blobs.forEach((blob) => {
    const gradient = context.createRadialGradient(blob.x, blob.y, 0, blob.x, blob.y, blob.radius);
    gradient.addColorStop(0, rgba(blob.rgb, blob.alpha));
    gradient.addColorStop(0.55, rgba(blob.rgb, blob.alpha * 0.35));
    gradient.addColorStop(1, rgba(blob.rgb, 0));
    context.fillStyle = gradient;
    context.fillRect(0, 0, width, height);
  });
}

function rgba(rgb, opacity) {
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${opacity})`;
}

function drawWelcomeSky(scene, now, delta) {
  const { context, width, height, stars } = scene;
  const starRgb = scene.starRgb || [232, 232, 232];
  const shineRgb = scene.shineRgb || [242, 242, 242];
  context.clearRect(0, 0, width, height);



  stars.forEach((star, index) => {
    const progress = (now - star.born) / star.duration;
    if (progress >= 1) {
      stars[index] = newCanvasStar(now);
      return;
    }
    if (progress < 0) return;
    const x = star.x * width;
    const y = star.y * height;
    const opacity = Math.pow(Math.sin(Math.PI * progress), 1.55)
      * star.brightness;
    if (opacity <= .01) return;
    context.beginPath();
    context.arc(x, y, star.radius, 0, Math.PI * 2);
    context.fillStyle = rgba(starRgb, opacity);
    context.fill();
  });

  if (!scene.shootingStar && now >= scene.nextShootingStar) {
    const angle = randomBetween(14, 24) * Math.PI / 180;
    const speed = randomBetween(210, 285);
    scene.shootingStar = {
      x: randomBetween(width * .04, width * .48),
      y: randomBetween(height * .06, height * .38),
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      age: 0,
      duration: randomBetween(2.1, 2.8),
      trail: randomBetween(100, 150)
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
  const fadeIn = Math.min(1, shot.age / .32);
  const fadeOut = Math.min(1, Math.max(0, (shot.duration - shot.age) / 1.1));
  const opacity = Math.min(fadeIn, fadeOut) * .5;
  const tailX = shot.x - ux * shot.trail;
  const tailY = shot.y - uy * shot.trail;
  const streak = context.createLinearGradient(tailX, tailY, shot.x, shot.y);
  streak.addColorStop(0, rgba(starRgb, 0));
  streak.addColorStop(.62, rgba(starRgb, opacity * .18));
  streak.addColorStop(1, rgba(shineRgb, opacity));
  context.beginPath();
  context.moveTo(tailX, tailY);
  context.lineTo(shot.x, shot.y);
  context.lineWidth = 1.8;
  context.strokeStyle = streak;
  context.stroke();
  context.beginPath();
  context.arc(shot.x, shot.y, 2.2, 0, Math.PI * 2);
  context.fillStyle = rgba(shineRgb, opacity);
  context.fill();

  if (shot.age >= shot.duration || shot.x > width + shot.trail || shot.y > height + shot.trail) {
    scene.shootingStar = null;
    scene.nextShootingStar = now + randomBetween(12000, 24000);
  }
}

function welcomeSkyVignette(x, y, width, height) {
  const normalizedX = (x - width / 2) / Math.max(1, width * .7);
  const normalizedY = (y - height / 2) / Math.max(1, height * .72);
  const distance = Math.hypot(normalizedX, normalizedY);
  const edgeFade = Math.max(0, Math.min(1, (1 - distance) / .38));
  return edgeFade * edgeFade * (3 - 2 * edgeFade);
}

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function appendUserMessage(text, scroll = true) {
  stopWelcomeSky();
  el.messages.querySelector(".welcome")?.remove();
  const row = messageShell("user", "You");
  row.stack.appendChild(bubble(text, false));
  appendMessageActions(row.stack, "edit");
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
        const row = toolCallRow({
          name: item.name,
          args: item.args,
          result: item.result,
          historical: true
        });
        // Duration is never persisted, so a historical row shows status only.
        row.dataset.status = item.result == null ? "unknown" : classifyToolResult(item.result);
        body?.appendChild(row);
      } else if (item.text) {
        body?.appendChild(thinkingContent(item.text));
      }
    });
    row.stack.appendChild(thinkingBlock);
  }

  if (message.content) {
    const responseBubble = bubble(message.content, true);
    if (message.error) responseBubble.classList.add("error");
    row.stack.appendChild(responseBubble);
    appendMessageActions(row.stack, "copy");
  }
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
  if (markdown) {
    renderResponseInto(node, content);
  } else {
    const text = displayText(content);
    node.dataset.messageText = text;
    node.innerHTML = escapeHTML(text).replace(/\n/g, "<br>");
  }
  return node;
}

function appendMessageActions(stack, action) {
  const actions = document.createElement("div");
  actions.className = "message-actions";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "message-action";
  button.dataset.action = action;
  button.setAttribute("aria-label", action === "edit" ? "Edit prompt" : "Copy response");
  button.title = action === "edit" ? "Edit prompt" : "Copy response";
  button.innerHTML = action === "edit"
    ? '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 11.8V13h1.2l7.1-7.1-1.2-1.2L3 11.8Zm8-8 1.2-1.2 1.2 1.2L12.2 5 11 3.8Z"/></svg><span>Edit</span>'
    : '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="5" y="5" width="8" height="8" rx="1.4"/><path d="M3 10.5H2.8A1.8 1.8 0 0 1 1 8.7V2.8A1.8 1.8 0 0 1 2.8 1h5.9a1.8 1.8 0 0 1 1.8 1.8V3"/></svg><span>Copy</span>';
  actions.appendChild(button);
  stack.appendChild(actions);
}

function messageActionText(button) {
  return button.closest(".message-actions")?.previousElementSibling?.dataset.messageText || "";
}

function editPromptMessage(button) {
  if (!el.input) return;
  const text = messageActionText(button);
  if (!text) return;
  el.input.value = text;
  resizeComposer();
  updateComposerState();
  el.input.focus({ preventScroll: true });
  el.input.setSelectionRange(text.length, text.length);
}

async function copyResponseMessage(button) {
  const text = messageActionText(button);
  if (!text) return;
  try {
    await writeClipboardText(text);
    button.classList.add("copied");
    const label = button.querySelector("span");
    if (label) label.textContent = "Copied";
    setTimeout(() => {
      if (!button.isConnected) return;
      button.classList.remove("copied");
      if (label) label.textContent = "Copy";
    }, 1400);
  } catch {
    toast("Could not copy that response.");
  }
}

function detailBlock(title, label, content, running) {
  const block = document.createElement("section");
  block.className = `detail-block${running ? " running" : ""}`;

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "block-toggle";
  toggle.setAttribute("aria-expanded", "false");
  toggle.innerHTML = `
    <span class="block-title">${escapeHTML(title)} <span class="pill">${escapeHTML(label)}</span></span>
    <span class="block-preview" aria-hidden="true"></span>
    <span class="block-chevron" aria-hidden="true">
      <svg viewBox="0 0 16 16"><path d="m6 4 4 4-4 4"/></svg>
    </span>
  `;

  // The reveal wraps the body rather than replacing it, so every existing
  // querySelector(".block-body") append site keeps working unchanged while the
  // wrapper handles the 0fr->1fr height animation.
  const reveal = document.createElement("div");
  reveal.className = "block-reveal";

  const body = document.createElement("div");
  body.className = "block-body";

  const blockContent = thinkingContent(content || (running ? "Running..." : ""));

  toggle.addEventListener("click", () => {
    const isOpen = block.classList.toggle("open");
    toggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
    if (isOpen) scrollThinkingToBottom(block);
  });
  body.appendChild(blockContent);
  reveal.appendChild(body);
  block.append(toggle, reveal);
  return block;
}

// A one-line tail of the reasoning shown while the block is collapsed, so the
// stream is legible without forcing it open.
function updateBlockPreview(block, text) {
  const preview = block?.querySelector(".block-preview");
  if (!preview) return;
  const flat = String(text || "").replace(/\s+/g, " ").trim();
  preview.textContent = flat.length > 90 ? `…${flat.slice(-90)}` : flat;
}

function thinkingContent(text = "") {
  const content = document.createElement("div");
  content.className = "block-content";
  content.textContent = displayText(text);
  return content;
}

function scrollThinkingToBottom(block) {
  if (!block?.classList.contains("open")) return;
  const body = block.querySelector(".block-body");
  if (!body) return;
  body.scrollTop = body.scrollHeight;
}

// Chars rendered into the DOM, and chars kept in dataset for "copy full".
const TOOL_INLINE_LIMIT = 8000;
const TOOL_STORE_LIMIT = 120000;

// Tool failures are structured: tool_runner._error_content emits
// {"ok": false, "error": ..., "error_code": ...}. Detection is therefore exact
// for our own tools, with a text heuristic only for non-JSON output.
function classifyToolResult(raw) {
  const text = displayText(raw);
  if (!text) return "unknown";
  try {
    const data = JSON.parse(text);
    if (data && typeof data === "object") {
      if (data.ok === false || data.error) return "error";
      return "ok";
    }
  } catch {
    /* not JSON — fall through to the text heuristic */
  }
  return /^(error|traceback|exception)\b/i.test(text.trim()) ? "error" : "ok";
}

function safeStringify(value) {
  const seen = new WeakSet();
  try {
    return JSON.stringify(value, (key, item) => {
      if (item && typeof item === "object") {
        if (seen.has(item)) return "[circular]";
        seen.add(item);
      }
      return item;
    }, 2);
  } catch {
    return String(value);
  }
}

function formatToolPayload(value) {
  if (value == null) return "";
  if (typeof value === "object") return safeStringify(value);
  const text = displayText(value);
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

function formatToolDuration(ms) {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(2)}s`;
}

// Sections are built on first expand, so a transcript with dozens of tool calls
// does not carry dozens of large <pre> nodes it may never show.
function buildToolSection(label, text) {
  const section = document.createElement("div");
  section.className = "tool-row-section";
  section.dataset.label = label;

  const pre = document.createElement("pre");
  const full = String(text || "");
  pre.textContent = full.length > TOOL_INLINE_LIMIT ? full.slice(0, TOOL_INLINE_LIMIT) : full;
  section.appendChild(pre);

  if (full.length > TOOL_INLINE_LIMIT) {
    const footer = document.createElement("div");
    footer.className = "tool-row-footer";
    const note = document.createElement("span");
    note.textContent =
      `Showing first ${TOOL_INLINE_LIMIT.toLocaleString()} of ${full.length.toLocaleString()} characters`;
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "tool-row-copy";
    copy.textContent = "Copy full";
    copy.addEventListener("click", async () => {
      try {
        await writeClipboardText(full);
        copy.textContent = "Copied";
        setTimeout(() => { if (copy.isConnected) copy.textContent = "Copy full"; }, 1400);
      } catch {
        toast("Could not copy that tool output.");
      }
    });
    footer.append(note, copy);
    section.appendChild(footer);
  }
  return section;
}

function expandToolRow(row) {
  const detail = row.querySelector(".tool-row-detail");
  if (!detail || detail.dataset.built === "true") return;
  detail.dataset.built = "true";
  const inner = detail.querySelector(".tool-row-detail-inner");
  if (row.dataset.args) inner.appendChild(buildToolSection("arguments", row.dataset.args));
  if (row.dataset.result) inner.appendChild(buildToolSection("result", row.dataset.result));
}

function setToolRowPayload(row, key, value) {
  const text = formatToolPayload(value);
  if (!text) return;
  // Truncate at store time too, so a tool that returns megabytes cannot pin
  // that much string data in a dataset attribute for the life of the page.
  row.dataset[key] = text.length > TOOL_STORE_LIMIT ? text.slice(0, TOOL_STORE_LIMIT) : text;
  const detail = row.querySelector(".tool-row-detail");
  if (detail?.dataset.built === "true") {
    detail.dataset.built = "false";
    const inner = detail.querySelector(".tool-row-detail-inner");
    if (inner) inner.replaceChildren();
    if (row.classList.contains("open")) expandToolRow(row);
  }
}

function toolCallRow({ id, name, args, result, running = false, historical = false } = {}) {
  const row = document.createElement("section");
  row.className = `tool-row${running ? " running" : ""}`;
  if (id != null) row.dataset.toolId = String(id);
  if (historical) row.dataset.historical = "true";
  row.dataset.status = running ? "running" : "unknown";

  const head = document.createElement("button");
  head.type = "button";
  head.className = "tool-row-head";
  head.setAttribute("aria-expanded", "false");
  head.innerHTML = `
    <span class="tool-status-dot" aria-hidden="true"></span>
    <span class="tool-row-name"></span>
    <span class="tool-row-time"></span>
    <span class="tool-row-chevron" aria-hidden="true">
      <svg viewBox="0 0 16 16"><path d="m6 4 4 4-4 4"/></svg>
    </span>
  `;
  head.querySelector(".tool-row-name").textContent = humanizeToolName(name);

  const detail = document.createElement("div");
  detail.className = "tool-row-detail";
  detail.dataset.built = "false";
  const inner = document.createElement("div");
  inner.className = "tool-row-detail-inner";
  detail.appendChild(inner);

  head.addEventListener("click", () => {
    const isOpen = row.classList.toggle("open");
    head.setAttribute("aria-expanded", isOpen ? "true" : "false");
    if (isOpen) expandToolRow(row);
  });

  row.append(head, detail);
  if (args != null) setToolRowPayload(row, "args", args);
  if (result != null) {
    setToolRowPayload(row, "result", result);
    row.dataset.status = classifyToolResult(result);
  }
  return row;
}

// Kept as a shim so historical replay and any other caller can migrate
// independently of the live streaming path.
function toolIndicator(name, running = false) {
  return toolCallRow({ name, running });
}

function settleToolRow(row, { status = "unknown", duration = null, result = null } = {}) {
  if (!row) return;
  row.classList.remove("running");
  row.dataset.status = status;
  if (result != null) setToolRowPayload(row, "result", result);
  const time = row.querySelector(".tool-row-time");
  if (time && duration != null) time.textContent = formatToolDuration(duration);
  // A failing tool opens itself — that is the one case where the detail is the
  // point rather than an aside.
  if (status === "error" && !row.classList.contains("open")) {
    row.classList.add("open");
    row.querySelector(".tool-row-head")?.setAttribute("aria-expanded", "true");
    expandToolRow(row);
  }
}

function humanizeToolName(name) {
  return String(name || "tool")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

async function sendMessage() {
  if (settingsOpen()) closeSettings();
  if (generationForSession()) return;
  const text = el.input.value.trim();
  if (!text) return;

  if (handleThemeCommand(text)) {
    el.input.value = "";
    resetPromptRecall();
    closeSlashMenu();
    resizeComposer();
    updateComposerState();
    return;
  }

  stopVoiceInput({ abort: true, silent: true });
  closeSlashMenu();
  closeModeMenu();
  appendUserMessage(text);
  if (!text.startsWith("/")) state.history.push({ role: "user", content: text });
  el.input.value = "";
  resetPromptRecall();
  resizeComposer();
  updateComposerState();

  const controller = new AbortController();
  let resolveIdentity;
  const identityReady = new Promise((resolve) => { resolveIdentity = resolve; });
  const generation = {
    controller,
    id: createRuntimeId("generation"),
    sessionName: state.activeSessionName,
    viewVersion: state.viewVersion,
    baseHistory: JSON.parse(JSON.stringify(state.history)),
    events: [],
    wasBackgrounded: false,
    identityReady,
    resolveIdentity
  };
  state.generations.set(generation.id, generation);
  syncActiveGeneration();
  resetStream();
  renderSessions();
  updateComposerState();

  try {
    // Mode/profile changes are persisted asynchronously. Do not let a quick
    // send race the backend and start with the previous mode.
    await settingsWriteChain;
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify({
        message: text,
        session_name: generation.sessionName,
        generation_id: generation.id,
        client_id: state.clientId
      }),
      signal: controller.signal
    });
    if (!response.ok) throw await apiError(response, "The model request could not be started.");
    bindGenerationSession(
      generation,
      response.headers.get("X-Selene-Session-Name")
    );
    generation.resolveIdentity();
    await readEventStream(response, generation);
  } catch (error) {
    generation.resolveIdentity();
    if (error.name !== "AbortError") {
      const detail = error.message
        || "The response stream was interrupted. Try again or select another model.";
      const errorEvent = { type: "content_chunk", text: detail, error: true };
      const visible = isGenerationVisible(generation);
      handleStreamEvent(errorEvent, generation);
      if (!visible) {
        toast(`Response failed in “${cleanSessionName(generation.sessionName)}”.`);
      }
      finishGeneration(generation);
    }
  }
}

function bindGenerationSession(generation, sessionName) {
  const normalized = String(sessionName || "").trim();
  if (!normalized || isNewConversationName(normalized)) return;
  const previousName = generation.sessionName;
  generation.sessionName = normalized;
  if (!state.savedSessions.includes(normalized)) {
    state.savedSessions = [normalized, ...state.savedSessions];
  }
  if (
    state.viewVersion === generation.viewVersion
    && state.activeSessionName === previousName
  ) {
    state.activeSessionName = normalized;
    el.title.textContent = cleanSessionName(normalized);
    document.title = titleForSession(normalized);
  }
  syncActiveGeneration();
  renderSessions();
  updateComposerState();
}

function recordGenerationEvent(generation, event) {
  if (!generation || ["conversation_started", "done"].includes(event.type)) return;
  const events = generation.events;
  const previous = events[events.length - 1];
  if (previous?.type === "token_usage" && event.type === "token_usage") {
    events[events.length - 1] = JSON.parse(JSON.stringify(event));
    return;
  }
  const chunkTypes = new Set(["thinking_chunk", "planning_chunk", "content_chunk"]);
  const previousKey = previous?.text !== undefined ? "text" : "content";
  const eventKey = event.text !== undefined ? "text" : "content";
  if (
    previous
    && previous.type === event.type
    && chunkTypes.has(event.type)
    && Boolean(previous.error) === Boolean(event.error)
    && previousKey === eventKey
  ) {
    previous[eventKey] = displayText(previous[eventKey]) + displayText(event[eventKey]);
    return;
  }
  events.push(JSON.parse(JSON.stringify(event)));
}

async function readEventStream(response, generation) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawDone = false;

  while (true) {
    const { value, done } = await reader.read();
    buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = done ? "" : (lines.pop() || "");

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      let event;
      try {
        event = JSON.parse(line.slice(5).trimStart());
      } catch {
        // A partial proxy frame should not discard later valid SSE events.
        continue;
      }
      if (event.type === "done") sawDone = true;
      // Keep JSON framing errors separate from display errors. Previously a
      // renderer exception was treated like malformed SSE and silently dropped
      // an otherwise valid provider chunk.
      handleStreamEvent(event, generation);
    }
    if (done) break;
  }
  if (!sawDone) throw new Error("The response stream ended before the model completed its reply.");
}

function handleStreamEvent(event, generation, { record = true, forceVisible = false, replay = false } = {}) {
  if (record) recordGenerationEvent(generation, event);
  const visible = forceVisible || isGenerationVisible(generation);
  switch (event.type) {
    case "conversation_started":
      bindGenerationSession(generation, event.session_name);
      generation.resolveIdentity();
      break;
    case "model_fallback": {
      const fallbackModel = event.model || {};
      if (fallbackModel.id) {
        state.settings.model_id = fallbackModel.id;
        const existing = state.models.findIndex((model) => model.id === fallbackModel.id);
        if (existing >= 0) state.models[existing] = { ...state.models[existing], ...fallbackModel };
        else state.models.push(fallbackModel);
        updateModelUI();
      }
      if (visible) appendStatus(
        event.message || "Model error. Switched to the next fallback model and continuing automatically."
      );
      break;
    }
    case "status":
      if (!visible) break;
      appendStatus(event.message || "", event.activity_mode || "");
      break;
    case "thinking_start":
      if (!visible) break;
      ensureAssistantStack();
      if (state.stream.assistantBubble) {
        scheduleStreamRender({ immediate: true });
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
      state.stream.thinkingText += displayText(event.text);
      scheduleStreamRender();
      break;
    case "thinking_end":
      if (!visible) break;
      scheduleStreamRender({ immediate: true });
      state.stream.thinkingBlock?.classList.remove("running");
      state.stream.thinkingBlock?.classList.remove("open");
      break;
    case "planning_start":
      handleStreamEvent(
        { ...event, type: "thinking_start" }, generation,
        { record: false, forceVisible: visible }
      );
      break;
    case "planning_chunk":
      handleStreamEvent(
        { ...event, type: "thinking_chunk" }, generation,
        { record: false, forceVisible: visible }
      );
      break;
    case "planning_promote":
      if (!visible) break;
      if (!state.stream.thinkingBlock && event.text) {
        handleStreamEvent(
          { ...event, type: "thinking_start" }, generation,
          { record: false, forceVisible: true }
        );
        handleStreamEvent(
          { ...event, type: "thinking_chunk" }, generation,
          { record: false, forceVisible: true }
        );
      }
      break;
    case "planning_end":
      handleStreamEvent(
        { ...event, type: "thinking_end" }, generation,
        { record: false, forceVisible: visible }
      );
      break;
    // Both are emitted by the backend and were previously ignored. They mark
    // the boundary between tool rounds, which is what makes the per-round
    // tool ids unambiguous.
    case "tool_calls_start":
    case "tool_parallel_start":
      state.stream.toolRound += 1;
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
      {
        const rawId = String(event.id ?? state.stream.activeToolBlocks.size);
        let key = `${state.stream.toolRound}:${rawId}`;
        // A serial multi-round loop emits neither round event, so the same id
        // can arrive again while the previous one is still open. Settle the
        // stale row and start a new round rather than overwriting it.
        if (state.stream.activeToolBlocks.has(key)) {
          const stale = state.stream.activeToolBlocks.get(key);
          settleToolRow(stale.row, { status: "unknown" });
          state.stream.activeToolBlocks.delete(key);
          state.stream.toolRound += 1;
          key = `${state.stream.toolRound}:${rawId}`;
        }
        const row = toolCallRow({
          id: rawId,
          name: event.name || "tool",
          args: event.arguments,
          running: true
        });
        state.stream.activeToolBlocks.set(key, {
          row,
          name: event.name || "tool",
          // Omitted on replay: the clock would measure from replay time, not
          // from when the tool actually ran.
          startedAt: replay ? null : performance.now()
        });
        state.stream.thinkingBlock.querySelector(".block-body")?.appendChild(row);
      }
      scrollThinkingToBottom(state.stream.thinkingBlock);
      scrollToBottom();
      break;
    case "tool_end":
      if (!visible) break;
      {
        const rawId = String(event.id ?? "");
        let key = `${state.stream.toolRound}:${rawId}`;
        let entry = state.stream.activeToolBlocks.get(key);
        if (!entry) {
          // The round may have advanced between start and end.
          for (const candidate of state.stream.activeToolBlocks.keys()) {
            if (candidate.endsWith(`:${rawId}`)) {
              key = candidate;
              entry = state.stream.activeToolBlocks.get(candidate);
              break;
            }
          }
        }
        if (entry) {
          settleToolRow(entry.row, {
            status: classifyToolResult(event.result),
            duration: entry.startedAt == null ? null : performance.now() - entry.startedAt,
            result: event.result
          });
          state.stream.activeToolBlocks.delete(key);
        } else {
          // Cache hits emit tool_end with no preceding tool_start. Render them
          // as their own completed row instead of hijacking an unrelated
          // running one, which is what the old fallback did.
          const row = toolCallRow({
            id: rawId,
            name: event.name || "tool",
            result: event.result
          });
          settleToolRow(row, { status: classifyToolResult(event.result), result: event.result });
          state.stream.thinkingBlock?.querySelector(".block-body")?.appendChild(row);
        }
      }
      if (!state.stream.activeToolBlocks.size) state.stream.thinkingBlock?.classList.remove("running");
      break;
    case "content_chunk":
      if (!visible) break;
      settleModeStatus();
      ensureAssistantStack();
      if (!state.stream.assistantBubble) {
        scheduleStreamRender({ immediate: true });
        state.stream.thinkingBlock?.classList.remove("open", "running");
        state.stream.assistantBubble = document.createElement("div");
        // .streaming drives the blinking caret; settleStreamActivity clears it
        // on every terminal path, including abort and transport failure.
        state.stream.assistantBubble.className = "bubble streaming";
        state.stream.assistantStack.appendChild(state.stream.assistantBubble);
        appendMessageActions(state.stream.assistantStack, "copy");
        state.stream.thinkingBlock = null;
        state.stream.thinkingContent = null;
        state.stream.thinkingText = "";
        state.stream.activeToolBlocks.clear();
      }
      state.stream.assistantText += displayText(event.text ?? event.content);
      state.stream.assistantBubble.classList.remove("has-cards");
      scheduleStreamRender();
      if (event.error) state.stream.assistantBubble.classList.add("error");
      break;
    case "command_result":
      if (!visible) break;
      renderCommandResultCards(event.payload);
      break;
    case "token_usage":
      if (!visible) break;
      updateContextMeter(event.total, event.budget);
      break;
    case "done":
      const viewedGeneratingConversation = state.activeSessionName === generation.sessionName;
      const finishedName = event.active_session_name || generation.sessionName;
      if (event.saved_sessions) state.savedSessions = event.saved_sessions;
      bindGenerationSession(generation, finishedName);
      if (visible) {
        scheduleStreamRender({ immediate: true });
        const previousModel = state.settings.model_id || "local:default";
        if (event.state === "failed" && event.error) {
          if (state.stream.assistantBubble) {
            state.stream.assistantBubble.classList.add("error");
          } else {
            appendStreamError(event.error);
          }
        }
        // A /model switch compacts history in place, so the server can return
        // fewer messages than are currently on screen.
        const historyShrank =
          Array.isArray(event.history) && event.history.length < state.history.length;
        if (event.history) state.history = event.history;
        if (event.settings) state.settings = mergeSettings(event.settings);
        if (event.runtime) {
          state.runtime = event.runtime;
          reportRuntimeWarnings(state.runtime);
        }
        state.activeSessionName = finishedName;
        el.title.textContent = cleanSessionName(state.activeSessionName);
        document.title = titleForSession(state.activeSessionName);
        syncSettingsUI();
        if (
          previousModel !== "local:default"
          && state.settings.model_id === "local:default"
        ) {
          showStartupProfileDialog();
        }
        if (generation.wasBackgrounded || historyShrank) {
          resetStream();
          renderMessages();
        }
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

function appendStreamError(detail) {
  settleModeStatus();
  ensureAssistantStack();
  state.stream.thinkingBlock?.classList.remove("open", "running");
  const normalized = String(detail || "The model response stopped unexpectedly.");
  if (!state.stream.assistantBubble) {
    state.stream.assistantBubble = document.createElement("div");
    state.stream.assistantBubble.className = "bubble error";
    state.stream.assistantStack.appendChild(state.stream.assistantBubble);
    appendMessageActions(state.stream.assistantStack, "copy");
  } else {
    state.stream.assistantBubble.classList.add("error");
  }
  if (!state.stream.assistantText.includes(normalized)) {
    const heading = state.stream.assistantText
      ? "\n\n---\n\n**Response interrupted.**\n\n"
      : "**Couldn’t complete this response.**\n\n";
    state.stream.assistantText += heading + normalized;
  }
  scheduleStreamRender({ immediate: true });
}

function scheduleStreamRender({ immediate = false } = {}) {
  const render = () => {
    state.stream.renderFrame = null;
    if (state.stream.thinkingContent) {
      state.stream.thinkingContent.textContent = state.stream.thinkingText;
      updateBlockPreview(state.stream.thinkingBlock, state.stream.thinkingText);
      scrollThinkingToBottom(state.stream.thinkingBlock);
    }
    if (state.stream.assistantBubble && !state.stream.assistantBubble.classList.contains("has-cards")) {
      renderResponseInto(state.stream.assistantBubble, state.stream.assistantText);
    }
    scrollToBottom();
  };

  if (immediate) {
    if (state.stream.renderFrame !== null) cancelAnimationFrame(state.stream.renderFrame);
    render();
    return;
  }
  if (state.stream.renderFrame === null) {
    state.stream.renderFrame = requestAnimationFrame(render);
  }
}

function resetStream() {
  if (state.stream.renderFrame !== null) cancelAnimationFrame(state.stream.renderFrame);
  state.stream.assistantStack = null;
  state.stream.assistantBubble = null;
  state.stream.assistantText = "";
  state.stream.thinkingBlock = null;
  state.stream.thinkingContent = null;
  state.stream.thinkingText = "";
  state.stream.renderFrame = null;
  state.stream.modeStatusLine = null;
  state.stream.activeToolBlocks.clear();
  state.stream.toolRound = 0;
}

function settleModeStatus() {
  state.stream.modeStatusLine?.remove();
  state.stream.modeStatusLine = null;
}

function settleStreamActivity(interrupted = false) {
  // Streaming can end without a thinking_end/tool_end event when fetch is
  // aborted or the connection fails. Settle the DOM before resetStream drops
  // the only references to these still-visible elements.
  state.stream.thinkingBlock?.classList.remove("open", "running");
  state.stream.assistantBubble?.classList.remove("streaming");
  settleModeStatus();
  for (const entry of state.stream.activeToolBlocks.values()) {
    settleToolRow(entry.row, { status: interrupted ? "unknown" : "ok" });
    if (interrupted) entry.row.dataset.status = "unknown";
  }
}

function finishGeneration(generation = generationForSession(), { interrupted = false } = {}) {
  if (!generation || !state.generations.has(generation.id)) return;
  const visible = isGenerationVisible(generation);
  if (visible) {
    scheduleStreamRender({ immediate: true });
    settleStreamActivity(interrupted);
  }
  state.generations.delete(generation.id);
  if (visible) resetStream();
  syncActiveGeneration();
  renderSessions();
  updateComposerState();
}

function resetComposer({ clear = true } = {}) {
  stopVoiceInput({ abort: true, silent: true });
  closeSlashMenu();
  resetPromptRecall();
  if (!el.input) return;
  el.input.disabled = false;
  el.input.readOnly = false;
  if (clear) el.input.value = "";
  resizeComposer();
  updateComposerState();
  requestAnimationFrame(() => el.input?.focus({ preventScroll: true }));
}

function stopGeneration({ generation = generationForSession(), refresh = true } = {}) {
  if (!generation || generation.stopping) return;
  generation.stopping = true;
  generation.controller?.abort();
  updateComposerState();
  toast("Stopping generation…");
  void (async () => {
    try {
      const response = await fetch("/api/cancel-generation", {
        method: "POST",
        headers: apiHeaders(true),
        body: JSON.stringify({ generation_id: generation.id, client_id: state.clientId }),
        keepalive: true
      });
      if (response.ok) await waitForGenerationRelease(generation);
    } catch {
      // Aborting the stream still tells the server to close the connection.
      // The server retains its ownership guard if the cancellation request
      // itself cannot be delivered.
    } finally {
      finishGeneration(generation, { interrupted: true });
      if (refresh) refreshSessions().catch(() => {});
    }
  })();
}

function waitForGenerationRelease(generation) {
  return new Promise((resolve) => {
    const poll = async () => {
      try {
        const response = await fetch(
          `/api/generations?client_id=${encodeURIComponent(state.clientId)}`,
          { headers: apiHeaders() }
        );
        if (!response.ok) return resolve();
        const payload = await response.json();
        const active = payload.active_operations || [];
        if (!active.some((operation) => operation.generation_id === generation.id)) {
          return resolve();
        }
      } catch {
        return resolve();
      }
      window.setTimeout(poll, 120);
    };
    poll();
  });
}

function appendStatus(text, activityMode = "") {
  if (activityMode === "ultra" || activityMode === "deep-research") {
    appendModeActivity(text, activityMode);
    return;
  }
  settleModeStatus();
  const status = document.createElement("div");
  status.className = "status-line";
  status.textContent = text;
  el.messages.appendChild(status);
  scrollToBottom();
}

function appendModeActivity(text, activityMode) {
  settleModeStatus();
  ensureAssistantStack();
  if (!state.stream.thinkingBlock) {
    state.stream.thinkingBlock = detailBlock("Thinking", "reasoning", "", false);
    state.stream.thinkingContent = state.stream.thinkingBlock.querySelector(".block-content");
    state.stream.assistantStack.appendChild(state.stream.thinkingBlock);
  }

  state.stream.thinkingBlock.classList.add("running");
  const activity = document.createElement("span");
  activity.className = "mode-activity-inline running";
  activity.dataset.mode = activityMode;
  activity.textContent = text;
  state.stream.thinkingBlock.querySelector(".block-title")?.appendChild(activity);
  state.stream.modeStatusLine = activity;
  scrollToBottom();
}

async function clearConversation() {
  const generation = generationForSession();
  if (generation) stopGeneration({ generation, refresh: false });
  try {
    await fetch("/api/clear-session", { method: "POST", headers: apiHeaders() });
    state.history = [];
    selectConversationView("New conversation");
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
  if (settingsOpen()) closeSettings();
  await waitForPendingConversationIdentity();
  markCurrentGenerationBackgrounded();
  const requestedView = ++state.viewVersion;
  try {
    const response = await fetch("/api/new-session", { method: "POST", headers: apiHeaders() });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (requestedView !== state.viewVersion) return;
    state.history = [];
    state.activeSessionName = "New conversation";
    state.savedSessions = data.saved_sessions || state.savedSessions;
    el.title.textContent = "New conversation";
    document.title = "Selene";
    state.followOutput = true;
    syncActiveGeneration();
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
  const generation = generationForSession(name);
  if (generation) stopGeneration({ generation, refresh: false });
  try {
    const response = await fetch("/api/delete-session", {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify({ name })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    state.savedSessions = data.saved_sessions || state.savedSessions.filter((session) => session !== name);
    const backendStartedNewSession = ["Active Session", "New conversation"].includes(data.active_session_name);
    if (deletingActiveSession || backendStartedNewSession) {
      state.history = [];
      selectConversationView("New conversation");
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
  const response = await fetch("/api/settings", { headers: apiHeaders() });
  if (!response.ok) return;
  const data = await response.json();
  state.savedSessions = data.saved_sessions || [];
  const backendName = data.active_session_name || state.activeSessionName;
  if (!generationForSession() || backendName === state.activeSessionName) {
    state.activeSessionName = backendName;
  }
  syncActiveGeneration();
  renderSessions();
}

// ── Vaults ──────────────────────────────────────────────────────────────
// The panel is read-only; every write still goes through the /vault slash
// commands, which are unchanged.
const VAULT_PANEL_KEY = "selene-vault-panel";

function vaultPanelOpen() {
  try {
    return localStorage.getItem(VAULT_PANEL_KEY) === "open";
  } catch {
    return false;
  }
}

function setVaultPanelOpen(open) {
  try {
    localStorage.setItem(VAULT_PANEL_KEY, open ? "open" : "closed");
  } catch {
    /* private browsing — the panel just won't remember across reloads */
  }
}

function toggleVaultPanel() {
  const section = el.vaultToggle?.closest(".vault-section");
  if (!section) return;
  const open = !section.classList.contains("open");
  section.classList.toggle("open", open);
  el.vaultToggle.setAttribute("aria-expanded", open ? "true" : "false");
  setVaultPanelOpen(open);
  // Fetched on first expand, never during loadState: the first call has to
  // import ChromaDB and can take seconds.
  if (open && !state.vaults.loaded) loadVaults();
}

async function loadVaults() {
  if (state.vaults.loading) return;
  state.vaults.loading = true;
  renderVaultList();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch("/api/vaults", { headers: apiHeaders(), signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.vaults.data = await response.json();
    state.vaults.loaded = true;
    state.vaults.error = "";
  } catch (error) {
    state.vaults.error = error.name === "AbortError" ? "Timed out." : String(error.message || error);
  } finally {
    clearTimeout(timeout);
    state.vaults.loading = false;
    renderVaultList();
  }
}

function renderVaultList() {
  if (!el.vaultList) return;
  el.vaultList.innerHTML = "";
  const data = state.vaults.data;

  const note = (text) => {
    const row = document.createElement("p");
    row.className = "vault-empty";
    row.textContent = text;
    el.vaultList.appendChild(row);
  };

  if (state.vaults.loading) {
    for (let i = 0; i < 3; i += 1) {
      const skeleton = document.createElement("div");
      skeleton.className = "vault-skeleton";
      el.vaultList.appendChild(skeleton);
    }
    return;
  }
  if (state.vaults.error) return note(state.vaults.error);
  if (!data) return;
  // Only the backend's own error string is shown — nothing is invented here.
  if (!data.available || data.error) return note(data.error || "Vaults unavailable.");

  const vaults = data.vaults || [];
  const aliases = data.aliases || [];
  if (el.vaultCount) {
    el.vaultCount.textContent = String(vaults.length);
    el.vaultCount.hidden = vaults.length === 0;
  }
  if (!vaults.length && !aliases.length) return note("No indexed vaults.");

  const maxChunks = Math.max(1, ...vaults.map((v) => Number(v.indexed_chunks) || 0));
  const byCollection = new Map();

  vaults.forEach((vault) => {
    const name = vault.collection || "unknown";
    const chunks = Number(vault.indexed_chunks);
    const row = document.createElement("button");
    row.type = "button";
    row.className = "vault-row";
    row.setAttribute("role", "listitem");
    row.innerHTML = `
      <span class="vault-row-main">
        <span class="vault-row-name"></span>
        <span class="vault-row-count"></span>
      </span>
      <span class="vault-bar"><span class="vault-bar-fill"></span></span>
    `;
    row.querySelector(".vault-row-name").textContent = name;
    row.querySelector(".vault-row-count").textContent =
      Number.isFinite(chunks) ? chunks.toLocaleString() : "—";
    row.querySelector(".vault-bar-fill").style.width =
      `${Math.round(((Number.isFinite(chunks) ? chunks : 0) / maxChunks) * 100)}%`;
    row.addEventListener("click", () => insertVaultQuery(name));
    el.vaultList.appendChild(row);
    byCollection.set(name, row);
  });

  const orphans = [];
  aliases.forEach((entry) => {
    const collection = entry.collection || "";
    const row = document.createElement("div");
    row.className = "vault-alias";
    row.innerHTML = `<span class="vault-alias-name"></span><span class="vault-alias-target"></span>`;
    row.querySelector(".vault-alias-name").textContent = entry.alias || "?";
    row.querySelector(".vault-alias-target").textContent = collection || "—";
    if (byCollection.has(collection)) {
      byCollection.get(collection).after(row);
    } else {
      // An alias whose collection is not indexed. Flagged, not hidden.
      row.classList.add("orphan");
      orphans.push(row);
    }
  });
  orphans.forEach((row) => el.vaultList.appendChild(row));
}

// Reuses the slash-insertion semantics: trailing space, caret at the end, and
// it never sends on the user's behalf.
function insertVaultQuery(collection) {
  if (!el.input) return;
  if (settingsOpen()) closeSettings();
  el.input.value = `/vault search --collection ${collection} `;
  el.input.focus();
  el.input.setSelectionRange(el.input.value.length, el.input.value.length);
  resizeComposer();
  updateComposerState();
  updateSlashMenu();
}

// Structured command output is an upgrade over the markdown that already
// rendered — never a requirement. Any failure leaves the markdown standing.
function renderCommandResultCards(payload) {
  const bubble = state.stream.assistantBubble;
  if (!bubble || !payload?.kind) return;
  let card = null;
  try {
    if (payload.kind === "vault.list") card = vaultListCard(payload);
    else if (payload.kind === "vault.aliases") card = vaultAliasesCard(payload);
    else if (payload.kind === "vault.search") card = vaultSearchCard(payload);
  } catch {
    card = null;
  }
  if (!card) return;
  // dataset.messageText is left intact so Copy still yields the raw markdown.
  bubble.replaceChildren(card);
  bubble.classList.add("has-cards");
  scrollToBottom();
}

function vaultCardShell(title, count) {
  const section = document.createElement("section");
  section.className = "vault-card";
  const header = document.createElement("header");
  header.className = "vault-card-head";
  const heading = document.createElement("span");
  heading.textContent = title;
  header.appendChild(heading);
  if (count != null) {
    const badge = document.createElement("span");
    badge.className = "vault-card-count";
    badge.textContent = String(count);
    header.appendChild(badge);
  }
  section.appendChild(header);
  return section;
}

function vaultListCard(payload) {
  const vaults = payload.vaults || [];
  if (!vaults.length) return null;
  const card = vaultCardShell("Indexed vaults", vaults.length);
  const max = Math.max(1, ...vaults.map((v) => Number(v.indexed_chunks) || 0));
  vaults.forEach((vault) => {
    const chunks = Number(vault.indexed_chunks);
    const row = document.createElement("div");
    row.className = "vault-card-row";
    row.innerHTML = `
      <span class="vault-card-name"></span>
      <span class="vault-bar"><span class="vault-bar-fill"></span></span>
      <span class="vault-card-count-text"></span>
    `;
    row.querySelector(".vault-card-name").textContent = vault.collection || "unknown";
    row.querySelector(".vault-card-count-text").textContent =
      Number.isFinite(chunks) ? `${chunks.toLocaleString()} chunks` : "unknown";
    row.querySelector(".vault-bar-fill").style.width =
      `${Math.round(((Number.isFinite(chunks) ? chunks : 0) / max) * 100)}%`;
    card.appendChild(row);
  });
  return card;
}

function vaultAliasesCard(payload) {
  const aliases = payload.aliases || [];
  if (!aliases.length) return null;
  const card = vaultCardShell("Vault aliases", aliases.length);
  aliases.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "vault-card-row alias";
    row.innerHTML = `
      <span class="vault-card-name"></span>
      <span class="vault-card-arrow" aria-hidden="true">→</span>
      <span class="vault-card-target"></span>
      <span class="vault-card-path"></span>
    `;
    row.querySelector(".vault-card-name").textContent = entry.alias || "?";
    row.querySelector(".vault-card-target").textContent = entry.collection || "—";
    const path = row.querySelector(".vault-card-path");
    if (entry.file_path) path.textContent = entry.file_path;
    else path.remove();
    card.appendChild(row);
  });
  return card;
}

function vaultSearchCard(payload) {
  const results = payload.results || [];
  if (!results.length) return null;
  const card = vaultCardShell(`Results for “${payload.query || ""}”`, results.length);
  card.classList.add("vault-search");
  results.forEach((result, index) => {
    const row = document.createElement("div");
    row.className = "vault-result";
    row.innerHTML = `
      <span class="vault-result-index"></span>
      <div class="vault-result-body">
        <div class="vault-result-head">
          <span class="vault-result-source"></span>
          <span class="vault-score"><span class="vault-score-fill"></span></span>
        </div>
        <p class="vault-result-text"></p>
      </div>
    `;
    row.querySelector(".vault-result-index").textContent = String(index + 1).padStart(2, "0");
    row.querySelector(".vault-result-source").textContent = result.source || "unknown";
    const score = Number(result.score);
    row.querySelector(".vault-score-fill").style.width =
      `${Math.round(Math.min(1, Math.max(0, Number.isFinite(score) ? score : 0)) * 100)}%`;

    const text = row.querySelector(".vault-result-text");
    const full = String(result.text || "");
    text.textContent = full;
    if (full.length > 240) {
      const more = document.createElement("button");
      more.type = "button";
      more.className = "vault-more";
      more.textContent = "Show more";
      more.addEventListener("click", () => {
        const expanded = text.classList.toggle("expanded");
        more.textContent = expanded ? "Show less" : "Show more";
      });
      row.querySelector(".vault-result-body").appendChild(more);
    } else {
      text.classList.add("expanded");
    }
    card.appendChild(row);
  });
  return card;
}

// Drives the filled portion of a range track, which CSS cannot derive alone.
function updateRangeFill(input) {
  if (!input) return;
  const min = Number(input.min || 0);
  const max = Number(input.max || 100);
  const span = max - min || 1;
  const ratio = (Number(input.value) - min) / span;
  input.style.setProperty("--fill", `${Math.min(100, Math.max(0, ratio * 100))}%`);
}

function settingsOpen() {
  return el.settingsView?.classList.contains("open") === true;
}

// Keeps the settings sub-nav in step with what is actually on screen. Without
// it the rail highlight is frozen on the first section.
function syncSettingsNav() {
  if (!el.settingsView || !settingsOpen()) return;
  const links = [...el.settingsView.querySelectorAll(".settings-nav-link")];
  if (!links.length) return;
  const top = el.settingsView.getBoundingClientRect().top;
  let current = links[0];
  links.forEach((link) => {
    const section = el.settingsView.querySelector(link.getAttribute("href"));
    if (!section) return;
    // 24px of slack so a section counts as current just before it hits the top.
    if (section.getBoundingClientRect().top - top <= 24) current = link;
  });
  links.forEach((link) => link.classList.toggle("active", link === current));
}

function openSettings() {
  if (!el.settingsView) return;
  el.settingsView.hidden = false;
  el.settingsView.classList.add("open");
  if (el.settingsBackdrop) {
    el.settingsBackdrop.hidden = false;
    requestAnimationFrame(() => el.settingsBackdrop.classList.add("open"));
  }
  el.settingsBtn?.setAttribute("aria-pressed", "true");
  refreshOllamaStatus();
  syncSettingsNav();
}

function closeSettings() {
  if (!el.settingsView) return;
  el.settingsView.classList.remove("open");
  if (el.settingsBackdrop) el.settingsBackdrop.classList.remove("open");
  el.settingsBtn?.setAttribute("aria-pressed", "false");
  setTimeout(() => {
    if (!settingsOpen()) {
      el.settingsView.hidden = true;
      if (el.settingsBackdrop) el.settingsBackdrop.hidden = true;
    }
  }, 380);
  el.settingsBtn?.focus();
}

// ollama_status/ollama_reason are returned by GET /api/settings and were never
// read. They are status, not settings, so they are surfaced read-only.
// POST /api/settings does not echo them — never clear the card from a POST.
async function refreshOllamaStatus() {
  applyOllamaStatus();
  try {
    // Awaiting the write chain first means this GET cannot interleave with an
    // in-flight settings POST. Only these two fields are read.
    await settingsWriteChain;
    const response = await fetch("/api/settings", { headers: apiHeaders() });
    if (!response.ok) return;
    const data = await response.json();
    state.ollama = { status: data.ollama_status || "", reason: data.ollama_reason || "" };
    applyOllamaStatus();
  } catch {
    /* status is a nicety; a failure here must not disturb the page */
  }
}

function applyOllamaStatus() {
  const status = state.ollama?.status || "";
  const reason = state.ollama?.reason || "";
  if (el.ollamaStatus) el.ollamaStatus.textContent = status || "—";
  if (el.ollamaDot) {
    el.ollamaDot.dataset.state = status === "Online" ? "online" : status ? "offline" : "unknown";
  }
  if (el.ollamaReason) {
    el.ollamaReason.textContent = reason;
    el.ollamaReason.hidden = !reason;
  }
}

function renderThemeOptions() {
  if (!el.themeOptions) return;
  el.themeOptions.innerHTML = "";

  PLACE_THEMES.forEach((theme) => {
    const button = document.createElement("button");
    const active = theme.id === state.theme;
    button.type = "button";
    button.className = `theme-option${active ? " active" : ""}`;
    button.dataset.theme = theme.id;
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(active));
    button.style.setProperty("--preview-bg", theme.background);
    button.style.setProperty("--preview-surface", theme.surface);
    button.style.setProperty("--preview-primary", theme.primary);
    button.style.setProperty("--preview-accent", theme.accent);
    button.innerHTML = `
      <span class="theme-swatches" aria-hidden="true"></span>
      <span class="theme-option-copy">
        <strong>${escapeHTML(theme.name)}${theme.id === "oslo" ? " · Default" : ""}</strong>
        <small>${escapeHTML(theme.description)}</small>
      </span>
      <span class="theme-option-check" aria-hidden="true">✓</span>
    `;
    button.addEventListener("click", () => {
      applyTheme(theme.id, { announce: true });
      closeThemeDialog();
    });
    el.themeOptions.appendChild(button);
  });
}

function openThemeDialog(trigger = document.activeElement) {
  if (!el.themeBackdrop || !el.themeDialog) return;
  if (themeCloseTimer !== null) {
    clearTimeout(themeCloseTimer);
    themeCloseTimer = null;
  }
  themeTriggerElement = trigger instanceof HTMLElement ? trigger : el.input;
  if (settingsOpen()) closeSettings();
  renderThemeOptions();
  el.themeBackdrop.hidden = false;
  el.themeDialog.setAttribute("aria-hidden", "false");
  el.themeButton?.setAttribute("aria-expanded", "true");
  requestAnimationFrame(() => {
    el.themeBackdrop.classList.add("open");
    const active = el.themeOptions?.querySelector(".theme-option.active");
    (active || el.themeOptions?.querySelector(".theme-option"))?.focus();
  });
}

function closeThemeDialog() {
  if (!el.themeBackdrop || !el.themeDialog || el.themeBackdrop.hidden) return;
  el.themeBackdrop.classList.remove("open");
  el.themeDialog.setAttribute("aria-hidden", "true");
  el.themeButton?.setAttribute("aria-expanded", "false");
  themeCloseTimer = window.setTimeout(() => {
    el.themeBackdrop.hidden = true;
    themeCloseTimer = null;
  }, 180);
  const returnTarget = themeTriggerElement?.isConnected ? themeTriggerElement : el.input;
  themeTriggerElement = null;
  returnTarget?.focus({ preventScroll: true });
}

function themeOptionButtons() {
  return Array.from(el.themeOptions?.querySelectorAll(".theme-option") || []);
}

function themeGridColumns() {
  if (!el.themeOptions) return 1;
  const template = getComputedStyle(el.themeOptions).gridTemplateColumns.trim();
  return Math.max(1, template ? template.split(/\s+/).length : 1);
}

function focusThemeOption(index) {
  const options = themeOptionButtons();
  if (!options.length) return;
  const normalized = (index + options.length) % options.length;
  options[normalized].focus({ preventScroll: true });
  options[normalized].scrollIntoView({ block: "nearest", inline: "nearest" });
}

function cycleThemeDialogFocus(event) {
  const focusable = [
    document.getElementById("theme-close"),
    ...themeOptionButtons()
  ].filter((element) => element && !element.disabled);
  if (!focusable.length) return;
  const current = focusable.indexOf(document.activeElement);
  const direction = event.shiftKey ? -1 : 1;
  const start = current >= 0 ? current : 0;
  focusable[(start + direction + focusable.length) % focusable.length].focus({ preventScroll: true });
}

function handleThemeDialogKeydown(event) {
  const options = themeOptionButtons();
  const current = options.indexOf(document.activeElement);
  const fallback = Math.max(0, options.findIndex((option) => option.classList.contains("active")));
  const index = current >= 0 ? current : fallback;
  const columns = themeGridColumns();

  if (event.key === "Escape") {
    event.preventDefault();
    closeThemeDialog();
    return;
  }
  if (event.key === "Tab") {
    event.preventDefault();
    cycleThemeDialogFocus(event);
    return;
  }
  if (event.key === "Home") {
    event.preventDefault();
    focusThemeOption(0);
    return;
  }
  if (event.key === "End") {
    event.preventDefault();
    focusThemeOption(options.length - 1);
    return;
  }

  const movements = {
    ArrowLeft: -1,
    ArrowRight: 1,
    ArrowUp: -columns,
    ArrowDown: columns,
    PageUp: -(columns * 3),
    PageDown: columns * 3
  };
  if (Object.prototype.hasOwnProperty.call(movements, event.key)) {
    event.preventDefault();
    focusThemeOption(index + movements[event.key]);
    return;
  }
  if ((event.key === "Enter" || event.key === " ") && current >= 0) {
    event.preventDefault();
    options[current].click();
  }
}

function applyTheme(name, { announce = false } = {}) {
  const normalized = String(name || "").trim().toLowerCase();
  const theme = PLACE_THEMES.find((entry) => entry.id === normalized);
  if (!theme) return false;

  state.theme = theme.id;
  document.documentElement.dataset.theme = theme.id;
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme.id);
  } catch {
    // The active page still changes even when persistence is unavailable.
  }
  refreshWelcomeSkyPalette(state.sky.scene);
  renderThemeOptions();
  if (announce) toast(`${theme.name} theme selected.`);
  return true;
}

function handleThemeCommand(text) {
  const match = String(text || "").match(/^\/theme(?:\s+(.+))?$/i);
  if (!match) return false;
  const requested = String(match[1] || "").trim().toLowerCase();
  if (!requested) {
    openThemeDialog(el.input);
    return true;
  }
  if (!applyTheme(requested, { announce: true })) {
    openThemeDialog(el.input);
    toast(`Unknown theme “${requested}”. Choose a place below.`);
  }
  return true;
}

async function saveSession() {
  const name = window.prompt("Save this chat as:", cleanSessionName(state.activeSessionName));
  if (name === null) return;
  try {
    const response = await fetch("/api/save-session", {
      method: "POST",
      headers: apiHeaders(true),
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
  if (settingsOpen()) closeSettings();
  if (name === state.activeSessionName) {
    renderActiveConversation();
    updateComposerState();
    return;
  }
  await waitForPendingConversationIdentity();
  markCurrentGenerationBackgrounded();
  const requestedView = ++state.viewVersion;
  try {
    const response = await fetch("/api/load-session", {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify({ name })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    if (requestedView !== state.viewVersion) return;
    await loadState();
    toast("Session loaded.");
  } catch {
    toast("Could not load that session.");
  }
}

function updateComposerState() {
  const stopping = Boolean(state.generation?.stopping);
  const hasText = Boolean(el.input?.value.trim());
  if (el.mic) el.mic.disabled = !state.voice.recognition || state.isGenerating;
  if (el.modeTrigger) el.modeTrigger.disabled = state.isGenerating;
  if (el.modeClear) el.modeClear.disabled = state.isGenerating;
  if (el.modelTrigger) el.modelTrigger.disabled = state.isGenerating;
  if (el.modePicker) {
    el.modePicker.classList.toggle(
      "running",
      state.isGenerating && activeAgentMode() !== "normal"
    );
  }
  if (state.isGenerating) {
    closeModeMenu();
    closeModelMenu();
  }
  if (el.send) {
    el.send.disabled = stopping || (!state.isGenerating && !hasText);
    el.send.textContent = stopping ? "Stopping…" : (state.isGenerating ? "Stop" : "Send");
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

function resetPromptRecall() {
  state.promptRecall.index = null;
}

function promptHistory() {
  return state.history
    .filter((message) => message?.role === "user")
    .map((message) => displayText(message.content).trim())
    .filter(Boolean);
}

function handlePromptHistoryKeydown(event) {
  if (!el.input || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
    return false;
  }
  if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return false;

  const prompts = promptHistory();
  if (!prompts.length) return false;

  if (event.key === "ArrowUp") {
    if (state.promptRecall.index === null) {
      if (el.input.value) return false;
      state.promptRecall.index = prompts.length;
    }
    state.promptRecall.index = Math.max(0, state.promptRecall.index - 1);
  } else {
    if (state.promptRecall.index === null) return false;
    state.promptRecall.index += 1;
    if (state.promptRecall.index >= prompts.length) {
      state.promptRecall.index = null;
      el.input.value = "";
      event.preventDefault();
      resizeComposer();
      updateComposerState();
      return true;
    }
  }

  el.input.value = prompts[state.promptRecall.index];
  const cursor = el.input.value.length;
  el.input.setSelectionRange(cursor, cursor);
  event.preventDefault();
  resizeComposer();
  updateComposerState();
  closeSlashMenu();
  return true;
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

  if (item.command === "/theme") {
    el.input.value = "";
    closeSlashMenu();
    resizeComposer();
    updateComposerState();
    openThemeDialog(el.input);
    return;
  }

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
  const candidate = Number(forcedBudget ?? contextBudget());
  const budget = Number.isFinite(candidate) && candidate > 0 ? candidate : null;
  const used = forcedUsed ?? estimatedContextTokens();
  const pct = budget ? Math.min(100, Math.round((used / budget) * 100)) : 0;

  if (el.contextLabel) el.contextLabel.textContent = `${used} / ${budget ?? "—"}`;
  if (el.contextFill) el.contextFill.style.width = `${pct}%`;
  if (el.contextMeter) {
    el.contextMeter.classList.toggle("warn", pct >= 75 && pct < 90);
    el.contextMeter.classList.toggle("hot", pct >= 90);
  }
}

function activeSystemPromptText() {
  // Match backend: session override, else the selected model's own default.
  const override = String(state.settings.system || "").trim();
  if (override) return override;
  const backendPrompt = String(
    state.runtime?.active_system_prompt
      || state.runtime?.default_system_prompt
      || ""
  );
  if (backendPrompt) return backendPrompt;
  const fromHistory = state.history.find((message) => message.role === "system")?.content;
  if (fromHistory) return String(fromHistory);
  return "";
}

function estimatedContextTokens() {
  // Count system once (filtered out of history below) so the meter reflects the
  // same baseline the model always receives, even on an empty new chat.
  const systemPrompt = activeSystemPromptText();
  let total = 0;
  if (systemPrompt) {
    // Role/message framing overhead mirrors agent.core._estimate_message_tokens.
    total += estimateTokens(JSON.stringify({ role: "system", content: systemPrompt })) + 4;
  }

  const historyText = state.history
    .filter((message) => message.role !== "system")
    .map((message) => [
      message.role || "",
      displayText(message.content),
      displayText(message.planning),
      displayText(message.thinking),
      JSON.stringify(message.tool_calls || "")
    ].join("\n"))
    .join("\n");
  total += estimateTokens(historyText);

  const draft = el.input?.value || "";
  if (draft) total += estimateTokens(draft);

  return total;
}

function estimateTokens(text) {
  // Align with agent.core._estimate_tokens: ~4 ASCII chars/token, ~1 non-ASCII.
  const value = String(text || "");
  if (!value) return 0;
  let ascii = 0;
  let nonAscii = 0;
  for (let index = 0; index < value.length; index += 1) {
    if (value.charCodeAt(index) < 128) ascii += 1;
    else nonAscii += 1;
  }
  return Math.floor(ascii / 4) + nonAscii + 1;
}

function contextBudget() {
  const budget = Number(
    state.settings.options?.num_ctx
      ?? state.runtime?.effective_options?.num_ctx
  );
  return Number.isFinite(budget) && budget > 0 ? budget : null;
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
    await writeClipboardText(code);
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

async function writeClipboardText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) throw new Error("Clipboard command was rejected");
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

function displayText(value) {
  if (value == null) return "";
  if (typeof value === "string") return repairTextEncoding(value);
  if (["number", "boolean", "bigint"].includes(typeof value)) return String(value);
  if (Array.isArray(value)) {
    return value.map(displayText).filter(Boolean).join("\n");
  }
  if (typeof value === "object") {
    for (const key of ["text", "content", "output_text", "value"]) {
      if (Object.prototype.hasOwnProperty.call(value, key)) return displayText(value[key]);
    }
    try {
      return JSON.stringify(value);
    } catch {
      return "[Unsupported response content]";
    }
  }
  return String(value);
}

const WINDOWS_1252_BYTES = Object.freeze({
  "€": 0x80, "‚": 0x82, "ƒ": 0x83, "„": 0x84, "…": 0x85,
  "†": 0x86, "‡": 0x87, "ˆ": 0x88, "‰": 0x89, "Š": 0x8a,
  "‹": 0x8b, "Œ": 0x8c, "Ž": 0x8e, "‘": 0x91, "’": 0x92,
  "“": 0x93, "”": 0x94, "•": 0x95, "–": 0x96, "—": 0x97,
  "˜": 0x98, "™": 0x99, "š": 0x9a, "›": 0x9b, "œ": 0x9c,
  "ž": 0x9e, "Ÿ": 0x9f
});

function decodeMojibakeRun(fragment) {
  const bytes = [];
  for (const character of fragment) {
    const code = character.codePointAt(0);
    if (code <= 0xff) {
      bytes.push(code);
    } else if (Object.prototype.hasOwnProperty.call(WINDOWS_1252_BYTES, character)) {
      bytes.push(WINDOWS_1252_BYTES[character]);
    } else {
      return fragment;
    }
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(new Uint8Array(bytes));
  } catch {
    return fragment;
  }
}

function repairTextEncoding(value) {
  let text = String(value || "");
  if (!/[ÃÂâðï]/u.test(text)) return text;
  for (let pass = 0; pass < 2; pass += 1) {
    const repaired = text.replace(/(?:Ã.|Â.|â..|ð...|ï..)/gsu, decodeMojibakeRun);
    if (repaired === text) break;
    text = repaired;
  }
  return text;
}

function renderResponseHTML(value) {
  const text = displayText(value);
  try {
    return renderMarkdown(text);
  } catch {
    // A malformed or incomplete markdown fragment must never hide model text.
    return escapeHTML(text).replace(/\n/g, "<br>");
  }
}

function renderResponseInto(node, value) {
  const text = displayText(value);
  node.dataset.messageText = text;
  try {
    node.innerHTML = renderResponseHTML(text);
  } catch {
    // DOM/CSP failures are display failures, not failed model responses.
    node.textContent = text;
  }
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
  // Files are either temporary (session_<timestamp>[_uuid].json) or agent-titled
  // (Title_Words_<timestamp>.json). Strip stamp/uuid so the sidebar shows the title.
  let base = String(name || "New conversation").replace(/\.json$/i, "");
  base = base
    .replace(/_[0-9a-f]{8}$/i, "")
    // Drop stamp and any numeric collision suffix (…_YYYYMMDD_HHMMSS[_us][_n]).
    .replace(/_\d{8}_\d{6}(?:_\d+)*$/u, "")
    .replace(/^session$/i, "New conversation")
    .replace(/_/g, " ")
    .replace(/^Active Session$/i, "New conversation")
    .trim();
  return base || "New conversation";
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
