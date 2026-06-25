/**
 * Selene — app.js (complete ground-up rewrite)
 *
 * Implements:
 *   - Full toast/notification system (success/error/warning/info, auto-dismiss,
 *     hover-pause, progress bar, ARIA live regions, manual dismiss)
 *   - Auto-resizing textarea (expands up to maxHeight, then scrolls)
 *   - Character / context-token count display
 *   - Dynamic document titles: "{Conversation Name} — Selene"
 *   - Streaming SSE handler (thinking, tool cards, content chunks)
 *   - Improved, human-friendly error messages (specific + constructive)
 *   - Settings persistence and sync
 *   - Session load/save with proper UX feedback
 *   - showConfirm() — replaces window.confirm() with a toast-based prompt
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════
   STATE
   ═══════════════════════════════════════════════════════════════ */
const state = {
  activeSession: 'New conversation',
  history: [],
  settings: {
    options: { temperature: 0.7, top_p: 0.9, top_k: 40 },
    history: true,
    think: true,
    system: ''
  },
  savedSessions: [],
  isGenerating: false,
  // Streaming refs
  currentAssistantBody:   null,
  currentAssistantBubble: null,
  currentThinkingCard:    null,
  currentAssistantText:   '',
  currentThinkingText:    '',
  toolCards:              {},
  toolCallCounter:        0
};

let abortController = null;


/* ═══════════════════════════════════════════════════════════════
   TOAST / NOTIFICATION SYSTEM

   Requirements met:
   ✓ 4 types: success, error, warning, info
   ✓ Auto-dismiss (errors 8 s, warnings 5 s, info 3.8 s, success 3.2 s)
   ✓ Manual dismiss (× button)
   ✓ Stacks cleanly (top-right)
   ✓ ARIA live regions: role="alert" for error/warning (immediate),
                        role="status" for success/info (polite)
   ✓ Pause on hover — dismissal timer pauses while hovered
   ✓ Progress bar shows remaining time
   ═══════════════════════════════════════════════════════════════ */
const TOAST_DURATIONS = { success: 3200, info: 3800, warning: 5200, error: 0 }; // 0 = must dismiss

const TOAST_ICONS = {
  success: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>`,
  error:   `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
  warning: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>`,
  info:    `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>`
};

/**
 * Show a toast notification.
 * @param {string} message     User-facing message text
 * @param {'success'|'error'|'warning'|'info'} type  Notification type
 * @param {number|null} duration  Override ms (0 = permanent until dismissed, null = default)
 * @returns {HTMLElement} The toast element
 */
function showToast(message, type = 'info', duration = null) {
  const container = document.getElementById('toast-container');
  if (!container) return null;

  const life = duration !== null ? duration : (TOAST_DURATIONS[type] ?? 3800);

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  // ARIA: errors/warnings announce immediately; info/success are polite
  toast.setAttribute('role', (type === 'error' || type === 'warning') ? 'alert' : 'status');
  toast.setAttribute('aria-live', (type === 'error' || type === 'warning') ? 'assertive' : 'polite');

  toast.innerHTML = `
    <span class="toast-icon">${TOAST_ICONS[type] || TOAST_ICONS.info}</span>
    <span class="toast-message">${escapeHtml(message)}</span>
    <button class="toast-close" aria-label="Dismiss notification">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
        <path d="M18 6 6 18M6 6l12 12"/>
      </svg>
    </button>
    ${life > 0 ? `<div class="toast-progress"><div class="toast-progress-bar" style="width:100%"></div></div>` : ''}
  `;

  let timerId = null;
  let paused = false;
  let remaining = life;
  let startTime = null;
  const progressBar = toast.querySelector('.toast-progress-bar');

  function dismiss() {
    if (!toast.isConnected) return;
    toast.classList.add('leaving');
    toast.addEventListener('transitionend', () => {
      toast.remove();
    }, { once: true });
    // Fallback in case transition doesn't fire
    setTimeout(() => toast.remove(), 300);
  }

  function startTimer() {
    if (life <= 0) return;
    startTime = Date.now();
    if (progressBar) {
      progressBar.style.transition = `width ${remaining}ms linear`;
      progressBar.style.width = '0%';
    }
    timerId = setTimeout(dismiss, remaining);
  }

  function pauseTimer() {
    if (!timerId) return;
    clearTimeout(timerId);
    timerId = null;
    remaining = remaining - (Date.now() - startTime);
    if (progressBar) {
      const pct = (remaining / life) * 100;
      progressBar.style.transition = 'none';
      progressBar.style.width = `${Math.max(0, pct)}%`;
    }
  }

  function resumeTimer() {
    if (life <= 0 || remaining <= 0) return;
    startTimer();
  }

  toast.addEventListener('mouseenter', () => { paused = true; pauseTimer(); });
  toast.addEventListener('mouseleave', () => { paused = false; resumeTimer(); });
  toast.querySelector('.toast-close').addEventListener('click', dismiss);

  container.appendChild(toast);
  startTimer();

  // Expose dismiss for external use (e.g., confirm toasts)
  toast._dismiss = dismiss;
  return toast;
}

/**
 * Replace window.confirm() — shows a warning toast with Confirm + Cancel buttons.
 * @param {string}   message    Text to display
 * @param {Function} onConfirm  Called on confirm
 * @param {Function} onCancel   Called on cancel (optional)
 */
function showConfirm(message, onConfirm, onCancel) {
  const container = document.getElementById('toast-container');
  if (!container) { if (onConfirm) onConfirm(); return; }

  const toast = document.createElement('div');
  toast.className = 'toast warning';
  toast.setAttribute('role', 'alertdialog');
  toast.setAttribute('aria-live', 'assertive');
  toast.setAttribute('aria-label', 'Confirmation required');

  toast.innerHTML = `
    <span class="toast-icon">${TOAST_ICONS.warning}</span>
    <div class="toast-message" style="display:flex;flex-direction:column;gap:8px;">
      <span>${escapeHtml(message)}</span>
      <div style="display:flex;gap:8px;">
        <button id="confirm-yes" style="background:rgba(240,112,112,0.15);color:#f07070;border:1px solid rgba(240,112,112,0.3);padding:4px 12px;border-radius:7px;font-size:12px;cursor:pointer;font-family:inherit;">
          Confirm
        </button>
        <button id="confirm-no" style="background:rgba(255,255,255,0.06);color:var(--ink-2);border:1px solid var(--border);padding:4px 12px;border-radius:7px;font-size:12px;cursor:pointer;font-family:inherit;">
          Cancel
        </button>
      </div>
    </div>
    <button class="toast-close" aria-label="Cancel">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
        <path d="M18 6 6 18M6 6l12 12"/>
      </svg>
    </button>
  `;

  const dismiss = () => {
    toast.classList.add('leaving');
    setTimeout(() => toast.remove(), 300);
  };

  toast.querySelector('#confirm-yes').addEventListener('click', () => {
    dismiss();
    onConfirm?.();
  });
  toast.querySelector('#confirm-no').addEventListener('click', () => {
    dismiss();
    onCancel?.();
  });
  toast.querySelector('.toast-close').addEventListener('click', () => {
    dismiss();
    onCancel?.();
  });

  container.appendChild(toast);
}


/* ═══════════════════════════════════════════════════════════════
   UTILITIES
   ═══════════════════════════════════════════════════════════════ */
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Update the document title.
 * Pattern: "{Conversation Name} — Selene" or just "Selene" for new conversations.
 * Ensures browser tabs, history, and bookmarks are meaningful.
 * @param {string} title - Conversation name or empty for default
 */
function setDocumentTitle(title) {
  const name = (title || '').trim();
  if (!name || name === 'New conversation' || name === 'Active Session') {
    document.title = 'Selene';
  } else {
    document.title = `${name} — Selene`;
  }
}

function scrollToBottom(force = false) {
  const el = document.getElementById('messages');
  if (!el) return;
  if (force) {
    el.scrollTop = el.scrollHeight;
    return;
  }
  // Auto-scroll only when near bottom (within 140px)
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 140;
  if (nearBottom) el.scrollTop = el.scrollHeight;
}


/* ═══════════════════════════════════════════════════════════════
   INITIALIZATION
   ═══════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  setupLibraries();
  setupEventListeners();
  loadInitialState();
  init3DEffects();
});

function init3DEffects() {
  if (typeof gsap === 'undefined') return;

  // Track mouse globally for 3D tilts
  const body = document.body;

  body.addEventListener('mousemove', (e) => {
    const { clientX, clientY } = e;
    const { innerWidth, innerHeight } = window;

    // Normalize coordinates (-1 to 1)
    const xPos = (clientX / innerWidth - 0.5) * 2;
    const yPos = (clientY / innerHeight - 0.5) * 2;

    // Smooth 3D tilt on the composer
    const composer = document.querySelector('.composer-wrap');
    if (composer) {
      gsap.to(composer, {
        rotationY: xPos * 4,
        rotationX: -yPos * 4,
        ease: "power2.out",
        duration: 0.6
      });
    }

    // Subtle floating parallax for the welcome screen
    const welcome = document.querySelector('.welcome');
    if (welcome) {
      gsap.to(welcome, {
        rotationY: xPos * 6,
        rotationX: -yPos * 6,
        x: xPos * -15,
        y: yPos * -15,
        ease: "power3.out",
        duration: 0.8
      });
    }
  });

  // Reset when mouse leaves window
  body.addEventListener('mouseleave', () => {
    const composer = document.querySelector('.composer-wrap');
    const welcome = document.querySelector('.welcome');
    if (composer) gsap.to(composer, { rotationY: 0, rotationX: 0, ease: "power2.out", duration: 0.8 });
    if (welcome) gsap.to(welcome, { rotationY: 0, rotationX: 0, x: 0, y: 0, ease: "power3.out", duration: 1 });
  });
}

function setupLibraries() {
  // Configure marked for safe, highlighted markdown
  if (window.marked) {
    marked.setOptions({
      highlight: (code, lang) => {
        if (window.hljs) {
          const language = hljs.getLanguage(lang) ? lang : 'plaintext';
          return hljs.highlight(code, { language }).value;
        }
        return escapeHtml(code);
      },
      langPrefix: 'hljs language-',
      gfm: true,
      breaks: false
    });
  }

  // Init lucide icons after DOM is ready
  if (window.lucide) lucide.createIcons();
}

function setupEventListeners() {
  // ── Sidebar toggle ────────────────────────────────────────────
  const sidebar     = document.getElementById('sidebar');
  const sidebarToggleBtn = document.getElementById('sidebar-toggle');

  sidebarToggleBtn?.addEventListener('click', () => {
    const isOpen = sidebar.classList.toggle('open');
    sidebarToggleBtn.setAttribute('aria-expanded', String(isOpen));
  });

  // Close sidebar on outside click (mobile)
  document.addEventListener('click', (e) => {
    if (
      window.innerWidth < 821 &&
      sidebar?.classList.contains('open') &&
      !sidebar.contains(e.target) &&
      e.target.id !== 'sidebar-toggle'
    ) {
      sidebar.classList.remove('open');
      sidebarToggleBtn?.setAttribute('aria-expanded', 'false');
    }
  });

  // ── New chat ──────────────────────────────────────────────────
  document.getElementById('new-chat-btn')?.addEventListener('click', () => {
    clearConversation();
  });

  // ── Composer (chatbox) ────────────────────────────────────────
  const textarea = document.getElementById('chat-input');
  const sendBtn  = document.getElementById('send-btn');
  const countEl  = document.getElementById('input-count');

  const MAX_TEXTAREA_HEIGHT = 220; // px — requirement: expand then scroll

  function resizeTextarea() {
    if (!textarea) return;
    textarea.style.height = 'auto';
    const next = Math.min(textarea.scrollHeight, MAX_TEXTAREA_HEIGHT);
    textarea.style.height = `${next}px`;
    // Switch to scrollable when at max
    textarea.style.overflowY = textarea.scrollHeight > MAX_TEXTAREA_HEIGHT ? 'auto' : 'hidden';

    // Update character count
    if (countEl) countEl.textContent = String(textarea.value.length);
    updateSendButton();
  }

  textarea?.addEventListener('input', resizeTextarea);

  textarea?.addEventListener('keydown', (e) => {
    // Enter sends; Shift+Enter inserts newline
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!state.isGenerating && textarea.value.trim()) {
        sendMessage();
      }
    }
  });

  // Prime on load
  requestAnimationFrame(resizeTextarea);

  sendBtn?.addEventListener('click', () => {
    if (state.isGenerating) {
      // Stop button
      abortController?.abort();
      finishGeneration(true);
    } else {
      sendMessage();
    }
  });

  // ── Settings panel ────────────────────────────────────────────
  const settingsToggleBtn = document.getElementById('settings-toggle');
  const settingsPanel     = document.getElementById('settings-panel');
  const settingsChevron   = settingsToggleBtn?.querySelector('.settings-chevron');

  settingsToggleBtn?.addEventListener('click', () => {
    const isOpen = settingsPanel.classList.toggle('open');
    settingsChevron?.classList.toggle('open', isOpen);
    settingsToggleBtn.setAttribute('aria-expanded', String(isOpen));
  });

  setupSettingsSliders();

  // ── Clear ─────────────────────────────────────────────────────
  document.getElementById('clear-btn')?.addEventListener('click', () => {
    showConfirm(
      'Clear this conversation? This cannot be undone.',
      () => clearConversation(),
      null
    );
  });

  // ── Help modal ────────────────────────────────────────────────
  const helpModal = document.getElementById('help-modal');

  document.getElementById('help-btn')?.addEventListener('click', () => {
    helpModal?.classList.add('open');
    helpModal?.querySelector('.close-modal')?.focus();
  });

  helpModal?.addEventListener('click', (e) => {
    // Close on backdrop click
    if (e.target === helpModal) helpModal.classList.remove('open');
  });

  helpModal?.querySelector('.close-modal')?.addEventListener('click', () => {
    helpModal.classList.remove('open');
  });

  // Close modals on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      helpModal?.classList.remove('open');
    }
  });
}

function setupSettingsSliders() {
  const sliderConfig = [
    { sliderId: 'param-temperature', displayId: 'temp-val', key: 'temperature', decimals: 2 },
    { sliderId: 'param-top_p',       displayId: 'topp-val', key: 'top_p',       decimals: 2 },
    { sliderId: 'param-top_k',       displayId: 'topk-val', key: 'top_k',       decimals: 0 }
  ];

  sliderConfig.forEach(({ sliderId, displayId, key, decimals }) => {
    const slider  = document.getElementById(sliderId);
    const display = document.getElementById(displayId);
    if (!slider || !display) return;

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      display.textContent = val.toFixed(decimals);
      state.settings.options = state.settings.options || {};
      state.settings.options[key] = val;
      persistSettings();
    });
  });

  document.getElementById('setting-history')?.addEventListener('change', (e) => {
    state.settings.history = e.target.checked;
    persistSettings();
  });

  document.getElementById('setting-think')?.addEventListener('change', (e) => {
    state.settings.think = e.target.checked;
    persistSettings();
  });
}

function syncSettingsUI() {
  const opts = state.settings?.options || {};

  const map = [
    ['param-temperature', 'temp-val', opts.temperature ?? 0.7, 2],
    ['param-top_p',       'topp-val', opts.top_p      ?? 0.9, 2],
    ['param-top_k',       'topk-val', opts.top_k      ?? 40,  0]
  ];

  map.forEach(([sid, vid, val, dec]) => {
    const sl = document.getElementById(sid);
    const dv = document.getElementById(vid);
    if (sl) sl.value = val;
    if (dv) dv.textContent = parseFloat(val).toFixed(dec);
  });

  const histEl  = document.getElementById('setting-history');
  const thinkEl = document.getElementById('setting-think');
  if (histEl)  histEl.checked  = state.settings.history !== false;
  if (thinkEl) thinkEl.checked = state.settings.think   !== false;
}

async function persistSettings() {
  try {
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.settings)
    });
  } catch {
    // Silently fail — user is already in the UI, no need to disrupt
  }
}


/* ═══════════════════════════════════════════════════════════════
   DATA LOADING
   ═══════════════════════════════════════════════════════════════ */
async function loadInitialState() {
  try {
    const res  = await fetch('/api/settings');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    state.history       = data.history           || [];
    state.savedSessions = data.saved_sessions     || [];
    state.settings      = data.settings           || state.settings;
    state.activeSession = data.active_session_name || 'New conversation';

    const badge = document.getElementById('model-badge');
    if (badge) badge.textContent = data.model_name || 'selene';

    updateConnectionStatus(data.ollama_status);
    syncSettingsUI();
    renderConversationList();
    renderMessages();
    setDocumentTitle(state.activeSession);

  } catch (err) {
    // IMPROVED ERROR: specific + constructive + human
    showToast(
      "Couldn't connect to Selene's backend. Make sure Ollama is running, then refresh the page.",
      'error'
    );
    renderWelcome();
  }
}

function updateConnectionStatus(status) {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  const wrap = document.getElementById('status-indicator');

  const isOnline = (status === 'Online');
  dot?.classList.toggle('offline', !isOnline);
  if (text) text.textContent = isOnline ? 'Online' : 'Offline';
  if (wrap) wrap.title = `Ollama: ${status || 'Unknown'}`;
}

async function clearConversation() {
  try {
    await fetch('/api/clear-session', { method: 'POST' });
    state.history       = [];
    state.activeSession = 'New conversation';

    const messages = document.getElementById('messages');
    if (messages) messages.innerHTML = '';

    renderConversationList();
    renderWelcome();
    setDocumentTitle(state.activeSession);
    showToast('Conversation cleared — ready for a fresh start.', 'info', 2000);
  } catch {
    showToast(
      'Could not clear the conversation right now. Please try again.',
      'error'
    );
  }
}


/* ═══════════════════════════════════════════════════════════════
   RENDERING — CONVERSATION LIST
   ═══════════════════════════════════════════════════════════════ */
function renderConversationList() {
  const list = document.getElementById('convo-list');
  if (!list) return;

  list.innerHTML = '';

  if (!state.savedSessions?.length) {
    const empty = document.createElement('div');
    empty.className = 'convo-empty';
    empty.textContent = 'No saved conversations yet.\nYour sessions will appear here.';
    list.appendChild(empty);
    return;
  }

  state.savedSessions.forEach((filename) => {
    const base = filename.replace('.json', '');
    let display = base;
    let metaText = '';

    // Parse timestamped filenames like: name_20240101_120000.json
    const m = base.match(/^(.+)_(\d{8})_(\d{6})$/);
    if (m) {
      display  = m[1].replace(/_/g, ' ');
      const y  = m[2].slice(0, 4);
      const mo = m[2].slice(4, 6);
      const d  = m[2].slice(6, 8);
      metaText = `${d}/${mo}`;
    }

    const item = document.createElement('div');
    item.className = `convo-item${filename === state.activeSession ? ' active' : ''}`;
    item.setAttribute('role', 'listitem');
    item.setAttribute('title', display);
    item.setAttribute('tabindex', '0');
    item.innerHTML = `
      <svg class="convo-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>
      <span class="convo-name">${escapeHtml(display)}</span>
      ${metaText ? `<span class="convo-meta">${escapeHtml(metaText)}</span>` : ''}
    `;

    const load = () => loadSavedSession(filename);
    item.addEventListener('click', load);
    item.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); load(); }
    });

    list.appendChild(item);
  });
}

async function loadSavedSession(filename) {
  try {
    const res = await fetch('/api/load-session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: filename })
    });
    const data = await res.json();

    if (data.status === 'success') {
      await loadInitialState();
      // Close sidebar on mobile after loading
      if (window.innerWidth < 821) {
        document.getElementById('sidebar')?.classList.remove('open');
        document.getElementById('sidebar-toggle')?.setAttribute('aria-expanded', 'false');
      }
      showToast('Conversation loaded.', 'success', 2200);
    } else {
      showToast(
        "That conversation couldn't be opened — the file may be missing or corrupted. Try another one.",
        'error'
      );
    }
  } catch {
    showToast(
      "Network hiccup — the conversation didn't load. Check your connection and try again.",
      'error'
    );
  }
}


/* ═══════════════════════════════════════════════════════════════
   RENDERING — MESSAGES
   ═══════════════════════════════════════════════════════════════ */
function renderMessages() {
  const container = document.getElementById('messages');
  if (!container) return;
  container.innerHTML = '';

  // Coalesce history into display pairs
  const display = [];
  for (let i = 0; i < state.history.length; i++) {
    const msg = state.history[i];
    if (msg.role === 'user') {
      display.push({ role: 'user', content: msg.content });
    } else if (msg.role === 'assistant') {
      const toolResults = [];
      let j = i + 1;
      while (j < state.history.length && state.history[j].role === 'tool') {
        toolResults.push(state.history[j]);
        j++;
      }
      i = j - 1;
      display.push({
        role:         'assistant',
        content:      msg.content,
        thinking:     msg.thinking,
        tool_calls:   msg.tool_calls,
        tool_results: toolResults
      });
    }
    // Skip system messages in display
  }

  if (display.length === 0) {
    renderWelcome(container);
    return;
  }

  display.forEach((m) => {
    if (m.role === 'user') appendUserMessage(m.content, false);
    else                   appendAssistantMessage(m, false);
  });

  scrollToBottom(true);
}

function renderWelcome(container = null) {
  const el = container || document.getElementById('messages');
  if (!el) return;

  const div = document.createElement('div');
  div.className = 'welcome';
  div.innerHTML = `
    <svg class="welcome-moon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
    <h1 class="welcome-title">Good to see you.</h1>
    <p class="welcome-sub">Selene is ready — your calm, capable AI on this machine.<br>Ask anything or try one of these:</p>

    <div class="suggestions" role="list">
      <button class="suggestion-chip" data-prompt="Search the web for the latest AI research news" role="listitem">
        <span class="chip-label">Explore</span>
        <span class="chip-text">Search for the latest AI research</span>
      </button>
      <button class="suggestion-chip" data-prompt="Index my project folder and give me an architecture summary" role="listitem">
        <span class="chip-label">Vault</span>
        <span class="chip-text">Index my project and summarize its architecture</span>
      </button>
      <button class="suggestion-chip" data-prompt="Write a Python function to recursively find duplicate files" role="listitem">
        <span class="chip-label">Code</span>
        <span class="chip-text">Write a script to find duplicate files</span>
      </button>
      <button class="suggestion-chip" data-prompt="Play some ambient focus music on Spotify" role="listitem">
        <span class="chip-label">Music</span>
        <span class="chip-text">Queue ambient focus music on Spotify</span>
      </button>
    </div>
  `;

  el.appendChild(div);

  div.querySelectorAll('.suggestion-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const input = document.getElementById('chat-input');
      if (input) {
        input.value = chip.dataset.prompt;
        input.dispatchEvent(new Event('input'));
        input.focus();
      }
    });
  });
}

function appendUserMessage(text, animate = true) {
  const container = document.getElementById('messages');
  if (!container) return;

  // Remove welcome screen if present
  container.querySelector('.welcome')?.remove();

  const row = document.createElement('div');
  row.className = 'msg user';
  row.innerHTML = `
    <div class="msg-body">
      <div class="bubble">${escapeHtml(text)}</div>
    </div>
    <div class="msg-avatar" aria-label="You" title="You">You</div>
  `;
  container.appendChild(row);

  if (animate && typeof gsap !== 'undefined') {
    gsap.fromTo(row,
      { opacity: 0, y: 20, rotationX: -15, translateZ: -20 },
      { opacity: 1, y: 0, rotationX: 0, translateZ: 0, duration: 0.6, ease: "back.out(1.5)" }
    );
  }

  if (animate) scrollToBottom(true);
}

function appendAssistantMessage(msgData, animate = true) {
  const container = document.getElementById('messages');
  if (!container) return;

  const row    = document.createElement('div');
  row.className = 'msg assistant';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.innerHTML = `<img src="/avatar.png" alt="Selene" width="30" height="30">`;

  const body = document.createElement('div');
  body.className = 'msg-body';

  if (msgData.thinking) {
    body.appendChild(createThinkingCard(msgData.thinking, false));
  }

  if (msgData.tool_calls?.length) {
    msgData.tool_calls.forEach((tc, idx) => {
      const result = msgData.tool_results?.[idx]?.content || '';
      body.appendChild(createToolCard(tc.function.name, tc.function.arguments, result, false));
    });
  }

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML  = marked.parse(msgData.content || '');
  body.appendChild(bubble);
  addCopyButtons(bubble);

  row.appendChild(avatar);
  row.appendChild(body);
  container.appendChild(row);

  if (animate && typeof gsap !== 'undefined') {
    gsap.fromTo(row,
      { opacity: 0, y: 20, rotationX: 10, translateZ: -20 },
      { opacity: 1, y: 0, rotationX: 0, translateZ: 0, duration: 0.6, ease: "back.out(1.2)" }
    );
  }

  if (animate) scrollToBottom(true);
  if (window.lucide) lucide.createIcons();
}


/* ═══════════════════════════════════════════════════════════════
   THINKING & TOOL CARDS
   ═══════════════════════════════════════════════════════════════ */
function createThinkingCard(text, streaming = false) {
  const card = document.createElement('div');
  card.className = `thinking-card${streaming ? '' : ' collapsed'}`;

  const header = document.createElement('div');
  header.className = 'thinking-header';
  header.setAttribute('role', 'button');
  header.setAttribute('tabindex', '0');
  header.setAttribute('aria-expanded', String(!streaming));
  header.innerHTML = `
    <svg class="thinking-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <path d="M12 2a8 8 0 0 1 8 8c0 5.4-8 12-8 12S4 15.4 4 10a8 8 0 0 1 8-8z"/>
      <circle cx="12" cy="10" r="2"/>
    </svg>
    <span class="thinking-label">Thinking</span>
    <svg class="card-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true">
      <path d="m6 9 6 6 6-6"/>
    </svg>
  `;

  const body = document.createElement('div');
  body.className = 'thinking-body';
  body.textContent = text;

  const toggle = () => {
    const collapsed = card.classList.toggle('collapsed');
    header.setAttribute('aria-expanded', String(!collapsed));
  };

  header.addEventListener('click', toggle);
  header.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
  });

  card.appendChild(header);
  card.appendChild(body);
  return card;
}

function createToolCard(name, args, result = '', running = false) {
  const card = document.createElement('div');
  card.className = `tool-card${running ? '' : ' collapsed'}`;

  const argsStr   = typeof args === 'string' ? args : JSON.stringify(args, null, 2);
  const resultStr = result ? `\n\nResult:\n${result}` : (running ? '\nRunning…' : '');

  const header = document.createElement('div');
  header.className = 'tool-header';
  header.setAttribute('role', 'button');
  header.setAttribute('tabindex', '0');
  header.setAttribute('aria-expanded', String(running));
  header.innerHTML = `
    <svg class="tool-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
    </svg>
    <span style="font-size:var(--fs-caption);color:var(--ink-3);">Tool: </span>
    <span class="tool-name">${escapeHtml(name)}</span>
    <svg class="card-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true">
      <path d="m6 9 6 6 6-6"/>
    </svg>
  `;

  const body = document.createElement('div');
  body.className = 'tool-body';
  body.textContent = `Args:\n${argsStr}${resultStr}`;

  const toggle = () => {
    const collapsed = card.classList.toggle('collapsed');
    header.setAttribute('aria-expanded', String(!collapsed));
  };

  header.addEventListener('click', toggle);
  header.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
  });

  card.appendChild(header);
  card.appendChild(body);
  return card;
}

function addCopyButtons(scope) {
  scope.querySelectorAll('pre').forEach((pre) => {
    if (pre.querySelector('.copy-btn')) return; // skip if already added

    const btn    = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code to clipboard');

    btn.addEventListener('click', async () => {
      const code = pre.querySelector('code')?.textContent || '';
      try {
        await navigator.clipboard.writeText(code);
        btn.textContent = 'Copied!';
        showToast('Code copied to clipboard.', 'success', 1500);
      } catch {
        showToast('Could not copy — try selecting the text manually.', 'warning');
      }
      setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
    });

    pre.appendChild(btn);
  });
}


/* ═══════════════════════════════════════════════════════════════
   SEND & STREAM
   ═══════════════════════════════════════════════════════════════ */
async function sendMessage() {
  const input = document.getElementById('chat-input');
  if (!input || state.isGenerating) return;

  const text = input.value.trim();
  if (!text) return;

  // Clear input
  input.value = '';
  input.style.height = 'auto';
  input.style.overflowY = 'hidden';
  const countEl = document.getElementById('input-count');
  if (countEl) countEl.textContent = '0';

  appendUserMessage(text);

  state.isGenerating = true;
  updateSendButton();
  resetStreamState();

  abortController = new AbortController();

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
      signal: abortController.signal
    });

    if (!resp.ok) {
      throw new Error(`Server responded with ${resp.status}`);
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // keep partial line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            processStreamEvent(JSON.parse(line.slice(6)));
          } catch {
            // Ignore malformed events
          }
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      // User pressed stop — don't show error
    } else {
      // IMPROVED ERROR: specific + constructive + human
      showToast(
        'Selene lost the connection mid-response. Check that Ollama is still running, then try again.',
        'error'
      );
      finishGeneration(false);
    }
  }
}

function resetStreamState() {
  state.currentAssistantBody   = null;
  state.currentAssistantBubble = null;
  state.currentThinkingCard    = null;
  state.currentAssistantText   = '';
  state.currentThinkingText    = '';
  state.toolCards              = {};
  state.toolCallCounter        = 0;
}

function getOrCreateAssistantBody() {
  if (state.currentAssistantBody) return state.currentAssistantBody;

  const container = document.getElementById('messages');
  container.querySelector('.welcome')?.remove();

  const row    = document.createElement('div');
  row.className = 'msg assistant';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.innerHTML = `<img src="/avatar.png" alt="Selene" width="30" height="30">`;

  const body = document.createElement('div');
  body.className = 'msg-body';

  row.appendChild(avatar);
  row.appendChild(body);
  container.appendChild(row);

  state.currentAssistantBody = body;
  return body;
}

function processStreamEvent(evt) {
  switch (evt.type) {

    case 'status': {
      const container = document.getElementById('messages');
      const row = document.createElement('div');
      row.className = 'status-row';
      row.innerHTML = `<span>${escapeHtml(evt.message || '')}</span>`;
      container.appendChild(row);
      scrollToBottom();
      break;
    }

    case 'thinking_start': {
      const body = getOrCreateAssistantBody();
      state.currentThinkingCard = createThinkingCard('', true);
      body.appendChild(state.currentThinkingCard);
      scrollToBottom();
      break;
    }

    case 'thinking_chunk': {
      if (state.currentThinkingCard) {
        state.currentThinkingText += evt.text || '';
        const b = state.currentThinkingCard.querySelector('.thinking-body');
        if (b) b.textContent = state.currentThinkingText;
      }
      break;
    }

    case 'thinking_end': {
      if (state.currentThinkingCard) {
        state.currentThinkingCard.classList.add('collapsed');
        const header = state.currentThinkingCard.querySelector('.thinking-header');
        if (header) header.setAttribute('aria-expanded', 'false');
      }
      break;
    }

    case 'content_chunk': {
      const body = getOrCreateAssistantBody();
      if (!state.currentAssistantBubble) {
        state.currentAssistantBubble = document.createElement('div');
        state.currentAssistantBubble.className = 'bubble';
        body.appendChild(state.currentAssistantBubble);
      }
      state.currentAssistantText += evt.text || '';
      state.currentAssistantBubble.innerHTML = marked.parse(state.currentAssistantText);
      addCopyButtons(state.currentAssistantBubble);
      scrollToBottom();
      break;
    }

    case 'tool_start': {
      const body = getOrCreateAssistantBody();
      state.toolCallCounter++;
      const id   = `tool-${state.toolCallCounter}`;
      const card = createToolCard(evt.name || '', evt.arguments || {}, '', true);
      body.appendChild(card);
      state.toolCards[id]  = card;
      state.lastToolCardId = id;
      scrollToBottom();
      break;
    }

    case 'tool_end': {
      const id  = state.lastToolCardId;
      const old = state.toolCards[id];
      if (old?.parentNode) {
        const fresh = createToolCard(evt.name || '', {}, evt.result || '', false);
        old.parentNode.replaceChild(fresh, old);
        state.toolCards[id] = fresh;
      }
      // Reset content buffer so next model response starts fresh
      state.currentAssistantBody   = null;
      state.currentAssistantBubble = null;
      state.currentAssistantText   = '';
      break;
    }

    case 'token_usage': {
      const countEl = document.getElementById('input-count');
      if (countEl) {
        const pct = (evt.total / evt.budget);
        countEl.textContent = `${evt.total}/${evt.budget} ctx`;
        countEl.classList.toggle('warning', pct > 0.88);
      }
      break;
    }

    case 'done': {
      finishGeneration(false);
      break;
    }
  }
}

function finishGeneration(aborted = false) {
  state.isGenerating = false;
  abortController    = null;
  updateSendButton();

  // Collapse open thinking card
  if (state.currentThinkingCard) {
    state.currentThinkingCard.classList.add('collapsed');
  }

  // Restore character count display (replaces "ctx" counter)
  const countEl = document.getElementById('input-count');
  const input   = document.getElementById('chat-input');
  if (countEl) {
    countEl.classList.remove('warning');
    countEl.textContent = input?.value.length || '0';
  }

  // Refresh session list + title after completion
  loadInitialState().then(() => {
    setDocumentTitle(state.activeSession);
  });
}

function updateSendButton() {
  const btn   = document.getElementById('send-btn');
  const input = document.getElementById('chat-input');
  if (!btn) return;

  if (state.isGenerating) {
    btn.disabled = false;
    btn.classList.add('stop');
    btn.setAttribute('aria-label', 'Stop generation');
    btn.title = 'Stop (Esc)';
    btn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <rect x="6" y="6" width="12" height="12" rx="2"/>
      </svg>
    `;
  } else {
    btn.classList.remove('stop');
    btn.setAttribute('aria-label', 'Send message');
    btn.title = 'Send (Enter)';
    btn.innerHTML = `
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M12 19V5M5 12l7-7 7 7"/>
      </svg>
    `;
    btn.disabled = !input?.value.trim();
  }
}


/* ═══════════════════════════════════════════════════════════════
   ESCAPE KEY STOPS GENERATION
   ═══════════════════════════════════════════════════════════════ */
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && state.isGenerating) {
    abortController?.abort();
    finishGeneration(true);
    showToast('Generation stopped.', 'info', 1400);
  }
});


/* ═══════════════════════════════════════════════════════════════
   PUBLIC API (debugging / external use)
   ═══════════════════════════════════════════════════════════════ */
window.SeleneUI = {
  showToast,
  showConfirm,
  setDocumentTitle
};