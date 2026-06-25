/*
 * Selene — Complete ground-up frontend (app.js)
 * - Toast system (success/error/warning/info)
 * - Auto-resizing chat input (max height then scroll) + size count
 * - Dynamic document titles
 * - All ad-hoc alerts/confirms replaced
 * - Improved human error messages
 * - Consolidated type scale usage
 */

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
  currentAssistantBody: null,
  currentAssistantBubble: null,
  currentThinkingBox: null,
  currentAssistantText: '',
  currentAssistantThinkingText: '',
  toolCards: {},
  toolCallCounter: 0
};

let currentAbortController = null;
let toasts = [];

/* ═══════════════════════════════════════════════════════════════════════
   TOAST / NOTIFICATION SYSTEM
   Requirements:
   - Types: success, error, warning, info
   - Auto dismiss (errors last longer)
   - Manual dismiss (X)
   - Stack cleanly (top-right)
   - ARIA live region via #toast-container (role=status + aria-live=polite)
   - Pause on hover
   ═══════════════════════════════════════════════════════════════════════ */
function showToast(message, type = 'info', duration = null) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.setAttribute('role', 'status');

  const icons = {
    success: '<i data-lucide="check-circle" style="width:17px;height:17px;color:#4ade80"></i>',
    error:   '<i data-lucide="alert-triangle" style="width:17px;height:17px;color:#f87171"></i>',
    warning: '<i data-lucide="alert-circle" style="width:17px;height:17px;color:#facc15"></i>',
    info:    '<i data-lucide="info" style="width:17px;height:17px;color:#60a5fa"></i>'
  };

  const defaultDurations = { success: 3200, info: 3800, warning: 5200, error: 8200 };
  const life = duration ?? defaultDurations[type] ?? 4200;

  toast.innerHTML = `
    <span class="icon">${icons[type] || icons.info}</span>
    <span class="msg">${escapeHtml(message)}</span>
    <button class="close" aria-label="Dismiss"><i data-lucide="x" style="width:15px;height:15px;"></i></button>
  `;

  let timeoutId = null;

  const dismiss = () => {
    toast.style.transition = 'opacity 120ms';
    toast.style.opacity = '0';
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
      toasts = toasts.filter(t => t !== toast);
    }, 120);
  };

  // Hover pause
  let paused = false;
  toast.addEventListener('mouseenter', () => { paused = true; if (timeoutId) clearTimeout(timeoutId); });
  toast.addEventListener('mouseleave', () => {
    paused = false;
    if (life > 0) timeoutId = setTimeout(dismiss, 900);
  });

  toast.querySelector('.close').addEventListener('click', dismiss);

  container.appendChild(toast);
  toasts.push(toast);

  if (window.lucide) lucide.createIcons();

  if (life > 0) {
    timeoutId = setTimeout(() => {
      if (!paused) dismiss();
    }, life);
  }

  // Allow manual control
  toast._dismiss = dismiss;
  return toast;
}

/* Replace ad-hoc alert / confirm with nice toasts + confirm toast */
function showConfirm(message, onConfirm, onCancel) {
  // Simple accessible confirm using two toasts stacked
  const t = showToast(message + '  —  Click X to cancel', 'warning', 0);

  const confirmBtn = document.createElement('button');
  confirmBtn.textContent = 'Confirm';
  confirmBtn.style.cssText = 'margin-left:8px;background:#334155;color:#fff;border:0;padding:2px 9px;border-radius:6px;font-size:12px;cursor:pointer;';
  confirmBtn.onclick = () => {
    t._dismiss?.();
    onConfirm?.();
  };

  const closeBtn = t.querySelector('.close');
  const orig = closeBtn.onclick;
  closeBtn.onclick = (e) => {
    t._dismiss?.();
    onCancel?.();
  };

  t.querySelector('.msg').appendChild(confirmBtn);
}

/* ═══════════════════════════════════════════════════════════════════════
   UTILITIES
   ═══════════════════════════════════════════════════════════════════════ */
function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function updateDocumentTitle(title) {
  const clean = (title || 'New conversation').trim();
  // Consistent, descriptive pattern: specific page/convo first, then app name
  document.title = clean === 'New conversation' || !clean
    ? 'Selene'
    : `${clean} — Selene`;
}

function scrollMessagesToBottom(force = false) {
  const el = document.getElementById('messages');
  if (!el) return;
  if (force) {
    el.scrollTop = el.scrollHeight;
    return;
  }
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  if (nearBottom) el.scrollTop = el.scrollHeight;
}

/* ═══════════════════════════════════════════════════════════════════════
   INITIALIZATION
   ═══════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initLibraries();
  initEventListeners();
  loadSessionState();
});

function initLibraries() {
  if (window.marked) {
    marked.setOptions({
      highlight: (code, lang) => {
        const language = (window.hljs && hljs.getLanguage(lang)) ? lang : 'plaintext';
        return window.hljs ? hljs.highlight(code, { language }).value : escapeHtml(code);
      },
      langPrefix: 'hljs language-'
    });
  }
  if (window.lucide) lucide.createIcons();
}

function initEventListeners() {
  // Sidebar toggle
  const sidebar = document.getElementById('sidebar');
  document.getElementById('sidebar-toggle')?.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });

  // New chat
  document.getElementById('new-chat-btn')?.addEventListener('click', () => {
    clearConversation();
  });

  // Composer
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const countEl = document.getElementById('input-count');

  if (input) {
    // Auto-resize + max height (requirement)
    const MAX_HEIGHT = 220;
    const onInputResize = () => {
      input.style.height = 'auto';
      const next = Math.min(input.scrollHeight, MAX_HEIGHT);
      input.style.height = next + 'px';
      input.style.overflowY = input.scrollHeight > MAX_HEIGHT ? 'auto' : 'hidden';

      if (countEl) countEl.textContent = String(input.value.length || 0);
      updateSendButton();
    };

    input.addEventListener('input', onInputResize);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!state.isGenerating) sendMessage();
      }
    });

    // Prime count
    setTimeout(onInputResize, 0);
  }

  if (sendBtn) {
    sendBtn.addEventListener('click', () => {
      if (state.isGenerating) {
        if (currentAbortController) currentAbortController.abort();
        finishGeneration(true);
      } else {
        sendMessage();
      }
    });
  }

  // Settings panel toggle
  const settingsToggle = document.getElementById('settings-toggle');
  const settingsPanel = document.getElementById('settings-panel');
  settingsToggle?.addEventListener('click', () => {
    settingsPanel.classList.toggle('open');
    const chev = settingsToggle.querySelector('[data-lucide="chevron-down"]');
    if (chev) chev.style.transform = settingsPanel.classList.contains('open') ? 'rotate(180deg)' : '';
  });

  // Model parameter sliders + switches
  bindSettingsControls();

  // Clear
  document.getElementById('clear-btn')?.addEventListener('click', () => {
    showConfirm('Clear this conversation?', () => clearConversation(), null);
  });

  // Help
  const helpModal = document.getElementById('help-modal');
  document.getElementById('help-btn')?.addEventListener('click', () => {
    if (helpModal) helpModal.classList.add('open');
  });
  helpModal?.querySelectorAll('.close-help, .modal-card').forEach(el => {
    el.addEventListener('click', (e) => {
      if (e.target === helpModal || e.currentTarget.classList.contains('close-help')) {
        helpModal.classList.remove('open');
      }
    });
  });

  // Click outside sidebar on mobile closes it
  document.addEventListener('click', (e) => {
    if (window.innerWidth < 821 && sidebar.classList.contains('open')) {
      if (!sidebar.contains(e.target) && e.target.id !== 'sidebar-toggle') {
        sidebar.classList.remove('open');
      }
    }
  });
}

function bindSettingsControls() {
  const sliders = [
    { id: 'param-temperature', val: 'temp-val', key: 'temperature' },
    { id: 'param-top_p', val: 'topp-val', key: 'top_p' },
    { id: 'param-top_k', val: 'topk-val', key: 'top_k' }
  ];

  sliders.forEach(s => {
    const input = document.getElementById(s.id);
    const display = document.getElementById(s.val);
    if (!input || !display) return;

    const update = (val) => {
      display.textContent = parseFloat(val).toFixed(s.key === 'top_k' ? 0 : 2);
      if (!state.settings.options) state.settings.options = {};
      state.settings.options[s.key] = parseFloat(val);
      saveSettings();
    };

    input.addEventListener('input', (e) => update(e.target.value));
  });

  const hist = document.getElementById('setting-history');
  const think = document.getElementById('setting-think');

  if (hist) hist.addEventListener('change', (e) => { state.settings.history = e.target.checked; saveSettings(); });
  if (think) think.addEventListener('change', (e) => { state.settings.think = e.target.checked; saveSettings(); });
}

function syncSettingsUI() {
  const s = state.settings || {};
  const opts = s.options || {};

  const map = [
    ['param-temperature', 'temp-val', opts.temperature ?? 0.7, 2],
    ['param-top_p', 'topp-val', opts.top_p ?? 0.9, 2],
    ['param-top_k', 'topk-val', opts.top_k ?? 40, 0]
  ];

  map.forEach(([id, vid, val, dec]) => {
    const el = document.getElementById(id);
    const v = document.getElementById(vid);
    if (el) el.value = val;
    if (v) v.textContent = parseFloat(val).toFixed(dec);
  });

  const hist = document.getElementById('setting-history');
  if (hist) hist.checked = s.history !== false;

  const think = document.getElementById('setting-think');
  if (think) think.checked = s.think !== false;
}

async function saveSettings() {
  try {
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.settings)
    });
  } catch (e) { /* silent — toast on load failure only */ }
}

/* ═══════════════════════════════════════════════════════════════════════
   SESSION / DATA LOADING
   ═══════════════════════════════════════════════════════════════════════ */
async function loadSessionState() {
  try {
    const res = await fetch('/api/settings');
    const data = await res.json();

    state.history = data.history || [];
    state.savedSessions = data.saved_sessions || [];
    state.settings = data.settings || state.settings;
    state.activeSession = data.active_session_name || 'New conversation';

    const badge = document.getElementById('model-badge');
    if (badge) badge.textContent = data.model_name || 'selene';

    updateConnectionStatus(data.ollama_status);

    syncSettingsUI();
    renderConversationList();
    renderMessages();
    updateDocumentTitle(state.activeSession);
  } catch (err) {
    showToast('Could not load your session. Is the backend running?', 'error');
  }
}

function updateConnectionStatus(status) {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('status-text');
  const wrap = document.getElementById('status-indicator');

  const isOnline = status === 'Online';
  if (dot) dot.style.background = isOnline ? '#4ade80' : '#f87171';
  if (txt) txt.textContent = isOnline ? 'Online' : 'Offline';
  if (wrap) wrap.title = `Ollama: ${status || 'Unknown'}`;
}

async function clearConversation() {
  try {
    await fetch('/api/clear-session', { method: 'POST' });
    state.history = [];
    state.activeSession = 'New conversation';
    document.getElementById('messages').innerHTML = '';
    renderConversationList();
    renderWelcome();
    updateDocumentTitle(state.activeSession);
    showToast('Conversation cleared', 'info', 1600);
  } catch (e) {
    showToast('Failed to clear conversation. Please try again.', 'error');
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   RENDERING
   ═══════════════════════════════════════════════════════════════════════ */
function renderConversationList() {
  const list = document.getElementById('convo-list');
  if (!list) return;

  list.innerHTML = '';

  if (!state.savedSessions || state.savedSessions.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'convo-empty';
    empty.textContent = 'No saved conversations yet';
    list.appendChild(empty);
    return;
  }

  state.savedSessions.forEach((filename) => {
    const base = filename.replace('.json', '');
    let display = base;
    let meta = '';

    const m = base.match(/^(.+)_(\d{8})_(\d{6})$/);
    if (m) {
      display = m[1].replace(/_/g, ' ');
      const [y, mo, d, hr, mn] = [m[2].slice(0,4), m[2].slice(4,6), m[2].slice(6,8), m[2].slice(0,2), m[2].slice(2,4)];
      meta = `${d}/${mo} ${hr}:${mn}`;
    }

    const item = document.createElement('div');
    item.className = `convo-item ${base + '.json' === state.activeSession ? 'active' : ''}`;
    item.innerHTML = `
      <span class="convo-name" title="${escapeHtml(display)}">${escapeHtml(display)}</span>
      <span class="text-caption mono" style="flex-shrink:0;">${meta}</span>
    `;
    item.addEventListener('click', () => loadSavedSession(filename, item));
    list.appendChild(item);
  });
}

async function loadSavedSession(filename, el) {
  try {
    const res = await fetch('/api/load-session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: filename })
    });
    const json = await res.json();
    if (json.status === 'success') {
      await loadSessionState();
      document.getElementById('sidebar')?.classList.remove('open');
    } else {
      showToast('Could not load that conversation.', 'error');
    }
  } catch (e) {
    showToast('Network error loading conversation.', 'error');
  }
}

function renderMessages() {
  const container = document.getElementById('messages');
  if (!container) return;
  container.innerHTML = '';

  const display = [];
  for (let i = 0; i < state.history.length; i++) {
    const msg = state.history[i];
    if (msg.role === 'user') {
      display.push({ role: 'user', content: msg.content });
    } else if (msg.role === 'assistant') {
      const tools = [];
      let j = i + 1;
      while (j < state.history.length && state.history[j].role === 'tool') {
        tools.push(state.history[j]);
        j++;
      }
      i = j - 1;
      display.push({
        role: 'assistant',
        content: msg.content,
        thinking: msg.thinking,
        tool_calls: msg.tool_calls,
        tool_results: tools
      });
    }
  }

  if (display.length === 0) {
    renderWelcome(container);
    return;
  }

  display.forEach(m => {
    if (m.role === 'user') appendUserMessage(m.content, false);
    else appendAssistantMessage(m, false);
  });
  scrollMessagesToBottom(true);
}

function renderWelcome(container = null) {
  const el = container || document.getElementById('messages');
  if (!el) return;

  const div = document.createElement('div');
  div.className = 'welcome';
  div.innerHTML = `
    <svg class="moon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
    <h1 class="text-h1">Selene</h1>
    <p>A calm, capable AI that lives on your machine.</p>

    <div class="suggestions">
      <div class="suggestion" data-cmd="Search the web for the latest AI news">
        <span class="label">Web</span>
        <div>Search the web for the latest AI news</div>
      </div>
      <div class="suggestion" data-cmd="Index my project folder and give me an architecture summary">
        <span class="label">Vault</span>
        <div>Index my project and summarize the architecture</div>
      </div>
      <div class="suggestion" data-cmd="Write a Python function to recursively find duplicate files">
        <span class="label">Code</span>
        <div>Write a script to find duplicate files</div>
      </div>
      <div class="suggestion" data-cmd="Play some ambient focus music">
        <span class="label">Music</span>
        <div>Play ambient focus music on Spotify</div>
      </div>
    </div>
  `;

  el.appendChild(div);

  div.querySelectorAll('.suggestion').forEach(s => {
    s.addEventListener('click', () => {
      const input = document.getElementById('chat-input');
      if (input) {
        input.value = s.getAttribute('data-cmd');
        input.dispatchEvent(new Event('input'));
        input.focus();
      }
    });
  });
}

function appendUserMessage(text, animate = true) {
  const container = document.getElementById('messages');
  const welcome = container.querySelector('.welcome');
  if (welcome) welcome.remove();

  const row = document.createElement('div');
  row.className = 'msg user';
  row.innerHTML = `
    <div class="bubble">${escapeHtml(text)}</div>
    <div class="avatar">U</div>
  `;
  container.appendChild(row);
  if (animate) scrollMessagesToBottom(true);
}

function appendAssistantMessage(msgData, animate = true) {
  const container = document.getElementById('messages');
  const row = document.createElement('div');
  row.className = 'msg assistant';

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.innerHTML = '<img src="/avatar.png" alt="S" style="width:28px;height:28px;object-fit:cover;border-radius:8px;">';

  const body = document.createElement('div');

  if (msgData.thinking) body.appendChild(createThinkingEl(msgData.thinking, false));
  if (msgData.tool_calls) {
    msgData.tool_calls.forEach((tc, idx) => {
      const res = (msgData.tool_results && msgData.tool_results[idx]) ? msgData.tool_results[idx].content : '';
      body.appendChild(createToolEl(tc.function.name, tc.function.arguments, res, false));
    });
  }

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = marked.parse(msgData.content || '');
  body.appendChild(bubble);

  row.appendChild(avatar);
  row.appendChild(body);
  container.appendChild(row);

  addCopyButtons(bubble);
  if (animate) scrollMessagesToBottom(true);
}

function createThinkingEl(text, streaming) {
  const box = document.createElement('div');
  box.className = `thinking ${streaming ? '' : 'collapsed'}`;

  const header = document.createElement('div');
  header.className = 'thinking-header';
  header.innerHTML = `<span>Thinking</span><i data-lucide="chevron-down" style="width:14px;height:14px;"></i>`;
  const content = document.createElement('div');
  content.className = 'thinking-content';
  content.textContent = text;

  box.appendChild(header);
  box.appendChild(content);

  header.addEventListener('click', () => box.classList.toggle('collapsed'));
  if (window.lucide) lucide.createIcons();
  return box;
}

function createToolEl(name, args, result = '', isRunning = false) {
  const card = document.createElement('div');
  card.className = `tool ${isRunning ? '' : 'collapsed'}`;

  const header = document.createElement('div');
  header.className = 'tool-header';
  header.innerHTML = `
    <span>Tool: <span style="color:#a5b4fc">${escapeHtml(name)}</span></span>
    <i data-lucide="chevron-down" style="width:15px;height:15px;"></i>
  `;
  const content = document.createElement('div');
  content.className = 'tool-content';
  content.textContent = `Args:\n${JSON.stringify(args, null, 2)}\n\n${result ? 'Result:\n' + result : (isRunning ? 'Running...' : '')}`;

  card.appendChild(header);
  card.appendChild(content);
  header.addEventListener('click', () => card.classList.toggle('collapsed'));
  if (window.lucide) lucide.createIcons();
  return card;
}

function addCopyButtons(scope) {
  scope.querySelectorAll('pre').forEach(pre => {
    if (pre.querySelector('.copy-btn')) return;
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.style.cssText = 'position:absolute;top:8px;right:8px;font-size:11px;padding:3px 8px;border-radius:6px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);color:#9aa3b8;cursor:pointer;';
    btn.innerHTML = 'Copy';
    pre.style.position = 'relative';
    pre.appendChild(btn);

    btn.addEventListener('click', async () => {
      const code = pre.querySelector('code')?.textContent || '';
      await navigator.clipboard.writeText(code);
      const old = btn.textContent;
      btn.textContent = 'Copied';
      setTimeout(() => { btn.textContent = old; }, 1500);
    });
  });
}

/* ═══════════════════════════════════════════════════════════════════════
   SENDING + STREAMING
   ═══════════════════════════════════════════════════════════════════════ */
async function sendMessage() {
  const input = document.getElementById('chat-input');
  if (!input || state.isGenerating) return;

  const text = input.value.trim();
  if (!text) return;

  appendUserMessage(text);
  input.value = '';
  input.style.height = 'auto';
  document.getElementById('input-count').textContent = '0';

  state.isGenerating = true;
  updateSendButton();
  resetStreamState();

  currentAbortController = new AbortController();

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
      signal: currentAbortController.signal
    });

    if (!resp.ok) throw new Error('Request failed');

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            handleStreamEvent(JSON.parse(line.slice(6)));
          } catch (_) {}
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      showToast('Generation stopped', 'info', 1400);
    } else {
      // IMPROVED ERROR MESSAGE (specific + constructive + human)
      showToast('Connection lost. Please make sure Ollama is running and try sending again.', 'error');
    }
    finishGeneration(false);
  }
}

function resetStreamState() {
  state.currentAssistantBody = null;
  state.currentAssistantBubble = null;
  state.currentThinkingBox = null;
  state.currentAssistantText = '';
  state.currentAssistantThinkingText = '';
  state.toolCards = {};
  state.toolCallCounter = 0;
}

function getAssistantBody() {
  if (state.currentAssistantBody) return state.currentAssistantBody;

  const container = document.getElementById('messages');
  const row = document.createElement('div');
  row.className = 'msg assistant';

  const av = document.createElement('div');
  av.className = 'avatar';
  av.innerHTML = '<img src="/avatar.png" alt="S" style="width:28px;height:28px;border-radius:8px;object-fit:cover">';

  const body = document.createElement('div');

  row.appendChild(av);
  row.appendChild(body);
  container.appendChild(row);

  state.currentAssistantBody = body;
  return body;
}

function handleStreamEvent(evt) {
  if (evt.type === 'status') {
    appendStatusRow(evt.message);
  }
  else if (evt.type === 'thinking_start') {
    const body = getAssistantBody();
    state.currentThinkingBox = createThinkingEl('', true);
    body.appendChild(state.currentThinkingBox);
    scrollMessagesToBottom();
  }
  else if (evt.type === 'thinking_chunk') {
    if (state.currentThinkingBox) {
      state.currentAssistantThinkingText += evt.text || '';
      const c = state.currentThinkingBox.querySelector('.thinking-content');
      if (c) c.textContent = state.currentAssistantThinkingText;
    }
  }
  else if (evt.type === 'thinking_end') {
    if (state.currentThinkingBox) state.currentThinkingBox.classList.add('collapsed');
  }
  else if (evt.type === 'content_chunk') {
    const body = getAssistantBody();
    if (!state.currentAssistantBubble) {
      state.currentAssistantBubble = document.createElement('div');
      state.currentAssistantBubble.className = 'bubble';
      body.appendChild(state.currentAssistantBubble);
    }
    state.currentAssistantText += evt.text || '';
    state.currentAssistantBubble.innerHTML = marked.parse(state.currentAssistantText);
    addCopyButtons(state.currentAssistantBubble);
    scrollMessagesToBottom();
  }
  else if (evt.type === 'tool_start') {
    const body = getAssistantBody();
    state.toolCallCounter += 1;
    const id = 't' + state.toolCallCounter;
    const card = createToolEl(evt.name, evt.arguments || {}, '', true);
    body.appendChild(card);
    state.toolCards[id] = card;
    state.lastToolId = id;
    scrollMessagesToBottom();
  }
  else if (evt.type === 'tool_end') {
    const id = state.lastToolId;
    const old = state.toolCards[id];
    if (old && old.parentNode) {
      const fresh = createToolEl(evt.name, {}, evt.result || '', false);
      old.parentNode.replaceChild(fresh, old);
      state.toolCards[id] = fresh;
    }
    state.currentAssistantBody = null;
    state.currentAssistantBubble = null;
    state.currentAssistantText = '';
  }
  else if (evt.type === 'token_usage') {
    const count = document.getElementById('input-count');
    if (count) {
      count.textContent = `${evt.total} / ${evt.budget} ctx`;
      count.style.color = evt.total > evt.budget * 0.9 ? 'var(--warning)' : '';
    }
  }
  else if (evt.type === 'done') {
    finishGeneration(false);
  }
}

function appendStatusRow(text) {
  const c = document.getElementById('messages');
  const row = document.createElement('div');
  row.className = 'status-row';
  row.innerHTML = `<span>${escapeHtml(text)}</span>`;
  c.appendChild(row);
  scrollMessagesToBottom();
}

function finishGeneration(aborted = false) {
  state.isGenerating = false;
  currentAbortController = null;
  updateSendButton();

  if (state.currentThinkingBox) state.currentThinkingBox.classList.add('collapsed');

  // Restore char count display (ctx badge only lasts during generation)
  const countEl = document.getElementById('input-count');
  const input = document.getElementById('chat-input');
  if (countEl && input) {
    countEl.style.color = '';
    countEl.textContent = String(input.value.length || 0);
  }

  // Refresh sidebar history + title
  loadSessionState().then(() => {
    updateDocumentTitle(state.activeSession);
  });
}

function updateSendButton() {
  const btn = document.getElementById('send-btn');
  const input = document.getElementById('chat-input');
  if (!btn || !input) return;

  if (state.isGenerating) {
    btn.disabled = false;
    btn.classList.add('stop');
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12" rx="2"></rect></svg>';
  } else {
    btn.classList.remove('stop');
    btn.innerHTML = '<i data-lucide="arrow-up" style="width:18px;height:18px;"></i>';
    btn.disabled = !input.value.trim();
  }
  if (window.lucide) lucide.createIcons();
}

/* Initial send button state */
setTimeout(() => {
  const inp = document.getElementById('chat-input');
  if (inp) inp.addEventListener('input', () => {
    const b = document.getElementById('send-btn');
    if (b && !state.isGenerating) b.disabled = !inp.value.trim();
  });
}, 300);

/* Expose a couple helpers for debugging if needed */
window.SeleneUI = { showToast };