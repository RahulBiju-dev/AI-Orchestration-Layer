/*
 * ────────────────────────────────────────────────────────────────────────
 * app.js
 * 
 * Main client-side script for the Selene AI Agent Web UI.
 * Handles DOM events, state management, SSE streaming from the backend,
 * and rendering of Markdown, tool cards, and thinking panels.
 * ────────────────────────────────────────────────────────────────────────
 */

// ── Global Application State ─────────────────────────────────────────
const state = {
    activeSession: 'Active Session',
    history: [],
    settings: {
        options: {
            temperature: 0.4,
            top_p: 0.9,
            top_k: 40
        },
        verbose: false,
        wordwrap: true,
        system: '',
        history: true,
        think: true
    },
    savedSessions: [],
    
    // Generation & Streaming State
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

// ── Initialization ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initLibraries();
    initEventListeners();
    loadSessionState();
});

/**
 * Initialize third-party libraries:
 * - marked.js with highlight.js for markdown/code rendering
 * - lucide.js for icon generation
 */
function initLibraries() {
    marked.setOptions({
        highlight: function (code, lang) {
            const language = hljs.getLanguage(lang) ? lang : 'plaintext';
            return hljs.highlight(code, { language }).value;
        },
        langPrefix: 'hljs language-'
    });
    
    if (window.lucide) {
        lucide.createIcons();
    }
}

// ── Event Listeners ──────────────────────────────────────────────────
/**
 * Attach DOM event listeners for the sidebar, settings, and chat inputs.
 * Sets up keybindings, slider value binding, and click events.
 */
function initEventListeners() {
    // Menu Sidebar toggle
    const menuToggle = document.getElementById('menu-toggle');
    const closeSidebar = document.getElementById('close-sidebar');
    const sidebar = document.getElementById('sidebar');
    
    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', () => sidebar.classList.toggle('collapsed'));
    }
    if (closeSidebar && sidebar) {
        closeSidebar.addEventListener('click', () => sidebar.classList.add('collapsed'));
    }
    
    // Parameter Sliders
    const sliders = [
        { id: 'param-temperature', valId: 'temp-val', key: 'temperature', isOption: true },
        { id: 'param-top_p', valId: 'topp-val', key: 'top_p', isOption: true },
        { id: 'param-top_k', valId: 'topk-val', key: 'top_k', isOption: true }
    ];
    
    sliders.forEach(sliderInfo => {
        const slider = document.getElementById(sliderInfo.id);
        const display = document.getElementById(sliderInfo.valId);
        if (slider && display) {
            slider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                display.textContent = val;
                
                if (sliderInfo.isOption) {
                    state.settings.options[sliderInfo.key] = val;
                } else {
                    state.settings[sliderInfo.key] = val;
                }
                saveSettingsOnBackend();
            });
        }
    });
    
    // Settings Switches
    const historySwitch = document.getElementById('setting-history');
    const thinkSwitch = document.getElementById('setting-think');
    
    if (historySwitch) {
        historySwitch.addEventListener('change', (e) => {
            state.settings.history = e.target.checked;
            saveSettingsOnBackend();
        });
    }
    if (thinkSwitch) {
        thinkSwitch.addEventListener('change', (e) => {
            state.settings.think = e.target.checked;
            saveSettingsOnBackend();
        });
    }
    
    // Custom System Prompt
    const systemPromptArea = document.getElementById('system-prompt');
    const applyPromptBtn = document.getElementById('apply-prompt-btn');
    
    if (applyPromptBtn && systemPromptArea) {
        applyPromptBtn.addEventListener('click', () => {
            state.settings.system = systemPromptArea.value.trim();
            saveSettingsOnBackend();
            
            const originalText = applyPromptBtn.textContent;
            applyPromptBtn.textContent = 'Applied ✓';
            setTimeout(() => applyPromptBtn.textContent = originalText, 1500);
        });
    }
    
    // Input Area
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    
    if (userInput) {
        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = (userInput.scrollHeight) + 'px';
            updateSendButtonState();
            showAutocomplete(userInput.value);
        });
        
        userInput.addEventListener('keydown', (e) => {
            const popup = document.getElementById('autocomplete-popup');
            const isPopupVisible = popup && popup.style.display === 'block';
            
            if (isPopupVisible) {
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    activeAutocompleteIndex = (activeAutocompleteIndex + 1) % filteredCommands.length;
                    renderAutocompleteItems();
                    return;
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    activeAutocompleteIndex = (activeAutocompleteIndex - 1 + filteredCommands.length) % filteredCommands.length;
                    renderAutocompleteItems();
                    return;
                } else if (e.key === 'Tab' || e.key === 'Enter') {
                    e.preventDefault();
                    if (activeAutocompleteIndex >= 0 && activeAutocompleteIndex < filteredCommands.length) {
                        selectAutocompleteItem(filteredCommands[activeAutocompleteIndex]);
                    }
                    return;
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    hideAutocomplete();
                    return;
                }
            }
            
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!state.isGenerating) {
                    sendMessage();
                }
            }
        });
    }
    
    if (sendBtn) {
        sendBtn.addEventListener('click', () => {
            if (state.isGenerating) {
                if (currentAbortController) {
                    currentAbortController.abort();
                }
                finishGeneration(true);
            } else {
                sendMessage();
            }
        });
    }
    
    // Buttons
    document.getElementById('new-chat-btn')?.addEventListener('click', clearSession);
    document.getElementById('clear-history-btn')?.addEventListener('click', () => {
        if (confirm('Clear this conversation history?')) clearSession();
    });
    
    // Modals
    setupModal('save-session-btn', 'save-modal');
    setupModal('help-btn', 'help-modal');
    
    // Close autocomplete on click outside
    document.addEventListener('click', (e) => {
        const popup = document.getElementById('autocomplete-popup');
        const userInputEl = document.getElementById('user-input');
        if (popup && e.target !== popup && e.target !== userInputEl && !popup.contains(e.target)) {
            hideAutocomplete();
        }
    });
    
    const confirmSaveBtn = document.getElementById('confirm-save-btn');
    const sessionNameInput = document.getElementById('save-session-name');
    if (confirmSaveBtn && sessionNameInput) {
        confirmSaveBtn.addEventListener('click', async () => {
            const name = sessionNameInput.value.trim();
            if (!name) return;
            
            try {
                const res = await fetch('/api/save-session', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    document.querySelectorAll('.modal-overlay').forEach(m => m.classList.remove('active'));
                    sessionNameInput.value = '';
                    loadSessionState();
                } else {
                    alert('Error saving session: ' + data.error);
                }
            } catch (err) {
                console.error(err);
            }
        });
    }
    
    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const target = document.getElementById(btn.getAttribute('data-target'));
            if (target) target.classList.add('active');
        });
    });
}

function setupModal(triggerId, modalId) {
    const trigger = document.getElementById(triggerId);
    const modal = document.getElementById(modalId);
    if (!trigger || !modal) return;
    
    trigger.addEventListener('click', () => {
        modal.classList.add('active');
        const firstInput = modal.querySelector('input');
        if (firstInput) setTimeout(() => firstInput.focus(), 100);
    });
    
    modal.querySelectorAll('.close-modal-btn').forEach(btn => {
        btn.addEventListener('click', () => modal.classList.remove('active'));
    });
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
    });
}

// ── Backend Communications ───────────────────────────────────────────
async function loadSessionState() {
    try {
        const res = await fetch('/api/settings');
        const data = await res.json();
        
        state.settings = data.settings;
        state.history = data.history;
        state.savedSessions = data.saved_sessions;
        state.activeSession = data.active_session_name || 'Active Session';
        
        const sessionTitle = document.getElementById('current-session-title');
        if (sessionTitle) {
            sessionTitle.textContent = state.activeSession;
        }
        const modelBadge = document.getElementById('model-badge');
        if (modelBadge) {
            modelBadge.textContent = data.model_name || 'selene';
        }
        
        const dot = document.querySelector('.status-dot');
        const statusIndicator = document.querySelector('.status-indicator');
        if (dot) {
            if (data.ollama_status === 'Online') {
                dot.className = 'status-dot online';
            } else {
                dot.className = 'status-dot offline';
            }
        }
        if (statusIndicator) {
            statusIndicator.title = `Ollama Status: ${data.ollama_status || 'Offline'}`;
        }
        const statusText = document.getElementById('connection-status');
        if (statusText) {
            statusText.textContent = data.ollama_status === 'Online' ? 'Online' : 'Ollama Offline';
        }
        
        syncSettingsToUI();
        renderSavedSessionsList();
        renderHistory();
        
    } catch (err) {
        console.error('Failed to load session settings:', err);
    }
}

async function saveSettingsOnBackend() {
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state.settings)
        });
    } catch (err) {
        console.error('Failed to save settings:', err);
    }
}

function syncSettingsToUI() {
    const ids = {
        'param-temperature': ['temp-val', state.settings.options.temperature ?? 0.4],
        'param-top_p': ['topp-val', state.settings.options.top_p ?? 0.9],
        'param-top_k': ['topk-val', state.settings.options.top_k ?? 40]
    };
    
    for (const [id, [valId, val]] of Object.entries(ids)) {
        const slider = document.getElementById(id);
        const display = document.getElementById(valId);
        if (slider && display) {
            slider.value = val;
            display.textContent = val;
        }
    }
    
    const hist = document.getElementById('setting-history');
    if (hist) hist.checked = state.settings.history;
    
    const think = document.getElementById('setting-think');
    if (think) think.checked = state.settings.think;
    
    const prompt = document.getElementById('system-prompt');
    if (prompt) prompt.value = state.settings.system || '';
}

async function clearSession() {
    try {
        await fetch('/api/clear-session', { method: 'POST' });
        loadSessionState();
    } catch (err) {
        console.error(err);
    }
}

function renderSavedSessionsList() {
    const container = document.getElementById('sessions-list');
    if (!container) return;
    
    if (state.savedSessions.length === 0) {
        container.innerHTML = '<div class="loading-placeholder">No saved sessions</div>';
        return;
    }
    
    container.innerHTML = '';
    state.savedSessions.forEach((sessionFile) => {
        const basename = sessionFile.replace('.json', '');
        let displayName = basename;
        let timeLabel = '';
        
        const match = basename.match(/^(.+)_(\d{8})_(\d{6})$/);
        if (match) {
            displayName = match[1].replace(/_/g, ' ');
            const [yr, mo, dy, hr, mn] = [
                match[2].substring(0,4), match[2].substring(4,6), match[2].substring(6,8),
                match[3].substring(0,2), match[3].substring(2,4)
            ];
            timeLabel = `${dy}/${mo}/${yr} ${hr}:${mn}`;
        }
        
        const item = document.createElement('div');
        item.className = 'session-item';
        if (basename + '.json' === state.activeSession) item.classList.add('active');
        
        item.innerHTML = `
            <div class="session-details">
                <span class="session-name" title="${displayName}">${displayName}</span>
                <span class="session-meta">${timeLabel || 'Session file'}</span>
            </div>
            <i data-lucide="chevron-right" style="width: 14px; height: 14px; color: var(--text-muted);"></i>
        `;
        
        item.addEventListener('click', async () => {
            try {
                const res = await fetch('/api/load-session', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: sessionFile })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    loadSessionState();
                    document.getElementById('sidebar')?.classList.add('collapsed');
                }
            } catch (err) {
                console.error(err);
            }
        });
        container.appendChild(item);
    });
    
    if (window.lucide) lucide.createIcons();
}

// ── Rendering Chat Logs ──────────────────────────────────────────────
function renderHistory() {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    
    container.innerHTML = '';
    const displayMessages = [];
    
    for (let i = 0; i < state.history.length; i++) {
        const msg = state.history[i];
        if (msg.role === 'user') {
            displayMessages.push({ role: 'user', content: msg.content });
        } else if (msg.role === 'assistant') {
            const tools = [];
            let j = i + 1;
            while (j < state.history.length && state.history[j].role === 'tool') {
                tools.push(state.history[j]);
                j++;
            }
            i = j - 1;
            
            displayMessages.push({
                role: 'assistant',
                content: msg.content,
                thinking: msg.thinking,
                tool_calls: msg.tool_calls,
                tool_results: tools
            });
        }
    }
    
    if (displayMessages.length === 0) {
        renderWelcomeScreen(container);
        return;
    }
    
    displayMessages.forEach(msg => {
        if (msg.role === 'user') appendUserMessage(msg.content, false);
        else appendAssistantMessage(msg, false);
    });
    
    scrollToBottom(true);
}

function renderWelcomeScreen(container) {
    const welcome = document.createElement('div');
    welcome.className = 'welcome-container';
    welcome.innerHTML = `
        <div class="welcome-logo">
            <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
        </div>
        <div>
            <h1 class="welcome-title">Selene</h1>
            <p class="welcome-subtitle">A professional AI agent interface with local tools, web search, and document reasoning capabilities.</p>
        </div>
        
        <div class="suggestions-grid">
            <div class="suggestion-card" data-cmd="Search the web for today's tech news">
                <div class="suggestion-header">
                    <i data-lucide="globe" style="width: 18px; height: 18px;"></i> Web Search
                </div>
                <div class="suggestion-text">"Search the web for today's tech news"</div>
            </div>
            <div class="suggestion-card" data-cmd="Index the project folder and summarize its architecture">
                <div class="suggestion-header">
                    <i data-lucide="database" style="width: 18px; height: 18px;"></i> Document Vault
                </div>
                <div class="suggestion-text">"Index the project folder and summarize it"</div>
            </div>
            <div class="suggestion-card" data-cmd="Play some focus music on Spotify">
                <div class="suggestion-header">
                    <i data-lucide="music" style="width: 18px; height: 18px;"></i> Spotify
                </div>
                <div class="suggestion-text">"Play some focus music on Spotify"</div>
            </div>
            <div class="suggestion-card" data-cmd="Show me a python script to list all files in a directory recursively">
                <div class="suggestion-header">
                    <i data-lucide="code" style="width: 18px; height: 18px;"></i> Coding
                </div>
                <div class="suggestion-text">"Write a python script to list files"</div>
            </div>
        </div>
    `;
    
    container.appendChild(welcome);
    
    welcome.querySelectorAll('.suggestion-card').forEach(card => {
        card.addEventListener('click', () => {
            const input = document.getElementById('user-input');
            if (input) {
                input.value = card.getAttribute('data-cmd');
                input.style.height = 'auto';
                input.style.height = input.scrollHeight + 'px';
                updateSendButtonState();
                input.focus();
            }
        });
    });
    
    if (window.lucide) lucide.createIcons();
}

function appendUserMessage(content, animate = true) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    
    const welcome = container.querySelector('.welcome-container');
    if (welcome) welcome.remove();
    
    const msg = document.createElement('div');
    msg.className = 'message user';
    if (!animate) msg.style.animation = 'none';
    
    msg.innerHTML = `
        <div class="message-avatar">U</div>
        <div class="message-body">
            <div class="message-bubble">${escapeHtml(content)}</div>
        </div>
    `;
    container.appendChild(msg);
    if (animate) scrollToBottom(true);
}

function appendAssistantMessage(msgData, animate = true) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    if (!animate) msg.style.animation = 'none';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<img src="/avatar.png" alt="S">';
    msg.appendChild(avatar);
    
    const body = document.createElement('div');
    body.className = 'message-body';
    
    if (msgData.thinking) {
        body.appendChild(createThinkingBoxElement(msgData.thinking, false));
    }
    
    if (msgData.tool_calls) {
        msgData.tool_calls.forEach((tc, idx) => {
            const args = tc.function.arguments;
            let result = '';
            if (msgData.tool_results && msgData.tool_results[idx]) {
                result = msgData.tool_results[idx].content;
            }
            body.appendChild(createToolCardElement(tc.function.name, args, result, false));
        });
    }
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = marked.parse(msgData.content || '');
    body.appendChild(bubble);
    
    msg.appendChild(body);
    container.appendChild(msg);
    addCopyButtons(bubble);
    
    if (animate) scrollToBottom(true);
}

function createThinkingBoxElement(text, isStreaming = false) {
    const box = document.createElement('div');
    box.className = 'thinking-box';
    if (!isStreaming) box.classList.add('collapsed');
    
    const header = document.createElement('div');
    header.className = 'thinking-header';
    header.innerHTML = `
        <div class="thinking-title">
            <i data-lucide="brain" style="width: 14px; height: 14px;"></i> Thinking Process
        </div>
        <div class="thinking-meta">
            ${isStreaming ? `
                <span class="thinking-pulse">
                    <span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span>
                </span>
            ` : `<i data-lucide="chevron-down" class="thinking-arrow" style="width: 14px; height: 14px;"></i>`}
        </div>
    `;
    box.appendChild(header);
    
    const content = document.createElement('div');
    content.className = 'thinking-content';
    content.textContent = text;
    box.appendChild(content);
    
    header.addEventListener('click', () => box.classList.toggle('collapsed'));
    
    if (window.lucide) lucide.createIcons();
    return box;
}

function createToolCardElement(name, args, result = '', isExecuting = false) {
    const card = document.createElement('div');
    card.className = 'tool-card';
    if (!isExecuting) card.classList.add('collapsed');
    
    let statusIcon = isExecuting ? '<i data-lucide="loader" class="thinking-pulse" style="width: 14px; height: 14px;"></i>' : 
                     (result && !result.toLowerCase().startsWith('error') ? '<i data-lucide="check" style="width: 14px; height: 14px; color: var(--accent-green);"></i>' : '<i data-lucide="alert-circle" style="width: 14px; height: 14px; color: var(--accent-red);"></i>');
    
    const header = document.createElement('div');
    header.className = 'tool-header';
    header.innerHTML = `
        <div class="tool-info">
            ${statusIcon}
            <span>Executed tool: <code>${name}</code></span>
        </div>
        <i data-lucide="chevron-down" class="tool-arrow" style="width: 16px; height: 16px;"></i>
    `;
    card.appendChild(header);
    
    const content = document.createElement('div');
    content.className = 'tool-result';
    let details = `Arguments:\n${JSON.stringify(args, null, 2)}\n\n`;
    if (result) details += `Result:\n${result}`;
    else if (isExecuting) details += `Running in background...`;
    
    content.textContent = details;
    card.appendChild(content);
    
    header.addEventListener('click', () => card.classList.toggle('collapsed'));
    if (window.lucide) lucide.createIcons();
    return card;
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function scrollToBottom(force = false) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    
    if (force) {
        container.scrollTop = container.scrollHeight;
        return;
    }
    
    // Smart scroll: only scroll if the user is already near the bottom (within 100px)
    const threshold = 100;
    const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight <= threshold;
    if (isNearBottom) {
        container.scrollTop = container.scrollHeight;
    }
}

function addCopyButtons(container) {
    container.querySelectorAll('pre').forEach(pre => {
        if (pre.querySelector('.copy-btn')) return;
        
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.innerHTML = `<i data-lucide="copy" style="width: 14px; height: 14px;"></i> Copy`;
        pre.appendChild(btn);
        
        btn.addEventListener('click', () => {
            const code = pre.querySelector('code').textContent;
            navigator.clipboard.writeText(code).then(() => {
                btn.innerHTML = `<i data-lucide="check" style="width: 14px; height: 14px;"></i> Copied`;
                setTimeout(() => {
                    btn.innerHTML = `<i data-lucide="copy" style="width: 14px; height: 14px;"></i> Copy`;
                    if (window.lucide) lucide.createIcons();
                }, 2000);
            });
        });
    });
    if (window.lucide) lucide.createIcons();
}

// ── SSE Streaming Logic ──────────────────────────────────────────────
function resetStreamingState() {
    state.currentAssistantBody = null;
    state.currentAssistantBubble = null;
    state.currentThinkingBox = null;
    state.currentAssistantText = '';
    state.currentAssistantThinkingText = '';
    state.toolCards = {};
    state.toolCallCounter = 0;
}

async function sendMessage() {
    const inputArea = document.getElementById('user-input');
    if (!inputArea || state.isGenerating) return;
    
    const text = inputArea.value.trim();
    if (!text) return;
    
    inputArea.value = '';
    inputArea.style.height = 'auto';
    updateSendButtonState();
    
    appendUserMessage(text);
    
    // Initialize state for the new response
    state.isGenerating = true;
    resetStreamingState();
    updateSendButtonState();
    
    currentAbortController = new AbortController();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text }),
            signal: currentAbortController.signal
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); 
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        handleSSEEvent(JSON.parse(line.substring(6)));
                    } catch (e) {
                        console.error('SSE JSON error:', e);
                    }
                }
            }
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            appendStatusMessage('Interrupted by user.', 'primary');
        } else {
            console.error('Stream error:', err);
            appendStatusMessage('Connection error. Is the server running?', 'red');
        }
        finishGeneration(false);
    }
}

function handleSSEEvent(event) {
    if (event.type === 'status') {
        appendStatusMessage(event.message, event.color === 'red' ? 'red' : 'primary');
    }
    else if (event.type === 'thinking_start') {
        const body = getOrCreateAssistantBodyElement();
        state.currentThinkingBox = createThinkingBoxElement('', true);
        body.appendChild(state.currentThinkingBox);
        scrollToBottom();
    }
    else if (event.type === 'thinking_chunk') {
        if (state.currentThinkingBox) {
            state.currentAssistantThinkingText += event.text;
            const contentDiv = state.currentThinkingBox.querySelector('.thinking-content');
            if (contentDiv) contentDiv.textContent = state.currentAssistantThinkingText;
        }
    }
    else if (event.type === 'thinking_end') {
        if (state.currentThinkingBox) {
            state.currentThinkingBox.classList.add('collapsed');
            const meta = state.currentThinkingBox.querySelector('.thinking-meta');
            if (meta) {
                meta.innerHTML = `<i data-lucide="chevron-down" class="thinking-arrow" style="width: 14px; height: 14px;"></i>`;
                if (window.lucide) lucide.createIcons();
            }
        }
    }
    else if (event.type === 'content_chunk') {
        const body = getOrCreateAssistantBodyElement();
        if (!state.currentAssistantBubble) {
            state.currentAssistantBubble = document.createElement('div');
            state.currentAssistantBubble.className = 'message-bubble';
            body.appendChild(state.currentAssistantBubble);
        }
        state.currentAssistantText += event.text;
        state.currentAssistantBubble.innerHTML = marked.parse(state.currentAssistantText);
        addCopyButtons(state.currentAssistantBubble);
        scrollToBottom();
    }
    else if (event.type === 'tool_start') {
        const body = getOrCreateAssistantBodyElement();
        state.toolCallCounter++;
        const callId = `tool_${state.toolCallCounter}`;
        
        const card = createToolCardElement(event.name, event.arguments, '', true);
        body.appendChild(card);
        state.toolCards[callId] = card;
        
        // Temporarily store the ID on the event name (a bit hacky but works since backend doesn't send IDs)
        // For multiple concurrent tools this would fail, but Selene tools run sequentially
        state.lastToolCallId = callId;
        scrollToBottom();
    }
    else if (event.type === 'tool_end') {
        const callId = state.lastToolCallId;
        const card = state.toolCards[callId];
        if (card) {
            const newCard = createToolCardElement(event.name, {}, event.result, false);
            card.parentNode.replaceChild(newCard, card);
            state.toolCards[callId] = newCard;
        }
        
        // Reset bubbles so the post-tool response gets a fresh bubble/body block
        state.currentAssistantBody = null;
        state.currentAssistantBubble = null;
        state.currentAssistantText = '';
        scrollToBottom();
    }
    else if (event.type === 'token_usage') {
        const usageDiv = document.getElementById('token-usage');
        if (usageDiv) {
            const pct = event.total / event.budget;
            usageDiv.textContent = `${event.total} / ${event.budget} ctx`;
            usageDiv.style.display = 'block';
            
            usageDiv.className = 'token-usage';
            if (pct >= 1.0) {
                usageDiv.classList.add('danger');
            } else if (pct >= 0.9) {
                usageDiv.classList.add('warning');
            }
        }
    }
    else if (event.type === 'done') {
        finishGeneration(false);
    }
}

function getOrCreateAssistantBodyElement() {
    // If we have an active body for this turn, return it
    if (state.currentAssistantBody) return state.currentAssistantBody;
    
    // Otherwise, create a new assistant message wrapper
    const container = document.getElementById('chat-messages');
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<img src="/avatar.png" alt="S">';
    msg.appendChild(avatar);
    
    const body = document.createElement('div');
    body.className = 'message-body';
    msg.appendChild(body);
    container.appendChild(msg);
    
    state.currentAssistantBody = body;
    return body;
}

function finishGeneration(wasAborted = false) {
    if (!state.isGenerating) return;
    
    state.isGenerating = false;
    currentAbortController = null;
    updateSendButtonState();
    
    if (state.currentThinkingBox) {
        state.currentThinkingBox.classList.add('collapsed');
        const meta = state.currentThinkingBox.querySelector('.thinking-meta');
        if (meta) {
            meta.innerHTML = `<i data-lucide="chevron-down" class="thinking-arrow" style="width: 14px; height: 14px;"></i>`;
        }
    }
    if (window.lucide) lucide.createIcons();
    
    loadSessionState();
}

function appendStatusMessage(text, type = 'primary') {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    
    const row = document.createElement('div');
    row.className = 'status-msg-bubble';
    
    let icon = 'info';
    if (type === 'red') icon = 'alert-triangle';
    if (text.startsWith('✓')) icon = 'check-circle';
    
    row.innerHTML = `
        <i data-lucide="${icon}" class="status-msg-icon" style="color: var(--accent-${type});"></i>
        <span>${escapeHtml(text.replace('✓ ', '').replace('⚠ ', ''))}</span>
    `;
    container.appendChild(row);
    if (window.lucide) lucide.createIcons();
    scrollToBottom();
}

function updateSendButtonState() {
    const sendBtn = document.getElementById('send-btn');
    const inputArea = document.getElementById('user-input');
    if (!sendBtn || !inputArea) return;
    
    if (state.isGenerating) {
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>';
        sendBtn.className = 'send-btn stop-btn';
    } else {
        const hasText = inputArea.value.trim().length > 0;
        sendBtn.disabled = !hasText;
        sendBtn.innerHTML = '<i data-lucide="arrow-up" style="width: 18px; height: 18px;"></i>';
        sendBtn.className = 'send-btn';
    }
    if (window.lucide) lucide.createIcons();
}

// ── Autocomplete Functionality ───────────────────────────────────────
const SLASH_COMMANDS = [
    { cmd: '/help', usage: '', desc: 'Show help menu' },
    { cmd: '/clear', usage: '', desc: 'Clear history & system override' },
    { cmd: '/save', usage: '[name]', desc: 'Save current session file' },
    { cmd: '/load', usage: '[name|index]', desc: 'Load a saved session file' },
    { cmd: '/set parameter', usage: '<name> <val>', desc: 'Set parameter (e.g. temperature 0.7)' },
    { cmd: '/set system', usage: '"<prompt>"', desc: 'Set custom system prompt override' },
    { cmd: '/set history', usage: '', desc: 'Enable conversation history (default)' },
    { cmd: '/set nohistory', usage: '', desc: 'Disable conversation history' },
    { cmd: '/set wordwrap', usage: '', desc: 'Enable word wrapping (default)' },
    { cmd: '/set nowordwrap', usage: '', desc: 'Disable word wrapping' },
    { cmd: '/set format json', usage: '', desc: 'Force JSON output from the model' },
    { cmd: '/set noformat', usage: '', desc: 'Disable forced output format (default)' },
    { cmd: '/set verbose', usage: '', desc: 'Show generation stats after each response' },
    { cmd: '/set quiet', usage: '', desc: 'Hide generation stats (default)' },
    { cmd: '/set think', usage: '', desc: 'Enable model thinking/reasoning (default)' },
    { cmd: '/set nothink', usage: '', desc: 'Disable model thinking' },
    { cmd: '/show parameters', usage: '', desc: 'Show current session parameters' },
    { cmd: '/show system', usage: '', desc: 'Show the active system prompt' },
    { cmd: '/show model', usage: '', desc: 'Show model info' },
    { cmd: '/vault list', usage: '', desc: 'List indexed vaults' },
    { cmd: '/vault aliases', usage: '', desc: 'List registered vault aliases' },
    { cmd: '/vault alias', usage: '<name> <coll>', desc: 'Register friendly alias for a collection' },
    { cmd: '/vault rename', usage: '<old> <new>', desc: 'Rename a vault collection' },
    { cmd: '/vault add', usage: '<path>', desc: 'Index folder/file to searchable vault' },
    { cmd: '/vault search', usage: '<query>', desc: 'Search indexed content' },
    { cmd: '/vault delete', usage: '<source>', desc: 'Delete indexed chunks' },
    { cmd: '/quit', usage: '', desc: 'Exit the agent session' },
    { cmd: '/exit', usage: '', desc: 'Exit the agent session' },
    { cmd: '/q', usage: '', desc: 'Exit the agent session' }
];

let activeAutocompleteIndex = -1;
let filteredCommands = [];

function showAutocomplete(filterText) {
    const popup = document.getElementById('autocomplete-popup');
    if (!popup) return;
    
    if (!filterText.startsWith('/')) {
        hideAutocomplete();
        return;
    }
    
    const query = filterText.toLowerCase();
    filteredCommands = SLASH_COMMANDS.filter(item => {
        return item.cmd.toLowerCase().startsWith(query);
    });
    
    if (filteredCommands.length === 0) {
        hideAutocomplete();
        return;
    }
    
    activeAutocompleteIndex = 0;
    renderAutocompleteItems();
    popup.style.display = 'block';
}

function hideAutocomplete() {
    const popup = document.getElementById('autocomplete-popup');
    if (popup) {
        popup.style.display = 'none';
        popup.innerHTML = '';
    }
    filteredCommands = [];
    activeAutocompleteIndex = -1;
}

function renderAutocompleteItems() {
    const popup = document.getElementById('autocomplete-popup');
    if (!popup) return;
    
    popup.innerHTML = '';
    filteredCommands.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'autocomplete-item' + (index === activeAutocompleteIndex ? ' active' : '');
        
        const cmdSpan = document.createElement('span');
        cmdSpan.className = 'autocomplete-cmd';
        cmdSpan.textContent = item.cmd;
        
        const usageSpan = document.createElement('span');
        usageSpan.className = 'autocomplete-usage';
        usageSpan.textContent = item.usage ? ' ' + item.usage : '';
        
        const descSpan = document.createElement('span');
        descSpan.className = 'autocomplete-desc';
        descSpan.textContent = `— ${item.desc}`;
        
        div.appendChild(cmdSpan);
        div.appendChild(usageSpan);
        div.appendChild(descSpan);
        
        div.addEventListener('click', () => {
            selectAutocompleteItem(item);
        });
        
        popup.appendChild(div);
    });
    
    const activeItem = popup.children[activeAutocompleteIndex];
    if (activeItem) {
        activeItem.scrollIntoView({ block: 'nearest' });
    }
}

function selectAutocompleteItem(item) {
    const userInput = document.getElementById('user-input');
    if (!userInput) return;
    
    const suffix = item.usage ? ' ' : '';
    userInput.value = item.cmd + suffix;
    
    hideAutocomplete();
    userInput.focus();
    
    userInput.style.height = 'auto';
    userInput.style.height = userInput.scrollHeight + 'px';
    updateSendButtonState();
}

