// ── Global Application State ─────────────────────────────────────────
const state = {
    activeSession: 'Active Session',
    history: [],
    settings: {
        options: {
            temperature: 0.7,
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
    isGenerating: false,
    currentThinkingBox: null,
    currentAssistantBubble: null,
    currentAssistantText: '',
    currentAssistantThinkingText: '',
    toolCards: {} // Maps tool call indexes or IDs to elements
};

// ── Initialization ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initLibraries();
    initEventListeners();
    loadSessionState();
});

// Configure markdown parsing & highlighting
function initLibraries() {
    marked.setOptions({
        highlight: function (code, lang) {
            const language = hljs.getLanguage(lang) ? lang : 'plaintext';
            return hljs.highlight(code, { language }).value;
        },
        langPrefix: 'hljs language-'
    });
    
    // Initial Lucide icons rendering
    if (window.lucide) {
        lucide.createIcons();
    }
}

// ── Event Listeners ──────────────────────────────────────────────────
function initEventListeners() {
    // Menu Sidebar toggle (for mobile)
    const menuToggle = document.getElementById('menu-toggle');
    const closeSidebar = document.getElementById('close-sidebar');
    const sidebar = document.getElementById('sidebar');
    
    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', () => sidebar.classList.add('active'));
    }
    if (closeSidebar && sidebar) {
        closeSidebar.addEventListener('click', () => sidebar.classList.remove('active'));
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
            
            // Show confirmation on button
            const originalText = applyPromptBtn.textContent;
            applyPromptBtn.textContent = 'Applied ✓';
            applyPromptBtn.style.borderColor = 'var(--accent-teal)';
            applyPromptBtn.style.color = 'var(--accent-teal)';
            setTimeout(() => {
                applyPromptBtn.textContent = originalText;
                applyPromptBtn.style.borderColor = '';
                applyPromptBtn.style.color = '';
            }, 1500);
        });
    }
    
    // Input Area Resizing & Enter key
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    
    if (userInput) {
        userInput.addEventListener('input', () => {
            // Auto resize
            userInput.style.height = 'auto';
            userInput.style.height = (userInput.scrollHeight - 4) + 'px';
            
            // Toggle send button
            sendBtn.disabled = !userInput.value.trim() || state.isGenerating;
        });
        
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
    
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }
    
    // New Chat Button
    const newChatBtn = document.getElementById('new-chat-btn');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', () => {
            clearSession();
        });
    }
    
    // Clear History Button
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to clear this conversation history?')) {
                clearSession();
            }
        });
    }
    
    // Modals
    setupModal('save-session-btn', 'save-modal');
    setupModal('help-btn', 'help-modal');
    
    // Confirm Save Session
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
                    closeAllModals();
                    sessionNameInput.value = '';
                    loadSessionState(); // Reload session list
                } else {
                    alert('Error saving session: ' + data.error);
                }
            } catch (err) {
                console.error(err);
                alert('Failed to save session');
            }
        });
    }
}

// Modal helper
function setupModal(triggerId, modalId) {
    const trigger = document.getElementById(triggerId);
    const modal = document.getElementById(modalId);
    if (!trigger || !modal) return;
    
    trigger.addEventListener('click', () => {
        modal.classList.add('active');
        const firstInput = modal.querySelector('input');
        if (firstInput) setTimeout(() => firstInput.focus(), 100);
    });
    
    const closeBtns = modal.querySelectorAll('.close-modal-btn');
    closeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modal.classList.remove('active');
        });
    });
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });
}

function closeAllModals() {
    const modals = document.querySelectorAll('.modal-overlay');
    modals.forEach(m => m.classList.remove('active'));
}

// ── Backend Communications ───────────────────────────────────────────

// Load current configuration and history from backend
async function loadSessionState() {
    try {
        const res = await fetch('/api/settings');
        const data = await res.json();
        
        state.settings = data.settings;
        state.history = data.history;
        state.savedSessions = data.saved_sessions;
        state.activeSession = data.active_session_name || 'Active Session';
        
        document.getElementById('current-session-title').textContent = state.activeSession;
        document.getElementById('model-badge').textContent = data.model_name || 'gemma-agent';
        
        // Sync UI inputs with loaded settings
        syncSettingsToUI();
        
        // Render saved sessions
        renderSavedSessionsList();
        
        // Render history
        renderHistory();
        
    } catch (err) {
        console.error('Failed to load session settings:', err);
        document.getElementById('connection-status').textContent = 'Ollama Offline';
        document.getElementById('connection-status').previousElementSibling.className = 'status-dot offline';
    }
}

// Push local settings updates to backend
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

// Sync settings variables to HTML form fields
function syncSettingsToUI() {
    const tempSlider = document.getElementById('param-temperature');
    const tempVal = document.getElementById('temp-val');
    if (tempSlider && tempVal) {
        const temp = state.settings.options.temperature ?? 0.7;
        tempSlider.value = temp;
        tempVal.textContent = temp;
    }
    
    const toppSlider = document.getElementById('param-top_p');
    const toppVal = document.getElementById('topp-val');
    if (toppSlider && toppVal) {
        const topp = state.settings.options.top_p ?? 0.9;
        toppSlider.value = topp;
        toppVal.textContent = topp;
    }
    
    const topkSlider = document.getElementById('param-top_k');
    const topkVal = document.getElementById('topk-val');
    if (topkSlider && topkVal) {
        const topk = state.settings.options.top_k ?? 40;
        topkSlider.value = topk;
        topkVal.textContent = topk;
    }
    
    const historySwitch = document.getElementById('setting-history');
    if (historySwitch) {
        historySwitch.checked = state.settings.history;
    }
    
    const thinkSwitch = document.getElementById('setting-think');
    if (thinkSwitch) {
        thinkSwitch.checked = state.settings.think;
    }
    
    const systemPrompt = document.getElementById('system-prompt');
    if (systemPrompt) {
        systemPrompt.value = state.settings.system || '';
    }
}

// Clear current session settings and history
async function clearSession() {
    try {
        const res = await fetch('/api/clear-session', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            loadSessionState();
        }
    } catch (err) {
        console.error(err);
    }
}

// Render the list of saved sessions in sidebar
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
        
        // Try parsing timestamp and cleaner name
        let displayName = basename;
        let timeLabel = '';
        
        // Format: [name]_[YYYYMMDD]_[HHMMSS]
        const match = basename.match(/^(.+)_(\d{8})_(\d{6})$/);
        if (match) {
            displayName = match[1].replace(/_/g, ' ');
            const yr = match[2].substring(0, 4);
            const mo = match[2].substring(4, 6);
            const dy = match[2].substring(6, 8);
            const hr = match[3].substring(0, 2);
            const mn = match[3].substring(2, 4);
            timeLabel = `${dy}/${mo}/${yr} ${hr}:${mn}`;
        }
        
        const item = document.createElement('div');
        item.className = 'session-item';
        if (basename + '.json' === state.activeSession) {
            item.classList.add('active');
        }
        
        item.innerHTML = `
            <div class="session-details">
                <span class="session-name" title="${displayName}">${displayName}</span>
                <span class="session-meta">${timeLabel || 'Session file'}</span>
            </div>
            <i data-lucide="chevron-right" style="width: 14px; height: 14px; color: var(--text-muted);"></i>
        `;
        
        item.addEventListener('click', () => loadSavedSession(sessionFile));
        container.appendChild(item);
    });
    
    if (window.lucide) lucide.createIcons({ attrs: { class: 'lucide-icon' } });
}

// Load a specific session JSON file
async function loadSavedSession(filename) {
    try {
        const res = await fetch('/api/load-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: filename })
        });
        const data = await res.json();
        if (data.status === 'success') {
            loadSessionState();
            // Close mobile sidebar if open
            document.getElementById('sidebar').classList.remove('active');
        } else {
            alert('Failed to load session: ' + data.error);
        }
    } catch (err) {
        console.error(err);
    }
}

// ── Rendering Chat Logs ──────────────────────────────────────────────

// Render welcome layout or chat history bubbles
function renderHistory() {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Filter history to display user & assistant replies, excluding system/tool messages
    // (tools will be integrated inside assistant messages)
    const displayMessages = [];
    
    for (let i = 0; i < state.history.length; i++) {
        const msg = state.history[i];
        
        if (msg.role === 'user') {
            displayMessages.push({
                role: 'user',
                content: msg.content
            });
        } else if (msg.role === 'assistant') {
            // Check for associated tools succeeding this assistant msg
            const tools = [];
            let j = i + 1;
            while (j < state.history.length && state.history[j].role === 'tool') {
                tools.push(state.history[j]);
                j++;
            }
            // Advance iterator past tool messages
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
        if (msg.role === 'user') {
            appendUserMessage(msg.content, false);
        } else {
            appendAssistantMessage(msg, false);
        }
    });
    
    scrollToBottom();
}

// Render empty dashboard with query suggestions
function renderWelcomeScreen(container) {
    const welcome = document.createElement('div');
    welcome.className = 'welcome-container';
    welcome.innerHTML = `
        <div class="welcome-logo">
            <i data-lucide="bot"></i>
        </div>
        <div>
            <h1 class="welcome-title">AI Agent Workspace</h1>
            <p class="welcome-subtitle">A modern, visual wrapper around your local Gemma-powered CLI Agent. Control your desktop, query documents, and listen to music.</p>
        </div>
        
        <div class="suggestions-grid">
            <div class="suggestion-card" data-cmd="play some synthwave music on spotify">
                <div class="suggestion-header">
                    <i data-lucide="music" style="width: 16px; height: 16px;"></i> Spotify Music
                </div>
                <div class="suggestion-text">"Play some synthwave music on Spotify"</div>
            </div>
            <div class="suggestion-card" data-cmd="open chrome">
                <div class="suggestion-header">
                    <i data-lucide="external-link" style="width: 16px; height: 16px;"></i> Launch App
                </div>
                <div class="suggestion-text">"Open Google Chrome browser"</div>
            </div>
            <div class="suggestion-card" data-cmd="search indexed vaults for project info">
                <div class="suggestion-header">
                    <i data-lucide="search" style="width: 16px; height: 16px;"></i> Vault Search
                </div>
                <div class="suggestion-text">"Search indexed vaults for project info"</div>
            </div>
            <div class="suggestion-card" data-cmd="show help info">
                <div class="suggestion-header">
                    <i data-lucide="help-circle" style="width: 16px; height: 16px;"></i> Show Help
                </div>
                <div class="suggestion-text">"How do I index my local PDF folders?"</div>
            </div>
        </div>
    `;
    
    container.appendChild(welcome);
    
    // Bind click events to suggestion cards
    const cards = welcome.querySelectorAll('.suggestion-card');
    cards.forEach(card => {
        card.addEventListener('click', () => {
            const cmd = card.getAttribute('data-cmd');
            const input = document.getElementById('user-input');
            if (input) {
                input.value = cmd;
                input.style.height = 'auto';
                input.style.height = (input.scrollHeight - 4) + 'px';
                document.getElementById('send-btn').disabled = false;
                input.focus();
            }
        });
    });
    
    if (window.lucide) lucide.createIcons();
}

// Append a user speech bubble
function appendUserMessage(content, animate = true) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    
    // Clear welcome screen if present
    const welcome = container.querySelector('.welcome-container');
    if (welcome) container.innerHTML = '';
    
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
    if (animate) scrollToBottom();
}

// Helper to escape HTML characters
function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Append an assistant reply bubble (which might include thinking logs and tool cards)
function appendAssistantMessage(msgData, animate = true) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    if (!animate) msg.style.animation = 'none';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = 'A';
    msg.appendChild(avatar);
    
    const body = document.createElement('div');
    body.className = 'message-body';
    
    // 1. Thinking block
    if (msgData.thinking) {
        const thinkingBox = createThinkingBoxElement(msgData.thinking, false);
        body.appendChild(thinkingBox);
    }
    
    // 2. Tool Calls
    if (msgData.tool_calls) {
        msgData.tool_calls.forEach((tc, idx) => {
            const toolName = tc.function.name;
            const args = tc.function.arguments;
            let result = '';
            
            // Try matching results
            if (msgData.tool_results && msgData.tool_results[idx]) {
                result = msgData.tool_results[idx].content;
            }
            
            const toolCard = createToolCardElement(toolName, args, result, false);
            body.appendChild(toolCard);
        });
    }
    
    // 3. Final Content
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = marked.parse(msgData.content || '');
    body.appendChild(bubble);
    
    msg.appendChild(body);
    container.appendChild(msg);
    
    // Enable copy buttons for code blocks
    addCopyButtons(bubble);
    
    if (animate) scrollToBottom();
}

// Create a collapsible Thinking Log panel
function createThinkingBoxElement(text, isStreaming = false) {
    const box = document.createElement('div');
    box.className = 'thinking-box';
    if (!isStreaming) box.classList.add('collapsed');
    
    const header = document.createElement('div');
    header.className = 'thinking-header';
    
    const title = document.createElement('div');
    title.className = 'thinking-title';
    title.innerHTML = `<i data-lucide="brain" style="width: 14px; height: 14px;"></i> Thinking Process`;
    
    const meta = document.createElement('div');
    meta.className = 'thinking-meta';
    
    if (isStreaming) {
        meta.innerHTML = `
            <span class="thinking-pulse">
                <span class="thinking-dot"></span>
                <span class="thinking-dot"></span>
                <span class="thinking-dot"></span>
            </span>
        `;
    } else {
        meta.innerHTML = `<i data-lucide="chevron-down" class="thinking-arrow" style="width: 14px; height: 14px;"></i>`;
    }
    
    header.appendChild(title);
    header.appendChild(meta);
    box.appendChild(header);
    
    const content = document.createElement('div');
    content.className = 'thinking-content';
    content.textContent = text;
    box.appendChild(content);
    
    // Event listener to toggle collapse
    header.addEventListener('click', () => {
        if (state.isGenerating && isStreaming) return; // Prevent collapse while streaming
        box.classList.toggle('collapsed');
        const arrow = meta.querySelector('.thinking-arrow');
        if (arrow) {
            // Done toggle via class, arrow is handled in CSS
        }
    });
    
    if (window.lucide) lucide.createIcons({ attrs: { class: 'lucide-icon' } });
    return box;
}

// Create an expandable Tool Execution card
function createToolCardElement(name, args, result = '', isExecuting = false) {
    const card = document.createElement('div');
    card.className = 'tool-card';
    if (!isExecuting) card.classList.add('collapsed');
    
    const header = document.createElement('div');
    header.className = 'tool-header';
    
    const info = document.createElement('div');
    info.className = 'tool-info';
    
    let statusIcon = '🔧';
    if (isExecuting) statusIcon = '⟳';
    else if (result && !result.toLowerCase().startsWith('error')) statusIcon = '✓';
    else if (result) statusIcon = '⚠';
    
    info.innerHTML = `
        <span class="tool-msg-icon">${statusIcon}</span>
        <span>Executed tool: <code style="color: var(--accent-amber); font-weight: bold; background: transparent; padding: 0;">${name}</code></span>
    `;
    
    const arrow = document.createElement('i');
    arrow.className = 'tool-arrow';
    arrow.setAttribute('data-lucide', 'chevron-down');
    arrow.style.width = '14px';
    arrow.style.height = '14px';
    arrow.style.color = 'var(--text-muted)';
    
    header.appendChild(info);
    header.appendChild(arrow);
    card.appendChild(header);
    
    const content = document.createElement('div');
    content.className = 'tool-result';
    
    // Format display content: args + result
    let details = `Arguments:\n${JSON.stringify(args, null, 2)}\n\n`;
    if (result) {
        details += `Result:\n${result}`;
    } else if (isExecuting) {
        details += `Running in background...`;
    }
    
    content.textContent = details;
    card.appendChild(content);
    
    header.addEventListener('click', () => {
        card.classList.toggle('collapsed');
    });
    
    if (window.lucide) lucide.createIcons({ attrs: { class: 'lucide-icon' } });
    return card;
}

// Scroll chat container to bottom
function scrollToBottom() {
    const container = document.getElementById('chat-messages');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

// Add copy buttons to code blocks
function addCopyButtons(container) {
    const preBlocks = container.querySelectorAll('pre');
    preBlocks.forEach(pre => {
        // Prevent adding multiple buttons
        if (pre.querySelector('.copy-btn')) return;
        
        const btn = document.createElement('button');
        btn.className = 'btn btn-secondary btn-sm copy-btn';
        btn.style.position = 'absolute';
        btn.style.top = '8px';
        btn.style.right = '8px';
        btn.style.width = 'auto';
        btn.style.padding = '4px 8px';
        btn.style.fontSize = '0.75rem';
        btn.innerHTML = `<i data-lucide="copy" style="width: 12px; height: 12px;"></i> Copy`;
        
        pre.appendChild(btn);
        
        btn.addEventListener('click', () => {
            const code = pre.querySelector('code').textContent;
            navigator.clipboard.writeText(code).then(() => {
                btn.innerHTML = 'Copied ✓';
                btn.style.color = 'var(--accent-teal)';
                setTimeout(() => {
                    btn.innerHTML = `<i data-lucide="copy" style="width: 12px; height: 12px;"></i> Copy`;
                    btn.style.color = '';
                    if (window.lucide) lucide.createIcons();
                }, 1500);
            });
        });
    });
    
    if (window.lucide) lucide.createIcons({ attrs: { class: 'lucide-icon' } });
}

// ── SSE Chat Streaming Logic ─────────────────────────────────────────

async function sendMessage() {
    const inputArea = document.getElementById('user-input');
    if (!inputArea || state.isGenerating) return;
    
    const text = inputArea.value.trim();
    if (!text) return;
    
    // Clear input
    inputArea.value = '';
    inputArea.style.height = 'auto';
    document.getElementById('send-btn').disabled = true;
    
    // Display user bubble
    appendUserMessage(text);
    
    // Lock state
    state.isGenerating = true;
    state.currentThinkingBox = null;
    state.currentAssistantBubble = null;
    state.currentAssistantText = '';
    state.currentAssistantThinkingText = '';
    
    // Start streaming from server
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        
        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Hold onto partial line
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const eventData = JSON.parse(line.substring(6));
                        handleSSEEvent(eventData);
                    } catch (e) {
                        console.error('Error parsing SSE event json:', e, line);
                    }
                }
            }
        }
    } catch (err) {
        console.error('Streaming failed:', err);
        appendStatusMessage('⚠ Connection error: Make sure the backend server is running and Ollama is active.', 'red');
    } finally {
        // Unlock state
        state.isGenerating = false;
        const sendBtn = document.getElementById('send-btn');
        if (sendBtn) sendBtn.disabled = !inputArea.value.trim();
        
        // Final refresh
        loadSessionState();
    }
}

// Handle individual Server-Sent Events from backend
function handleSSEEvent(event) {
    const chatContainer = document.getElementById('chat-messages');
    if (!chatContainer) return;
    
    // 1. Status notices
    if (event.type === 'status') {
        appendStatusMessage(event.message, event.color === 'red' ? 'red' : 'cyan');
    }
    
    // 2. Thinking phase start
    else if (event.type === 'thinking_start') {
        // If we already have a bubble, create a new container in it. Otherwise, create a new assistant block.
        const body = getOrCreateAssistantBodyElement();
        state.currentThinkingBox = createThinkingBoxElement('', true);
        body.appendChild(state.currentThinkingBox);
        scrollToBottom();
    }
    
    // 3. Thinking content chunk
    else if (event.type === 'thinking_chunk') {
        if (state.currentThinkingBox) {
            state.currentAssistantThinkingText += event.text;
            const contentDiv = state.currentThinkingBox.querySelector('.thinking-content');
            if (contentDiv) contentDiv.textContent = state.currentAssistantThinkingText;
        }
    }
    
    // 4. Thinking phase end
    else if (event.type === 'thinking_end') {
        if (state.currentThinkingBox) {
            state.currentThinkingBox.classList.add('collapsed');
            const meta = state.currentThinkingBox.querySelector('.thinking-meta');
            if (meta) {
                meta.innerHTML = `<i data-lucide="chevron-down" class="thinking-arrow" style="width: 14px; height: 14px;"></i>`;
                if (window.lucide) lucide.createIcons({ attrs: { class: 'lucide-icon' } });
            }
        }
    }
    
    // 5. Final content text chunk
    else if (event.type === 'content_chunk') {
        const body = getOrCreateAssistantBodyElement();
        
        // If the markdown bubble isn't created yet, create it
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
    
    // 6. Tool execution start
    else if (event.type === 'tool_start') {
        const body = getOrCreateAssistantBodyElement();
        const card = createToolCardElement(event.name, event.arguments, '', true);
        body.appendChild(card);
        state.toolCards[event.name] = card;
        scrollToBottom();
    }
    
    // 7. Tool execution end
    else if (event.type === 'tool_end') {
        const card = state.toolCards[event.name];
        if (card) {
            const name = event.name;
            const res = event.result;
            
            // Replace with completed tool card
            const newCard = createToolCardElement(name, {}, res, false);
            card.parentNode.replaceChild(newCard, card);
            state.toolCards[name] = newCard;
        }
        scrollToBottom();
    }
}

// Get or spawn the active assistant DOM message wrapper
function getOrCreateAssistantBodyElement() {
    const chatContainer = document.getElementById('chat-messages');
    
    // If there is an active assistant bubble we are streaming into, return its body
    const messages = chatContainer.querySelectorAll('.message.assistant');
    if (messages.length > 0 && state.isGenerating) {
        const lastMsg = messages[messages.length - 1];
        // Ensure this message was spawned in the current user turn
        const body = lastMsg.querySelector('.message-body');
        if (body) return body;
    }
    
    // Otherwise, create a new assistant block
    const welcome = chatContainer.querySelector('.welcome-container');
    if (welcome) chatContainer.innerHTML = '';
    
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'A';
    msg.appendChild(avatar);
    
    const body = document.createElement('div');
    body.className = 'message-body';
    msg.appendChild(body);
    
    chatContainer.appendChild(msg);
    return body;
}

// Append a system status bubble
function appendStatusMessage(text, type = 'cyan') {
    const chatContainer = document.getElementById('chat-messages');
    if (!chatContainer) return;
    
    const welcome = chatContainer.querySelector('.welcome-container');
    if (welcome) chatContainer.innerHTML = '';
    
    const row = document.createElement('div');
    row.className = 'message status-msg';
    
    let icon = '🔧';
    if (text.startsWith('✓')) icon = '✓';
    else if (text.startsWith('⚠')) icon = '⚠';
    
    row.innerHTML = `
        <div class="status-msg-bubble">
            <span class="status-msg-icon" style="color: var(--accent-${type});">${icon}</span>
            <span>${escapeHtml(text)}</span>
        </div>
    `;
    
    chatContainer.appendChild(row);
    scrollToBottom();
}
