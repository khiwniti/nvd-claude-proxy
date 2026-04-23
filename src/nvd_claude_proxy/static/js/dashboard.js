/**
 * nvd-claude-proxy Dashboard
 * Minimalist SPA logic using standard JS and template literals
 */

const state = {
    activeTab: 'sessions',
    sessions: [],
    models: {
        static_mappings: {},
        dynamic_mappings: [],
        available_nvidia_models: []
    },
    transformers: [],
    loading: true,
    ws: null,
    monitor: {
        openai: "",
        anthropic: "",
        fixes: []
    }
};

const API_BASE = '/api/dashboard';

async function fetchData(endpoint) {
    const res = await fetch(`${API_BASE}${endpoint}`);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return await res.json();
}

async function postData(endpoint, data) {
    const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return await res.json();
}

function render() {
    const contentArea = document.getElementById('content-area');
    if (!contentArea) return;

    if (state.loading) {
        contentArea.innerHTML = `
            <div class="flex items-center justify-center h-full">
                <div class="flex flex-col items-center">
                    <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4"></div>
                    <p class="text-gray-500 font-medium">Loading ${state.activeTab}...</p>
                </div>
            </div>`;
        return;
    }

    switch (state.activeTab) {
        case 'sessions':
            renderSessions(contentArea);
            break;
        case 'models':
            renderModels(contentArea);
            break;
        case 'transformers':
            renderTransformers(contentArea);
            break;
        case 'monitor':
            renderMonitor(contentArea);
            break;
        default:
            contentArea.innerHTML = `<div class="p-8 text-center text-gray-500">Tab "${state.activeTab}" not implemented.</div>`;
    }
    
    // Update navigation styles
    document.querySelectorAll('.nav-btn').forEach(btn => {
        if (btn.dataset.tab === state.activeTab) {
            btn.classList.add('bg-indigo-600', 'text-white');
            btn.classList.remove('text-gray-400', 'hover:bg-gray-800');
        } else {
            btn.classList.remove('bg-indigo-600', 'text-white');
            btn.classList.add('text-gray-400', 'hover:bg-gray-800');
        }
    });
    
    if (window.lucide) {
        lucide.createIcons();
    }
}

function renderSessions(container) {
    container.innerHTML = `
        <div class="mb-8 flex justify-between items-center">
            <div>
                <h2 class="text-2xl font-bold text-gray-900">Active Sessions</h2>
                <p class="text-sm text-gray-500">History of unique client connections identified by API key.</p>
            </div>
            <button onclick="refreshSessions()" class="flex items-center px-4 py-2 bg-white border border-gray-300 rounded-lg shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 transition">
                <i data-lucide="rotate-cw" class="w-4 h-4 mr-2"></i> Refresh
            </button>
        </div>
        <div class="bg-white shadow-sm rounded-xl border border-gray-200 overflow-hidden">
            <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                    <tr>
                        <th class="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Session</th>
                        <th class="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Usage</th>
                        <th class="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Status</th>
                        <th class="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Last Activity</th>
                        <th class="px-6 py-3 text-right text-xs font-semibold text-gray-500 uppercase tracking-wider">Actions</th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                    ${state.sessions.length === 0 ? `
                        <tr><td colspan="5" class="px-6 py-12 text-center text-gray-500 italic">No sessions found in database.</td></tr>
                    ` : state.sessions.map(s => `
                        <tr class="hover:bg-gray-50 transition-colors">
                            <td class="px-6 py-4">
                                <div class="flex flex-col">
                                    <span class="text-sm font-bold text-gray-900">${s.friendly_name || 'Unnamed Session'}</span>
                                    <span class="text-xs font-mono text-gray-400">Key: ${s.api_key.substring(0, 12)}...</span>
                                </div>
                            </td>
                            <td class="px-6 py-4">
                                <div class="flex items-center text-sm text-gray-600">
                                    <div class="mr-4">
                                        <p class="text-[10px] uppercase text-gray-400 font-bold">Total Tokens</p>
                                        <p class="font-mono">${s.tokens_used.toLocaleString()}</p>
                                    </div>
                                    ${s.model_alias ? `
                                    <div>
                                        <p class="text-[10px] uppercase text-gray-400 font-bold">Target Model</p>
                                        <p class="font-mono text-xs">${s.model_alias}</p>
                                    </div>` : ''}
                                </div>
                            </td>
                            <td class="px-6 py-4">
                                <span class="px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">Active</span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                ${new Date(s.last_active).toLocaleString()}
                            </td>
                            <td class="px-6 py-4 text-right text-sm font-medium">
                                <button onclick="editFriendlyName('${s.api_key}', '${s.friendly_name || ''}')" class="text-indigo-600 hover:text-indigo-900">Rename</button>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function renderModels(container) {
    container.innerHTML = `
        <div class="mb-8">
            <h2 class="text-2xl font-bold text-gray-900">Model Routing</h2>
            <p class="text-sm text-gray-500">Manage how requests for Anthropic models are routed to NVIDIA NIM.</p>
        </div>
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Dynamic Mappings -->
            <div class="bg-white shadow-sm rounded-xl border border-gray-200 p-6">
                <div class="flex justify-between items-center mb-6">
                    <h3 class="text-lg font-bold text-gray-900">Dynamic Overrides</h3>
                    <button onclick="addMapping()" class="inline-flex items-center px-3 py-1.5 bg-indigo-50 text-indigo-700 rounded-lg text-sm font-bold hover:bg-indigo-100 transition">
                        <i data-lucide="plus" class="w-4 h-4 mr-1"></i> New
                    </button>
                </div>
                <div class="space-y-4">
                    ${state.models.dynamic_mappings.length === 0 ? `
                        <p class="text-center py-8 text-gray-400 italic text-sm border-2 border-dashed rounded-xl">No dynamic overrides defined.</p>
                    ` : state.models.dynamic_mappings.map(m => `
                        <div class="flex items-center justify-between p-4 bg-gray-50 rounded-xl border border-gray-100 group">
                            <div class="flex items-center">
                                <div class="bg-indigo-100 p-2 rounded-lg mr-4">
                                    <i data-lucide="repeat" class="w-4 h-4 text-indigo-600"></i>
                                </div>
                                <div>
                                    <p class="font-mono text-xs font-bold text-gray-400 uppercase tracking-tighter">Alias</p>
                                    <p class="font-mono text-sm font-bold text-indigo-600">${m.anthropic_model}</p>
                                </div>
                                <i data-lucide="chevron-right" class="mx-4 text-gray-300 w-4 h-4"></i>
                                <div>
                                    <p class="font-mono text-xs font-bold text-gray-400 uppercase tracking-tighter">NVIDIA Target</p>
                                    <p class="font-mono text-sm text-gray-700">${m.nvd_model}</p>
                                </div>
                            </div>
                            <button onclick="editMapping('${m.anthropic_model}', '${m.nvd_model}')" class="opacity-0 group-hover:opacity-100 transition-opacity p-2 text-gray-400 hover:text-indigo-600">
                                <i data-lucide="edit-3" class="w-4 h-4"></i>
                            </button>
                        </div>
                    `).join('')}
                </div>
            </div>

            <!-- Static Mappings -->
            <div class="bg-white shadow-sm rounded-xl border border-gray-200 p-6 flex flex-col">
                <h3 class="text-lg font-bold text-gray-900 mb-6">Static Registry</h3>
                <div class="flex-grow space-y-2 overflow-y-auto max-h-[400px] pr-2">
                    ${Object.entries(state.models.static_mappings).map(([alias, target]) => `
                        <div class="flex items-center justify-between text-xs p-2 rounded-lg hover:bg-gray-50">
                            <span class="font-mono font-bold text-gray-600">${alias}</span>
                            <span class="text-gray-300 font-mono">⋯⋯</span>
                            <span class="font-mono text-gray-400 text-right">${target}</span>
                        </div>
                    `).join('')}
                </div>
                <div class="mt-6 pt-4 border-t border-gray-100 text-[10px] text-gray-400 font-medium">
                    Defined in <code class="bg-gray-100 px-1 rounded">config/models.yaml</code>. Dynamic overrides take precedence.
                </div>
            </div>
        </div>
    `;
}

function renderTransformers(container) {
    container.innerHTML = `
        <div class="mb-8">
            <h2 class="text-2xl font-bold text-gray-900">Transformer Policies</h2>
            <p class="text-sm text-gray-500">Toggle behavioral modifiers for requests and responses.</p>
        </div>
        <div class="bg-white shadow-sm rounded-xl border border-gray-200 overflow-hidden max-w-2xl">
            <div class="divide-y divide-gray-100">
                ${state.transformers.length === 0 ? `
                    <div class="p-12 text-center">
                        <i data-lucide="settings-2" class="w-12 h-12 text-gray-200 mx-auto mb-4"></i>
                        <p class="text-gray-500">No active transformer overrides. Using system defaults.</p>
                    </div>
                ` : state.transformers.map(t => `
                    <div class="p-6 flex items-center justify-between hover:bg-gray-50 transition-colors">
                        <div class="flex items-center">
                            <div class="bg-indigo-50 p-3 rounded-xl mr-4 text-indigo-600">
                                <i data-lucide="zap" class="w-5 h-5"></i>
                            </div>
                            <div>
                                <p class="font-bold text-gray-900">${t.transformer_name}</p>
                                <div class="flex items-center mt-0.5">
                                    <span class="text-[10px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded ${t.session_id ? 'bg-orange-100 text-orange-700' : 'bg-blue-100 text-blue-700'}">
                                        ${t.session_id ? 'Session-Level' : 'Global Default'}
                                    </span>
                                    ${t.session_id ? `<span class="text-xs text-gray-400 ml-2">ID: ${t.session_id}</span>` : ''}
                                </div>
                            </div>
                        </div>
                        <button onclick="toggleTransformer(${t.id}, '${t.transformer_name}', ${t.enabled}, ${t.session_id})" 
                                class="relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-600 focus:ring-offset-2 ${t.enabled ? 'bg-indigo-600' : 'bg-gray-200'}">
                            <span class="pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${t.enabled ? 'translate-x-5' : 'translate-x-0'}"></span>
                        </button>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

function renderMonitor(container) {
    container.innerHTML = `
        <div class="flex flex-col h-full max-h-[calc(100vh-120px)]">
            <div class="mb-4 flex justify-between items-center">
                <div>
                    <h2 class="text-2xl font-bold text-gray-900">Live Traffic Monitor</h2>
                    <p class="text-sm text-gray-500">Streaming OpenAI (In) &rarr; Anthropic (Out) events.</p>
                </div>
                <div id="connection-status" class="flex items-center text-xs font-bold uppercase tracking-widest text-gray-500">
                    <span class="w-2 h-2 rounded-full bg-gray-400 mr-2 animate-pulse"></span> Initializing...
                </div>
            </div>

            <!-- Fix Ticker -->
            <div class="bg-indigo-900 text-indigo-100 px-4 py-2 rounded-lg mb-4 overflow-hidden relative h-10 flex items-center shadow-inner border border-indigo-700">
                <div class="absolute left-0 top-0 bottom-0 px-3 bg-indigo-800 flex items-center z-10 border-r border-indigo-700 shadow-xl">
                    <i data-lucide="zap" class="w-4 h-4 text-yellow-400 mr-2"></i>
                    <span class="text-[10px] font-black uppercase italic">FIX TICKER</span>
                </div>
                <div id="fix-ticker" class="whitespace-nowrap flex space-x-8 pl-32">
                    <span class="text-indigo-300 italic opacity-50">Waiting for transformer fixes...</span>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-grow overflow-hidden min-h-0">
                <!-- OpenAI Window (In) -->
                <div class="bg-gray-900 rounded-xl border border-gray-800 flex flex-col overflow-hidden shadow-2xl">
                    <div class="bg-gray-800 px-4 py-2 flex items-center justify-between">
                        <span class="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Upstream (OpenAI)</span>
                        <button onclick="clearMonitor('openai')" class="text-gray-500 hover:text-white transition">
                            <i data-lucide="trash-2" class="w-3 h-3"></i>
                        </button>
                    </div>
                    <pre id="openai-window" class="flex-grow p-4 text-[11px] text-blue-300 overflow-y-auto font-mono leading-relaxed scroll-smooth">${state.monitor.openai || 'Waiting for upstream chunks...'}</pre>
                </div>

                <!-- Anthropic Window (Out) -->
                <div class="bg-gray-900 rounded-xl border border-gray-800 flex flex-col overflow-hidden shadow-2xl">
                    <div class="bg-gray-800 px-4 py-2 flex items-center justify-between">
                        <span class="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Client (Anthropic)</span>
                        <button onclick="clearMonitor('anthropic')" class="text-gray-500 hover:text-white transition">
                            <i data-lucide="trash-2" class="w-3 h-3"></i>
                        </button>
                    </div>
                    <pre id="anthropic-window" class="flex-grow p-4 text-[11px] text-green-300 overflow-y-auto font-mono leading-relaxed scroll-smooth">${state.monitor.anthropic || 'Waiting for downstream events...'}</pre>
                </div>
            </div>
        </div>
        <style>
            @keyframes marquee {
                0% { transform: translateX(100%); }
                100% { transform: translateX(-100%); }
            }
            .fix-item {
                display: inline-block;
                animation: marquee 20s linear infinite;
            }
            #openai-window::-webkit-scrollbar, #anthropic-window::-webkit-scrollbar {
                width: 4px;
            }
            #openai-window::-webkit-scrollbar-thumb, #anthropic-window::-webkit-scrollbar-thumb {
                background: #374151;
                border-radius: 10px;
            }
        </style>
    `;
    
    initMonitorWS();
    if (window.lucide) lucide.createIcons();
}

function initMonitorWS() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        updateConnectionStatus(true);
        return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/monitor`;
    
    console.log('Connecting to monitor WS:', wsUrl);
    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        console.log('Monitor WS Connected');
        updateConnectionStatus(true);
    };

    state.ws.onclose = () => {
        console.log('Monitor WS Disconnected');
        updateConnectionStatus(false);
        // Try to reconnect after 5 seconds
        setTimeout(initMonitorWS, 5000);
    };

    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWSMessage(data);
        } catch (e) {
            console.error('Failed to parse WS message', e);
        }
    };
}

function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connection-status');
    if (!statusEl) return;
    
    if (connected) {
        statusEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-green-500 mr-2"></span> Connected`;
        statusEl.classList.remove('text-gray-500');
        statusEl.classList.add('text-green-600');
    } else {
        statusEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-red-500 mr-2"></span> Disconnected`;
        statusEl.classList.remove('text-green-600');
        statusEl.classList.add('text-red-600');
    }
}

function handleWSMessage(msg) {
    const { type, payload, request_id } = msg;
    
    if (type === 'openai_chunk') {
        const text = JSON.stringify(payload, null, 2);
        state.monitor.openai += `\n[${request_id.substring(0,8)}] ${text}`;
        const win = document.getElementById('openai-window');
        if (win) {
            win.textContent = state.monitor.openai;
            win.scrollTop = win.scrollHeight;
        }
    } else if (type === 'anthropic_event') {
        const text = JSON.stringify(payload, null, 2);
        state.monitor.anthropic += `\n[${request_id.substring(0,8)}] ${text}`;
        const win = document.getElementById('anthropic-window');
        if (win) {
            win.textContent = state.monitor.anthropic;
            win.scrollTop = win.scrollHeight;
        }
    } else if (type === 'transformer_fix') {
        const fixText = `${msg.fix_type}: ${JSON.stringify(payload)}`;
        addFixToTicker(fixText);
    } else if (type === 'error') {
        state.monitor.anthropic += `\n[ERROR] ${JSON.stringify(payload, null, 2)}`;
        const win = document.getElementById('anthropic-window');
        if (win) {
            win.textContent = state.monitor.anthropic;
            win.scrollTop = win.scrollHeight;
        }
    }
}

function addFixToTicker(text) {
    const ticker = document.getElementById('fix-ticker');
    if (!ticker) return;

    // Clear initial message
    if (state.monitor.fixes.length === 0) {
        ticker.innerHTML = '';
    }

    state.monitor.fixes.push(text);
    if (state.monitor.fixes.length > 10) state.monitor.fixes.shift();

    const span = document.createElement('span');
    span.className = 'text-yellow-400 font-bold mr-12';
    span.textContent = `⚡ ${text}`;
    ticker.appendChild(span);
    
    // Auto-scroll ticker if it gets too long
    ticker.scrollLeft = ticker.scrollWidth;
}

window.clearMonitor = (type) => {
    state.monitor[type] = "";
    const win = document.getElementById(`${type}-window`);
    if (win) win.textContent = `Cleared. Waiting for new ${type} events...`;
};

// Global window-scoped actions for HTML handlers
window.refreshSessions = async () => {
    state.loading = true; render();
    try {
        state.sessions = await fetchData('/sessions');
    } catch(e) { alert('Failed to fetch sessions'); }
    state.loading = false; render();
};

window.editFriendlyName = async (apiKey, current) => {
    const newName = prompt('Enter a recognizable name for this API key:', current);
    if (newName !== null) {
        try {
            await postData(`/sessions/${apiKey}/friendly_name`, { friendly_name: newName });
            window.refreshSessions();
        } catch(e) { alert('Failed to update name'); }
    }
};

window.editMapping = async (anthropicModel, currentNvd) => {
    const nvdModel = prompt(`Map "${anthropicModel}" to which NVIDIA model?`, currentNvd);
    if (nvdModel) {
        try {
            await postData('/models/map', { anthropic_model: anthropicModel, nvd_model: nvdModel });
            loadTab('models');
        } catch(e) { alert('Failed to update mapping'); }
    }
};

window.addMapping = async () => {
    const anthropicModel = prompt('Anthropic Model Name (e.g. claude-3-5-sonnet):');
    if (!anthropicModel) return;
    const nvdModel = prompt('Target NVIDIA NIM Identifier:');
    if (!nvdModel) return;
    try {
        await postData('/models/map', { anthropic_model: anthropicModel, nvd_model: nvdModel });
        loadTab('models');
    } catch(e) { alert('Failed to create mapping'); }
};

window.toggleTransformer = async (id, name, current, sessionId) => {
    try {
        await postData('/transformers/toggle', { 
            transformer_name: name, 
            enabled: !current,
            session_id: sessionId
        });
        loadTab('transformers');
    } catch(e) { alert('Failed to toggle transformer'); }
};

async function loadTab(tab) {
    state.activeTab = tab;
    state.loading = true;
    render();
    
    try {
        if (tab === 'sessions') {
            state.sessions = await fetchData('/sessions');
        } else if (tab === 'models') {
            state.models = await fetchData('/models');
        } else if (tab === 'transformers') {
            state.transformers = await fetchData('/transformers');
        }
    } catch (err) {
        console.error('Failed to fetch data', err);
    }
    
    state.loading = false;
    render();
}

// Global Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    document.addEventListener('click', e => {
        const btn = e.target.closest('.nav-btn');
        if (btn) {
            loadTab(btn.dataset.tab);
        }
    });

    // Initial Load
    loadTab('sessions');
    
    // Proactively start WS for monitor
    initMonitorWS();
});
