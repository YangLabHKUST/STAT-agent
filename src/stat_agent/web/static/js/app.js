(function() {
  'use strict';

  // ============================================================
  // SECTION 1: STATE & CONSTANTS
  // ============================================================

  const state = {
    initialized: false,
    sessionSummary: null,
    currentSliceId: null,
    currentModality: 'gene',
    geneList: [],

    // ROI
    roiTool: null,             // null | 'bbox' | 'polygon' | 'freehand'
    roiDrawingState: null,     // in-progress drawing coords
    rois: [],
    highlightedROI: null,      // name of ROI being hovered in list

    // Chat
    chatMessages: [],
    chatStreaming: false,
    chatAbortController: null,
    currentAssistantMsg: null,  // DOM element being built during streaming

    // Canvas cache: keyed by "slice_{id}_{modality}"
    canvases: {},

    // Coordinate scale factor for point rendering
    coordScale: 1.0,

    // Interaction
    isPanning: false,
    panStart: null,

    // Skill panel
    skillPanelOpen: false,
    skills: [],  // [{slug, title, description, enabled, ...}]

    // v2: interaction mode, provenance
    mode: 'interactive',
    ledgerVersion: 0,
    stale: new Set(),
  };

  // ---- Provider configuration ----
  // Default model IDs as of 2026-05. Every value here has been verified
  // end-to-end against the live provider API. Users type the bare model ID;
  // routing is determined by the Provider dropdown, not by a name prefix.
  // Less common providers (Moonshot, xAI/Grok, Zhipu, Qwen) are intentionally
  // omitted — they are reachable through Poe.
  const PROVIDER_CONFIG = {
    openai:    { models: 'gpt-5.4, gpt-5.5, gpt-4o',                                                        placeholder: 'gpt-5.4' },
    anthropic: { models: 'claude-opus-4-7, claude-opus-4-6',                                                placeholder: 'claude-opus-4-7' },
    google:    { models: 'gemini-3.1-pro-preview, gemini-2.5-pro',                                          placeholder: 'gemini-3.1-pro-preview' },
    deepseek:  { models: 'deepseek-v4-pro, deepseek-v4-flash',                                              placeholder: 'deepseek-v4-pro' },
    poe:       { models: 'claude-sonnet-4.5, claude-opus-4.7, gpt-5.5, gemini-3.1-pro, deepseek-v4-pro-el', placeholder: 'claude-sonnet-4.5' },
    custom:    { models: 'Enter any model ID',                                                              placeholder: '' },
  };

  // Updates placeholder + base-url visibility for the currently selected
  // provider. Does NOT touch the model / API key values — used during page
  // load when we want to keep restored values intact.
  function updateProviderUI() {
    const provider = document.getElementById('provider').value;
    const cfg = PROVIDER_CONFIG[provider] || PROVIDER_CONFIG.custom;
    const modelInput = document.getElementById('model');
    const baseUrlInput = document.getElementById('base-url');
    const baseUrlGroup = document.getElementById('base-url-group');
    const modelHint = document.getElementById('model-hint');

    modelInput.placeholder = cfg.placeholder || 'model-name';
    modelHint.textContent = 'e.g. ' + cfg.models;

    // Base URL is only meaningful for the Custom provider (built-in providers
    // route to the right endpoint internally). Hide the field otherwise and
    // clear any stale value so it isn't sent to the backend.
    if (provider === 'custom') {
      baseUrlGroup.classList.remove('hidden');
    } else {
      baseUrlGroup.classList.add('hidden');
      baseUrlInput.value = '';
    }
  }

  // Fires when the user picks a different provider in the dropdown. Clears
  // the model + API key inputs so they fall back to the provider's defaults
  // (different providers expect different IDs and different keys).
  function onProviderChange() {
    updateProviderUI();
    document.getElementById('model').value = '';
    document.getElementById('api-key').value = '';
  }

  // ---- localStorage helpers for form persistence ----
  const STORAGE_PREFIX = 'stat_';
  const PERSISTED_FIELDS = ['dataset-dir', 'session-name', 'api-key', 'model', 'base-url', 'provider'];

  function saveFormToStorage() {
    PERSISTED_FIELDS.forEach(id => {
      const el = document.getElementById(id);
      if (el) localStorage.setItem(STORAGE_PREFIX + id, el.value);
    });
  }

  function loadFormFromStorage() {
    PERSISTED_FIELDS.forEach(id => {
      const el = document.getElementById(id);
      const saved = localStorage.getItem(STORAGE_PREFIX + id);
      if (el && saved) el.value = saved;
    });
    // Refresh placeholder / base-url visibility WITHOUT clearing the values
    // we just restored.
    updateProviderUI();
  }

  // ---- Session reconnect on page refresh ----
  function firstSliceOf(summary) {
    const slices = (summary && summary.available_slices) || [];
    return slices.length ? slices[0] : null;
  }

  async function enterApp(res) {
    state.sessionSummary = res.summary;
    const first = firstSliceOf(res.summary);
    state.currentSliceId = first ? first.slice_id : null;
    state.currentModality = first ? first.modality : 'gene';
    state.initialized = true;
    state.mode = res.mode || state.mode || 'interactive';
    state.stale = new Set();
    stream.generation = res.generation != null ? res.generation : stream.generation;

    document.getElementById('init-overlay').classList.add('hidden');
    document.getElementById('app').classList.remove('hidden');

    const agentActive = res.agent_active !== false;
    document.getElementById('send-btn').disabled = !agentActive;
    document.getElementById('chat-input').placeholder = agentActive ? 'Ask about your data...' : 'Agent not available (no API key)';
    const modeSel = document.getElementById('mode-select');
    if (modeSel) modeSel.value = state.mode;

    updateSessionInfo();
    updateSliceTabs();
    updateModalityToggle();
    updateVizModeState();
    setupResizeObserver();
    // history first (sets stream.lastSeq), then the live stream continues from that seq: nothing renders
    // twice; both before the slice data, whose requests wait on the session lock while a block runs
    await restoreChatHistory();
    openEventStream();
    await loadSliceData(state.currentSliceId, state.currentModality);
    updateCelltypeCheckboxes();
    updateDataInfo();
    updateVizModeState();
    api.getGeneList(state.currentSliceId).then(r => { if (r.success) state.geneList = r.genes || []; });
    refreshROIs();
    loadSkillList();
    setTimeout(preloadAllSlices, 500);
  }

  async function tryReconnectSession() {
    try {
      const res = await api.getSessionSummary();
      if (res.success && res.summary) {
        await enterApp(res);
        return true;
      }
    } catch (_) { /* backend not ready or no session */ }
    return false;
  }

  // --- History restore: replay every turn's persisted events through the reducer ---
  async function restoreChatHistory() {
    try {
      const list = await api.listTurns();
      if (!list.success) return;
      const turns = (list.turns || []).slice(-20);
      const el = chatEl();
      if (turns.length) { const welcome = el.querySelector('.chat-welcome'); if (welcome) welcome.remove(); }
      for (const t of turns) {
        const res = await api.getTurnEvents(t.turn_id);
        if (!res.success) continue;
        const live = t.state === 'running' || t.state === 'awaiting_input';
        for (const ev of res.events || []) {
          handleEvent(ev, !live);
          if (ev.seq) stream.lastSeq = Math.max(stream.lastSeq, ev.seq);
        }
        if (live) {
          state.chatStreaming = t.state === 'running';
          if (t.state === 'awaiting_input' && t.pending_prompt) turnUI.pendingPrompt = t.pending_prompt;
          startStatusTimer();
        }
      }
      updateChatButtons();
      scrollChatToBottom();
    } catch (e) { console.warn('history restore failed', e); }
  }

  async function logout() {
    closeEventStream();
    stream.lastSeq = 0; stream.generation = null; stream.seen = new Set();
    turnUI.turnId = null; turnUI.body = null; turnUI.pendingPrompt = null;
    stopStatusTimer(); setStatus('', 'idle'); state.chatStreaming = false;
    state.stale = new Set();
    // Call backend reset to clear memory/session (force: a running turn is stopped, not left running)
    try { await api.resetSession(true); } catch (_) {}

    // Reset state
    state.initialized = false;
    state.sessionSummary = null;
    state.currentSliceId = null;
    state.currentModality = 'gene';
    state.geneList = [];
    state.rois = [];
    state.chatMessages = [];
    state.chatStreaming = false;
    state.canvases = {};
    state.currentAssistantMsg = null;
    state.skillPanelOpen = false;
    state.skills = [];

    // Show init, hide app
    document.getElementById('app').classList.add('hidden');
    document.getElementById('init-overlay').classList.remove('hidden');

    // Clear chat messages
    const chatEl = document.getElementById('chat-messages');
    chatEl.innerHTML = '<div class="chat-welcome"><p>Ask questions about your spatial transcriptomics data.</p><p class="small-info">Examples: "Annotate cell types", "Find spatially variable genes", "Show BRCA1 expression"</p></div>';

    // Restore saved form values
    loadFormFromStorage();
  }

  // ============================================================
  // SECTION: SKILL PANEL
  // ============================================================

  let _skillTooltipEl = null;
  let _skillTooltipTimer = null;

  async function loadSkillList() {
    try {
      const res = await fetch('/api/skills/list');
      const data = await res.json();
      if (data.skills) {
        state.skills = data.skills;
        renderSkillList();
      }
    } catch (e) {
      console.warn('Failed to load skills:', e);
    }
  }

  function toggleSkillPanel() {
    state.skillPanelOpen = !state.skillPanelOpen;
    const panel = document.getElementById('skill-panel');
    if (state.skillPanelOpen) {
      panel.classList.add('open');
    } else {
      panel.classList.remove('open');
      hideSkillTooltip();
    }
  }

  function renderSkillList() {
    const container = document.getElementById('skill-list');
    if (!container) return;
    container.innerHTML = '';
    for (const skill of state.skills) {
      const row = document.createElement('div');
      row.className = 'skill-row';
      row.dataset.slug = skill.slug;

      const label = document.createElement('label');
      label.className = 'skill-toggle';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = skill.enabled;
      cb.addEventListener('change', () => toggleSkill(skill.slug, cb.checked));
      label.appendChild(cb);

      const nameSpan = document.createElement('span');
      nameSpan.className = 'skill-name';
      nameSpan.textContent = skill.title;
      nameSpan.addEventListener('mouseenter', (e) => showSkillTooltip(skill, e));
      nameSpan.addEventListener('mouseleave', () => scheduleHideTooltip());

      const infoBtn = document.createElement('button');
      infoBtn.className = 'skill-info-btn';
      infoBtn.title = 'View full details';
      infoBtn.innerHTML = '&#8505;';
      infoBtn.addEventListener('click', (e) => { e.stopPropagation(); showSkillDetail(skill.slug); });

      row.appendChild(label);
      row.appendChild(nameSpan);
      row.appendChild(infoBtn);
      container.appendChild(row);
    }
  }

  async function toggleSkill(slug, enabled) {
    try {
      await fetch('/api/skills/toggle', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({slug, enabled})
      });
      // Update local state and re-sort: enabled first, then alphabetical
      const skill = state.skills.find(s => s.slug === slug);
      if (skill) skill.enabled = enabled;
      state.skills.sort((a, b) => (a.enabled === b.enabled) ? a.title.localeCompare(b.title) : (b.enabled - a.enabled));
      renderSkillList();
    } catch (e) {
      console.warn('Failed to toggle skill:', e);
    }
  }

  function showSkillTooltip(skill, event) {
    clearTimeout(_skillTooltipTimer);
    hideSkillTooltip();
    const tt = document.createElement('div');
    tt.className = 'skill-tooltip';

    let html = `<div class="tt-title">${escapeHtml(skill.title)}</div>`;
    html += `<div class="tt-desc">${escapeHtml(skill.description)}</div>`;

    if (skill.filter_requirements) {
      const fr = skill.filter_requirements;
      html += '<div class="tt-section"><div class="tt-label">Requirements</div>';
      if (fr.num_slices != null) html += `<span class="tt-tag">slices: ${fr.num_slices}</span> `;
      if (fr.modalities) html += fr.modalities.map(m => `<span class="tt-tag">${escapeHtml(m)}</span>`).join(' ');
      if (fr.data_levels) html += ' ' + fr.data_levels.map(d => `<span class="tt-tag">${escapeHtml(d)}</span>`).join(' ');
      html += '</div>';
    }

    if (skill.prerequisites && skill.prerequisites.length) {
      html += '<div class="tt-section"><div class="tt-label">Prerequisites</div>';
      html += '<ul style="margin:2px 0;padding-left:16px">';
      for (const p of skill.prerequisites) html += `<li>${escapeHtml(p)}</li>`;
      html += '</ul></div>';
    }

    tt.innerHTML = html;
    document.body.appendChild(tt);
    _skillTooltipEl = tt;

    // Position to the right of the skill panel (never overlapping it)
    const panel = document.getElementById('skill-panel');
    const panelRect = panel.getBoundingClientRect();
    const rowRect = event.target.closest('.skill-row').getBoundingClientRect();
    let left = panelRect.right + 6;
    let top = rowRect.top;
    // Keep within viewport
    if (left + 310 > window.innerWidth) left = panelRect.left - 310;
    if (top + tt.offsetHeight > window.innerHeight) top = Math.max(8, window.innerHeight - tt.offsetHeight - 8);
    tt.style.left = left + 'px';
    tt.style.top = top + 'px';
  }

  function scheduleHideTooltip() {
    _skillTooltipTimer = setTimeout(hideSkillTooltip, 200);
  }

  function hideSkillTooltip() {
    clearTimeout(_skillTooltipTimer);
    if (_skillTooltipEl) {
      _skillTooltipEl.remove();
      _skillTooltipEl = null;
    }
  }

  async function showSkillDetail(slug) {
    const modal = document.getElementById('skill-detail-modal');
    const titleEl = document.getElementById('skill-detail-title');
    const bodyEl = document.getElementById('skill-detail-body');
    titleEl.textContent = 'Loading...';
    bodyEl.innerHTML = '<div class="small-info">Loading skill details...</div>';
    modal.classList.remove('hidden');

    try {
      const res = await fetch(`/api/skills/${encodeURIComponent(slug)}/detail`);
      const data = await res.json();
      if (data.error) {
        titleEl.textContent = 'Error';
        bodyEl.innerHTML = `<div class="small-info">${escapeHtml(data.error)}</div>`;
        return;
      }
      titleEl.textContent = data.title || slug;
      bodyEl.innerHTML = renderMarkdownFull(data.content_md || data.body || '');
    } catch (e) {
      titleEl.textContent = 'Error';
      bodyEl.innerHTML = `<div class="small-info">Failed to load: ${escapeHtml(e.message)}</div>`;
    }
  }

  function closeSkillDetail() {
    document.getElementById('skill-detail-modal').classList.add('hidden');
  }

  /** Render markdown for skill detail modal (more complete than chat renderer).
   *  Splits on code fences first so their content is never processed as markdown. */
  function renderMarkdownFull(md) {
    if (!md) return '';

    // Split by code fences: alternating [text, lang, code, text, lang, code, ...]
    const parts = md.split(/(```\w*\n[\s\S]*?```)/g);
    let html = '';
    for (const part of parts) {
      const fenceMatch = part.match(/^```(\w*)\n([\s\S]*?)```$/);
      if (fenceMatch) {
        // Code block — escape content, wrap in <pre><code>, done
        html += `<pre><code>${escapeHtml(fenceMatch[2])}</code></pre>`;
      } else {
        // Regular markdown text — escape then apply formatting
        html += _renderMarkdownText(part);
      }
    }
    return html;
  }

  /** Render a non-code-fence markdown fragment into HTML */
  function _renderMarkdownText(text) {
    let html = escapeHtml(text);
    // Inline code (must come before other inline transforms)
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    // Bold & italic
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Unordered lists
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (m, label, url) => safeLink(label, url));
    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr>');
    // Paragraphs (double newline → break)
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';
    // Cleanup: unwrap block elements from <p>
    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/<p>\s*(<(?:h[123]|ul|ol|hr|pre)[\s>])/g, '$1');
    html = html.replace(/(<\/(?:h[123]|ul|ol|hr|pre)>)\s*<\/p>/g, '$1');
    return html;
  }

  // ============================================================
  // SECTION: LOGS MODAL
  // ============================================================

  let _logsTab = 'turns'; // 'turns' | 'notebook'

  async function openLogsModal() {
    const modal = document.getElementById('logs-modal');
    const pathEl = document.getElementById('logs-path');
    const listEl = document.getElementById('logs-turn-list');
    const detailEl = document.getElementById('logs-detail');
    pathEl.textContent = 'Loading...';
    listEl.innerHTML = '';
    detailEl.innerHTML = '<div class="logs-placeholder">Loading...</div>';
    _logsTab = 'turns';
    _updateLogsNavActive();
    modal.classList.remove('hidden');

    try {
      const res = await fetch('/api/logs/info');
      const data = await res.json();
      if (data.error) {
        pathEl.textContent = 'Error: ' + errText(data.error, 'unknown');
        return;
      }
      pathEl.textContent = data.session_dir || 'No session directory';
      _renderTurnList(data.turns || []);
      detailEl.innerHTML = '<div class="logs-placeholder">Select a turn to view its prompt log.</div>';
    } catch (e) {
      pathEl.textContent = 'Failed to load: ' + e.message;
    }
  }

  function closeLogsModal() {
    document.getElementById('logs-modal').classList.add('hidden');
  }

  function _updateLogsNavActive() {
    document.querySelectorAll('.logs-nav-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === _logsTab);
    });
  }

  function _renderTurnList(turns) {
    const listEl = document.getElementById('logs-turn-list');
    listEl.innerHTML = '';
    if (!turns.length) {
      listEl.innerHTML = '<div class="logs-placeholder" style="margin-top:20px">No turns logged yet.</div>';
      return;
    }
    for (const t of turns) {
      const item = document.createElement('div');
      item.className = 'logs-turn-item';
      item.dataset.filename = t.filename;
      item.innerHTML = `<div class="turn-num">Turn ${t.turn_number} <span class="turn-size">${t.size_kb} KB</span></div>` +
        `<div class="turn-preview">${escapeHtml(t.preview)}</div>`;
      item.addEventListener('click', () => _loadTurnDetail(t.filename, item));
      listEl.appendChild(item);
    }
  }

  async function _loadTurnDetail(filename, itemEl) {
    // Highlight active
    document.querySelectorAll('.logs-turn-item').forEach(el => el.classList.remove('active'));
    if (itemEl) itemEl.classList.add('active');

    const detailEl = document.getElementById('logs-detail');
    detailEl.innerHTML = '<div class="logs-placeholder">Loading...</div>';

    try {
      const res = await fetch(`/api/logs/turn/${encodeURIComponent(filename)}`);
      const data = await res.json();
      if (data.error) {
        detailEl.innerHTML = `<div class="logs-placeholder">${escapeHtml(errText(data.error, 'unknown'))}</div>`;
        return;
      }
      detailEl.innerHTML = '<div class="markdown-body">' + renderMarkdownFull(data.content) + '</div>';
    } catch (e) {
      detailEl.innerHTML = `<div class="logs-placeholder">Failed: ${escapeHtml(e.message)}</div>`;
    }
  }

  async function _loadNotebookView() {
    const detailEl = document.getElementById('logs-detail');
    detailEl.innerHTML = '<div class="logs-placeholder">Loading notebook...</div>';
    // Deselect turns
    document.querySelectorAll('.logs-turn-item').forEach(el => el.classList.remove('active'));

    try {
      const res = await fetch('/api/logs/notebook');
      const data = await res.json();
      if (data.error) {
        detailEl.innerHTML = `<div class="logs-placeholder">${escapeHtml(errText(data.error, 'unknown'))}</div>`;
        return;
      }
      detailEl.innerHTML = _renderNotebook(data.notebook || data);
    } catch (e) {
      detailEl.innerHTML = `<div class="logs-placeholder">Failed: ${escapeHtml(e.message)}</div>`;
    }
  }

  function _renderNotebook(nb) {
    if (!nb || !nb.cells || !nb.cells.length) return '<div class="logs-placeholder">Notebook is empty.</div>';
    let html = '';
    for (let i = 0; i < nb.cells.length; i++) {
      const cell = nb.cells[i];
      const src = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
      if (!src.trim()) continue;

      if (cell.cell_type === 'markdown') {
        html += `<div class="nb-cell"><div class="nb-markdown-cell">${renderMarkdownFull(src)}</div></div>`;
        continue;
      }

      // Code cell
      html += '<div class="nb-cell">';
      html += `<div class="nb-cell-header"><span class="nb-cell-type">In [${cell.execution_count || ' '}]</span></div>`;
      html += `<div class="nb-cell-source">${escapeHtml(src)}</div>`;

      // Outputs
      if (cell.outputs && cell.outputs.length) {
        html += '<div class="nb-cell-outputs">';
        for (const out of cell.outputs) {
          if (out.output_type === 'stream') {
            const text = Array.isArray(out.text) ? out.text.join('') : (out.text || '');
            const cls = out.name === 'stderr' ? ' class="nb-stderr"' : '';
            html += `<span${cls}>${escapeHtml(text)}</span>`;
          } else if (out.output_type === 'display_data' || out.output_type === 'execute_result') {
            const imgData = out.data && out.data['image/png'];
            if (imgData) {
              html += `<img src="data:image/png;base64,${imgData}" alt="plot">`;
            }
            const textData = out.data && out.data['text/plain'];
            if (textData && !imgData) {
              const t = Array.isArray(textData) ? textData.join('') : textData;
              html += `<span>${escapeHtml(t)}</span>`;
            }
          } else if (out.output_type === 'error') {
            const tb = Array.isArray(out.traceback) ? out.traceback.join('\n') : (out.traceback || '');
            // Strip ANSI escape codes
            html += `<span class="nb-stderr">${escapeHtml(tb.replace(/\x1b\[[0-9;]*m/g, ''))}</span>`;
          }
        }
        html += '</div>';
      }
      html += '</div>';
    }
    return html;
  }

  const CELL_RADIUS = 3;
  const SPOT_SCALE = 0.5;       // spot_diameter * SPOT_SCALE = canvas radius
  const MIN_ZOOM = 0.1;
  const MAX_ZOOM = 50;
  const ZOOM_FACTOR = 1.15;

  // Viridis-like color LUT (256 entries)
  let VIRIDIS_LUT = null;

  function buildViridisLUT() {
    // Simplified viridis: dark purple -> teal -> yellow
    const stops = [
      [68, 1, 84],
      [59, 82, 139],
      [33, 145, 140],
      [94, 201, 98],
      [253, 231, 37]
    ];
    const lut = new Array(256);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      const seg = t * (stops.length - 1);
      const idx = Math.min(Math.floor(seg), stops.length - 2);
      const frac = seg - idx;
      const a = stops[idx], b = stops[idx + 1];
      lut[i] = `rgb(${Math.round(a[0] + (b[0]-a[0])*frac)},${Math.round(a[1] + (b[1]-a[1])*frac)},${Math.round(a[2] + (b[2]-a[2])*frac)})`;
    }
    VIRIDIS_LUT = lut;
  }

  // ============================================================
  // SECTION 2: UTILITY FUNCTIONS
  // ============================================================

  function getCanvasKey(sliceId, modality) {
    return `slice_${sliceId}_${modality}`;
  }

  function getCurrentCanvas() {
    const key = getCanvasKey(state.currentSliceId, state.currentModality);
    if (!state.canvases[key]) {
      state.canvases[key] = createEmptyCanvas();
    }
    return state.canvases[key];
  }

  function createEmptyCanvas() {
    return {
      image: null, imageWidth: 0, imageHeight: 0, isScatter: false,
      viewX: 0, viewY: 0, viewWidth: 0, viewHeight: 0, zoom: 1.0,
      cells: null, nCells: 0, hasCelltype: false,
      celltypes: [], celltypeColors: {}, selectedCelltypes: new Set(),
      geneExpression: null, geneExpressionRange: null,
      customVmax: null,  // user-controlled max for color scale (null = auto)
      isSpotData: false, spotInfo: null,
      // Per-slice view settings
      vizMode: 'celltype',    // 'celltype' | 'gene' | 'proportion'
      selectedGene: null,
      opacity: 0.8,           // cell overlay opacity (0-1)
      pointSize: 1.0,         // point size multiplier (0.2-3)
      hideBackground: false,  // hide tissue image
      proportionCelltype: null, // single celltype for proportion heatmap (null = pie chart)
      loaded: false, loading: false,
    };
  }

  function getCurrentROIs() {
    return state.rois.filter(r =>
      r.slice_id === state.currentSliceId &&
      r.modality === state.currentModality
    );
  }

  // Per-canvas accessors for settings that used to be global
  function getVizMode() { const c = getCurrentCanvas(); return c ? c.vizMode : 'celltype'; }
  function getSelectedGene() { const c = getCurrentCanvas(); return c ? c.selectedGene : null; }

  function imageToCanvas(imgX, imgY) {
    const c = getCurrentCanvas();
    const el = document.getElementById('spatial-canvas');
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;
    return { x: (imgX - c.viewX) * sx, y: (imgY - c.viewY) * sy };
  }

  function canvasToImage(cx, cy) {
    const c = getCurrentCanvas();
    const el = document.getElementById('spatial-canvas');
    const sx = c.viewWidth / el.width;
    const sy = c.viewHeight / el.height;
    return { x: c.viewX + cx * sx, y: c.viewY + cy * sy };
  }

  function debounce(fn, ms) {
    let timer;
    return function(...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  // Global color registry — ensures same celltype name = same color across all slices
  const _globalCelltypeColors = {};

  function getOrAssignColor(name, canvasColors) {
    // 1. Backend authority — colour from this overlay response wins, refresh the cache
    if (canvasColors && canvasColors[name]) {
      _globalCelltypeColors[name] = canvasColors[name];
      return canvasColors[name];
    }
    // 2. Reuse globally remembered colour so the same celltype stays consistent across canvases
    if (_globalCelltypeColors[name]) return _globalCelltypeColors[name];
    // 3. Adopt a colour already assigned on another loaded canvas
    for (const key of Object.keys(state.canvases)) {
      const cc = state.canvases[key].celltypeColors;
      if (cc && cc[name]) {
        _globalCelltypeColors[name] = cc[name];
        return cc[name];
      }
    }
    // 4. Generate a stable colour from name hash
    let hash = 0;
    for (let i = 0; i < name.length; i++) {
      hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
    }
    const hue = ((Math.abs(hash) * 0.618033988749895) % 1) * 360;
    const s = 65 + (Math.abs(hash >> 8) % 20);
    const l = 45 + (Math.abs(hash >> 16) % 15);
    const color = `hsl(${Math.round(hue)}, ${s}%, ${l}%)`;
    _globalCelltypeColors[name] = color;
    return color;
  }

  // Kept for backward compat — delegates to global registry
  function generateColor(name) {
    return getOrAssignColor(name, null);
  }

  // the v2 error envelope is {code, message}; older call sites expected a string
  function errText(x, fallback) {
    if (!x) return fallback || '';
    if (typeof x === 'string') return x;
    return x.message || x.code || fallback || 'Error';
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  // markdown links: only web and mail URLs, never an unescaped quote inside the attribute
  function safeLink(label, url) {
    const u = String(url).trim();
    if (!/^(https?:|mailto:)/i.test(u)) return label;
    return `<a href="${u.replace(/"/g, '&quot;')}" target="_blank" rel="noopener">${label}</a>`;
  }

  function expressionToColor(value, min, max) {
    if (!VIRIDIS_LUT) buildViridisLUT();
    if (max === min) return VIRIDIS_LUT[0];
    const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
    return VIRIDIS_LUT[Math.round(t * 255)];
  }

  function showSpinner(msg) {
    const el = document.getElementById('canvas-spinner');
    el.querySelector('span').textContent = msg || 'Loading...';
    el.classList.remove('hidden');
  }

  function hideSpinner() {
    document.getElementById('canvas-spinner').classList.add('hidden');
  }

  // ============================================================
  // SECTION 3: API CLIENT
  // ============================================================

  const api = {
    async initDataset(params) {
      const res = await fetch('/api/init_dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      return res.json();
    },

    async getSessionSummary() {
      const res = await fetch('/api/session/summary');
      return res.json();
    },

    async getChatHistory() {
      const res = await fetch('/api/chat/history');
      return res.json();
    },

    async resetSession(force) {
      const res = await fetch('/api/session/reset' + (force ? '?force=1' : ''), { method: 'POST' });
      return res.json();
    },

    async getImageData(sliceId, modality) {
      const params = new URLSearchParams();
      if (sliceId != null) params.set('slice_id', sliceId);
      if (modality) params.set('modality', modality);
      const res = await fetch(`/api/image/data?${params}`);
      return res.json();
    },

    async getCellOverlay(sliceId, modality, selectedCelltypes) {
      const params = new URLSearchParams();
      if (sliceId != null) params.set('slice_id', sliceId);
      if (modality) params.set('modality', modality);
      if (selectedCelltypes) params.set('selected_celltypes', selectedCelltypes);
      const res = await fetch(`/api/cells/overlay?${params}`);
      return res.json();
    },

    async getGeneExpression(gene, sliceId) {
      const sid = sliceId != null ? sliceId : state.currentSliceId;
      const params = new URLSearchParams({ gene });
      const res = await fetch(`/api/slices/${sid}/expression?${params}`);
      return res.json();
    },

    async getGeneList(sliceId) {
      const sid = sliceId != null ? sliceId : state.currentSliceId;
      const res = await fetch(`/api/slices/${sid}/genes`);
      return res.json();
    },

    async getSnapshot() {
      const res = await fetch('/api/session/snapshot');
      return res.json();
    },

    async createTurn(body) {
      const res = await fetch('/api/turns', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      return res.json();
    },

    async replyTurn(turnId, decision) {
      const res = await fetch(`/api/turns/${encodeURIComponent(turnId)}/reply`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(decision) });
      return res.json();
    },

    async cancelTurn(turnId) {
      const res = await fetch(`/api/turns/${encodeURIComponent(turnId)}/cancel`, { method: 'POST' });
      return res.json();
    },

    async listTurns() {
      const res = await fetch('/api/turns');
      return res.json();
    },

    async getTurnEvents(turnId) {
      const res = await fetch(`/api/turns/${encodeURIComponent(turnId)}/events`, { headers: { Accept: 'application/json' } });
      return res.json();
    },

    async selectSlice(sliceId) {
      const res = await fetch('/api/select_slice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slice_id: sliceId }),
      });
      return res.json();
    },

    async selectModality(modality) {
      const res = await fetch('/api/select_modality', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modality }),
      });
      return res.json();
    },

    async addROI(name, type, params) {
      const res = await fetch('/api/rois', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, type, params, slice_id: state.currentSliceId }),
      });
      return res.json();
    },

    async listROIs() {
      const res = await fetch('/api/roi/list');
      return res.json();
    },

    async testLLM(config) {
      const res = await fetch('/api/test-llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      return res.json();
    },

    async deleteROI(name) {
      const res = await fetch('/api/roi/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      return res.json();
    },

    async renameROI(oldName, newName) {
      const res = await fetch('/api/roi/rename', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ old_name: oldName, new_name: newName }),
      });
      return res.json();
    },

    async chatAbort() {
      return abortTurn();
    },

    async getNotebookURL() {
      const res = await fetch('/api/notebook/url');
      return res.json();
    },

    saveChatHistory() {
      window.open('/api/export/report.html', '_blank');
    },
  };

  // ============================================================
  // SECTION 4: CANVAS RENDERING ENGINE
  // ============================================================

  function render() {
    const t0 = performance.now();
    const el = document.getElementById('spatial-canvas');
    if (!el) return;
    const ctx = el.getContext('2d');
    const c = getCurrentCanvas();
    if (!c || !c.loaded) {
      ctx.clearRect(0, 0, el.width, el.height);
      return;
    }

    ctx.clearRect(0, 0, el.width, el.height);
    ctx.save();

    // Draw image (or white background)
    if (c.image && !c.hideBackground) {
      ctx.drawImage(
        c.image,
        c.viewX, c.viewY, c.viewWidth, c.viewHeight,
        0, 0, el.width, el.height
      );
    } else {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, el.width, el.height);
    }

    // Draw cells/spots with per-canvas opacity
    ctx.globalAlpha = c.opacity;
    if (c.vizMode === 'celltype') {
      renderCells(ctx, c, el);
    } else if (c.vizMode === 'gene' && c.geneExpression) {
      renderGeneExpression(ctx, c, el);
    } else if (c.vizMode === 'proportion' && c.isSpotData && c.spotInfo) {
      renderProportions(ctx, c, el);
    }
    ctx.globalAlpha = 1.0;

    // Draw ROIs
    renderROIs(ctx, el);

    // Draw in-progress ROI
    if (state.roiDrawingState) {
      renderROIDrawing(ctx, el);
    }

    ctx.restore();

    // Update legend
    updateLegend(c);
    const dt = performance.now() - t0;
    if (dt > 10) console.log(`render: ${dt.toFixed(1)}ms`);
  }

  function renderCells(ctx, c, el) {
    if (!c.cells || !c.cells.x) return;
    const xs = c.cells.x, ys = c.cells.y, cts = c.cells.celltype;
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;
    const rImg = (c.isSpotData && c.spotInfo ? (c.spotInfo.spot_diameter * SPOT_SCALE) : CELL_RADIUS) * c.pointSize * state.coordScale;
    const r = Math.max(rImg * sx, 1);
    const isSquare = c.isSpotData && c.spotInfo && c.spotInfo.spot_shape === 'square';
    const useCircles = !isSquare && r > 3;
    const cullPad = rImg + 1;

    // Batch by color for performance
    const batches = {};
    const vxMin = c.viewX - cullPad, vxMax = c.viewX + c.viewWidth + cullPad;
    const vyMin = c.viewY - cullPad, vyMax = c.viewY + c.viewHeight + cullPad;

    for (let i = 0; i < xs.length; i++) {
      const x = xs[i], y = ys[i];
      if (x < vxMin || x > vxMax || y < vyMin || y > vyMax) continue;

      const ct = cts ? cts[i] : null;
      if (ct && c.selectedCelltypes.size > 0 && !c.selectedCelltypes.has(ct)) continue;

      const color = (ct && c.celltypeColors[ct]) || 'rgb(150,150,150)';
      if (!batches[color]) batches[color] = [];
      batches[color].push((x - c.viewX) * sx, (y - c.viewY) * sy);
    }

    if (useCircles) {
      for (const [color, coords] of Object.entries(batches)) {
        ctx.beginPath();
        for (let i = 0; i < coords.length; i += 2) {
          ctx.moveTo(coords[i] + r, coords[i+1]);
          ctx.arc(coords[i], coords[i+1], r, 0, Math.PI * 2);
        }
        ctx.fillStyle = color;
        ctx.fill();
      }
    } else {
      const size = Math.max(Math.round(r * 2), 1);
      for (const [color, coords] of Object.entries(batches)) {
        ctx.fillStyle = color;
        for (let i = 0; i < coords.length; i += 2) {
          ctx.fillRect(coords[i] - r, coords[i+1] - r, size, size);
        }
      }
    }
  }

  function renderGeneExpression(ctx, c, el) {
    const ge = c.geneExpression;
    if (!ge || !ge.x) return;
    const xs = ge.x, ys = ge.y, expr = ge.expression;
    const [emin, rawMax] = c.geneExpressionRange || [0, 1];
    const emax = c.customVmax != null ? c.customVmax : rawMax;
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;
    const rImg = (c.isSpotData && c.spotInfo ? (c.spotInfo.spot_diameter * SPOT_SCALE) : CELL_RADIUS) * c.pointSize * state.coordScale;
    const r = Math.max(rImg * sx, 1);
    const isSquare = c.isSpotData && c.spotInfo && c.spotInfo.spot_shape === 'square';
    const useCircles = !isSquare && r > 3;
    const cullPad = rImg + 1;
    const vxMin = c.viewX - cullPad, vxMax = c.viewX + c.viewWidth + cullPad;
    const vyMin = c.viewY - cullPad, vyMax = c.viewY + c.viewHeight + cullPad;

    // Bucket by color index for batching
    if (!VIRIDIS_LUT) buildViridisLUT();
    const buckets = new Array(256);
    for (let b = 0; b < 256; b++) buckets[b] = [];

    for (let i = 0; i < xs.length; i++) {
      const x = xs[i], y = ys[i];
      if (x < vxMin || x > vxMax || y < vyMin || y > vyMax) continue;

      const t = emax === emin ? 0 : (expr[i] - emin) / (emax - emin);
      const bi = Math.max(0, Math.min(255, Math.round(t * 255)));
      buckets[bi].push((x - c.viewX) * sx, (y - c.viewY) * sy);
    }

    const size = Math.max(Math.round(r * 2), 1);
    for (let b = 0; b < 256; b++) {
      const coords = buckets[b];
      if (coords.length === 0) continue;
      ctx.fillStyle = VIRIDIS_LUT[b];
      if (useCircles) {
        ctx.beginPath();
        for (let i = 0; i < coords.length; i += 2) {
          ctx.moveTo(coords[i] + r, coords[i+1]);
          ctx.arc(coords[i], coords[i+1], r, 0, Math.PI * 2);
        }
        ctx.fill();
      } else {
        for (let i = 0; i < coords.length; i += 2) {
          ctx.fillRect(coords[i] - r, coords[i+1] - r, size, size);
        }
      }
    }
  }

  function renderProportions(ctx, c, el) {
    if (!c.cells || !c.spotInfo || !c.spotInfo.deconv_weights) return;

    // Single-celltype heatmap mode
    if (c.proportionCelltype) {
      renderProportionHeatmap(ctx, c, el);
      return;
    }

    // Pie chart / stacked-bar mode
    const xs = c.cells.x, ys = c.cells.y;
    const weights = c.spotInfo.deconv_weights;
    const order = c.spotInfo.celltype_order || Object.keys(weights);
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;
    const rImg = (c.spotInfo.spot_diameter ? (c.spotInfo.spot_diameter * SPOT_SCALE) : 8) * c.pointSize * state.coordScale;
    const r = Math.max(rImg * sx, 2);
    const isSquare = c.spotInfo.spot_shape === 'square';

    for (let i = 0; i < xs.length; i++) {
      const x = xs[i], y = ys[i];
      if (x < c.viewX || x > c.viewX + c.viewWidth ||
          y < c.viewY || y > c.viewY + c.viewHeight) continue;

      const cx = (x - c.viewX) * sx;
      const cy = (y - c.viewY) * sy;

      let total = 0;
      for (const ct of order) {
        total += (weights[ct] ? (weights[ct][i] || 0) : 0);
      }
      if (total === 0) continue;

      if (isSquare) {
        // Stacked horizontal bars inside the square
        const size = r * 2;
        const left = cx - r, top = cy - r;
        let xOff = 0;
        for (const ct of order) {
          const val = weights[ct] ? (weights[ct][i] || 0) : 0;
          const prop = val / total;
          if (prop < 0.01) continue;
          const w = prop * size;
          ctx.fillStyle = c.celltypeColors[ct] || '#999';
          ctx.fillRect(left + xOff, top, w, size);
          xOff += w;
        }
        ctx.strokeStyle = 'rgba(0,0,0,0.2)';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(left, top, size, size);
      } else {
        // Pie chart
        let startAngle = -Math.PI / 2;
        for (const ct of order) {
          const val = weights[ct] ? (weights[ct][i] || 0) : 0;
          const prop = val / total;
          if (prop < 0.01) { startAngle += prop * Math.PI * 2; continue; }
          const sliceAngle = prop * Math.PI * 2;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.arc(cx, cy, r, startAngle, startAngle + sliceAngle);
          ctx.closePath();
          ctx.fillStyle = c.celltypeColors[ct] || '#999';
          ctx.fill();
          startAngle += sliceAngle;
        }
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0,0,0,0.2)';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }
  }

  function renderProportionHeatmap(ctx, c, el) {
    const ct = c.proportionCelltype;
    const weights = c.spotInfo.deconv_weights;
    if (!weights[ct]) return;
    const xs = c.cells.x, ys = c.cells.y;
    const vals = weights[ct];
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;
    const rImg = (c.isSpotData && c.spotInfo ? (c.spotInfo.spot_diameter * SPOT_SCALE) : CELL_RADIUS) * c.pointSize * state.coordScale;
    const r = Math.max(rImg * sx, 1);
    const isSquare = c.isSpotData && c.spotInfo && c.spotInfo.spot_shape === 'square';
    const useCircles = !isSquare && r > 3;
    const vmax = c.customVmax != null ? c.customVmax : 1.0;
    const cullPad = rImg + 1;
    const vxMin = c.viewX - cullPad, vxMax = c.viewX + c.viewWidth + cullPad;
    const vyMin = c.viewY - cullPad, vyMax = c.viewY + c.viewHeight + cullPad;

    if (!VIRIDIS_LUT) buildViridisLUT();
    const buckets = new Array(256);
    for (let b = 0; b < 256; b++) buckets[b] = [];

    for (let i = 0; i < xs.length; i++) {
      const x = xs[i], y = ys[i];
      if (x < vxMin || x > vxMax || y < vyMin || y > vyMax) continue;
      const val = vals[i] || 0;
      const t = vmax === 0 ? 0 : Math.min(1, val / vmax);
      const bi = Math.max(0, Math.min(255, Math.round(t * 255)));
      buckets[bi].push((x - c.viewX) * sx, (y - c.viewY) * sy);
    }

    const size = Math.max(Math.round(r * 2), 1);
    for (let b = 0; b < 256; b++) {
      const coords = buckets[b];
      if (coords.length === 0) continue;
      ctx.fillStyle = VIRIDIS_LUT[b];
      if (useCircles) {
        ctx.beginPath();
        for (let i = 0; i < coords.length; i += 2) {
          ctx.moveTo(coords[i] + r, coords[i+1]);
          ctx.arc(coords[i], coords[i+1], r, 0, Math.PI * 2);
        }
        ctx.fill();
      } else {
        for (let i = 0; i < coords.length; i += 2) {
          ctx.fillRect(coords[i] - r, coords[i+1] - r, size, size);
        }
      }
    }
  }

  function renderROIs(ctx, el) {
    const rois = getCurrentROIs();
    const c = getCurrentCanvas();
    if (!c || rois.length === 0) return;
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;

    for (const roi of rois) {
      const isHighlighted = state.highlightedROI === roi.name;
      ctx.save();

      // Highlighted ROI: solid thick line, stronger fill
      if (isHighlighted) {
        ctx.setLineDash([]);
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(220, 38, 38, 0.9)';
        ctx.fillStyle = 'rgba(220, 38, 38, 0.12)';
      } else {
        ctx.setLineDash([6, 3]);
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.9)';
        ctx.fillStyle = 'rgba(245, 158, 11, 0.1)';
      }

      if (roi.type === 'bbox' && roi.bounds) {
        const [x1, y1, x2, y2] = roi.bounds;
        const rx = (x1 - c.viewX) * sx;
        const ry = (y1 - c.viewY) * sy;
        const rw = (x2 - x1) * sx;
        const rh = (y2 - y1) * sy;
        ctx.fillRect(rx, ry, rw, rh);
        ctx.strokeRect(rx, ry, rw, rh);
      } else if (roi.vertices && roi.vertices.length > 0) {
        ctx.beginPath();
        const [fx, fy] = roi.vertices[0];
        ctx.moveTo((fx - c.viewX) * sx, (fy - c.viewY) * sy);
        for (let i = 1; i < roi.vertices.length; i++) {
          const [vx, vy] = roi.vertices[i];
          ctx.lineTo((vx - c.viewX) * sx, (vy - c.viewY) * sy);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // Label — with background for readability
      if (roi.name) {
        const bx = roi.bounds ? roi.bounds[0] : (roi.vertices ? roi.vertices[0][0] : 0);
        const by = roi.bounds ? roi.bounds[1] : (roi.vertices ? roi.vertices[0][1] : 0);
        const lx = (bx - c.viewX) * sx + 4;
        const ly = (by - c.viewY) * sy - 6;
        ctx.setLineDash([]);
        ctx.font = 'bold 12px sans-serif';
        const textW = ctx.measureText(roi.name).width;
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.fillRect(lx - 2, ly - 11, textW + 4, 14);
        ctx.fillStyle = isHighlighted ? 'rgba(220, 38, 38, 1)' : 'rgba(180, 100, 0, 1)';
        ctx.fillText(roi.name, lx, ly);
      }

      ctx.restore();
    }
  }

  function renderROIDrawing(ctx, el) {
    const ds = state.roiDrawingState;
    const c = getCurrentCanvas();
    if (!ds || !c) return;
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;

    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(220, 38, 38, 0.9)';
    ctx.fillStyle = 'rgba(220, 38, 38, 0.08)';

    if (ds.type === 'bbox' && ds.start && ds.current) {
      const x1 = Math.min(ds.start.x, ds.current.x);
      const y1 = Math.min(ds.start.y, ds.current.y);
      const x2 = Math.max(ds.start.x, ds.current.x);
      const y2 = Math.max(ds.start.y, ds.current.y);
      const rx = (x1 - c.viewX) * sx;
      const ry = (y1 - c.viewY) * sy;
      const rw = (x2 - x1) * sx;
      const rh = (y2 - y1) * sy;
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeRect(rx, ry, rw, rh);
    } else if ((ds.type === 'polygon' || ds.type === 'freehand') && ds.vertices.length > 0) {
      ctx.beginPath();
      const first = ds.vertices[0];
      ctx.moveTo((first.x - c.viewX) * sx, (first.y - c.viewY) * sy);
      for (let i = 1; i < ds.vertices.length; i++) {
        ctx.lineTo((ds.vertices[i].x - c.viewX) * sx, (ds.vertices[i].y - c.viewY) * sy);
      }
      if (ds.cursorPos && ds.type === 'polygon') {
        ctx.lineTo((ds.cursorPos.x - c.viewX) * sx, (ds.cursorPos.y - c.viewY) * sy);
      }
      // Auto-close preview if near first vertex (polygon or freehand)
      if (ds.type === 'polygon' && ds.vertices.length >= 3 && ds.cursorPos) {
        const dist = Math.hypot(ds.cursorPos.x - first.x, ds.cursorPos.y - first.y);
        if (dist < 80) {
          ctx.lineTo((first.x - c.viewX) * sx, (first.y - c.viewY) * sy);
        }
      }
      if (ds.type === 'freehand' && ds.vertices.length >= 5) {
        const lastV = ds.vertices[ds.vertices.length - 1];
        const distF = Math.hypot(lastV.x - first.x, lastV.y - first.y);
        if (distF < 80) {
          ctx.lineTo((first.x - c.viewX) * sx, (first.y - c.viewY) * sy);
        }
      }
      ctx.stroke();
      if (ds.vertices.length > 2) {
        ctx.closePath();
        ctx.fill();
      }
      // Draw vertices for polygon
      if (ds.type === 'polygon') {
        for (let i = 0; i < ds.vertices.length; i++) {
          const v = ds.vertices[i];
          const isFirst = i === 0 && ds.vertices.length >= 3;
          ctx.beginPath();
          ctx.arc((v.x - c.viewX) * sx, (v.y - c.viewY) * sy, isFirst ? 5 : 3, 0, Math.PI * 2);
          ctx.fillStyle = isFirst ? 'rgba(220, 38, 38, 1)' : 'rgba(220, 38, 38, 0.7)';
          ctx.fill();
          // Snap indicator on first vertex
          if (isFirst && ds.cursorPos) {
            const dist = Math.hypot(ds.cursorPos.x - v.x, ds.cursorPos.y - v.y);
            if (dist < 80) {
              ctx.beginPath();
              ctx.arc((v.x - c.viewX) * sx, (v.y - c.viewY) * sy, 8, 0, Math.PI * 2);
              ctx.strokeStyle = 'rgba(220, 38, 38, 0.5)';
              ctx.setLineDash([]);
              ctx.lineWidth = 1.5;
              ctx.stroke();
            }
          }
        }
      }
    }

    ctx.restore();
  }

  // ============================================================
  // SECTION 5: LEGEND
  // ============================================================

  function updateLegend(c) {
    // Legend removed — celltypes are shown in the controls panel
  }

  // ============================================================
  // SECTION 6: INTERACTION HANDLERS (ZOOM / PAN / ROI)
  // ============================================================

  function setupCanvasInteraction() {
    const el = document.getElementById('spatial-canvas');
    const container = document.getElementById('canvas-container');

    // Wheel zoom
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      const c = getCurrentCanvas();
      if (!c || !c.loaded) return;

      const rect = el.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (el.width / rect.width);
      const my = (e.clientY - rect.top) * (el.height / rect.height);
      const imgPos = canvasToImage(mx, my);

      const factor = e.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
      const newWidth = c.viewWidth / factor;
      const newHeight = c.viewHeight / factor;
      const newZoom = c.imageWidth / newWidth;
      if (newZoom < MIN_ZOOM || newZoom > MAX_ZOOM) return;

      c.viewX = imgPos.x - mx * (newWidth / el.width);
      c.viewY = imgPos.y - my * (newHeight / el.height);
      c.viewWidth = newWidth;
      c.viewHeight = newHeight;
      c.zoom = newZoom;

      render();
      updateZoomDisplay();
    }, { passive: false });

    // Mouse down
    el.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      const c = getCurrentCanvas();
      if (!c || !c.loaded) return;

      const rect = el.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (el.width / rect.width);
      const my = (e.clientY - rect.top) * (el.height / rect.height);
      const imgPos = canvasToImage(mx, my);

      if (state.roiTool === 'bbox') {
        state.roiDrawingState = { type: 'bbox', start: imgPos, current: imgPos };
        return;
      }

      if (state.roiTool === 'freehand') {
        state.roiDrawingState = { type: 'freehand', vertices: [imgPos] };
        return;
      }

      // Pan
      if (!state.roiTool) {
        state.isPanning = true;
        state.panStart = { mx, my, vx: c.viewX, vy: c.viewY };
        container.classList.add('panning');
      }
    });

    // Mouse move
    el.addEventListener('mousemove', (e) => {
      const c = getCurrentCanvas();
      if (!c || !c.loaded) return;
      const rect = el.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (el.width / rect.width);
      const my = (e.clientY - rect.top) * (el.height / rect.height);
      const imgPos = canvasToImage(mx, my);

      // Coordinate display
      document.getElementById('cursor-coords').textContent =
        `(${Math.round(imgPos.x)}, ${Math.round(imgPos.y)})`;

      if (state.isPanning && state.panStart) {
        const dx = (mx - state.panStart.mx) * c.viewWidth / el.width;
        const dy = (my - state.panStart.my) * c.viewHeight / el.height;
        c.viewX = state.panStart.vx - dx;
        c.viewY = state.panStart.vy - dy;
        render();
        return;
      }

      if (state.roiDrawingState) {
        if (state.roiDrawingState.type === 'bbox') {
          state.roiDrawingState.current = imgPos;
          render();
        } else if (state.roiDrawingState.type === 'polygon') {
          state.roiDrawingState.cursorPos = imgPos;
          render();
        } else if (state.roiDrawingState.type === 'freehand') {
          // Sample points while dragging (throttle to avoid too many)
          const verts = state.roiDrawingState.vertices;
          const last = verts[verts.length - 1];
          const dist = Math.hypot(imgPos.x - last.x, imgPos.y - last.y);
          if (dist > 3) { // min distance between points
            verts.push(imgPos);
            render();
          }
        }
      }
    });

    // Mouse up
    el.addEventListener('mouseup', (e) => {
      if (state.isPanning) {
        state.isPanning = false;
        state.panStart = null;
        container.classList.remove('panning');
        return;
      }

      // Complete bbox ROI
      if (state.roiDrawingState && state.roiDrawingState.type === 'bbox') {
        const ds = state.roiDrawingState;
        const minX = Math.min(ds.start.x, ds.current.x);
        const minY = Math.min(ds.start.y, ds.current.y);
        const maxX = Math.max(ds.start.x, ds.current.x);
        const maxY = Math.max(ds.start.y, ds.current.y);
        if (Math.abs(maxX - minX) > 5 && Math.abs(maxY - minY) > 5) {
          completeROI('bbox', { min_x: minX, min_y: minY, max_x: maxX, max_y: maxY });
        }
        state.roiDrawingState = null;
        render();
      }

      // Complete freehand ROI on mouse up
      if (state.roiDrawingState && state.roiDrawingState.type === 'freehand') {
        const ds = state.roiDrawingState;
        if (ds.vertices.length >= 5) {
          // Simplify path: take every Nth point to reduce vertex count
          const step = Math.max(1, Math.floor(ds.vertices.length / 100));
          const simplified = [];
          for (let i = 0; i < ds.vertices.length; i += step) {
            simplified.push([ds.vertices[i].x, ds.vertices[i].y]);
          }
          // Ensure last point is included
          const lastV = ds.vertices[ds.vertices.length - 1];
          const lastS = simplified[simplified.length - 1];
          if (lastS[0] !== lastV.x || lastS[1] !== lastV.y) {
            simplified.push([lastV.x, lastV.y]);
          }
          completeROI('polygon', { vertices: simplified });
        }
        state.roiDrawingState = null;
        render();
      }
    });

    // Click for polygon
    el.addEventListener('click', (e) => {
      if (state.roiTool !== 'polygon') return;
      const c = getCurrentCanvas();
      if (!c || !c.loaded) return;
      const rect = el.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (el.width / rect.width);
      const my = (e.clientY - rect.top) * (el.height / rect.height);
      const imgPos = canvasToImage(mx, my);

      if (!state.roiDrawingState) {
        state.roiDrawingState = { type: 'polygon', vertices: [imgPos], cursorPos: null };
      } else {
        // Check if close to first vertex (complete polygon)
        const first = state.roiDrawingState.vertices[0];
        const dist = Math.hypot(imgPos.x - first.x, imgPos.y - first.y);
        if (state.roiDrawingState.vertices.length >= 3 && dist < 80) {
          const verts = state.roiDrawingState.vertices.map(v => [v.x, v.y]);
          completeROI('polygon', { vertices: verts });
          state.roiDrawingState = null;
          render();
        } else {
          state.roiDrawingState.vertices.push(imgPos);
        }
      }
      render();
    });

    // Double-click completes polygon
    el.addEventListener('dblclick', (e) => {
      if (state.roiTool === 'polygon' && state.roiDrawingState &&
          state.roiDrawingState.vertices.length >= 3) {
        e.preventDefault();
        const verts = state.roiDrawingState.vertices.map(v => [v.x, v.y]);
        completeROI('polygon', { vertices: verts });
        state.roiDrawingState = null;
        render();
      }
    });
  }

  async function completeROI(type, params) {
    const name = prompt('ROI name:');
    if (!name) return;
    try {
      const result = await api.addROI(name, type, params);
      if (result.success) {
        await refreshROIs();
        render();
      } else {
        alert('Failed to create ROI: ' + errText(result.error, 'Unknown error'));
      }
    } catch (err) {
      alert('Error creating ROI: ' + err.message);
    }
    cancelROITool();
  }

  function cancelROITool() {
    state.roiTool = null;
    state.roiDrawingState = null;
    document.getElementById('canvas-container').classList.remove('roi-drawing');
    document.getElementById('roi-bbox-btn').classList.remove('active');
    document.getElementById('roi-polygon-btn').classList.remove('active');
    document.getElementById('roi-freehand-btn').classList.remove('active');
    document.getElementById('roi-cancel-btn').classList.add('hidden');
    render();
  }

  function resetView() {
    const c = getCurrentCanvas();
    const el = document.getElementById('spatial-canvas');
    if (!c || !c.imageWidth || !el) return;
    const aspect = el.width / el.height;
    const imgAspect = c.imageWidth / c.imageHeight;
    if (imgAspect > aspect) {
      c.viewWidth = c.imageWidth;
      c.viewHeight = c.imageWidth / aspect;
      c.viewX = 0;
      c.viewY = -(c.viewHeight - c.imageHeight) / 2;
    } else {
      c.viewHeight = c.imageHeight;
      c.viewWidth = c.imageHeight * aspect;
      c.viewX = -(c.viewWidth - c.imageWidth) / 2;
      c.viewY = 0;
    }
    c.zoom = el.width / c.viewWidth;
    render();
    updateZoomDisplay();
  }

  function updateZoomDisplay() {
    const c = getCurrentCanvas();
    document.getElementById('zoom-level').textContent =
      c ? `${Math.round(c.zoom * 100)}%` : '100%';
  }

  // ============================================================
  // SECTION 7: EVENT STREAM & CHAT ENGINE (protocol v2)
  // ============================================================
  //
  // One EventSource on /api/events carries every event of the session in a
  // single ordered stream (seq). Turns are started with POST /api/turns,
  // clarifications are answered with POST /api/turns/{id}/reply, and the same
  // reducer (handleEvent) renders live events and replayed history.

  const EVENT_TYPES = [
    'turn_started', 'phase', 'plan_updated', 'step_started', 'step_done',
    'filter_complete', 'match_complete', 'skill_selected', 'verify_complete',
    'prompt', 'prompt_answered', 'assumption', 'code_block_started', 'exec_output',
    'code_block_done', 'figure', 'execution_issue', 'reflection', 'text', 'text_delta',
    'provenance_update', 'usage', 'notebook_cell_added', 'advice', 'warning',
    'lock_state', 'session_reset', 'error', 'turn_done', 'turn_error', 'turn_cancelled',
    'method_card', 'unguarded_mode', 'refusal', 'validate_result',
    'draft_offer',
  ];
  const TERMINAL_EVENTS = new Set(['turn_done', 'turn_error', 'turn_cancelled']);
  const PHASE_LABELS = {
    planning: 'Planning', filtering: 'Filtering skills', matching: 'Matching skills',
    synthesizing: 'Describing method', verifying: 'Checking prerequisites', generating: 'Writing code',
    running: 'Running', validating: 'Validating', reflecting: 'Reflecting on error', analysing: 'Analysing results',
    waiting: 'Waiting for you', idle: '',
  };

  const stream = { es: null, lastSeq: 0, connected: false, attempts: 0, watchdog: null, lastFrameAt: 0, generation: null, seen: new Set() };
  // Per-turn rendering state (the assistant bubble currently being built).
  const turnUI = {
    turnId: null, body: null, pendingPrompt: null, startedAt: null, timer: null,
    streamEl: null, streamBuf: '', streamRaf: 0, userScrolledUp: false, unseen: false, execOutputEl: null, blockAnchor: null, planEl: null, statusEl: null, seen: new Set(), lastPhase: 'idle',
  };

  function chatEl() { return document.getElementById('chat-messages'); }

  // ---- connection -------------------------------------------------------------
  function openEventStream() {
    if (stream.es) { try { stream.es.close(); } catch (_) {} }
    const url = '/api/events' + (stream.lastSeq ? `?since=${stream.lastSeq}` : '');
    const es = new EventSource(url);
    stream.es = es;
    stream.lastFrameAt = Date.now();
    es.addEventListener('hello', (e) => {
      stream.connected = true;
      stream.attempts = 0;
      stream.lastFrameAt = Date.now();
      hideReconnectBanner();
      let d = {};
      try { d = JSON.parse(e.data); } catch (_) {}
      if (stream.generation != null && d.generation != null && d.generation !== stream.generation) {
        // the backend session was replaced; start over
        window.location.reload();
        return;
      }
      stream.generation = d.generation;
      if (d.active_turn && d.active_turn.state === 'awaiting_input' && d.active_turn.pending_prompt && !turnUI.pendingPrompt) {
        turnUI.turnId = d.active_turn.turn_id;
        turnUI.pendingPrompt = d.active_turn.pending_prompt;
      }
      updateChatButtons();
    });
    es.addEventListener('ping', () => { stream.lastFrameAt = Date.now(); });
    es.addEventListener('resync', () => { scheduleResync(); });
    for (const t of EVENT_TYPES) {
      es.addEventListener(t, (e) => {
        stream.lastFrameAt = Date.now();
        const seq = parseInt(e.lastEventId || '0', 10);
        if (seq && seq <= stream.lastSeq && stream.seen.has(seq)) return;   // duplicate after reconnect
        if (seq) stream.lastSeq = seq;
        let ev;
        try { ev = JSON.parse(e.data); } catch (err) { console.warn('bad frame', err); return; }
        handleEvent(ev, false);
      });
    }
    es.onerror = () => {
      stream.connected = false;
      stream.attempts += 1;
      showReconnectBanner(`Reconnecting (attempt ${stream.attempts})...`);
      // EventSource reconnects on its own with Last-Event-ID
    };
    if (!stream.watchdog) {
      stream.watchdog = setInterval(() => {
        if (stream.es && Date.now() - stream.lastFrameAt > 45000) {
          showReconnectBanner('Connection stalled, reconnecting...');
          openEventStream();
        }
      }, 10000);
    }
  }

  function closeEventStream() {
    if (stream.es) { try { stream.es.close(); } catch (_) {} stream.es = null; }
    if (stream.watchdog) { clearInterval(stream.watchdog); stream.watchdog = null; }
  }

  function showReconnectBanner(text) {
    const el = document.getElementById('reconnect-banner');
    if (!el) return;
    el.textContent = text;
    el.classList.remove('hidden');
  }
  function hideReconnectBanner() {
    const el = document.getElementById('reconnect-banner');
    if (el) el.classList.add('hidden');
  }

  // a resync replays everything from the snapshot: run at most one at a time and coalesce bursts
  const resyncState = { timer: null, running: false, again: false };
  function scheduleResync() {
    if (resyncState.timer) return;
    resyncState.timer = setTimeout(() => { resyncState.timer = null; resyncFromSnapshot(); }, 300);
  }
  async function resyncFromSnapshot() {
    if (resyncState.running) { resyncState.again = true; return; }
    resyncState.running = true;
    try {
      const snap = await api.getSnapshot();
      if (!snap.success) return;
      state.sessionSummary = snap.summary;
      state.rois = snap.rois || [];
      state.stale = new Set((snap.ledger && snap.ledger.stale) || []);
      updateROIList();
      updateStaleInfo();
      const el = chatEl();
      const wasAtBottom = !turnUI.userScrolledUp, oldTop = el.scrollTop;
      el.innerHTML = '';
      stream.seen = new Set();
      changes.length = 0; assumptions.length = 0;  // the replay below rebuilds the Changes drawer
      turnUI.turnId = null; turnUI.body = null; turnUI.streamEl = null; turnUI.streamRaf = 0; turnUI.planEl = null; turnUI.execOutputEl = null; turnUI.blockAnchor = null;
      await restoreChatHistory();
      if (wasAtBottom) scrollChatToBottom(true); else el.scrollTop = oldTop;  // the rebuilt chat keeps the reader's place
      stream.lastSeq = Math.max(stream.lastSeq, snap.last_seq || 0);
    } catch (e) { console.warn('resync failed', e); }
    finally {
      resyncState.running = false;
      if (resyncState.again) { resyncState.again = false; scheduleResync(); }
    }
  }

  // ---- turn lifecycle ---------------------------------------------------------
  function interfaceState() {
    const c = getCurrentCanvas();
    return {
      current_slice_id: state.currentSliceId,
      current_modality: state.currentModality,
      viz_mode: c ? c.vizMode : null,
      selected_gene: c ? c.selectedGene : null,
      highlighted_roi: state.highlightedROI,
    };
  }

  async function sendChatMessage(text) {
    if (!text || !text.trim()) return;
    text = text.trim();
    if (turnUI.pendingPrompt) {
      return sendReply({ kind: 'answer', text });  // any pending prompt takes typed text (a number picks an option)
    }
    state.chatStreaming = true;
    updateChatButtons();
    try {
      const res = await api.createTurn({
        message: text,
        mode: state.mode,
        client_turn_id: 'c_' + Date.now().toString(36),
        interface_state: interfaceState(),
      });
      if (!res.success) {
        state.chatStreaming = false;
        updateChatButtons();
        const msg = (res.error && res.error.message) || res.error_message || 'Request refused';
        appendSystemNotice(msg, 'error');
      }
    } catch (err) {
      state.chatStreaming = false;
      updateChatButtons();
      appendSystemNotice('Connection error: ' + err.message, 'error');
    }
  }

  async function sendReply(decision) {
    const pending = turnUI.pendingPrompt;
    if (!pending || !turnUI.turnId) return;
    const body = Object.assign({ prompt_id: pending.prompt_id }, decision);
    turnUI.pendingPrompt = null;  // one reply per prompt: a second Enter must not race this one
    state.chatStreaming = true;
    updateChatButtons();
    try {
      const res = await api.replyTurn(turnUI.turnId, body);
      if (!res.success) {
        const msg = (res.error && res.error.message) || 'Reply refused';
        appendSystemNotice(msg, 'error');
        const code = res.error && res.error.code;
        if (code === 'validation' || code === 'prompt_mismatch') { turnUI.pendingPrompt = pending; state.chatStreaming = false; updateChatButtons(); }
        // no_pending_prompt: the turn moved on by itself; its own events set the buttons
      }
    } catch (err) {
      turnUI.pendingPrompt = pending; state.chatStreaming = false; updateChatButtons();
      appendSystemNotice('Connection error: ' + err.message, 'error');
    }
  }

  async function abortTurn() {
    if (!turnUI.turnId) return;
    const id = turnUI.turnId;
    setStatus('Stopping...', 'cancelling');  // before the request: a waiting turn closes synchronously and sets idle
    try { await api.cancelTurn(id); } catch (_) {}
  }

  // ---- the reducer -------------------------------------------------------------
  function handleEvent(ev, replay) {
    const t = ev.type;
    if (ev.seq) {
      if (stream.seen.has(ev.seq)) return;
      stream.seen.add(ev.seq);
      if (stream.seen.size > 20000) stream.seen = new Set([...stream.seen].slice(-10000));
    }
    if (ev.turn_id && ev.turn_id !== turnUI.turnId && t !== 'turn_started' && !turnUI.body) {
      // events for a turn we have no bubble for (e.g. reconnect mid-turn): create one
      beginTurnBubble(ev.turn_id, null);
    }
    switch (t) {
      case 'turn_started':
        beginTurnBubble(ev.turn_id, ev.message);
        turnUI.startedAt = ev.ts ? ev.ts * 1000 : Date.now();
        if (!replay) { state.chatStreaming = true; startStatusTimer(); }
        setStatus('Starting', 'planning');
        break;
      case 'phase':
        turnUI.lastPhase = ev.phase;
        setStatus(PHASE_LABELS[ev.phase] || ev.phase, ev.phase, ev.detail);
        break;
      case 'plan_updated':
        renderPlanCard(ev);
        break;
      case 'step_started':
        if (ev.total > 1) appendToAssistantMsg(`<div class="small-info step-line"><strong>Step ${ev.index}/${ev.total}:</strong> ${escapeHtml(ev.description || '')}</div>`);
        turnUI.execOutputEl = null;
        turnUI.blockAnchor = null;
        break;
      case 'filter_complete':
        appendPipelineLog(`Filter: ${ev.compatible.length} compatible skill(s)${ev.bypassed ? ' (filter disabled)' : ''}${ev.rejected.length ? ', excluded: ' + ev.rejected.map(r => r.slug).join(', ') : ''}`);
        break;
      case 'match_complete':
        appendPipelineLog(ev.matched.length ? `Matched: ${ev.matched.join(', ')}` : 'No registered skill matched');
        break;
      case 'skill_selected':
        appendPipelineLog(`Using skill: ${ev.name || ev.skill}`);
        break;
      case 'verify_complete':
        appendPipelineLog(`Prerequisites: ${ev.decided_by === 'bypass' ? 'not checked' : (ev.verdicts.length ? ev.verdicts.map(v => `${v.prerequisite} - ${v.status}`).join('; ') : 'met')}`);
        break;
      case 'prompt':
        turnUI.pendingPrompt = ev;
        renderPrompt(ev, replay);
        setStatus('Waiting for you', 'waiting');
        if (!replay) { state.chatStreaming = false; updateChatButtons(); }
        break;
      case 'prompt_answered': {
        const card = turnUI.body && turnUI.body.querySelector(`.clarification-ui[data-prompt="${ev.prompt_id}"]`);
        if (card) { card.querySelectorAll('button, input').forEach(b => b.disabled = true); card.classList.add('answered'); }
        const d = ev.decision || {};
        const text = d.answers ? Object.values(d.answers).join(', ') : (d.option_index != null ? `option ${d.option_index}` : (d.answer || ''));
        const who = ev.source === 'user' ? '' : ` <span class="tag">${escapeHtml(ev.source)}</span>`;
        appendToAssistantMsg(`<div class="small-info reply-line">&#8627; ${escapeHtml(text)}${who}</div>`);
        turnUI.pendingPrompt = null;
        if (!replay) { state.chatStreaming = true; updateChatButtons(); }
        break;
      }
      case 'assumption':
        appendToAssistantMsg(`<div class="clarification-ui assumption-card"><div class="question"><strong>Assumed</strong> (${escapeHtml(ev.source || 'auto')}): ${escapeHtml(ev.question)}</div><div class="assumption-answer">&#8627; ${escapeHtml(ev.chosen)}${ev.rationale ? ` <span class="small-info">- ${escapeHtml(ev.rationale)}</span>` : ''}</div></div>`);
        addAssumption(ev);
        break;
      case 'text_delta': {
        if (!turnUI.streamEl) {
          turnUI.streamEl = document.createElement('div');
          turnUI.streamEl.className = 'stream-text';
          turnUI.streamBuf = '';
          appendNode(turnUI.streamEl);
        }
        turnUI.streamBuf += ev.delta;
        if (!turnUI.streamRaf) turnUI.streamRaf = requestAnimationFrame(() => {
          turnUI.streamRaf = 0;
          if (turnUI.streamEl) { turnUI.streamEl.innerHTML = renderStreamPreview(turnUI.streamBuf); scrollChatToBottom(); }
        });
        break;
      }
      case 'text': {
        turnUI.execOutputEl = null;
        turnUI.blockAnchor = null;
        const el = document.createElement('div');
        if (ev.role === 'analysis') {
          el.innerHTML = `<hr class="chat-hr-light"><div><em><strong>Analysis</strong></em></div><div>${renderMarkdown(ev.text)}</div>`;
        } else if (ev.role === 'note') {
          el.innerHTML = `<div class="alert-card advice">${renderMarkdown(ev.text)}</div>`;
        } else {
          el.innerHTML = renderMarkdown(ev.text);
          const seg = segmentIndexOf(ev);
          if (seg != null) { el.dataset.step = ev.step_id || ''; el.dataset.seg = String(seg); }
        }
        if (turnUI.streamEl) { dropStreamPreview(el); } else { appendNode(el); }
        break;
      }
      case 'code_block_started': {
        if (turnUI.streamEl) dropStreamPreview(null);
        const label = ev.attempt > 1 ? `Code (retry ${ev.attempt - 1})` : 'Code';
        const el = document.createElement('details');
        el.className = 'code-details';
        el.dataset.block = ev.block_id;
        el.innerHTML = `<summary>${label}</summary><div class="chat-code-block"><pre>${escapeHtml(ev.code)}</pre></div>`;
        insertSegment(el, ev.step_id, ev.segment_index);
        turnUI.blockAnchor = el;
        turnUI.execOutputEl = null;
        break;
      }
      case 'exec_output': {
        if (!turnUI.execOutputEl) {
          if (!turnUI.body) break;
          const details = document.createElement('details');
          details.className = 'exec-details';
          details.open = true;
          const summary = document.createElement('summary');
          summary.textContent = 'Output';
          details.appendChild(summary);
          const pre = document.createElement('div');
          pre.className = 'exec-output';
          details.appendChild(pre);
          turnUI.execOutputEl = pre;
          insertAfterAnchor(details);
        }
        for (const ln of (ev.lines || [])) {
          if (/^\s*\d+%\|/.test(ln.text) || /\d+\/\d+ \[.*\]/.test(ln.text)) continue;   // progress bars
          const lineEl = document.createElement('div');
          if (ln.stream === 'stderr') lineEl.className = 'stderr';
          lineEl.textContent = ln.text;
          turnUI.execOutputEl.appendChild(lineEl);
        }
        turnUI.execOutputEl.scrollTop = turnUI.execOutputEl.scrollHeight;
        break;
      }
      case 'code_block_done':
        if (turnUI.execOutputEl) { const d = turnUI.execOutputEl.parentElement; if (d && d.tagName === 'DETAILS') d.open = !ev.success; }
        turnUI.execOutputEl = null;
        if (!ev.success && ev.error) {
          const cls = ev.will_retry ? 'warning' : 'error';
          insertHtmlAfterAnchor(`<div class="alert-card ${cls}"><strong>${ev.will_retry ? 'Error (retrying)' : 'Execution Error'}</strong> ${escapeHtml(ev.error.type || '')}<br>${escapeHtml(ev.error.message || '')}</div>`);
        }
        break;
      case 'reflection':
        insertHtmlAfterAnchor(`<div class="small-info" style="color:var(--warning)">Reflecting on the error (attempt ${ev.attempt}/${ev.max_attempts})${ev.summary ? ': ' + escapeHtml(ev.summary) : ''}${ev.will_retry ? ' - retrying' : ''}</div>`);
        break;
      case 'figure': {
        // the size reserves the figure's space before the image loads, so nothing shifts while the reader scrolls
        const size = ev.width && ev.height ? ` width="${ev.width}" height="${ev.height}"` : '';
        insertHtmlAfterAnchor(`<div class="chat-plot"><img src="/api/figures/${encodeURIComponent(ev.figure_id)}.png" alt="Figure" title="Click to enlarge"${size}></div>`);
        break;
      }
      case 'execution_issue':
        if (ev.explanation) appendToAssistantMsg(`<div class="alert-card error"><strong>Execution Issue</strong> (${escapeHtml(ev.issue_type || 'error')})<br>${renderMarkdown(ev.explanation)}</div>`);
        break;
      case 'advice':
        appendToAssistantMsg(`<div class="alert-card advice">${renderMarkdown(ev.message)}</div>`);
        break;
      case 'warning':
        appendToAssistantMsg(`<div class="alert-card warning">${renderMarkdown(ev.message)}</div>`);
        break;
      case 'refusal':
        appendToAssistantMsg(`<div class="alert-card warning"><strong>Cannot run this analysis</strong>${ev.requirement ? ': ' + escapeHtml(ev.requirement) : ''}<br>${renderMarkdown(ev.reason || '')}</div>`);
        break;
      case 'unguarded_mode':
        appendToAssistantMsg(`<div class="alert-card warning"><strong>Provisional mode</strong>: ${escapeHtml(ev.reason)}</div>`);
        break;
      case 'method_card':
        appendPipelineLog(`Method card: ${(ev.card && ev.card.method_name) || 'unregistered method'}`);
        break;
      case 'validate_result':
        if (!ev.passed) appendPipelineLog('Step outputs still missing: ' + ev.failures.join('; '));
        break;
      case 'provenance_update':
        applyProvenanceUpdate(ev, replay);
        break;
      case 'usage':
        turnUI.usage = ev;
        updateStatusLine();
        break;
      case 'error':
        appendToAssistantMsg(`<div class="alert-card error">${escapeHtml(ev.message)}</div>`);
        break;
      case 'turn_done':
      case 'turn_error':
      case 'turn_cancelled':
        if (t === 'turn_error') appendToAssistantMsg(`<div class="alert-card error"><strong>Turn failed</strong>: ${escapeHtml(ev.message || '')}</div>`);
        if (t === 'turn_cancelled') appendToAssistantMsg('<div class="small-info">Stopped.</div>');
        if (t === 'turn_done' && ev.state === 'blocked') appendToAssistantMsg('<div class="small-info">Turn ended early: a step could not run (see the note above).</div>');
        if (turnUI.streamEl) { turnUI.streamEl.remove(); turnUI.streamEl = null; turnUI.streamBuf = ''; }
        finishTurn(ev, replay);
        break;
      case 'session_reset':
        if (!replay) window.location.reload();
        break;
      default:
        break;
    }
    if (!replay) scrollChatToBottom();
  }

  function beginTurnBubble(turnId, message) {
    if (turnUI.turnId === turnId && turnUI.body) return;
    const welcome = document.querySelector('.chat-welcome');
    if (welcome) welcome.remove();
    if (message != null) appendUserMessage(message);
    turnUI.turnId = turnId;
    turnUI.body = createAssistantMessage();
    turnUI.pendingPrompt = null;
    turnUI.streamEl = null;
    turnUI.streamBuf = '';
    turnUI.streamRaf = 0;
    turnUI.userScrolledUp = false;
    turnUI.unseen = false;
    turnUI.execOutputEl = null;
    turnUI.blockAnchor = null;
    turnUI.planEl = null;
    turnUI.usage = null;
    state.currentAssistantMsg = turnUI.body;
  }

  function finishTurn(ev, replay) {
    turnUI.pendingPrompt = null;
    if (turnUI.body) turnUI.body.querySelectorAll('.clarification-ui:not(.answered) button, .clarification-ui:not(.answered) input').forEach(b => { b.disabled = true; });
    stopStatusTimer();
    setStatus('', 'idle');
    if (!replay) {
      state.chatStreaming = false;
      updateChatButtons();
    }
  }

  // ---- status line ---------------------------------------------------------------
  function setStatus(text, phase, detail) {
    turnUI.status = { text, phase, detail };
    updateStatusLine();
  }
  function updateStatusLine() {
    const el = document.getElementById('turn-status');
    if (!el) return;
    const s = turnUI.status || { text: '', phase: 'idle' };
    if (!s.text) { el.innerHTML = ''; el.className = 'turn-status idle'; return; }
    const elapsed = turnUI.startedAt ? Math.round((Date.now() - turnUI.startedAt) / 1000) : 0;
    const mm = String(Math.floor(elapsed / 60)).padStart(2, '0'), ss = String(elapsed % 60).padStart(2, '0');
    const u = turnUI.usage;
    const tokens = u ? ` &middot; ${u.llm_calls} calls, ${(u.tokens_in + u.tokens_out).toLocaleString()} tokens` : '';
    el.className = 'turn-status ' + (s.phase || '');
    el.innerHTML = `<span class="phase-chip">${escapeHtml(s.text)}</span>${s.detail ? ` <span class="small-info">${escapeHtml(s.detail)}</span>` : ''} <span class="small-info">${mm}:${ss}${tokens}</span>`;
  }
  function startStatusTimer() { stopStatusTimer(); turnUI.timer = setInterval(updateStatusLine, 1000); }
  function stopStatusTimer() { if (turnUI.timer) { clearInterval(turnUI.timer); turnUI.timer = null; } }

  // ---- provenance / staleness ------------------------------------------------------
  function applyProvenanceUpdate(ev, replay) {
    state.ledgerVersion = ev.ledger_version;
    if (!state.stale) state.stale = new Set();
    (ev.stale_cleared || []).forEach(id => state.stale.delete(id));
    (ev.stale_now || []).forEach(id => state.stale.add(id));
    if (!replay) {
      handleStateChanges(ev.summary);
      if (ev.origin !== 'gui' && ev.changes && ev.changes.length) {
        const ids = ev.changes.map(c => `${c.op === 'deleted' ? '-' : ''}${c.artifact.slice_id != null ? 's' + c.artifact.slice_id + ':' : ''}${c.artifact.kind}:${c.artifact.key}`);
        appendPipelineLog('Changed: ' + ids.slice(0, 12).join(', ') + (ids.length > 12 ? ` (+${ids.length - 12})` : ''));
      }
      if (ev.stale_now && ev.stale_now.length) {
        appendToAssistantMsg(`<div class="alert-card warning"><strong>Now stale</strong> (upstream changed): ${escapeHtml(ev.stale_now.join(', '))}. <button class="btn-tiny rerun-downstream-btn">Re-run downstream</button></div>`);
        const btn = turnUI.body && turnUI.body.querySelector('.rerun-downstream-btn:not([data-bound])');
        if (btn) { btn.setAttribute('data-bound', '1'); btn.addEventListener('click', () => rerunDownstream(ev.stale_now)); }
      }
    }
    updateStaleInfo();
    addChangeEntry(ev);
  }

  function updateStaleInfo() {
    const el = document.getElementById('stale-info');
    if (!el) return;
    const stale = state.stale ? [...state.stale] : [];
    if (!stale.length) { el.classList.add('hidden'); el.innerHTML = ''; return; }
    el.classList.remove('hidden');
    el.innerHTML = `<span class="stale-badge">${stale.length} stale</span> ${escapeHtml(stale.join(', '))} <button class="btn-tiny" id="rerun-stale-btn">Re-run</button>`;
    el.querySelector('#rerun-stale-btn').addEventListener('click', () => rerunDownstream(stale));
  }

  async function rerunDownstream(artifacts) {
    if (state.chatStreaming) return;
    state.chatStreaming = true;
    updateChatButtons();
    try {
      const res = await api.createTurn({ action: { kind: 'rerun_downstream', artifacts }, mode: state.mode, interface_state: interfaceState() });
      if (!res.success) { state.chatStreaming = false; updateChatButtons(); appendSystemNotice(errText(res.error, 'Re-run refused'), 'error'); }
    } catch (e) { state.chatStreaming = false; updateChatButtons(); appendSystemNotice('Connection error: ' + e.message, 'error'); }
  }

  const changes = [];
  const assumptions = [];
  function addChangeEntry(ev) {
    changes.push(ev);
    renderChangesDrawer();
  }
  function addAssumption(ev) {
    assumptions.push(ev);
    renderChangesDrawer();
  }
  function renderChangesDrawer() {
    const el = document.getElementById('changes-body');
    if (!el) return;
    const rows = changes.slice(-60).reverse().map(ev => {
      const items = (ev.changes || []).map(c => `${c.op === 'deleted' ? '<s>' : ''}${c.artifact.slice_id != null ? 's' + c.artifact.slice_id + ':' : ''}${escapeHtml(c.artifact.kind)}:${escapeHtml(c.artifact.key)}${c.op === 'deleted' ? '</s>' : ''}`).join(', ');
      const stale = (ev.stale_now || []).length ? ` <span class="stale-badge">stale: ${escapeHtml(ev.stale_now.join(', '))}</span>` : '';
      return `<div class="change-row"><span class="tag">v${ev.ledger_version}</span> <span class="tag">${escapeHtml(ev.origin)}</span> ${items || '<em>no artefact changes</em>'}${stale}</div>`;
    }).join('');
    const arows = assumptions.slice(-30).reverse().map(a => `<div class="change-row"><span class="tag">${escapeHtml(a.kind)}</span> ${escapeHtml(a.question)} &#8627; <strong>${escapeHtml(a.chosen)}</strong></div>`).join('');
    el.innerHTML = `<h4>Provenance</h4>${rows || '<div class="small-info">No changes yet.</div>'}<h4>Assumptions</h4>${arows || '<div class="small-info">None.</div>'}`;
  }
  function toggleChangesDrawer() {
    const d = document.getElementById('changes-drawer');
    if (!d) return;
    d.classList.toggle('open');
    renderChangesDrawer();
  }

  // ---- cards -----------------------------------------------------------------------
  function renderPlanCard(ev) {
    if (!turnUI.body) return;
    const steps = ev.steps || [];
    if (steps.length < 2 && !turnUI.planEl) return;
    const html = steps.map(s => `<li class="plan-step ${s.status === 'active' ? 'active' : ''} ${s.status === 'done' ? 'done' : ''} ${s.status === 'failed' ? 'failed' : ''} ${['skipped', 'blocked', 'cancelled'].includes(s.status) ? s.status : ''}" data-step="${s.index}"><span class="plan-step-num">${s.index}</span>${escapeHtml(s.description || '')}${s.skill ? ` <span class="tag">${escapeHtml(s.skill)}</span>` : ''}</li>`).join('');
    if (!turnUI.planEl) {
      turnUI.body.insertAdjacentHTML('beforeend', `<details class="plan-card" open><summary>Plan (${steps.length} steps)</summary><ol class="plan-steps">${html}</ol></details>`);
      turnUI.planEl = turnUI.body.querySelector('.plan-card:last-of-type');
    } else {
      turnUI.planEl.querySelector('.plan-steps').innerHTML = html;
      turnUI.planEl.querySelector('summary').textContent = `Plan (${steps.length} steps)`;
    }
  }

  function renderPrompt(ev, replay) {
    if (!turnUI.body) return;
    let html = `<div class="clarification-ui" data-prompt="${escapeHtml(ev.prompt_id)}">`;
    if (ev.kind === 'skill_selection') {
      html += `<div class="question">${escapeHtml(ev.question || 'Multiple methods match. Which one should I use?')}</div><div class="clarification-options">`;
      html += (ev.options || []).map(o => `<button class="clarification-option-btn" data-index="${o.index}" title="${escapeHtml(o.description || '')}">${o.index}. ${escapeHtml(o.label || o.slug)}</button>`).join('');
      html += '</div>';
    } else if (ev.kind === 'verifier') {
      html += `<div class="question">Prerequisites needed${ev.skill ? ` for <strong>${escapeHtml(ev.skill)}</strong>` : ''}:</div><div class="clarification-options">`;
      html += (ev.questions || []).map(q => `<div class="prereq-row"><label>${escapeHtml(q.text)}</label><input type="text" class="prereq-input" data-qid="${escapeHtml(q.id)}"></div>`).join('');
      html += '<button class="btn-secondary prereq-submit">Submit</button></div>';
    } else {
      html += `<div class="question">${renderMarkdown(ev.question || '')}</div>`;
      if (ev.options && ev.options.length) {
        html += '<div class="clarification-options">' + ev.options.map(o => `<button class="clarification-option-btn" data-index="${o.index}">${escapeHtml(o.label)}</button>`).join('') + '</div>';
      } else {
        html += '<div class="clarification-options"><input type="text" class="clarification-text-input" placeholder="Type your answer (or use the chat box)..."><button class="btn-secondary clarification-submit">Submit</button></div>';
      }
    }
    html += '</div>';
    appendToAssistantMsg(html);
    const card = turnUI.body.querySelector(`.clarification-ui[data-prompt="${ev.prompt_id}"]`);
    if (!card) return;
    if (replay) { card.querySelectorAll('button, input').forEach(b => b.disabled = true); return; }
    card.querySelectorAll('.clarification-option-btn').forEach(btn => btn.addEventListener('click', () => {
      card.querySelectorAll('button').forEach(b => b.disabled = true);
      btn.classList.add('active');
      sendReply({ kind: 'option', index: parseInt(btn.dataset.index, 10) });
    }));
    const submit = card.querySelector('.prereq-submit');
    if (submit) submit.addEventListener('click', () => {
      const answers = {};
      card.querySelectorAll('.prereq-input').forEach(inp => { answers[inp.dataset.qid] = inp.value; });
      card.querySelectorAll('button, input').forEach(b => b.disabled = true);
      sendReply({ kind: 'prerequisites', answers });
    });
    const textSubmit = card.querySelector('.clarification-submit');
    if (textSubmit) {
      const input = card.querySelector('.clarification-text-input');
      const go = () => { if (input.value.trim()) { card.querySelectorAll('button, input').forEach(b => b.disabled = true); sendReply({ kind: 'answer', text: input.value.trim() }); } };
      textSubmit.addEventListener('click', go);
      input.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
      setTimeout(() => input.focus(), 0);
    }
  }

  // --- Chat UI helpers ---

  function appendUserMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-msg user';
    msgDiv.innerHTML = `<div class="chat-msg-body">${escapeHtml(text)}</div>`;
    chatEl().appendChild(msgDiv);
    scrollChatToBottom(true);
  }

  function appendSystemNotice(text, cls) {
    const div = document.createElement('div');
    div.className = 'chat-msg system';
    div.innerHTML = `<div class="chat-msg-body ${cls === 'error' ? 'alert-card error' : 'small-info'}">${escapeHtml(text)}</div>`;
    chatEl().appendChild(div);
    scrollChatToBottom(true);
  }

  function createAssistantMessage() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-msg assistant';
    const body = document.createElement('div');
    body.className = 'chat-msg-body';
    msgDiv.appendChild(body);
    chatEl().appendChild(msgDiv);
    const welcome = document.querySelector('.chat-welcome');
    if (welcome) welcome.remove();
    return body;
  }

  function appendToAssistantMsg(html) {
    if (!turnUI.body) return;
    turnUI.body.insertAdjacentHTML('beforeend', html);
  }
  function appendNode(node) {
    if (!turnUI.body) return;
    turnUI.body.appendChild(node);
  }

  // Prose and code are rendered in the order the model wrote them. Every prose segment and every code block
  // carries its position in the step's generation (segment_index); the prose arrives first, the code blocks
  // as they run, so a block is inserted before the first later prose segment of the same step. Its output,
  // errors, reflections and figures then follow the block (the "anchor") rather than the end of the bubble.
  function segmentIndexOf(ev) {
    if (ev.segment_index != null) return ev.segment_index;
    const m = /-t(\d+)$/.exec(ev.segment_id || '');
    return m ? parseInt(m[1], 10) : null;
  }
  function insertSegment(node, stepId, seg) {
    if (!turnUI.body) return;
    if (seg == null) { turnUI.body.appendChild(node); return; }
    node.dataset.step = stepId || '';
    node.dataset.seg = String(seg);
    const later = Array.from(turnUI.body.children).find(c => c.dataset && c.dataset.seg != null && c.dataset.step === (stepId || '') && parseInt(c.dataset.seg, 10) > seg);
    if (later) turnUI.body.insertBefore(node, later); else turnUI.body.appendChild(node);
  }
  function insertAfterAnchor(node) {
    if (!turnUI.body) return;
    if (turnUI.blockAnchor && turnUI.blockAnchor.parentNode === turnUI.body) { turnUI.blockAnchor.after(node); turnUI.blockAnchor = node; }
    else turnUI.body.appendChild(node);
  }
  function insertHtmlAfterAnchor(html) {
    const tpl = document.createElement('template');
    tpl.innerHTML = html.trim();
    const node = tpl.content.firstElementChild;
    if (node) insertAfterAnchor(node);
  }
  // The streamed preview is replaced in place by the rendered prose (or removed), so the bubble never
  // shrinks and grows back, which would move the reader's position.
  function dropStreamPreview(replacement) {
    if (turnUI.streamRaf) { cancelAnimationFrame(turnUI.streamRaf); turnUI.streamRaf = 0; }
    if (turnUI.streamEl) {
      if (replacement) turnUI.streamEl.replaceWith(replacement); else turnUI.streamEl.remove();
      turnUI.streamEl = null;
    } else if (replacement) {
      appendNode(replacement);
    }
    turnUI.streamBuf = '';
  }
  function openLightbox(src) {
    const lb = document.getElementById('figure-lightbox');
    if (!lb) return;
    lb.querySelector('img').src = src;
    lb.classList.remove('hidden');
  }
  function closeLightbox() {
    const lb = document.getElementById('figure-lightbox');
    if (!lb) return;
    lb.classList.add('hidden');
    lb.querySelector('img').removeAttribute('src');
  }

  function appendPipelineLog(msg) {
    if (!turnUI.body) return;
    let log = turnUI.body.querySelector('.pipeline-log');
    if (!log) {
      log = document.createElement('details');
      log.className = 'pipeline-log';
      log.innerHTML = '<summary>Pipeline details</summary><div class="pipeline-log-content"></div>';
      turnUI.body.appendChild(log);
    }
    log.querySelector('.pipeline-log-content').textContent += msg + '\n';
  }

  function renderMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<div class="chat-code-block"><pre>$2</pre></div>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/(^|\n)######\s+(.+)/g, '$1<h6 class="chat-heading">$2</h6>');
    html = html.replace(/(^|\n)#####\s+(.+)/g, '$1<h5 class="chat-heading">$2</h5>');
    html = html.replace(/(^|\n)####\s+(.+)/g, '$1<h4 class="chat-heading">$2</h4>');
    html = html.replace(/(^|\n)###\s+(.+)/g, '$1<h3 class="chat-heading">$2</h3>');
    html = html.replace(/(^|\n)##\s+(.+)/g, '$1<h2 class="chat-heading">$2</h2>');
    html = html.replace(/(^|\n)#\s+(.+)/g, '$1<h1 class="chat-heading">$2</h1>');
    html = html.replace(/(^|\n)---+(\n|$)/g, '$1<hr class="chat-hr">$2');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (m, label, url) => safeLink(label, url));
    html = html.replace(/((?:^|\n)\|.+\|(?:\n\|.+\|)+)/g, function(tableBlock) {
      const rows = tableBlock.trim().split('\n').filter(r => r.trim());
      if (rows.length < 2) return tableBlock;
      let table = '<table class="chat-table">';
      rows.forEach((row, i) => {
        if (/^\|[\s\-:]+\|/.test(row.replace(/<br>/g, ''))) return;
        const cells = row.split('|').filter((_, j, a) => j > 0 && j < a.length - 1);
        const tag = i === 0 ? 'th' : 'td';
        table += '<tr>' + cells.map(c => `<${tag}>${c.trim()}</${tag}>`).join('') + '</tr>';
      });
      table += '</table>';
      return table;
    });
    html = html.replace(/((?:^|\n)&gt; .+(?:\n&gt; .+)*)/g, function(block) {
      const content = block.trim().split('\n').map(l => l.replace(/^&gt; /, '')).join('<br>');
      return `<blockquote class="chat-blockquote">${content}</blockquote>`;
    });
    html = html.replace(/((?:^|\n)[*\-] .+(?:\n[*\-] .+)*)/g, function(block) {
      const items = block.trim().split('\n').map(l => l.replace(/^[*\-] /, ''));
      return '<ul>' + items.map(i => `<li>${i}</li>`).join('') + '</ul>';
    });
    html = html.replace(/((?:^|\n)\d+\. .+(?:\n\d+\. .+)*)/g, function(block) {
      const items = block.trim().split('\n').map(l => l.replace(/^\d+\. /, ''));
      return '<ol>' + items.map(i => `<li>${i}</li>`).join('') + '</ol>';
    });
    html = html.replace(/\n/g, '<br>');
    return html;
  }

  // Streamed generator text: show prose as it arrives, but never dump raw code into the bubble;
  // an open fence becomes a "Writing code" chip until the block is executed (code_block_started).
  function renderStreamPreview(buf) {
    const parts = String(buf).replace(/^__INTERPRET__:.*$/gm, '').split('```');
    let html = '';
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 0) {
        if (parts[i]) html += escapeHtml(parts[i]);
      } else {
        const closed = i < parts.length - 1;
        const lines = Math.max(parts[i].split('\n').length - 1, 1);
        html += `<span class="stream-code-chip">${closed ? 'Code block' : 'Writing code'} · ${lines} line${lines === 1 ? '' : 's'}${closed ? '' : '…'}</span>`;
      }
    }
    return html;
  }

  // The chat follows new content only while the reader is at the bottom. Any scroll that leaves the bottom,
  // a trackpad step of a few pixels included, stops the following until the reader returns to the bottom or
  // taps "Latest"; sending a message forces it. Our own scrolls land at the bottom, so they never flip the state.
  function chatAtBottom(el) { return el.scrollHeight - el.scrollTop - el.clientHeight <= 4; }
  function scrollChatToBottom(force) {
    const el = chatEl();
    if (!el) return;
    if (!turnUI._scrollBound) {
      turnUI._scrollBound = true;
      el.addEventListener('scroll', () => {
        turnUI.userScrolledUp = !chatAtBottom(el);
        if (!turnUI.userScrolledUp) turnUI.unseen = false;
        updateLatestPill();
      }, { passive: true });
    }
    if (force) { turnUI.userScrolledUp = false; turnUI.unseen = false; }
    if (!turnUI.userScrolledUp) {
      if (!chatAtBottom(el)) el.scrollTop = el.scrollHeight;
    } else {
      turnUI.unseen = true;
    }
    updateLatestPill();
  }
  function updateLatestPill() {
    const pill = document.getElementById('chat-latest');
    if (pill) pill.classList.toggle('hidden', !(turnUI.userScrolledUp && turnUI.unseen));
  }

  function updateChatButtons() {
    const waiting = !!turnUI.pendingPrompt;
    const busy = state.chatStreaming && !waiting;
    document.getElementById('send-btn').classList.toggle('hidden', busy);
    document.getElementById('abort-btn').classList.toggle('hidden', !state.chatStreaming && !waiting);
    const input = document.getElementById('chat-input');
    input.disabled = busy;
    input.placeholder = waiting && turnUI.pendingPrompt.kind !== 'skill_selection' ? 'Type your answer...' : (waiting ? 'Pick an option above or type its number...' : 'Ask about your data...');
  }

  // ============================================================
  // SECTION 8: DATA LOADING & STATE CHANGES
  // ============================================================

  async function loadSliceData(sliceId, modality) {
    const key = getCanvasKey(sliceId, modality);
    console.log(`loadSliceData: ${key}`);
    let c = state.canvases[key];
    if (!c) {
      c = createEmptyCanvas();
      state.canvases[key] = c;
    }
    if (c.loaded || c.loading) { console.log(`  skip: loaded=${c.loaded} loading=${c.loading}`); return; }
    c.loading = true;

    const isCurrentSlice = sliceId === state.currentSliceId && modality === state.currentModality;
    if (isCurrentSlice) showSpinner('Loading slice data...');

    try {
      console.time('fetch-data');
      // Fetch image and cells in parallel
      const [imgRes, cellRes] = await Promise.all([
        api.getImageData(sliceId, modality),
        api.getCellOverlay(sliceId, modality),
      ]);
      console.timeEnd('fetch-data');

      // Process image
      console.time('decode-image');
      if (imgRes.success && (imgRes.image_data || imgRes.is_scatter)) {
        c.imageWidth = imgRes.image_width;
        c.imageHeight = imgRes.image_height;
        c.isScatter = imgRes.is_scatter || false;

        if (c.isScatter) {
          // Scatter-only (no real tissue image) — skip decoding the blank white image.
          // render() will fill white background directly on canvas.
          c.image = null;
          console.log(`Scatter mode: skip image decode (${c.imageWidth}x${c.imageHeight})`);
        } else {
          // Real tissue image — decode via blob for speed
          const binary = atob(imgRes.image_data);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
          const blob = new Blob([bytes], { type: 'image/jpeg' });

          if (typeof createImageBitmap === 'function') {
            c.image = await createImageBitmap(blob);
          } else {
            const img = new Image();
            const url = URL.createObjectURL(blob);
            await new Promise((resolve, reject) => {
              img.onload = () => { URL.revokeObjectURL(url); resolve(); };
              img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Image decode failed')); };
              img.src = url;
            });
            c.image = img;
          }
        }
      }
      console.timeEnd('decode-image');

      // Process cells
      console.time('process-cells');
      if (cellRes.success) {
        c.cells = cellRes.cells;
        c.nCells = cellRes.n_cells || 0;
        c.hasCelltype = cellRes.has_celltype || false;
        c.celltypes = cellRes.celltypes || [];
        const backendColors = cellRes.celltype_colors || {};
        // Register backend colors in global registry, then normalize canvas to use global
        for (const [ct, color] of Object.entries(backendColors)) {
          if (!_globalCelltypeColors[ct]) _globalCelltypeColors[ct] = color;
        }
        // Always use global colors so all slices are consistent
        c.celltypeColors = {};
        for (const ct of (cellRes.celltypes || [])) {
          c.celltypeColors[ct] = getOrAssignColor(ct, backendColors);
        }
        c.selectedCelltypes = new Set(); // Empty = show all (no filtering)
        c.isSpotData = cellRes.is_spot_data || false;
        c.spotInfo = cellRes.spot_info || null;
      }
      console.timeEnd('process-cells');

      // a 409 (the agent holds the session) is an envelope, not an exception: retry later instead of blanking the slice
      const busy = [imgRes, cellRes].find(r => r && !r.success && r.error && (r.error.code === 'locked' || r.error.code === 'busy'));
      if (busy) {
        c.loading = false;
        if (isCurrentSlice) showSpinner('Session busy (a step is running); retrying…');
        setTimeout(() => loadSliceData(sliceId, modality), 5000);
        return;
      }
      c.loaded = !!(imgRes.success && cellRes.success);
    } catch (err) {
      console.error('Error loading slice data:', err);
    }

    c.loading = false;
    if (isCurrentSlice) {
      hideSpinner();
      resetView();
    }
  }

  async function switchSlice(sliceId, modality) {
    // Update local state first so getCurrentCanvas() returns the right one
    state.currentSliceId = sliceId;
    if (modality) state.currentModality = modality;

    // Notify backend
    api.selectSlice(sliceId).catch(() => {});
    if (modality) api.selectModality(modality).catch(() => {});

    // Load data if needed
    const c = getCurrentCanvas();
    if (!c.loaded && !c.loading) {
      await loadSliceData(sliceId, state.currentModality);
    }

    // Always render and sync UI
    if (c.loaded) {
      // If view was never initialized (preloaded in background), do it now
      if (!c.viewWidth) resetView();
      else { render(); updateZoomDisplay(); }
    }

    updateSliceTabs();
    updateModalityToggle();
    updateCelltypeCheckboxes();
    updateDataInfo();
    updateVizModeState();
    updateROIList();
    if (view3d.active) render3DView();
  }

  async function preloadAllSlices() {
    if (!state.sessionSummary) return;
    const slices = state.sessionSummary.available_slices || [];
    for (const s of slices) {
      const key = getCanvasKey(s.slice_id, s.modality);
      if (!state.canvases[key] || !state.canvases[key].loaded) {
        // Load in background (no await, just fire)
        loadSliceData(s.slice_id, s.modality);
      }
    }
  }

  async function refreshROIs() {
    try {
      const res = await api.listROIs();
      if (res.success) {
        state.rois = res.rois || [];
        updateROIList();
      }
    } catch (e) { /* ignore */ }
  }

  function handleStateChanges(changes) {
    if (!changes) return;
    const affected = new Set();
    (changes.celltypes_updated || []).forEach(sid => affected.add(sid));
    (changes.deconv_weights_updated || []).forEach(sid => affected.add(sid));
    (changes.celltype_colors_updated || []).forEach(sid => affected.add(sid));

    // Invalidate affected caches
    for (const sid of affected) {
      const mods = state.sessionSummary ? state.sessionSummary.modalities || ['gene'] : ['gene'];
      for (const mod of mods) {
        const key = getCanvasKey(sid, mod);
        if (state.canvases[key]) {
          state.canvases[key].loaded = false;
          state.canvases[key].loading = false;
        }
      }
    }

    // Reload current if affected
    if (affected.has(state.currentSliceId)) {
      loadSliceData(state.currentSliceId, state.currentModality).then(() => {
        render();
        updateCelltypeCheckboxes();
        updateDataInfo();
        if (view3d.active) render3DView();
      });
    }

    // Refresh ROIs if changed
    if (changes.rois_added || changes.rois_deleted) {
      refreshROIs().then(render);
    }
  }

  // ============================================================
  // SECTION 9: UI COMPONENT CONTROLLERS
  // ============================================================

  function updateSliceTabs() {
    const container = document.getElementById('slice-tabs');
    if (!state.sessionSummary) { container.innerHTML = ''; return; }
    const slices = state.sessionSummary.available_slices || [];
    // Group by modality - show each unique slice_id
    const seen = new Set();
    const uniqueSlices = slices.filter(s => {
      if (seen.has(s.slice_id)) return false;
      seen.add(s.slice_id);
      return true;
    });

    container.innerHTML = uniqueSlices.map(s =>
      `<button class="slice-tab ${s.slice_id === state.currentSliceId ? 'active' : ''}" data-slice-id="${s.slice_id}" data-modality="${s.modality}">${escapeHtml(s.tissue_name || 'Slice ' + s.slice_id)}</button>`
    ).join('');

    container.querySelectorAll('.slice-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        const sid = parseInt(btn.dataset.sliceId);
        const mod = btn.dataset.modality || state.currentModality;
        if (sid !== state.currentSliceId) switchSlice(sid, mod);
      });
    });
  }

  function updateModalityToggle() {
    const group = document.getElementById('modality-group');
    if (!state.sessionSummary) return;
    const hasProtein = state.sessionSummary.has_protein;
    group.classList.toggle('hidden', !hasProtein);

    document.querySelectorAll('#modality-toggle .seg-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.modality === state.currentModality);
    });
  }

  function updateVizModeState() {
    const c = getCurrentCanvas();
    const vm = c ? c.vizMode : 'celltype';
    const propRadio = document.querySelector('#proportion-option input');
    if (propRadio) {
      propRadio.disabled = !(c && c.isSpotData && c.spotInfo && c.spotInfo.has_deconv_weights);
    }
    // Sync radio buttons to per-canvas vizMode
    document.querySelectorAll('input[name="viz-mode"]').forEach(r => {
      r.checked = r.value === vm;
    });
    // Show/hide gene search
    document.getElementById('gene-search-group').classList.toggle('hidden', vm !== 'gene');
    // Sync gene search input
    document.getElementById('gene-search').value = (c && c.selectedGene) || '';
    // Show/hide proportion celltype selector
    const showPropCt = vm === 'proportion' && c && c.isSpotData && c.spotInfo && c.spotInfo.deconv_weights;
    document.getElementById('proportion-ct-group').classList.toggle('hidden', !showPropCt);
    if (showPropCt) updateProportionCtSelect();
    // Show/hide colorbar for gene and proportion modes
    const showColorbar = vm === 'gene' || (vm === 'proportion' && c && c.proportionCelltype);
    document.getElementById('colorbar-group').classList.toggle('hidden', !showColorbar);
    if (showColorbar) updateColorbar();
    // Sync opacity and point size sliders
    updateOverlaySliders();
  }

  function updateOverlaySliders() {
    const c = getCurrentCanvas();
    if (!c) return;
    document.getElementById('opacity-slider').value = Math.round(c.opacity * 100);
    document.getElementById('size-slider').value = Math.round(c.pointSize * 100);
    document.getElementById('hide-bg-cb').checked = c.hideBackground;
  }

  function updateProportionCtSelect() {
    const c = getCurrentCanvas();
    const select = document.getElementById('proportion-ct-select');
    if (!c || !c.spotInfo || !c.spotInfo.deconv_weights) return;

    const order = c.spotInfo.celltype_order || Object.keys(c.spotInfo.deconv_weights);
    const sorted = [...order].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
    select.innerHTML = '<option value="">Pie Chart (all types)</option>' +
      sorted.map(ct => `<option value="${escapeHtml(ct)}" ${ct === c.proportionCelltype ? 'selected' : ''}>${escapeHtml(ct)}</option>`).join('');
  }

  function updateColorbar() {
    const c = getCurrentCanvas();
    if (!c) return;

    const bar = document.getElementById('colorbar-bar');
    const minLabel = document.getElementById('colorbar-min');
    const maxLabel = document.getElementById('colorbar-max');
    const slider = document.getElementById('vmax-slider');
    const valueLabel = document.getElementById('vmax-value');

    if (!VIRIDIS_LUT) buildViridisLUT();

    if (c.vizMode === 'gene' && c.geneExpressionRange) {
      const [emin, rawMax] = c.geneExpressionRange;
      const emax = c.customVmax != null ? c.customVmax : rawMax;

      // Build viridis gradient for the bar
      const stops = [0, 0.25, 0.5, 0.75, 1].map(t =>
        VIRIDIS_LUT[Math.round(t * 255)]
      ).join(', ');
      bar.style.background = `linear-gradient(to right, ${stops})`;

      minLabel.textContent = emin.toFixed(2);
      maxLabel.textContent = emax.toFixed(2);

      // Set slider range: 1% to 100% of rawMax
      slider.min = 1;
      slider.max = 100;
      slider.value = c.customVmax != null ? Math.round((c.customVmax / rawMax) * 100) : 100;
      valueLabel.textContent = c.customVmax != null ? emax.toFixed(2) : 'auto';
    } else if (c.vizMode === 'proportion') {
      const stops = [0, 0.25, 0.5, 0.75, 1].map(t =>
        VIRIDIS_LUT[Math.round(t * 255)]
      ).join(', ');
      if (c.proportionCelltype) {
        // Single celltype heatmap — tunable vmax (0-1 range)
        const vmax = c.customVmax != null ? c.customVmax : 1.0;
        bar.style.background = `linear-gradient(to right, ${stops})`;
        minLabel.textContent = '0';
        maxLabel.textContent = vmax.toFixed(2);
        slider.disabled = false;
        slider.min = 1;
        slider.max = 100;
        slider.value = c.customVmax != null ? Math.round(c.customVmax * 100) : 100;
        valueLabel.textContent = c.customVmax != null ? vmax.toFixed(2) : 'auto';
      } else {
        // Pie chart mode — no vmax control
        bar.style.background = 'var(--bg-muted)';
        minLabel.textContent = '';
        maxLabel.textContent = 'Pie chart mode';
        slider.disabled = true;
        slider.value = 100;
        valueLabel.textContent = '';
      }
    } else {
      slider.disabled = false;
    }
  }

  function updateCelltypeCheckboxes() {
    const c = getCurrentCanvas();
    const container = document.getElementById('celltype-checkboxes');
    const noMsg = document.getElementById('no-celltype-msg');

    // Build full celltype list: merge obs celltypes + deconv weight celltypes
    let allCelltypes = c ? [...c.celltypes] : [];
    if (c && c.spotInfo && c.spotInfo.deconv_weights) {
      const deconvTypes = c.spotInfo.celltype_order || Object.keys(c.spotInfo.deconv_weights);
      for (const ct of deconvTypes) {
        if (!allCelltypes.includes(ct)) allCelltypes.push(ct);
      }
    }
    allCelltypes.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));

    if (!c || allCelltypes.length === 0) {
      container.innerHTML = '';
      noMsg.classList.remove('hidden');
      return;
    }
    noMsg.classList.add('hidden');

    // Ensure all celltypes have consistent colors via global registry
    for (const ct of allCelltypes) {
      c.celltypeColors[ct] = getOrAssignColor(ct, c.celltypeColors);
    }

    container.innerHTML = allCelltypes.map(ct => {
      const color = c.celltypeColors[ct];
      // Empty set = all shown (no filtering active)
      const checked = (c.selectedCelltypes.size === 0 || c.selectedCelltypes.has(ct)) ? 'checked' : '';
      return `<label class="ct-checkbox"><input type="checkbox" ${checked} data-ct="${escapeHtml(ct)}"><span class="ct-swatch" style="background:${color}"></span>${escapeHtml(ct)}</label>`;
    }).join('');

    const allCts = allCelltypes; // closure capture
    container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      cb.addEventListener('change', () => {
        const ct = cb.dataset.ct;
        if (c.selectedCelltypes.size === 0 && !cb.checked) {
          allCts.forEach(t => c.selectedCelltypes.add(t));
        }
        if (cb.checked) c.selectedCelltypes.add(ct);
        else c.selectedCelltypes.delete(ct);
        render();
      });
    });

    // Add resize handle if not present
    let resizeHandle = container.parentElement.querySelector('.ct-resize-handle');
    if (!resizeHandle) {
      resizeHandle = document.createElement('div');
      resizeHandle.className = 'ct-resize-handle';
      container.after(resizeHandle);
      setupCtResizeHandle(resizeHandle, container);
    }
  }

  function setupCtResizeHandle(handle, list) {
    let startY, startH;
    handle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      handle.classList.add('active');
      startY = e.clientY;
      startH = list.offsetHeight;
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';

      function onMove(e) {
        const delta = e.clientY - startY;
        const newH = Math.max(40, Math.min(600, startH + delta));
        list.style.maxHeight = newH + 'px';
      }
      function onUp() {
        handle.classList.remove('active');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }

  function getSliceName(sliceId) {
    if (!state.sessionSummary) return 'Slice ' + sliceId;
    const s = (state.sessionSummary.available_slices || []).find(s => s.slice_id === sliceId);
    return s ? (s.tissue_name || 'Slice ' + sliceId) : 'Slice ' + sliceId;
  }

  function updateROIList() {
    const container = document.getElementById('roi-list');
    const allRois = state.rois;
    if (allRois.length === 0) {
      container.innerHTML = '<div class="small-info">No ROIs defined</div>';
      return;
    }

    // Show all ROIs — active slice ones first, others dimmed
    const currentRois = [];
    const otherRois = [];
    for (const r of allRois) {
      if (r.slice_id === state.currentSliceId && r.modality === state.currentModality) {
        currentRois.push(r);
      } else {
        otherRois.push(r);
      }
    }

    function roiItemHtml(r, isActive) {
      const sliceLabel = getSliceName(r.slice_id);
      const cls = isActive ? 'roi-item' : 'roi-item roi-inactive';
      return `<div class="${cls}" data-roi="${escapeHtml(r.name)}" data-active="${isActive}">
        <div class="roi-item-main">
          <span class="roi-item-name">${escapeHtml(r.name)}</span>
          <span class="roi-item-slice">${escapeHtml(sliceLabel)}</span>
        </div>
        <div class="roi-item-detail hidden">${r.type} &middot; ${r.n_cells || '?'} cells</div>
        <span class="roi-item-actions">
          <button class="roi-action-btn roi-rename-btn" title="Rename">&#9998;</button>
          <button class="roi-action-btn roi-delete-btn" title="Delete">&times;</button>
        </span>
      </div>`;
    }

    container.innerHTML =
      currentRois.map(r => roiItemHtml(r, true)).join('') +
      otherRois.map(r => roiItemHtml(r, false)).join('');

    // Event listeners
    container.querySelectorAll('.roi-item').forEach(item => {
      const name = item.dataset.roi;
      const isActive = item.dataset.active === 'true';

      // Hover: highlight on canvas (only active slice) + show detail
      item.addEventListener('mouseenter', () => {
        item.querySelector('.roi-item-detail').classList.remove('hidden');
        if (isActive) {
          state.highlightedROI = name;
          item.classList.add('highlighted');
          render();
        }
      });
      item.addEventListener('mouseleave', () => {
        item.querySelector('.roi-item-detail').classList.add('hidden');
        if (isActive) {
          state.highlightedROI = null;
          item.classList.remove('highlighted');
          render();
        }
      });

      // Rename
      item.querySelector('.roi-rename-btn').addEventListener('click', async (e) => {
        e.stopPropagation();
        const newName = prompt('Rename ROI:', name);
        if (newName && newName !== name) {
          const res = await api.renameROI(name, newName);
          if (res.success) {
            await refreshROIs();
            render();
          } else {
            alert(errText(res.error, 'Rename failed'));
          }
        }
      });

      // Delete
      item.querySelector('.roi-delete-btn').addEventListener('click', async (e) => {
        e.stopPropagation();
        if (confirm(`Delete ROI "${name}"?`)) {
          const res = await api.deleteROI(name);
          if (res.success) {
            state.highlightedROI = null;
            await refreshROIs();
            render();
          } else {
            alert(errText(res.error, 'Delete failed'));
          }
        }
      });
    });
  }

  function updateDataInfo() {
    const c = getCurrentCanvas();
    const el = document.getElementById('data-info');
    if (!c || !c.loaded) { el.textContent = ''; return; }
    const parts = [];
    parts.push(`${c.nCells.toLocaleString()} cells`);
    if (c.isSpotData) parts.push('Spot data');
    if (c.hasCelltype) parts.push(`${c.celltypes.length} types`);
    el.textContent = parts.join(' \u00b7 ');
  }

  function updateSessionInfo() {
    const el = document.getElementById('session-info');
    if (!state.sessionSummary) return;
    const s = state.sessionSummary;
    const parts = [s.name || 'Session'];
    if (s.n_slices > 1) parts.push(`${s.n_slices} slices`);
    if (s.n_cells) parts.push(`${s.n_cells.toLocaleString()} cells`);
    if (s.n_genes) parts.push(`${s.n_genes.toLocaleString()} genes`);
    el.textContent = parts.join(' \u00b7 ');
  }

  function setupGeneSearch() {
    const input = document.getElementById('gene-search');
    const dropdown = document.getElementById('gene-autocomplete');
    let activeIdx = -1;
    // Pre-sort gene list alphabetically once loaded
    let sortedGenes = null;

    function getSortedGenes() {
      if (!sortedGenes || sortedGenes.length !== state.geneList.length) {
        sortedGenes = [...state.geneList].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
      }
      return sortedGenes;
    }

    function showDropdown(query) {
      const genes = getSortedGenes();
      const q = (query || '').trim().toLowerCase();
      let matches;
      if (!q) {
        // Empty input: show all genes alphabetically (first 30)
        matches = genes.slice(0, 30);
      } else {
        // Filter: prefix matches first, then substring matches
        const prefix = [], substring = [];
        for (const g of genes) {
          const gl = g.toLowerCase();
          if (gl.startsWith(q)) prefix.push(g);
          else if (gl.includes(q)) substring.push(g);
          if (prefix.length + substring.length >= 30) break;
        }
        matches = [...prefix, ...substring].slice(0, 30);
      }

      if (matches.length === 0) { dropdown.classList.add('hidden'); return; }
      activeIdx = -1;
      dropdown.innerHTML = matches.map(g =>
        `<div class="autocomplete-item" data-gene="${escapeHtml(g)}">${escapeHtml(g)}</div>`
      ).join('');
      dropdown.classList.remove('hidden');

      dropdown.querySelectorAll('.autocomplete-item').forEach(item => {
        item.addEventListener('click', () => selectGene(item.dataset.gene));
      });
    }

    input.addEventListener('input', debounce(() => showDropdown(input.value), 100));

    // Show all genes on focus
    input.addEventListener('focus', () => showDropdown(input.value));

    input.addEventListener('keydown', (e) => {
      const items = dropdown.querySelectorAll('.autocomplete-item');
      if (!items.length) return;
      if (e.key === 'ArrowDown') { e.preventDefault(); activeIdx = Math.min(activeIdx + 1, items.length - 1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); activeIdx = Math.max(activeIdx - 1, 0); }
      else if (e.key === 'Enter' && activeIdx >= 0) { e.preventDefault(); selectGene(items[activeIdx].dataset.gene); return; }
      else return;
      items.forEach((it, i) => it.classList.toggle('active', i === activeIdx));
      // Scroll active item into view
      if (items[activeIdx]) items[activeIdx].scrollIntoView({ block: 'nearest' });
    });

    document.addEventListener('click', (e) => {
      if (!e.target.closest('#gene-search-group')) dropdown.classList.add('hidden');
    });
  }

  async function selectGene(gene) {
    document.getElementById('gene-autocomplete').classList.add('hidden');
    document.getElementById('gene-search').value = gene;
    const c = getCurrentCanvas();
    if (c) { c.selectedGene = gene; c.vizMode = 'gene'; }
    document.querySelector('input[name="viz-mode"][value="gene"]').checked = true;
    updateVizModeState();

    showSpinner('Loading gene expression...');
    try {
      const res = await api.getGeneExpression(gene);
      if (res.success) {
        const c = getCurrentCanvas();
        c.geneExpression = res.cells;
        c.geneExpressionRange = res.expression_range;
        c.customVmax = null; // Reset vmax for new gene
        document.getElementById('gene-info').textContent =
          `${res.n_cells} cells, range: ${res.expression_range[0].toFixed(2)} - ${res.expression_range[1].toFixed(2)}`;
      }
    } catch (err) {
      console.error('Gene expression error:', err);
    }
    hideSpinner();
    updateColorbar();
    render();
  }

  // ============================================================
  // SECTION 10: INITIALIZATION
  // ============================================================

  function setupResizeObserver() {
    const container = document.getElementById('canvas-container');
    const canvasEl = document.getElementById('spatial-canvas');
    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1;
      canvasEl.width = container.clientWidth * dpr;
      canvasEl.height = container.clientHeight * dpr;
      canvasEl.style.width = container.clientWidth + 'px';
      canvasEl.style.height = container.clientHeight + 'px';
      if (state.initialized) render();
    });
    ro.observe(container);
  }

  function setupEventListeners() {
    // Init form
    document.getElementById('init-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const form = e.target;
      const submitBtn = document.getElementById('init-submit-btn');
      const errorEl = document.getElementById('init-error');
      const loadingEl = document.getElementById('init-loading');

      errorEl.classList.add('hidden');
      loadingEl.classList.remove('hidden');
      submitBtn.disabled = true;

      const providerVal = form.provider.value || undefined;
      const params = {
        dataset_dir: form.dataset_dir.value,
        session_name: form.session_name.value,
        provider: providerVal,
        api_key: form.api_key.value || undefined,
        model: form.model.value || undefined,
        base_url: providerVal === 'custom' ? (form.base_url.value || undefined) : undefined,
      };

      try {
        const res = await api.initDataset(params);
        if (res.success) {
          // Persist form values for next time
          saveFormToStorage();
          // one entry path for a fresh dataset and for a reconnect: enterApp opens the event stream and restores history
          await enterApp(res);
        } else {
          errorEl.textContent = errText(res.error, res.message || 'Failed to load dataset');
          errorEl.classList.remove('hidden');
        }
      } catch (err) {
        errorEl.textContent = 'Connection error: ' + err.message;
        errorEl.classList.remove('hidden');
      }

      loadingEl.classList.add('hidden');
      submitBtn.disabled = false;
    });

    // Provider dropdown
    document.getElementById('provider').addEventListener('change', () => onProviderChange());

    // Test LLM
    document.getElementById('test-llm-btn').addEventListener('click', async () => {
      const resultEl = document.getElementById('test-llm-result');
      resultEl.textContent = 'Testing...';
      resultEl.style.color = 'var(--text-secondary)';
      try {
        const provider = document.getElementById('provider').value;
        const modelVal = document.getElementById('model').value;
        const baseUrl = document.getElementById('base-url').value;
        const res = await api.testLLM({
          provider: provider,
          api_key: document.getElementById('api-key').value,
          model: modelVal,
          base_url: provider === 'custom' ? baseUrl : '',
        });
        if (res.success) {
          resultEl.textContent = 'Connected!';
          resultEl.style.color = 'var(--success)';
        } else {
          resultEl.textContent = res.error || 'Failed';
          resultEl.style.color = 'var(--danger)';
        }
      } catch (e) {
        resultEl.textContent = 'Error: ' + e.message;
        resultEl.style.color = 'var(--danger)';
      }
    });

    // Logout
    document.getElementById('logout-btn').addEventListener('click', () => logout());

    // Logs modal
    document.getElementById('logs-btn').addEventListener('click', () => openLogsModal());
    document.getElementById('logs-modal-close').addEventListener('click', () => closeLogsModal());
    document.getElementById('logs-modal').addEventListener('click', (e) => {
      if (e.target === e.currentTarget) closeLogsModal();
    });
    document.querySelectorAll('.logs-nav-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        _logsTab = btn.dataset.tab;
        _updateLogsNavActive();
        if (_logsTab === 'notebook') _loadNotebookView();
        else document.getElementById('logs-detail').innerHTML = '<div class="logs-placeholder">Select a turn to view its prompt log.</div>';
      });
    });

    // Skill panel
    document.getElementById('skill-panel-tab').addEventListener('click', () => toggleSkillPanel());
    document.getElementById('skill-detail-close').addEventListener('click', () => closeSkillDetail());
    document.getElementById('skill-detail-modal').addEventListener('click', (e) => {
      if (e.target === e.currentTarget) closeSkillDetail();
    });

    // Modality toggle
    document.querySelectorAll('#modality-toggle .seg-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const mod = btn.dataset.modality;
        if (mod !== state.currentModality) {
          state.currentModality = mod;
          updateModalityToggle();
          // Find a slice with this modality
          const slices = state.sessionSummary.available_slices || [];
          const match = slices.find(s => s.modality === mod);
          if (match) switchSlice(match.slice_id, mod);
        }
      });
    });

    // Viz mode
    document.querySelectorAll('input[name="viz-mode"]').forEach(radio => {
      radio.addEventListener('change', () => {
        const c = getCurrentCanvas();
        if (c) c.vizMode = radio.value;
        updateVizModeState();
        render();
      });
    });

    // Proportion celltype select
    document.getElementById('proportion-ct-select').addEventListener('change', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      const val = document.getElementById('proportion-ct-select').value;
      c.proportionCelltype = val || null;
      c.customVmax = null; // reset vmax when switching
      updateVizModeState();
      render();
    });

    // Vmax slider
    document.getElementById('vmax-slider').addEventListener('input', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      const pct = parseInt(document.getElementById('vmax-slider').value);

      if (c.vizMode === 'gene' && c.geneExpressionRange) {
        const [emin, rawMax] = c.geneExpressionRange;
        if (pct >= 100) {
          c.customVmax = null;
          document.getElementById('vmax-value').textContent = 'auto';
          document.getElementById('colorbar-max').textContent = rawMax.toFixed(2);
        } else {
          c.customVmax = rawMax * (pct / 100);
          document.getElementById('vmax-value').textContent = c.customVmax.toFixed(2);
          document.getElementById('colorbar-max').textContent = c.customVmax.toFixed(2);
        }
      } else if (c.vizMode === 'proportion' && c.proportionCelltype) {
        // Proportion heatmap: vmax is 0-1
        if (pct >= 100) {
          c.customVmax = null;
          document.getElementById('vmax-value').textContent = 'auto';
          document.getElementById('colorbar-max').textContent = '1.00';
        } else {
          c.customVmax = pct / 100;
          document.getElementById('vmax-value').textContent = c.customVmax.toFixed(2);
          document.getElementById('colorbar-max').textContent = c.customVmax.toFixed(2);
        }
      }
      render();
    });

    // Opacity slider
    document.getElementById('opacity-slider').addEventListener('input', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      c.opacity = parseInt(document.getElementById('opacity-slider').value) / 100;
      render();
    });

    // Point size slider
    document.getElementById('size-slider').addEventListener('input', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      c.pointSize = parseInt(document.getElementById('size-slider').value) / 100;
      render();
    });

    // Coordinate scale factor
    document.getElementById('coord-scale').addEventListener('change', () => {
      const val = parseFloat(document.getElementById('coord-scale').value);
      if (val > 0 && isFinite(val)) {
        state.coordScale = val;
        render();
      }
    });

    // Hide background checkbox
    document.getElementById('hide-bg-cb').addEventListener('change', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      c.hideBackground = document.getElementById('hide-bg-cb').checked;
      render();
    });

    // Celltype select all / none
    document.getElementById('ct-select-all').addEventListener('click', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      c.selectedCelltypes = new Set(); // Empty = show all
      updateCelltypeCheckboxes();
      render();
    });

    document.getElementById('ct-select-none').addEventListener('click', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      // Use a special marker set with a dummy entry to distinguish from "show all" empty set
      c.selectedCelltypes = new Set(['__none__']);
      updateCelltypeCheckboxes();
      render();
    });

    // ROI tools
    document.getElementById('roi-bbox-btn').addEventListener('click', () => {
      if (state.roiTool === 'bbox') { cancelROITool(); return; }
      state.roiTool = 'bbox';
      state.roiDrawingState = null;
      document.getElementById('canvas-container').classList.add('roi-drawing');
      document.getElementById('roi-bbox-btn').classList.add('active');
      document.getElementById('roi-polygon-btn').classList.remove('active');
      document.getElementById('roi-freehand-btn').classList.remove('active');
      document.getElementById('roi-cancel-btn').classList.remove('hidden');
    });

    document.getElementById('roi-polygon-btn').addEventListener('click', () => {
      if (state.roiTool === 'polygon') { cancelROITool(); return; }
      state.roiTool = 'polygon';
      state.roiDrawingState = null;
      document.getElementById('canvas-container').classList.add('roi-drawing');
      document.getElementById('roi-polygon-btn').classList.add('active');
      document.getElementById('roi-bbox-btn').classList.remove('active');
      document.getElementById('roi-freehand-btn').classList.remove('active');
      document.getElementById('roi-cancel-btn').classList.remove('hidden');
    });

    document.getElementById('roi-freehand-btn').addEventListener('click', () => {
      if (state.roiTool === 'freehand') { cancelROITool(); return; }
      state.roiTool = 'freehand';
      state.roiDrawingState = null;
      document.getElementById('canvas-container').classList.add('roi-drawing');
      document.getElementById('roi-freehand-btn').classList.add('active');
      document.getElementById('roi-bbox-btn').classList.remove('active');
      document.getElementById('roi-polygon-btn').classList.remove('active');
      document.getElementById('roi-cancel-btn').classList.remove('hidden');
    });

    document.getElementById('roi-cancel-btn').addEventListener('click', cancelROITool);

    // Zoom buttons
    document.getElementById('zoom-in-btn').addEventListener('click', () => {
      const c = getCurrentCanvas();
      if (!c || !c.loaded) return;
      const el = document.getElementById('spatial-canvas');
      const centerX = c.viewX + c.viewWidth / 2;
      const centerY = c.viewY + c.viewHeight / 2;
      c.viewWidth /= ZOOM_FACTOR;
      c.viewHeight /= ZOOM_FACTOR;
      c.viewX = centerX - c.viewWidth / 2;
      c.viewY = centerY - c.viewHeight / 2;
      c.zoom = c.imageWidth / c.viewWidth;
      render();
      updateZoomDisplay();
    });

    document.getElementById('zoom-out-btn').addEventListener('click', () => {
      const c = getCurrentCanvas();
      if (!c || !c.loaded) return;
      c.viewWidth *= ZOOM_FACTOR;
      c.viewHeight *= ZOOM_FACTOR;
      const centerX = c.viewX + c.viewWidth / (2 * ZOOM_FACTOR);
      const centerY = c.viewY + c.viewHeight / (2 * ZOOM_FACTOR);
      c.viewX = centerX - c.viewWidth / 2;
      c.viewY = centerY - c.viewHeight / 2;
      c.zoom = c.imageWidth / c.viewWidth;
      render();
      updateZoomDisplay();
    });

    document.getElementById('zoom-fit-btn').addEventListener('click', resetView);

    // 3D toggle
    document.getElementById('toggle-3d-btn').addEventListener('click', toggle3DView);

    // Chat
    const chatInput = document.getElementById('chat-input');
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (text && (!state.chatStreaming || turnUI.pendingPrompt)) {
          chatInput.value = '';
          sendChatMessage(text);
        }
      }
    });

    chatInput.addEventListener('input', () => {
      // Auto-resize
      chatInput.style.height = 'auto';
      chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    });

    document.getElementById('send-btn').addEventListener('click', () => {
      const text = chatInput.value.trim();
      if (text && (!state.chatStreaming || turnUI.pendingPrompt)) {
        chatInput.value = '';
        chatInput.style.height = 'auto';
        sendChatMessage(text);
      }
    });

    document.getElementById('abort-btn').addEventListener('click', () => { abortTurn(); });
    const latestBtn = document.getElementById('chat-latest');
    if (latestBtn) latestBtn.addEventListener('click', () => scrollChatToBottom(true));
    chatEl().addEventListener('click', (e) => { const img = e.target.closest('.chat-plot img'); if (img) openLightbox(img.currentSrc || img.src); });
    const lightbox = document.getElementById('figure-lightbox');
    if (lightbox) {
      lightbox.addEventListener('click', closeLightbox);
      document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !lightbox.classList.contains('hidden')) closeLightbox(); });
    }

    // Mode selector (interactive / auto) and Changes drawer
    const modeSel = document.getElementById('mode-select');
    if (modeSel) {
      modeSel.value = localStorage.getItem('stat_mode') || 'interactive';
      state.mode = modeSel.value;
      modeSel.addEventListener('change', () => { state.mode = modeSel.value; localStorage.setItem('stat_mode', state.mode); });
    }
    const changesBtn = document.getElementById('changes-btn');
    if (changesBtn) changesBtn.addEventListener('click', toggleChangesDrawer);
    const changesClose = document.getElementById('changes-close');
    if (changesClose) changesClose.addEventListener('click', toggleChangesDrawer);

    document.getElementById('save-chat-btn').addEventListener('click', () => {
      api.saveChatHistory();
    });

    // Notebook
    document.getElementById('notebook-btn').addEventListener('click', async () => {
      try {
        const res = await api.getNotebookURL();
        if (res.success) {
          window.open(res.url || `http://localhost:${res.jupyter_port}`, '_blank');
        } else {
          alert(errText(res.error, 'Notebook not available'));
        }
      } catch (e) {
        alert('Notebook not available');
      }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (state.roiTool) cancelROITool();
      }
    });
  }

  // ============================================================
  // SECTION: 3D VIEWER
  // ============================================================

  const view3d = {
    active: false,
    rotX: -30,
    rotY: 20,
    zoom: 1.0,
    spacing: 140,
    orbiting: false,
    dragMoved: false,    // distinguish click vs drag
    orbitStart: null,
    orbitStartRot: null,
  };

  function toggle3DView() {
    view3d.active = !view3d.active;
    const viewer3d = document.getElementById('viewer-3d');
    const canvas2d = document.getElementById('spatial-canvas');
    const btn = document.getElementById('toggle-3d-btn');

    if (view3d.active) {
      viewer3d.classList.remove('hidden');
      canvas2d.style.visibility = 'hidden';
      btn.classList.add('active');
      btn.textContent = '2D View';
      render3DView();
      setup3DInteraction();
    } else {
      viewer3d.classList.add('hidden');
      canvas2d.style.visibility = '';
      btn.classList.remove('active');
      btn.textContent = '3D View';
      teardown3DInteraction();
      render();
    }
  }

  function renderSliceToCanvas(sliceId, modality, displayW, displayH, bounds) {
    const key = getCanvasKey(sliceId, modality);
    const c = state.canvases[key];
    if (!c || !c.loaded) return null;

    // Use data bounds if provided, otherwise full image
    const vx = bounds ? bounds.vx : 0;
    const vy = bounds ? bounds.vy : 0;
    const vw = bounds ? bounds.w : (c.imageWidth || displayW);
    const vh = bounds ? bounds.h : (c.imageHeight || displayH);

    // Render resolution capped at 2048 for perf
    const maxDim = 2048;
    const resFactor = Math.min(maxDim / Math.max(vw, vh), 1);
    const renderW = Math.round(vw * resFactor);
    const renderH = Math.round(vh * resFactor);

    const offscreen = document.createElement('canvas');
    offscreen.width = renderW;
    offscreen.height = renderH;
    offscreen.style.width = displayW + 'px';
    offscreen.style.height = displayH + 'px';
    const ctx = offscreen.getContext('2d');

    // Draw image cropped to the data bounds region
    if (c.image && !c.hideBackground) {
      ctx.drawImage(c.image, vx, vy, vw, vh, 0, 0, renderW, renderH);
    } else {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, renderW, renderH);
    }

    // Draw cells within the data bounds viewport
    const fakeEl = { width: renderW, height: renderH };
    const saved = { vx: c.viewX, vy: c.viewY, vw: c.viewWidth, vh: c.viewHeight };
    c.viewX = vx; c.viewY = vy;
    c.viewWidth = vw;
    c.viewHeight = vh;

    ctx.globalAlpha = c.opacity;
    if (c.vizMode === 'celltype') {
      renderCells(ctx, c, fakeEl);
    } else if (c.vizMode === 'gene' && c.geneExpression) {
      renderGeneExpression(ctx, c, fakeEl);
    } else if (c.vizMode === 'proportion' && c.isSpotData && c.spotInfo) {
      renderProportions(ctx, c, fakeEl);
    }
    ctx.globalAlpha = 1.0;

    // Restore view
    c.viewX = saved.vx; c.viewY = saved.vy;
    c.viewWidth = saved.vw; c.viewHeight = saved.vh;

    return offscreen;
  }

  function render3DView() {
    if (!view3d.active) return;

    const scene = document.getElementById('scene-3d');
    const container = document.getElementById('viewer-3d');
    const cw = container.clientWidth;
    const ch = container.clientHeight;

    // Gather all loaded slices
    const slices = (state.sessionSummary?.available_slices || []).filter(s => {
      const key = getCanvasKey(s.slice_id, s.modality);
      return state.canvases[key]?.loaded;
    });
    if (slices.length === 0) return;

    // Get per-slice dimensions from image size (or cell bounds for scatter-only data)
    const sliceBounds = slices.map(s => {
      const c = state.canvases[getCanvasKey(s.slice_id, s.modality)];
      if (!c) return { vx: 0, vy: 0, w: 400, h: 400 };
      const imgW = c.imageWidth || 0, imgH = c.imageHeight || 0;
      // Use image size when available (most common case)
      if (imgW > 0 && imgH > 0 && c.image) {
        return { vx: 0, vy: 0, w: imgW, h: imgH };
      }
      // No image (scatter only): use cell coordinate bounds
      if (c.cells && c.cells.x && c.cells.x.length > 0) {
        let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
        const xs = c.cells.x, ys = c.cells.y;
        for (let i = 0; i < xs.length; i++) {
          if (xs[i] < xmin) xmin = xs[i];
          if (xs[i] > xmax) xmax = xs[i];
          if (ys[i] < ymin) ymin = ys[i];
          if (ys[i] > ymax) ymax = ys[i];
        }
        const pad = Math.max((xmax - xmin), (ymax - ymin)) * 0.05;
        return { vx: xmin - pad, vy: ymin - pad, w: (xmax - xmin) + pad * 2, h: (ymax - ymin) + pad * 2 };
      }
      return { vx: 0, vy: 0, w: imgW || 400, h: imgH || 400 };
    });
    const maxW = Math.max(...sliceBounds.map(d => d.w));
    const maxH = Math.max(...sliceBounds.map(d => d.h));

    // Size planes to fill the view well:
    // - Horizontally: use up to 75% of container width
    // - Vertically: account for slice height + total stack depth projected at the view angle
    const stackDepth = (slices.length - 1) * view3d.spacing;
    const projectedStackH = maxH * (maxW / maxH) + Math.abs(Math.sin(view3d.rotX * Math.PI / 180)) * stackDepth;
    const fitScale = Math.min((cw * 0.75) / maxW, (ch * 0.7) / projectedStackH);
    const displayW = Math.round(maxW * fitScale);

    scene.innerHTML = '';

    // Stack slices vertically in Z
    const totalSpacing = (slices.length - 1) * view3d.spacing;
    const startZ = totalSpacing / 2;

    slices.forEach((s, i) => {
      const c = state.canvases[getCanvasKey(s.slice_id, s.modality)];
      if (!c) return;

      // Per-slice dimensions scaled to its own data bounds (not uniform maxW)
      const bounds = sliceBounds[i];
      const planeW = Math.round(bounds.w * fitScale);
      const planeH = Math.round(bounds.h * fitScale);

      const offscreen = renderSliceToCanvas(s.slice_id, s.modality, planeW, planeH, bounds);
      if (!offscreen) return;

      const plane = document.createElement('div');
      plane.className = 'slice-plane' + (s.slice_id === state.currentSliceId ? ' active-slice' : '');
      plane.style.width = planeW + 'px';
      plane.style.height = planeH + 'px';
      plane.style.left = ((cw - planeW) / 2) + 'px';
      plane.style.top = ((ch - planeH) / 2) + 'px';

      const z = startZ - i * view3d.spacing;
      plane.style.transform = `translateZ(${z}px)`;

      plane.appendChild(offscreen);

      // Label
      const label = document.createElement('div');
      label.className = 'slice-label';
      label.textContent = `${s.tissue_name || 'Slice ' + s.slice_id} (${s.modality})`;
      plane.appendChild(label);

      // Click: switch to that slice in 2D view (only if not dragging)
      plane.addEventListener('click', () => {
        if (view3d.dragMoved) return;
        switchSlice(s.slice_id, s.modality);
        toggle3DView(); // Exit 3D, show selected slice in 2D
      });

      scene.appendChild(plane);
    });

    updateSceneTransform();
  }

  function updateSceneTransform() {
    const scene = document.getElementById('scene-3d');
    if (!scene) return;
    const container = document.getElementById('viewer-3d');
    const cx = container.clientWidth / 2;
    const cy = container.clientHeight / 2;
    scene.style.perspective = `${1200 / view3d.zoom}px`;
    scene.style.perspectiveOrigin = `${cx}px ${cy}px`;
    scene.style.transform = `scale(${view3d.zoom}) rotateX(${view3d.rotX}deg) rotateY(${view3d.rotY}deg)`;
  }

  let _3dHandlers = {};

  function setup3DInteraction() {
    const viewer = document.getElementById('viewer-3d');

    _3dHandlers.mousedown = (e) => {
      if (e.target.closest('.viewer-3d-controls')) return;
      view3d.orbiting = true;
      view3d.dragMoved = false;
      view3d.orbitStart = { x: e.clientX, y: e.clientY };
      view3d.orbitStartRot = { x: view3d.rotX, y: view3d.rotY };
      viewer.classList.add('orbiting');
      e.preventDefault();
    };

    _3dHandlers.mousemove = (e) => {
      if (!view3d.orbiting) return;
      const dx = e.clientX - view3d.orbitStart.x;
      const dy = e.clientY - view3d.orbitStart.y;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) view3d.dragMoved = true;
      view3d.rotY = view3d.orbitStartRot.y + dx * 0.3;
      // Allow full rotation — no clamping
      view3d.rotX = view3d.orbitStartRot.x - dy * 0.3;
      updateSceneTransform();
    };

    _3dHandlers.mouseup = () => {
      view3d.orbiting = false;
      viewer.classList.remove('orbiting');
    };

    _3dHandlers.wheel = (e) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.92 : 1.08;
      view3d.zoom = Math.max(0.2, Math.min(6, view3d.zoom * factor));
      updateSceneTransform();
    };

    _3dHandlers.spacing = () => {
      view3d.spacing = parseInt(document.getElementById('spacing-slider').value);
      render3DView();
    };

    viewer.addEventListener('mousedown', _3dHandlers.mousedown);
    document.addEventListener('mousemove', _3dHandlers.mousemove);
    document.addEventListener('mouseup', _3dHandlers.mouseup);
    viewer.addEventListener('wheel', _3dHandlers.wheel, { passive: false });
    document.getElementById('spacing-slider').addEventListener('input', _3dHandlers.spacing);
  }

  function teardown3DInteraction() {
    const viewer = document.getElementById('viewer-3d');
    if (!viewer) return;
    viewer.removeEventListener('mousedown', _3dHandlers.mousedown);
    document.removeEventListener('mousemove', _3dHandlers.mousemove);
    document.removeEventListener('mouseup', _3dHandlers.mouseup);
    viewer.removeEventListener('wheel', _3dHandlers.wheel);
    const spacingSlider = document.getElementById('spacing-slider');
    if (spacingSlider) spacingSlider.removeEventListener('input', _3dHandlers.spacing);
    _3dHandlers = {};
  }

  // ============================================================
  // SECTION: RESIZABLE PANELS
  // ============================================================

  function setupResizableHandles() {
    const main = document.querySelector('.main-layout');
    if (!main) return;

    function setupHandle(handleId, colIndex, minPx, maxPx) {
      const handle = document.getElementById(handleId);
      if (!handle) return;

      let startX, startSize, cols;

      handle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        handle.classList.add('active');
        startX = e.clientX;
        cols = main.style.gridTemplateColumns
          ? main.style.gridTemplateColumns.split(/\s+/)
          : getComputedStyle(main).gridTemplateColumns.split(/\s+/);
        startSize = parseFloat(cols[colIndex]);
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';

        function onMove(e) {
          const delta = e.clientX - startX;
          // Both panels are to the RIGHT of their handle:
          // Dragging handle right = panel shrinks (negative direction)
          const direction = -1;
          let newSize = Math.max(minPx, Math.min(maxPx, startSize + delta * direction));
          cols[colIndex] = newSize + 'px';
          // Keep viewer as 1fr
          cols[0] = '1fr';
          main.style.gridTemplateColumns = cols.join(' ');
          // Trigger canvas resize
          if (state.initialized) {
            const container = document.getElementById('canvas-container');
            const canvasEl = document.getElementById('spatial-canvas');
            const dpr = window.devicePixelRatio || 1;
            canvasEl.width = container.clientWidth * dpr;
            canvasEl.height = container.clientHeight * dpr;
            render();
          }
        }

        function onUp() {
          handle.classList.remove('active');
          document.body.style.cursor = '';
          document.body.style.userSelect = '';
          document.removeEventListener('mousemove', onMove);
          document.removeEventListener('mouseup', onUp);
        }

        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
      });
    }

    // Left handle controls the controls-panel width (column index 2)
    setupHandle('resize-left', 2, 150, 400);
    // Right handle controls the chat-panel width (column index 4)
    setupHandle('resize-right', 4, 300, 900);
  }

  async function init() {
    buildViridisLUT();
    setupEventListeners();
    setupCanvasInteraction();
    setupGeneSearch();
    setupResizableHandles();

    // Restore saved form values
    loadFormFromStorage();

    // If backend session is still alive, reconnect without re-init
    await tryReconnectSession();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
