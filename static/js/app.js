let curM = null, models = [], tpls = [], curTpl = null, attF = [], attI = [];
let backendType = 'none';
let webSearchEnabled = false;
let planModeEnabled = false; // Plan 模式开关：开启后工具调用不执行，返回计划供审批
let planModeActive = false; // SSE 流中是否已收到 plan 事件
let vendorDefs = {};   // { vendorId: { name, models, has_server_key, base_url, default_model } }
let vendorModels = []; // 当前 vendor 的模型列表
let currentSession = null; // 当前会话
let sessionList = [];    // 会话列表缓存
let vendorId = null;   // 当前选中的 vendor ID
let vendorCreds = {};  // { vendorId: { api_key, base_url } } — 多厂商凭据内存
let abortCtrl = null;  // AbortController for stopping generation
let _tokenCount = 0;  // 当前对话 token 计数（估算）
let _reasoningText = '';  // 当前 reasoning 内容
let _compressedMsgs = null; // 压缩后的消息（null=未压缩）
let _fullMessages = []; // 完整消息历史（含 tool calls/results）
let _currentAssistantMsg = null; // 当前助手消息（streaming 中构建）
let _lastToolCallId = null;
let auxConfig = { enabled: false, provider: '', model: '', tasks: ['compression'] };
let auxProviders = [];  // available providers for aux dropdown
var toolsEnabled = false;  // 工具调用开关（var 使 window.toolsEnabled 生效，_SF 注册表通过 window[d.var] 访问）
let toolSchemas = [];      // 从服务端加载的工具定义

// ──────────────────────────────────────────────────────────────────────────────
// 设置持久化
// ──────────────────────────────────────────────────────────────────────────────
// ── 设置字段注册表: 新增持久化设置只需在此添加 ──────────────────────────
// { el: DOM id, var?: 全局变量名, type?: 'input'(默认)|'checkbox' }
// 18 个参数 = 本地 9 项 (3 滑块 + 6 开关) + 厂商 9 项
const _SF_LOCAL = {
  's-temp-local':          { el: 's-temp-local' },
  's-max-local':           { el: 's-max-local' },
  's-topp-local':          { el: 's-topp-local' },
  'tools_local':           { el: 'tools-local-enable',     var: 'toolsEnabled',     type: 'checkbox' },
  'plan_mode_local':       { el: 'plan-mode-local',        var: 'planModeEnabled',  type: 'checkbox' },
  'think_output_local':    { el: 'think-output-local',     var: 'thinkOutputEnabled',type: 'checkbox' },
  'auto_review_local':     { el: 'auto-review-local',      var: 'autoReviewEnabled',type: 'checkbox' },
  'min_prompt_local':      { el: 'min-prompt-local',       var: 'minPromptEnabled', type: 'checkbox' },
  'ctx_ext_local':         { el: 'ctx-ext-local',          var: 'ctxExtEnabled',    type: 'checkbox' },
};
const _SF_VENDOR = {
  's-temp-vendor':        { el: 's-temp-vendor' },
  's-max-vendor':         { el: 's-max-vendor' },
  's-topp-vendor':        { el: 's-topp-vendor' },
  'tools_vendor':         { el: 'tools-vendor-enable',     var: 'toolsEnabledVendor',  type: 'checkbox' },
  'plan_mode_vendor':     { el: 'plan-mode-vendor',        var: 'planModeVendor',      type: 'checkbox' },
  'think_output_vendor':  { el: 'think-output-vendor',     var: 'thinkOutputVendor',   type: 'checkbox' },
  'auto_review_vendor':   { el: 'auto-review-vendor',      var: 'autoReviewVendor',    type: 'checkbox' },
  'min_prompt_vendor':    { el: 'min-prompt-vendor',       var: 'minPromptVendor',     type: 'checkbox' },
  'ctx_ext_vendor':       { el: 'ctx-ext-vendor',          var: 'ctxExtVendor',        type: 'checkbox' },
};

async function saveSettings() {
  // 先收集当前厂商凭据（从设置面板）
  if (vendorId && isVendorBackend(backendType)) {
    const key = document.getElementById('set-api-key')?.value || '';
    const url = document.getElementById('set-base-url')?.value || '';
    vendorCreds[vendorId] = { api_key: key, base_url: url };
  }
  collectAuxConfig();
  const settings = { vendor_creds: vendorCreds, backend: backendType, aux_config: auxConfig };
  // 自动收集所有注册表中注册的字段
  for (const [k, d] of Object.entries({..._SF_LOCAL, ..._SF_VENDOR})) {
    if (d.type === 'checkbox') settings[k] = document.getElementById(d.el)?.checked || false;
    else settings[k] = document.getElementById(d.el)?.value || '';
  }
  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    });
    if (res.ok) {
      const msg = document.getElementById('s-msg');
      if (msg) { msg.textContent = '✓ 已保存到项目目录'; setTimeout(() => msg.textContent = '', 2000); }
    } else {
      throw new Error('保存失败');
    }
  } catch (e) {
    const msg = document.getElementById('s-msg');
    if (msg) { msg.textContent = '✗ ' + e.message; setTimeout(() => msg.textContent = '', 3000); }
  }
}

async function loadSettings() {
  try {
    const res = await fetch('/api/settings');
    if (!res.ok) return;
    const s = await res.json();
    // 自动恢复所有注册表中注册的字段
    for (const [k, d] of Object.entries({..._SF_LOCAL, ..._SF_VENDOR})) {
      if (s[k] === undefined) continue;
      const el = document.getElementById(d.el);
      if (!el) continue;
      if (d.type === 'checkbox') {
        el.checked = s[k];
        if (d.var) window[d.var] = s[k];
      } else {
        el.value = s[k];
      }
    }
    // 更新 slider 值标签
    us('temp','local'); us('max','local'); us('topp','local');
    us('temp','vendor'); us('max','vendor'); us('topp','vendor');
    if (s.vendor_creds) {
      vendorCreds = { ...s.vendor_creds };
      if (vendorId && vendorCreds[vendorId]) {
        const vc = vendorCreds[vendorId];
        if (document.getElementById('set-api-key')) document.getElementById('set-api-key').value = vc.api_key || '';
        if (document.getElementById('set-base-url')) document.getElementById('set-base-url').value = vc.base_url || '';
      }
    }
  } catch (e) {}
  syncSettingsPanels();
}

// ──────────────────────────────────────────────────────────────────────────────
// 辅助模型配置
// ──────────────────────────────────────────────────────────────────────────────
function renderAuxProviders() {
  const sel = document.getElementById('aux-provider');
  if (!sel) return;
  const curVal = sel.value;
  sel.innerHTML = '<option value="">跟随主模型</option>';
  auxProviders.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = p.name + (p.hasKey ? ' 🔑' : ' ⚠️');
    sel.appendChild(opt);
  });
  sel.value = curVal;
}

function collectAuxProviders() {
  const list = [];
  for (const [id, def] of Object.entries(vendorDefs)) {
    const hasKey = !!(vendorCreds[id]?.api_key);
    list.push({ id, name: def.name || id, hasKey });
  }
  // 也加上本地后端选项
  auxProviders = list;
  renderAuxProviders();
}

function collectAuxConfig() {
  const tasks = [];
  document.querySelectorAll('.aux-task:checked').forEach(cb => tasks.push(cb.value));
  const provider = document.getElementById('aux-provider')?.value || '';
  const model = document.getElementById('aux-model')?.value || '';
  auxConfig = {
    enabled: document.getElementById('aux-enabled')?.checked || false,
    provider,
    model,
    tasks
  };
  return auxConfig;
}

async function loadAuxConfig() {
  try {
    const res = await fetch('/api/aux-config');
    if (!res.ok) return;
    const ac = await res.json();
    auxConfig = { enabled: false, provider: '', model: '', tasks: ['compression'], ...ac };
    const enabledEl = document.getElementById('aux-enabled');
    const providerEl = document.getElementById('aux-provider');
    const modelEl = document.getElementById('aux-model');
    if (enabledEl) enabledEl.checked = auxConfig.enabled;
    if (providerEl) providerEl.value = auxConfig.provider || '';
    if (modelEl) modelEl.value = auxConfig.model || '';
    // tasks checkboxes
    document.querySelectorAll('.aux-task').forEach(cb => {
      cb.checked = auxConfig.tasks.includes(cb.value);
    });
    updateAuxStatus();
  } catch (e) { console.warn('加载辅助配置失败:', e); }
}

async function saveAuxConfig() {
  collectAuxConfig();
  try {
    const res = await fetch('/api/aux-config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(auxConfig)
    });
    updateAuxStatus();
    return res.ok;
  } catch (e) { return false; }
}

function onAuxChange() { collectAuxConfig(); updateAuxStatus(); }

function onAuxProviderChange() {
  const provider = document.getElementById('aux-provider')?.value || '';
  // 自动填充默认模型
  if (provider && vendorDefs[provider]) {
    const def = vendorDefs[provider];
    const modelEl = document.getElementById('aux-model');
    if (modelEl && !modelEl.value) {
      // 优先用 vendor 的 default_model
      modelEl.placeholder = def.default_model || '';
    }
    // 显示凭据状态
    const hasKey = !!(vendorCreds[provider]?.api_key);
    const statusEl = document.getElementById('aux-status');
    if (statusEl) {
      if (hasKey) {
        statusEl.style.display = 'block';
        statusEl.innerHTML = '✅ 凭据已配置 — 直接从设置面板的 API Key 读取';
        statusEl.className = 'aux-stats aux-ok';
      } else {
        statusEl.style.display = 'block';
        statusEl.innerHTML = '⚠️ 未配置 API Key — 请在左侧厂商面板中先填入';
        statusEl.className = 'aux-stats aux-warn';
      }
    }
  } else {
    const statusEl = document.getElementById('aux-status');
    if (statusEl) statusEl.style.display = 'none';
  }
  onAuxChange();
}

function updateAuxStatus() {
  const statusEl = document.getElementById('aux-status');
  if (!statusEl) return;
  if (!auxConfig.enabled) {
    statusEl.style.display = 'none';
    return;
  }
  const provider = auxConfig.provider;
  const model = auxConfig.model || (vendorDefs[provider]?.default_model || '(默认)');
  const hasKey = !!(vendorCreds[provider]?.api_key);
  statusEl.style.display = 'block';
  if (provider && hasKey) {
    statusEl.innerHTML = `✅ 辅助模型: ${vendorDefs[provider]?.name || provider} / ${model} — ${auxConfig.tasks.join(', ')}`;
    statusEl.className = 'aux-stats aux-ok';
  } else if (provider && !hasKey) {
    statusEl.innerHTML = `⚠️ ${vendorDefs[provider]?.name || provider} 缺少 API Key`;
    statusEl.className = 'aux-stats aux-warn';
  } else {
    statusEl.innerHTML = 'ℹ️ 未选择厂商 — 辅助模型未生效';
    statusEl.className = 'aux-stats aux-info';
  }
}

function stopGen() {
  if (abortCtrl) { abortCtrl.abort(); abortCtrl = null; }
}

// ──────────────────────────────────────────────────────────────────────────────
// 初始化
// ──────────────────────────────────────────────────────────────────────────────
async function init() {
  await loadSettings();
  await detectBackend();
  await initSession();    // 会话持久化：加载/创建会话
  await loadT();
  await loadVendors(); // 预加载厂商定义
  if (backendType === 'llama-cpp') {
    await loadLlamaModels();
  }
  collectAuxProviders();   // 辅助模型可用厂商列表（先填选项）
  await loadAuxConfig();   // 辅助模型配置（再设值 — 选项已存在才能正确匹配）
  await loadToolSchemas();
  updateBackendStatus();
  syncSettingsPanels();
}

async function detectBackend() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    backendType = data.current_backend || data.backend || 'none';
    const bkSel = document.getElementById('bk-sel');
    if (bkSel) {
      const avail = data.available_backends || [];
      bkSel.querySelectorAll('option').forEach(opt => {
        if (isVendorBackend(opt.value)) return; // 厂商 API 永远可选
        opt.disabled = !avail.includes(opt.value);
        opt.hidden = opt.disabled;
      });
      bkSel.value = backendType;
    }
    const dot = document.getElementById('st-dot');
    const txt = document.getElementById('st-txt');
    const lc = data.llama_cpp || {};
    if (backendType === 'llama-cpp') {
      dot.classList.add('on');
      txt.textContent = lc.gpu_available ? 'llama-cpp (GPU)' : 'llama-cpp (CPU)';
    } else {
      dot.classList.remove('on');
      txt.textContent = '无可用后端';
    }
  } catch (e) {
    console.error('检测后端失败:', e);
    backendType = 'none';
  }
}

function updateBackendStatus() {
  const badge = document.getElementById('cur-m');
  if (backendType === 'llama-cpp') {
    fetch('/api/status').then(r => r.json()).then(data => {
      const lc = data.llama_cpp || {};
      if (lc.cpu_mode) badge.title = 'CPU 模式';
      else if (lc.gpu_available) badge.title = 'GPU 模式';
    }).catch(() => { });
  }
}

// ── 厂商 API 加载 ──────────────────────────────────────────────────────
async function loadVendors() {
  try {
    const res = await fetch('/api/vendors');
    const data = await res.json();
    (data.vendors || []).forEach(v => { vendorDefs[v.id] = v; });
  } catch (e) {
    console.error('加载厂商列表失败:', e);
  }
}

function isVendorBackend(t) {
  return ['openai', 'deepseek', 'anthropic', 'gemini', 'qwen', 'zhipu', 'moonshot', 'ollama-cloud', 'custom',].includes(t);
}


function syncSettingsPanels() {
  const localInfer = document.getElementById('set-local-infer');
  const vendorInfer = document.getElementById('set-vendor-infer');
  const vendorSec = document.getElementById('set-vendor');
  if (!localInfer || !vendorInfer) return;
  const isVendor = isVendorBackend(backendType);
  localInfer.style.display = isVendor ? 'none' : 'block';
  vendorInfer.style.display = isVendor ? 'block' : 'none';
  if (vendorSec) {
    vendorSec.style.display = isVendor ? 'block' : 'none';
  }
}

async function switchBackend(target) {
  // ── 厂商 API 后端（不需要服务端切换）──
  if (isVendorBackend(target)) {
    vendorId = target;
    backendType = target;
    curM = null;
    vendorModels = vendorDefs[target]?.models || [];
    const vdef = vendorDefs[target] || {};

    // 加载已保存凭据
    const savedCreds = vendorCreds[target] || {};
    const keyInput = savedCreds.api_key || '';
    const baseUrl = savedCreds.base_url || vdef.base_url || '';
    
    // key 状态提示（内存中）
    if (vdef.has_server_key) {
      // 环境变量已配，静默
    } else if (!keyInput) {
      // 需要输入 — 后续在设置面板提示
    }
    
    // 同步到设置面板
    syncVendorToSettings(vdef);

    // 更新模型选择器
    renderVendorModels(vendorModels, vdef.default_model);

    // 状态栏
    const dot = document.getElementById('st-dot');
    const txt = document.getElementById('st-txt');
    dot.classList.add('on');
    txt.textContent = vdef.name || target;
    syncSettingsPanels();
    return;
  }

  // ── 本地后端（llama-cpp）──
  vendorId = null;
  document.getElementById('set-vendor').style.display = 'none';

  try {
    const res = await fetch('/api/switch_backend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backend: target })
    });
    const data = await res.json();
    if (res.ok) {
      backendType = data.backend;
      curM = null;
      models = [];
      const sel = document.getElementById('m-sel');
      sel.innerHTML = '<option value="">选择模型...</option>';
      document.getElementById('cur-m').textContent = '未选择';
      if (backendType === 'llama-cpp') {
        await loadLlamaModels();
      }
      const dot = document.getElementById('st-dot');
      const txt = document.getElementById('st-txt');
      dot.classList.add('on');
      txt.textContent = 'llama-cpp';
      syncSettingsPanels();
    } else {
      alert(data.error || '切换失败');
      document.getElementById('bk-sel').value = backendType;
    syncSettingsPanels();
    }
  } catch (e) {
    alert('切换失败: ' + e.message);
    document.getElementById('bk-sel').value = backendType;
    syncSettingsPanels();
  }
}

function renderVendorModels(modelList, defaultModel) {
  const sel = document.getElementById('m-sel');
  sel.innerHTML = '<option value="">选择模型...</option>';
  (modelList || []).forEach(m => {
    sel.innerHTML += `<option value="${esc(m)}">${esc(m)}</option>`;
  });
  // 自定义厂商 + 手动输入框
  if (vendorId === 'custom') {
    sel.innerHTML += '<option value="__custom__">自定义模型名...</option>';
  }
  if (defaultModel) {
    selModel(defaultModel);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// 模型管理
// ──────────────────────────────────────────────────────────────────────────────
async function loadLlamaModels() {
  try {
    const res = await fetch('/api/llama/models');
    const data = await res.json();
    models = data.models || [];
    renderModelSelect();
    if (!curM && models.length) selModel(models[0]);
  } catch (e) {
    console.error('加载 llama 模型失败:', e);
  }
}

function renderModelSelect() {
  const sel = document.getElementById('m-sel');
  sel.innerHTML = '<option value="">选择模型...</option>' +
    models.map(m => {
      const path = m.path || m;
      const vision = m.has_vision ? ' 👁' : '';
      return `<option value="${esc(path)}">${esc(path)}${vision}</option>`;
    }).join('');
  if (curM) sel.value = curM;
}

function selModel(n) {
  // 自定义模型名
  if (n === '__custom__') {
    const customName = prompt('输入模型名:') || '';
    if (!customName) return;
    n = customName;
  }
  const modelPath = (typeof n === 'object') ? n.path : n;
  const modelObj = (typeof n === 'object') ? n : models.find(m => (m.path || m) === n);
  curM = modelPath;
  document.getElementById('m-sel').value = modelPath;
  document.getElementById('cur-m').textContent = modelPath || '未选择';
  if (backendType === 'llama-cpp' && modelPath) {
    loadLlamaModel(modelPath, modelObj);
  }
}

async function loadLlamaModel(modelName, modelObj) {
  const btn = document.getElementById('cur-m');
  const originalText = btn.textContent;
  btn.textContent = '加载中...';
  try {
    const body = { model: modelName, chat_handler: 'auto' };
    const res = await fetch('/api/llama/load_model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (res.ok) {
      const mmprojInfo = data.config?.mmproj_loaded ? ' (视觉已启用)' : '';
      btn.textContent = modelName + mmprojInfo;
    } else {
      btn.textContent = originalText;
      alert('加载失败: ' + (data.error || '未知错误'));
    }
  } catch (e) {
    btn.textContent = originalText;
    alert('加载失败: ' + e.message);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// 对话发送
// ──────────────────────────────────────────────────────────────────────────────

function toggleWebSearch() {
  webSearchEnabled = !webSearchEnabled;
  const btn = document.getElementById('wbtn');
  btn.classList.toggle('on', webSearchEnabled);
  btn.title = webSearchEnabled ? '联网搜索：开' : '联网搜索：关';
}

function togglePlanMode() {
  planModeEnabled = !planModeEnabled;
  console.log('[Plan] togglePlanMode →', planModeEnabled);
  const btn = document.getElementById('pbtn');
  btn.classList.toggle('on', planModeEnabled);
  btn.title = planModeEnabled ? 'Plan 模式：开（工具不执行，生成计划）' : 'Plan 模式：关';
  // 切换时给用户提示
  const inp = document.getElementById('inp');
  if (planModeEnabled) {
    inp.placeholder = 'Plan 模式已开启 — 工具调用将生成计划供审批...';
  } else {
    inp.placeholder = '输入消息，Shift+Enter 换行...';
  }
}

function shouldSearch(query) {
  if (query.length < 5) return false;
  const questionPattern = /[？?吗呢]/;
  if (questionPattern.test(query)) return true;
  const infoKeywords = /最新|今天|现在|当前|最近|新闻|发生|价格|天气|股价|排名|几点|多少|哪里|什么|怎么|如何|为什么|推荐|哪个|什么时候|是谁|几月|几号|周几|星期|今年|明年|去年|本月|上月|刚刚|实时|行情|走势|预报|预测|公布|发布|上市|开盘|收盘|涨幅|跌幅|市值|汇率|利率|指数|数据|统计|报告|通知|政策|法规|规定|调整|变化|更新/;
  return infoKeywords.test(query);
}

async function autoSearch(query) {
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const data = await res.json();
    if (!res.ok || !data.results || data.results.length === 0) return '';
    let ctx = '\n\n[联网搜索结果 — 请参考以下最新信息回答用户问题]\n';
    data.results.forEach((r, i) => {
      ctx += `${i + 1}. ${r.title}\n   ${r.snippet}\n   来源: ${r.url}\n`;
    });
    return ctx;
  } catch (e) {
    console.error('autoSearch failed:', e);
    return '';
  }
}

async function send() {
  const inp = document.getElementById('inp');
  const txt = inp.value.trim();
  if (!txt && !attF.length && !attI.length) return;
  if (!curM) { alert('请先选择模型'); return; }
  document.getElementById('empty')?.classList.add('hid');

  let content = txt;
  if (attF.length) {
    const filesText = attF.map(f => `--- 文件: ${f.name} ---\n${f.content}`).join('\n\n');
    content = txt ? txt + '\n\n' + filesText : filesText;
  }

  const tplId = document.getElementById('tpl-sel').value;
  let systemPrompt = '';
  if (tplId) {
    const t = tpls.find(t => t.id === tplId);
    if (t && t.system) systemPrompt = t.system;
  }

  // 联网搜索：开关开启 + 智能判断
  if (webSearchEnabled && shouldSearch(txt)) {
    const searchCtx = await autoSearch(txt);
    if (searchCtx) systemPrompt = (systemPrompt ? systemPrompt + searchCtx : searchCtx.trim());
  }

  // 记忆召回：搜索相关记忆并注入 system prompt
  const memCtx = await fetchMemoryContext(txt);
  if (memCtx) systemPrompt = (systemPrompt ? memCtx + '\n\n' + systemPrompt : memCtx);

  addMsg('usr', txt, [...attI, ...attF]);
  if (!currentSession) await createNewSession();  // 有内容才建会话记录
  await saveUserMsg(txt);
  inp.value = '';
  inp.style.height = 'auto';
  const savedImages = [...attI];
  const savedFiles = [...attF];
  attF = [];
  attI = [];
  document.getElementById('atchs').classList.add('hid');
  document.getElementById('atchs').innerHTML = '';

  const assistantMsg = addMsg('ast', '');
  planModeActive = false; // 每次发送重置 Plan 活动标记

  // AbortController for stopping
  abortCtrl = new AbortController();
  const sbtn = document.getElementById('sbtn');
  const stbtn = document.getElementById('stbtn');
  sbtn.style.display = 'none';
  stbtn.style.display = 'flex';

  // 上下文压缩
  _compressedMsgs = null;
  if (isVendorBackend(backendType) || backendType === 'llama-cpp') {
    _compressedMsgs = await maybeCompress(131072);
    if (_compressedMsgs) {
      addSystemMsg('⚡ 上下文已压缩（节省 token）', 'compressed');
    }
  }

  // 同步工具开关（根据当前后端读取对应 checkbox）
  toolsEnabled = isVendorBackend(backendType)
    ? document.getElementById('tools-vendor-enable')?.checked || false
    : document.getElementById('tools-local-enable')?.checked || false;

  // 加载工具定义（如果启用工具调用）
  if (toolsEnabled) await loadToolSchemas();

  try {
    if (backendType === 'llama-cpp') {
      await sendLlama(content, systemPrompt, savedImages, assistantMsg, abortCtrl.signal);
    } else if (isVendorBackend(backendType)) {
      await sendVendor(content, systemPrompt, savedImages, assistantMsg, abortCtrl.signal);
    } else {
      throw new Error('无可用后端');
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      const bubble = assistantMsg.querySelector('.msg-bubble');
      if (bubble && bubble.textContent.trim()) {
        bubble.innerHTML += '<br><span style="color:var(--muted);font-style:italic">[已停止]</span>';
      } else if (bubble) {
        bubble.innerHTML = '<span style="color:var(--muted);font-style:italic">[已停止]</span>';
      }
    } else {
      assistantMsg.querySelector('.ct').innerHTML = `<span class="err">请求失败: ${esc(e.message)}</span>`;
    }
  } finally {
    // 持久化：保存助手消息
    saveAssistantMsg(assistantMsg);
    // 空内容不留在历史里（Plan 模式下保留：计划卡片在 ct 中）
    const bubble = assistantMsg.querySelector('.msg-bubble');
    if (bubble && !bubble.textContent.trim() && !planModeActive) {
      assistantMsg.remove();
    }
    abortCtrl = null;
    sbtn.style.display = 'flex';
    stbtn.style.display = 'none';
  }
}

async function sendLlama(content, systemPrompt, images, msgEl, signal) {
  // 收集对话历史（多轮对话）
  const messages = [];
  if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
  // 优先使用压缩后的消息
  if (_compressedMsgs && _compressedMsgs.length) {
    for (const m of _compressedMsgs) messages.push(m);
  } else {
    const history = collectMessages();
    for (const m of history) messages.push(m);
  }
  // ── 工具调用: 后端注入提示词, 前端解析 XML ──
  const maxToolTurns = toolsEnabled && toolSchemas.length ? 5 : 1;

  const ctElement = msgEl.querySelector('.ct');
  let bubble = ctElement.querySelector('.msg-bubble');
  if (bubble) {
    bubble.innerHTML = '';
  } else {
    bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    const actions = ctElement.querySelector('.msg-actions');
    if (actions) ctElement.insertBefore(bubble, actions);
    else ctElement.appendChild(bubble);
  }

  for (let turn = 0; turn < maxToolTurns; turn++) {
    const body = {
      messages: messages,
      max_tokens: parseInt(document.getElementById('s-max-local').value),
      temperature: parseFloat(document.getElementById('s-temp-local').value),
      top_p: parseFloat(document.getElementById('s-topp-local').value),
      top_k: 40,
      repeat_penalty: 1.0,
      images: images.map(img => img.base64),
      stream: true
    };
    if (toolsEnabled && toolSchemas.length) body.tools = toolSchemas;

    const res = await fetch('/api/llama/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status}: ${err.slice(0, 200)}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let turnText = '';
    _tpsStart = Date.now();
    _tokenCount = 0;
    const bubbleBase = turn === 0 ? '' : bubble.innerHTML + '\n\n';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'reasoning') {
            _ensureReasoningBlock(msgEl, data.content);
            _tokenCount++;
            updateMsgTps(msgEl, _tokenCount, _tpsStart);
          } else if (data.content) {
            turnText += data.content;
            _tokenCount++;
            updateMsgTps(msgEl, _tokenCount, _tpsStart);
            // 流式渲染时隐藏可能存在的 tool_call 标签
            const displayText = turnText.replace(/<tool_call\s+name="[^"]*">[\s\S]*<\/tool_call>/gi, '⚙️ ...');
            bubble.innerHTML = bubbleBase + renderMarkdown(displayText);
            msgEl.scrollIntoView({ behavior: 'smooth', block: 'end' });
          }
          if (data.error) throw new Error(data.error);
        } catch (e) {
          if (e.message) throw e;
        }
      }
    }

    // ── 解析 XML 工具调用 ──
    const xmlCalls = extractXmlToolCalls(turnText);

    if (xmlCalls.length > 0) {
      // 移除 XML 标签后的纯文本作为 assistant content
      let cleanText = turnText.replace(/<tool_call\s+name="[^"]*">[\s\S]*?<\/tool_call>/gi, '').trim();

      messages.push({
        role: 'assistant',
        content: cleanText || null,
        tool_calls: xmlCalls.map(c => ({ function: { name: c.name, arguments: c.params } }))
      });

      for (const call of xmlCalls) {
        displayToolCall(msgEl, call.name, call.params);

        try {
          const params = JSON.parse(call.params || '{}');
          const result = await executeTool(call.name, params);
          messages.push({
            role: 'tool',
            name: call.name,
            content: JSON.stringify(result),
            tool_call_id: 'lc_' + call.name + '_' + turn
          });
          displayToolResult(msgEl, call.name, result);
        } catch (e) {
          messages.push({
            role: 'tool',
            name: call.name,
            content: JSON.stringify({ error: e.message }),
            tool_call_id: 'lc_' + call.name + '_' + turn
          });
          displayToolResult(msgEl, call.name, { error: e.message });
        }
      }

      continue;
    }

    // ── 无工具调用 → 最终响应 ──
    updateMsgTps(msgEl, _tokenCount, _tpsStart, true);
    bubble.dataset.turnText = turnText;
    if (!turnText) bubble.innerHTML = '(空响应)';
    else {
      _lastTurnText = turnText;
      if (_currentAssistantMsg) _currentAssistantMsg._turnText = turnText;
      renderFinal(msgEl, turnText);
    }
    return;
  }

  console.warn(`[llama-cpp] 工具调用已达 ${maxToolTurns} 轮上限`);
  bubble.innerHTML += '<div class="tool-warn"><span>⚠️</span> 工具轮次已达上限</div>';
}

// ── 厂商 API 发送 ──────────────────────────────────────────────────────
async function sendVendor(content, systemPrompt, images, msgEl, signal, overrideMessages) {
  let messages = [];
  if (overrideMessages) {
    // 使用传入的完整消息（clarify 续接等），深拷贝避免引用问题
    messages = overrideMessages.map(m => ({ ...m }));
  } else {
  if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
  // 优先使用压缩后的消息
  if (_compressedMsgs && _compressedMsgs.length) {
    for (const m of _compressedMsgs) {
      messages.push({ role: m.role, content: m.content });
    }
  } else {
    const prevMsgs = document.querySelectorAll('#msgs .msg');
    for (const m of prevMsgs) {
      const bubble = m.querySelector('.msg-bubble');
      if (!bubble) continue;
      const role = m.classList.contains('usr') ? 'user' : 'assistant';
      messages.push({ role, content: bubble.innerText || '' });
    }
  }
  if (images && images.length) {
    const imgsContent = images.map(img => ({
      type: 'image_url',
      image_url: { url: img.base64 }
    }));
    messages.push({
      role: 'user',
      content: [{ type: 'text', text: content || 'Describe this image.' }, ...imgsContent]
    });
  } else {
    messages.push({ role: 'user', content: content });
  }
  } // end else (not overrideMessages)

  // 保存完整消息（含 tool calls/results），用于 clarify 等需要继续的交互
  _fullMessages = messages.map(m => ({...m}));
  _currentAssistantMsg = null;
  _lastToolCallId = null;

  const creds = vendorCreds[vendorId] || {};
  const apiKey = creds.api_key || '';
  const baseUrl = creds.base_url || '';

  // 处理自定义模型名
  let model = curM;
  if (model === '__custom__') {
    model = prompt('输入模型名:') || '';
    if (!model) { msgEl.querySelector('.ct').innerHTML = '<span class="err">已取消</span>'; return; }
  }

  const maxToolTurns = toolsEnabled && toolSchemas.length ? 5 : 1;

  // 创建/清空 bubble（多轮工具调用复用同一个 bubble）
  const ctElement = msgEl.querySelector('.ct');
  let bubble = ctElement.querySelector('.msg-bubble');
  if (bubble) {
    bubble.innerHTML = '';
  } else {
    bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    const actions = ctElement.querySelector('.msg-actions');
    if (actions) ctElement.insertBefore(bubble, actions);
    else ctElement.appendChild(bubble);
  }

  // ── AgentLoop 服务端工具执行 ──
  const body = {
    vendor_id: vendorId,
    backend_type: "vendor",
    model: model,
    messages: messages,
    api_key: apiKey,
    base_url: baseUrl,
    max_tokens: parseInt(document.getElementById('s-max-vendor').value),
    temperature: parseFloat(document.getElementById('s-temp-vendor').value),
    top_p: parseFloat(document.getElementById('s-topp-vendor').value),
    plan_mode: planModeEnabled,  // Plan 模式：不执行工具，返回计划
    web_search: webSearchEnabled,  // 联网搜索开关：传至后端控制厂商原生搜索
  };

  const res = await fetch('/api/agent/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  let turnText = '';
  _tpsStart = Date.now();
  _tokenCount = 0;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let hasContent = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (line === 'data: [DONE]') continue;
      if (!line.startsWith('data: ')) continue;
      try {
        const event = JSON.parse(line.slice(6));
        switch (event.type) {
          case 'token':
            turnText += event.content;
            hasContent = true;
            bubble.innerHTML = renderMarkdown(turnText);
            msgEl.scrollIntoView({ behavior: 'smooth', block: 'end' });
            // Track in _fullMessages
            if (!_currentAssistantMsg) {
              _currentAssistantMsg = { role: 'assistant', content: '' };
              _fullMessages.push(_currentAssistantMsg);
            }
            _currentAssistantMsg.content += event.content;
            break;
          case 'tool_call':
            displayToolCall(msgEl, event.name, event.args);
            // Track in _fullMessages
            if (!_currentAssistantMsg) {
              _currentAssistantMsg = { role: 'assistant', content: null, tool_calls: [] };
              _fullMessages.push(_currentAssistantMsg);
            }
            if (!_currentAssistantMsg.tool_calls) _currentAssistantMsg.tool_calls = [];
            _lastToolCallId = 'call_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
            _currentAssistantMsg.tool_calls.push({
              id: _lastToolCallId,
              type: 'function',
              function: { name: event.name, arguments: JSON.stringify(event.args) }
            });
            break;
          case 'tool_result': {
            const parsed = JSON.parse(event.content);
            if (parsed && parsed.type === 'clarify_request') {
              displayClarifyRequest(msgEl, parsed);
            } else {
              displayToolResult(msgEl, event.name, parsed);
            }
            // Track in _fullMessages
            _fullMessages.push({
              role: 'tool',
              tool_call_id: _lastToolCallId || '',
              name: event.name,
              content: event.content
            });
            break;
          }
          case 'error':
            bubble.innerHTML += `<div class="err">${esc(event.content)}</div>`;
            return;
          case 'done':
            // 重置追踪状态
            _currentAssistantMsg = null;
            _lastToolCallId = null;
            _reasoningText = '';
            // 更新 token 计数显示
            if (event.reasoning_per_turn) {
              updateTokenDisplay(event.reasoning_per_turn);
            }
            break;
          case 'plan':
            // ── Plan 模式：渲染计划供用户审批 ──
            console.log('[Plan] SSE event received', event);
            hasContent = false;
            renderPlan(msgEl, event.items, event.assistant_message, event.turn);
            planModeActive = true;
            break;
          case 'plan_result':
            // Plan 执行结果：展示工具执行结果
            displayPlanResult(msgEl, event.tool, event.result);
            break;
          case 'plan_execute_done':
            // Plan 执行完成，后续正常流式对话
            break;
          case 'reasoning':
            _ensureReasoningBlock(msgEl, event.content);
            _tokenCount++;
            updateMsgTps(msgEl, _tokenCount, _tpsStart);
            if (!_currentAssistantMsg) {
              _currentAssistantMsg = { role: 'assistant', content: '' };
              _fullMessages.push(_currentAssistantMsg);
            }
            _reasoningText += event.content;
            break;
        }
      } catch (e) {
        if (e.message && !e.message.startsWith('[')) throw e;
      }
    }
  }

  // 渲染最终结果
  if (hasContent) {
    renderFinal(msgEl, turnText);
  } else if (!planModeActive) {
    // Plan 模式：保留空气泡（计划卡片已在 renderPlan 中渲染到 ct 中）
    bubble.innerHTML = '(空响应)';
  }
}

// ── 厂商 API 发送结束 ──────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────────────
// Markdown 渲染
// ──────────────────────────────────────────────────────────────────────────────
function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  var tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    var inputArea = document.querySelector('.input-area');
    if (inputArea) inputArea.appendChild(tokenEl);
  }
  var total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  let tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    let inputArea = document.querySelector('.input-area');
    if (inputArea) { inputArea.appendChild(tokenEl); }
  }
  let total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  let tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    let inputArea = document.querySelector('.input-area');
    if (inputArea) { inputArea.appendChild(tokenEl); }
  }
  let total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  let tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    let inputArea = document.querySelector('.input-area');
    if (inputArea) { inputArea.appendChild(tokenEl); }
  }
  let total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  let tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    let inputArea = document.querySelector('.input-area');
    if (inputArea) { inputArea.appendChild(tokenEl); }
  }
  let total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function updateTokenDisplay(reasoningPerTurn) {
  // 更新 token 计数显示（在输入框下方）
  let tokenEl = document.getElementById('token-count');
  if (!tokenEl) {
    tokenEl = document.createElement('div');
    tokenEl.id = 'token-count';
    tokenEl.style.cssText = 'font-size:11px;color:#888;margin-top:4px;';
    let inputArea = document.querySelector('.input-area');
    if (inputArea) { inputArea.appendChild(tokenEl); }
  }
  let total = reasoningPerTurn.reduce(function(a, b) { return a + b; }, 0);
  tokenEl.textContent = 'Tokens: ' + total + ' (this turn: ' + (reasoningPerTurn[reasoningPerTurn.length-1] || 0) + ')';
}

function renderMarkdown(text) {
  let html = esc(text);
  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<div class="code-block"><div class="code-hd"><span>${lang || 'code'}</span><button class="code-copy" onclick="copyCode(this)">复制</button></div><pre><code>${code.trim()}</code></pre></div>`;
  });
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" style="color:var(--accent);text-decoration:none">$1</a>');
  // Lists
  html = html.replace(/^\s*[-*]\s+(.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul style="margin:8px 0;padding-left:20px">$&</ul>');
  // Line breaks
  html = html.replace(/\n/g, '<br>');
  return html;
}

// ──────────────────────────────────────────────────────────────────────────────
// 最终渲染
// ──────────────────────────────────────────────────────────────────────────────
function renderFinal(msgEl, rawText) {
  const bubble = msgEl.querySelector('.msg-bubble');
  if (!bubble) return;

  // 移除 think 标签，只保留回复内容
  let cleanText = rawText.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
  cleanText = cleanText.replace(/[\s\S]*?<\/think>/, '').trim();
  
  bubble.innerHTML = renderMarkdown(cleanText || rawText);
}

// ──────────────────────────────────────────────────────────────────────────────
// 消息操作
// ──────────────────────────────────────────────────────────────────────────────
function copyMsg(btn) {
  const msgEl = btn.closest('.msg');
  const bubble = msgEl.querySelector('.msg-bubble');
  var text = bubble.innerText;
  // 追加附件信息
  try {
    var attData = bubble.dataset.attachments;
    if (attData) {
      var att = JSON.parse(attData);
      var names = [];
      (att.images || []).forEach(function(img) { names.push(img.name || 'image.jpg'); });
      (att.files || []).forEach(function(f) { names.push(f.name || 'file'); });
      if (names.length) text += '\n\n[附件: ' + names.join(', ') + ']';
    }
  } catch(e) {}
  navigator.clipboard.writeText(text.trim()).then(() => {
    const original = btn.textContent;
    btn.textContent = '✓';
    setTimeout(() => btn.textContent = original, 1500);
  });
}

function editMsg(btn) {
  const msgEl = btn.closest('.msg');
  const bubble = msgEl.querySelector('.msg-bubble');
  const text = bubble.innerText;
  restoreAttachments(bubble);
  const inp = document.getElementById('inp');
  inp.value = text;
  inp.focus();
  ar(inp);
  // Remove this message and all after it
  let remove = false;
  document.querySelectorAll('.msg').forEach(m => {
    if (remove) m.remove();
    if (m === msgEl) {
      remove = true;
      m.remove();
    }
  });
}

function deleteMsg(btn) {
  const msgEl = btn.closest('.msg');
  if (confirm('删除这条消息?')) {
    msgEl.remove();
  }
}

function regenerateMsg(btn) {
  const msgEl = btn.closest('.msg');
  if (!msgEl) return;

  // 恢复附件
  const msgBubble = msgEl.querySelector('.msg-bubble');
  restoreAttachments(msgBubble);

  // Find the previous user message
  let prevUserMsg = null;
  const allMsgs = Array.from(document.querySelectorAll('.msg'));
  const idx = allMsgs.indexOf(msgEl);
  for (let i = idx - 1; i >= 0; i--) {
    if (allMsgs[i].classList.contains('usr')) {
      prevUserMsg = allMsgs[i];
      break;
    }
  }
  if (!prevUserMsg) return;

  const bubble = prevUserMsg.querySelector('.msg-bubble');
  const text = bubble ? bubble.innerText.trim() : '';
  if (!text) return;

  // Remove this assistant message and all after it
  let remove = false;
  allMsgs.forEach(m => {
    if (remove) m.remove();
    if (m === msgEl) {
      remove = true;
      m.remove();
    }
  });

  // Remove the old user message (send() will re-add it)
  prevUserMsg.remove();

  // Re-send
  const inp = document.getElementById('inp');
  inp.value = text;
  send();
}

function copyCode(btn) {
  const code = btn.closest('.code-block').querySelector('code');
  navigator.clipboard.writeText(code.textContent).then(() => {
    btn.textContent = '已复制';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.textContent = '复制';
      btn.classList.remove('copied');
    }, 1500);
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// 消息添加
// ──────────────────────────────────────────────────────────────────────────────
function addMsg(role, txt, att = []) {
  const c = document.getElementById('msgs');
  const d = document.createElement('div');
  d.className = 'msg ' + (role === 'usr' ? 'usr' : 'ast');

  // Content container
  const ct = document.createElement('div');
  ct.className = 'ct';

  // Bubble
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  let h = '';
  if (att.length) {
    const imgs = att.filter(a => a.type === 'image');
    const files = att.filter(a => a.type === 'file');
    h += imgs.map(a => `<img src="${esc(a.preview)}" style="max-width:200px;border-radius:8px;margin:4px;display:block">`).join('');
    h += files.map(a => `<div style="background:var(--input);padding:4px 8px;border-radius:4px;font-size:12px;margin:4px;color:var(--text2);display:inline-block">📄 ${esc(a.name)}</div>`).join('');
  }
  if (txt) {
    if (role === 'ast') {
      h += renderMarkdown(txt);
    } else {
      h += esc(txt);
    }
  }
  bubble.innerHTML = h || '...';
  ct.appendChild(bubble);

  // Actions
  const actions = document.createElement('div');
  actions.className = 'msg-actions';
  if (role === 'ast') {
    actions.innerHTML = `
      <button class="msg-btn" onclick="copyMsg(this)" title="复制">📋</button>
      <button class="msg-btn" onclick="regenerateMsg(this)" title="重新生成">🔄</button>
      <button class="msg-btn" onclick="deleteMsg(this)" title="删除">🗑️</button>
    `;
  } else {
    actions.innerHTML = `
      <button class="msg-btn" onclick="copyMsg(this)" title="复制">📋</button>
      <button class="msg-btn" onclick="editMsg(this)" title="编辑">✏️</button>
      <button class="msg-btn" onclick="deleteMsg(this)" title="删除">🗑️</button>
    `;
  }
  ct.appendChild(actions);

  d.appendChild(ct);
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
  return d;
}

// ── 系统消息条 ───────────────────────────────────────────────────────────────
function addSystemMsg(txt, cls = '') {
  const c = document.getElementById('msgs');
  const d = document.createElement('div');
  d.className = 'sys-msg' + (cls ? ' ' + cls : '');
  d.textContent = txt;
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
  // 3 秒后自动淡出
  setTimeout(() => {
    d.style.opacity = '0';
    d.style.transition = 'opacity 0.5s';
    setTimeout(() => d.remove(), 500);
  }, 3000);
  return d;
}

// ──────────────────────────────────────────────────────────────────────────────
// Web 搜索
// ──────────────────────────────────────────────────────────────────────────────
async function webSearch() {
  const query = prompt('请输入搜索关键词:');
  if (!query) return;
  const searchBtn = document.querySelector('.sbtn');
  const originalText = searchBtn.textContent;
  searchBtn.textContent = '🔍';
  searchBtn.disabled = true;
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const data = await res.json();
    if (!res.ok) {
      alert('搜索失败: ' + (data.error || '未知错误'));
      return;
    }
    showSearchResults(data);
  } catch (e) {
    alert('搜索请求失败: ' + e.message);
  } finally {
    searchBtn.textContent = originalText;
    searchBtn.disabled = false;
  }
}

function showSearchResults(data) {
  const resultsDiv = document.createElement('div');
  resultsDiv.className = 'search-results';
  resultsDiv.style.cssText = 'background:var(--input);border-radius:12px;padding:12px;margin:8px 0;max-height:300px;overflow-y:auto';
  let html = `<div style="display:flex;justify-content:space-between;margin-bottom:8px">
    <strong>🔍 搜索结果: ${esc(data.query)}</strong>
    <button onclick="this.parentElement.parentElement.remove()" style="background:none;border:none;color:var(--muted);cursor:pointer">✕</button>
  </div>`;
  if (data.results.length === 0) {
    html += '<p>没有找到结果</p>';
  } else {
    for (const r of data.results) {
      html += `
        <div style="margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border)">
          <a href="${esc(r.url)}" target="_blank" style="font-weight:bold;color:var(--accent);text-decoration:none">${esc(r.title)}</a>
          <div style="font-size:12px;color:var(--muted);margin:4px 0">${esc(r.url)}</div>
          <div style="font-size:13px;color:var(--text2)">${esc(r.snippet)}</div>
          <button class="use-search" data-url="${esc(r.url)}" data-title="${esc(r.title)}" style="margin-top:4px;padding:2px 8px;font-size:12px;background:var(--accent);border:none;border-radius:4px;cursor:pointer">📋 插入链接</button>
        </div>
      `;
    }
  }
  resultsDiv.innerHTML = html;
  const msgsDiv = document.getElementById('msgs');
  msgsDiv.appendChild(resultsDiv);
  msgsDiv.scrollTop = msgsDiv.scrollHeight;
  resultsDiv.querySelectorAll('.use-search').forEach(btn => {
    btn.onclick = () => {
      const url = btn.dataset.url;
      const title = btn.dataset.title;
      const inp = document.getElementById('inp');
      const link = `[${title}](${url})`;
      inp.value = inp.value ? inp.value + '\n' + link : link;
      inp.focus();
      resultsDiv.remove();
    };
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// UI 辅助
// ──────────────────────────────────────────────────────────────────────────────
function nav(n) {
  document.querySelectorAll('.nav').forEach(e => e.classList.toggle('act', e.dataset.n === n));
  document.querySelectorAll('.panel').forEach(e => e.classList.toggle('act', e.id === 'p-' + n));
  const titles = { chat: '对话', translate: '翻译', templates: '模板', skills: '技能', cron: '定时任务', plugins: '插件', settings: '设置', 'api-docs': 'API 文档', help: '帮助', monitor: '监控', 'approval-page': '审批', logs: '日志' };
  if (n === 'skills') loadSkillsList();
  if (n === 'cron') loadCronJobs();
  if (n === 'plugins') loadPlugins();
  document.getElementById('pg-ttl').textContent = titles[n] || n;
}

function toggleSB() {
  document.getElementById('sidebar').classList.toggle('collapsed');
}

function onKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
}

function ar(e) {
  e.style.height = 'auto';
  e.style.height = Math.min(e.scrollHeight, 200) + 'px';
}

function esc(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// 文件上传
function hFiles(inp) {
  for (const f of inp.files) {
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    if (f.type.startsWith('image/')) {
      const r = new FileReader();
      r.onload = e => {
        attI.push({ name: f.name, preview: e.target.result, base64: e.target.result, type: 'image' });
        renderAttachments();
      };
      r.readAsDataURL(f);
    } else if (['.txt', '.srt', '.vtt', '.ass', '.ssa', '.sub', '.py', '.json', '.cpp', '.html', '.js'].includes(ext)) {
      const r = new FileReader();
      r.onload = e => {
        attF.push({ name: f.name, content: e.target.result, type: 'file' });
        renderAttachments();
      };
      r.readAsText(f);
    }
  }
  inp.value = '';
}

function renderAttachments() {
  const c = document.getElementById('atchs');
  const all = [...attI.map(a => ({ ...a, type: 'image' })), ...attF];
  if (!all.length) {
    c.classList.add('hid');
    return;
  }
  c.classList.remove('hid');
  c.innerHTML = all.map((a, i) => `<div class="atch">${a.type === 'image' ? '<img src="' + esc(a.preview) + '">' : '📄'}<span>${esc(a.name)}</span><span class="rm" onclick="rmA(${i})">✕</span></div>`).join('');
}

function rmA(i) {
  if (i < attI.length) attI.splice(i, 1);
  else attF.splice(i - attI.length, 1);
  renderAttachments();
}

// 拖拽上传
(function () {
  const z = document.getElementById('dz');
  if (!z) return;
  z.addEventListener('dragover', e => { e.preventDefault(); z.style.borderColor = 'var(--accent)'; });
  z.addEventListener('dragleave', e => { e.preventDefault(); z.style.borderColor = ''; });
  z.addEventListener('drop', e => {
    e.preventDefault();
    z.style.borderColor = '';
    hFiles({ files: e.dataTransfer.files, value: '' });
  });
})();

// 模板
async function loadT() {
  const r = await fetch('/api/prompt_templates').then(r => r.json()).catch(() => ({}));
  tpls = r.templates || [];
  renderTemplatesList();
  renderTemplateSelect();
}

function renderTemplateSelect() {
  const s = document.getElementById('tpl-sel');
  s.innerHTML = '<option value="">普通对话</option>' + tpls.map(t => `<option value="${esc(t.id)}">${esc(t.name)}</option>`).join('');
}

function renderTemplatesList() {
  document.getElementById('tl-list').innerHTML = tpls.map(t => `<div class="tpl-i${t.id === curTpl ? ' act' : ''}" onclick="edTpl('${esc(t.id)}')"><div class="n">${esc(t.name)}${t.builtin ? '<span class="bdg">内置</span>' : ''}</div><div class="d">${esc(t.description || '')}</div></div>`).join('');
}

async function edTpl(id) {
  curTpl = id;
  renderTemplatesList();
  const r = await fetch('/api/prompt_templates/' + encodeURIComponent(id)).then(r => r.json());
  const t = r.template;
  const ed = document.getElementById('tpl-ed');
  ed.innerHTML = `<div class="tpl-form">
    <div class="fg"><label>ID</label><input id="ti" value="${esc(id)}" ${t.builtin ? 'readonly' : ''}></div>
    <div class="fg"><label>名称</label><input id="tn" value="${esc(t.name || '')}"></div>
    <div class="fg"><label>描述</label><input id="td" value="${esc(t.description || '')}"></div>
    <div class="fg"><label>System Prompt</label><textarea id="ts">${esc(t.system || '')}</textarea></div>
    <div class="fg"><label>前缀</label><input id="tp" value="${esc(t.prefix || '')}"></div>
    <div class="fg"><label>后缀</label><input id="tf" value="${esc(t.suffix || '')}"></div>
    <div class="fa"><button class="pri" onclick="svTpl()">保存</button>${t.builtin ? '' : '<button class="dan" onclick="dlTpl()">删除</button>'}<button onclick="loadT()">取消</button></div></div>`;
}

function newTpl() {
  curTpl = null;
  renderTemplatesList();
  document.getElementById('tpl-ed').innerHTML = `<div class="tpl-form">
    <div class="fg"><label>ID</label><input id="ti" placeholder="my_template"></div>
    <div class="fg"><label>名称</label><input id="tn" placeholder="我的模板"></div>
    <div class="fg"><label>描述</label><input id="td"></div>
    <div class="fg"><label>System Prompt</label><textarea id="ts"></textarea></div>
    <div class="fg"><label>前缀</label><input id="tp"></div>
    <div class="fg"><label>后缀</label><input id="tf"></div>
    <div class="fa"><button class="pri" onclick="svTpl()">保存</button><button onclick="loadT()">取消</button></div></div>`;
}

async function svTpl() {
  const id = document.getElementById('ti').value.trim();
  const name = document.getElementById('tn').value.trim();
  if (!id || !name) { alert('ID和名称必填'); return; }
  await fetch('/api/prompt_templates', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id, name,
      description: document.getElementById('td').value.trim(),
      system: document.getElementById('ts').value,
      prefix: document.getElementById('tp').value,
      suffix: document.getElementById('tf').value
    })
  });
  await loadT();
  curTpl = id;
  renderTemplatesList();
}

async function dlTpl() {
  if (!curTpl || !confirm('删除此模板?')) return;
  await fetch('/api/prompt_templates/' + encodeURIComponent(curTpl), { method: 'DELETE' });
  curTpl = null;
  await loadT();
}

// 翻译
async function doTr() {
  const txt = document.getElementById('st').value.trim();
  if (!txt) return;
  if (!curM) { alert('请先选择模型'); return; }
  const sl = document.getElementById('sl').value;
  const tl = document.getElementById('tl').value;
  const names = { auto: '自动', ja: '日语', ko: '韩语', zh: '中文', en: '英语' };
  const prompt = `你是专业的${names[sl]}到${names[tl]}翻译专家。只输出翻译结果。\n\n${txt}`;
  const out = document.getElementById('to');
  out.value = '';
  if (backendType === 'llama-cpp') {
    const res = await fetch('/api/llama/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, temperature: 0.3, max_tokens: 4096 })
    });
    const data = await res.json();
    out.value = data.output || '';
  } else if (isVendorBackend(backendType)) {
    await doTrVendor(prompt, out);
  } else {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: curM, messages: [{ role: 'user', content: prompt }], temperature: 0.3 })
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '', full = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const d = JSON.parse(line.slice(6));
          if (d.message?.content) {
            full += d.message.content;
            out.value = full;
          }
        } catch { }
      }
    }
  }
}

function clrTr() {
  document.getElementById('st').value = '';
  document.getElementById('to').value = '';
}

async function doTrVendor(prompt, outEl) {
  const creds = vendorCreds[vendorId] || {};
  const apiKey = creds.api_key || '';
  const baseUrl = creds.base_url || '';
  const body = {
    vendor: vendorId,
    model: curM,
    messages: [{ role: 'user', content: prompt }],
    api_key: apiKey,
    base_url: baseUrl,
    temperature: 0.3,
    max_tokens: 4096,
  };
  const res = await fetch('/api/vendors/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    outEl.value = '错误: ' + (err.error || '请求失败');
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', full = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (line === 'data: [DONE]') continue;
      if (!line.startsWith('data: ')) continue;
      try {
        const d = JSON.parse(line.slice(6));
        if (d.content) { full += d.content; outEl.value = full; }
      } catch { }
    }
  }
}

// 设置
function us(k, sfx) {
  const s = sfx || '';
  const ids = { temp: 's-temp' + (s ? '-' + s : ''), max: 's-max' + (s ? '-' + s : ''), topp: 's-topp' + (s ? '-' + s : '') };
  const vals = { temp: 'v-temp' + (s ? '-' + s : ''), max: 'v-max' + (s ? '-' + s : ''), topp: 'v-topp' + (s ? '-' + s : '') };
  document.getElementById(vals[k]).textContent = document.getElementById(ids[k]).value;
}

function syncVendorToSettings(vdef) {
  const sec = document.getElementById('set-vendor');
  const vdef_ = vdef || vendorDefs[vendorId] || {};
  const vc = vendorCreds[vendorId] || {};
  if (isVendorBackend(backendType)) {
    sec.style.display = 'block';
    document.getElementById('set-vendor-name').textContent = vdef_.name ? ' — ' + vdef_.name : '';
    document.getElementById('set-api-key').value = vc.api_key || '';
    document.getElementById('set-base-url').value = vc.base_url || vdef_.base_url || '';
    // key 状态提示
    const hasServerKey = vdef_.has_server_key;
    const hasLocalKey = vc.api_key;
    const statusEl = document.getElementById('set-key-status');
    if (hasServerKey) {
      statusEl.innerHTML = '<span style="color:#4ade80">✓ 已通过环境变量配置</span>';
    } else if (hasLocalKey) {
      statusEl.innerHTML = '<span style="color:#4ade80">✓ 已保存</span>';
    } else {
      statusEl.innerHTML = '<span style="color:#f59e0b">⚠ 需要输入 API Key</span>';
    }
  } else {
    sec.style.display = 'none';
  }
}

function syncVendorKey() {
  // 不再同步到侧边栏（侧边栏已移除）
}

function syncVendorUrl() {
  // 不再同步到侧边栏（侧边栏已移除）
}

// 输入框变动时同步到内存凭据 + 更新状态
function onVendorCredChanged() {
  if (!vendorId || !isVendorBackend(backendType)) return;
  const key = document.getElementById('set-api-key')?.value || '';
  const url = document.getElementById('set-base-url')?.value || '';
  vendorCreds[vendorId] = { api_key: key, base_url: url };
  const vdef = vendorDefs[vendorId] || {};
  const statusEl = document.getElementById('set-key-status');
  if (vdef.has_server_key) {
    statusEl.innerHTML = '<span style="color:#4ade80">✓ 已通过环境变量配置</span>';
  } else if (key) {
    statusEl.innerHTML = '<span style="color:#4ade80">✓ 已保存</span>';
  } else {
    statusEl.innerHTML = '<span style="color:#f59e0b">⚠ 需要输入 API Key</span>';
  }
}

// ── 导航覆写 ────────────────────────────────────────────────────────
let _skipChatNewSession = false;
const _origNav = nav;
nav = function (n) {
  _origNav(n);
  if (n === 'settings') syncVendorToSettings();
  if (n === 'chat' && !_skipChatNewSession) newChat();  // 点击对话 → 新会话
};

// 新对话：清空界面，不发消息不建会话记录
async function newChat() {
  document.getElementById('msgs').innerHTML = '<div class="empty" id="empty"><h2>开始对话</h2><p>选择模型后输入消息，或拖入文件</p></div>';
  localStorage.removeItem('currentSessionId');
  currentSession = null;
  // 不建会话 — 等用户发出第一条消息时再建
  renderSessionList();
}

// 启动
init();

// ── 会话持久化 ──────────────────────────────────────────────────────

async function initSession() {
  let sessionId = localStorage.getItem('currentSessionId');
  if (sessionId) {
    try {
      const r = await fetch('/api/sessions/' + sessionId);
      if (r.ok) {
        currentSession = await r.json();
        await loadSessionMessages(sessionId);
        renderSessionList();
        return;
      }
    } catch (e) { /* 会话已过期 */ }
  }
  await newChat();  // 不自动建会话 — 等用户发消息再建
}

async function createNewSession() {
  const r = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ backend: backendType, model: curM }),
  });
  currentSession = await r.json();
  localStorage.setItem('currentSessionId', currentSession.id);
  renderSessionList();
}

async function loadSessionMessages(sessionId) {
  try {
    const r = await fetch('/api/sessions/' + sessionId + '/messages');
    if (!r.ok) return;
    const data = await r.json();
    if (!data.messages || data.messages.length === 0) return;
    const ct = document.getElementById('msgs');
    ct.innerHTML = '';
    for (const m of data.messages) {
      if (m.role === 'user') addMsg('usr', m.content || '');
      else if (m.role === 'assistant') addMsg('ast', m.content || '');
    }
    scroller();
  } catch (e) {}
}

async function switchSession(sessionId) {
  // 同一会话 + 已在对话页：无需操作；在其他页面（如设置）仍需切过去
  if (currentSession && currentSession.id === sessionId && document.getElementById('p-chat').classList.contains('act')) return;
  _skipChatNewSession = true;
  nav('chat');
  _skipChatNewSession = false;
  localStorage.setItem('currentSessionId', sessionId);
  document.getElementById('msgs').innerHTML = '<div class="empty">加载中...</div>';
  const r = await fetch('/api/sessions/' + sessionId);
  if (!r.ok) return;
  currentSession = await r.json();
  await loadSessionMessages(sessionId);
  renderSessionList();
}

async function renameSession(title) {
  if (!currentSession) return;
  title = (title || '').trim() || '未命名会话';
  await fetch('/api/sessions/' + currentSession.id, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  currentSession.title = title;
  renderSessionList();
}

async function deleteSession(sessionId) {
  if (!confirm('确定删除此会话？')) return;
  await fetch('/api/sessions/' + sessionId, { method: 'DELETE' });
  if (currentSession && currentSession.id === sessionId) {
    localStorage.removeItem('currentSessionId');
    currentSession = null;
    const others = sessionList.filter(s => s.id !== sessionId);
    if (others.length > 0) {
      await switchSession(others[0].id);
    } else {
      await newChat();  // 等用户发消息再建
    }
  } else {
    renderSessionList();
  }
}

async function renderSessionList() {
  try {
    const r = await fetch('/api/sessions?limit=30');
    const data = await r.json();
    sessionList = data.sessions || [];
    const container = document.getElementById('session-list');
    if (!container) return;
    container.innerHTML = sessionList.map(s => {
      const active = currentSession && currentSession.id === s.id ? ' active' : '';
      const preview = (s.preview || '').substring(0, 25);
      const time = s.updated_at ? new Date(s.updated_at * 1000).toLocaleString('zh-CN', {month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '';
      return `<div class="sitem${active}" onclick="switchSession('${s.id}')">
        <span class="sitem-name">${esc((s.title || s.id||'').substring(0,15))}</span>
        <span class="sitem-count">${s.message_count||0}条</span>
        <button class="sitem-del" onclick="event.stopPropagation();deleteSession('${s.id}')">×</button>
      </div>`;
    }).join('') || '<div class="sitem empty-hint">暂无历史会话</div>';
  } catch (e) {}
}

async function saveMsg(role, content) {
  if (!currentSession || !currentSession.id) return;
  try {
    await fetch('/api/sessions/' + currentSession.id + '/messages', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role, content }),
    });
    if (currentSession) currentSession.message_count = (currentSession.message_count||0) + 1;
  } catch (e) {}
}

function saveAssistantMsg(bubble, msgEl) {
  if (!bubble) return;
  // Store reasoning text from dataset
  const reasoningText = bubble.dataset.reasoningText || _reasoningText || '';
  const turnText = _lastTurnText || bubble.dataset.turnText || bubble.innerText || '';
  const content = reasoningText ? (reasoningText + '\n\n---\n\n' + turnText) : turnText;
  // Save with attachments
  if (attI.length || attF.length) {
    bubble.dataset.attachments = JSON.stringify({ attI: attI, attF: attF });
  }
  saveMsg('assistant', content);
  // Clear per-turn state
  _reasoningText = '';
  _tokenCount = 0;
  _tpsStart = 0;
}

async function saveUserMsg(txt) {
  if (!currentSession) return;
  await saveMsg('user', txt);
  // 从首条用户消息生成标题
  if (currentSession.message_count <= 2 && (!currentSession.title || currentSession.title.startsWith('202'))) {
    const title = txt.substring(0, 30) || '未命名会话';
    await renameSession(title);
  }
}

// ── 记忆管理 ────────────────────────────────────────────────────────

async function loadMemoryList() {
  const container = document.getElementById('memory-list');
  if (!container) return;
  try {
    const r = await fetch('/api/memory');
    const data = await r.json();
    container.innerHTML = (data.memories||[]).map(m =>
      `<div class="mem-item">
        <div class="mem-key">${esc(m.key)} <span class="mem-cat">[${esc(m.category)}]</span></div>
        <div class="mem-val">${esc((m.value||'').substring(0,100))}</div>
        <button class="mem-del" onclick="deleteMemory('${esc(m.key)}')">删除</button>
      </div>`
    ).join('') || '<div class="sitem empty-hint">暂无记忆</div>';
  } catch (e) {}
}

async function saveMemoryItem() {
  const key = document.getElementById('mem-key')?.value?.trim();
  const val = document.getElementById('mem-val')?.value?.trim();
  if (!key || !val) return alert('请输入 key 和 value');
  await fetch('/api/memory', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, value: val, category: document.getElementById('mem-cat')?.value||'general' }),
  });
  document.getElementById('mem-key').value = '';
  document.getElementById('mem-val').value = '';
  loadMemoryList();
}

async function deleteMemory(key) {
  if (!confirm('删除: ' + key + '?')) return;
  await fetch('/api/memory/' + encodeURIComponent(key), { method: 'DELETE' });
  loadMemoryList();
}

// 从记忆库召回相关上下文（注入到 system prompt）
async function fetchMemoryContext(query) {
  if (!query || !query.trim()) return '';
  try {
    const r = await fetch('/api/memory/search?q=' + encodeURIComponent(query));
    if (!r.ok) return '';
    const data = await r.json();
    if (!data.results || data.results.length === 0) return '';
    return '<memory-context>\n[System note: 以下是根据当前对话召回的已存储记忆——作为背景参考，不是新的用户输入]\n\n' +
      data.results.map(m => `- [${m.category}] ${m.key}: ${m.value}`).join('\n') +
      '\n</memory-context>';
  } catch (e) { return ''; }
}

// ── 搜索 ────────────────────────────────────────────────────────────

async function doSearch(query) {
  const resEl = document.getElementById('search-results');
  if (!resEl) return;
  if (!query || !query.trim()) { resEl.innerHTML = ''; return; }
  try {
    const r = await fetch('/api/messages/search', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit: 15 }),
    });
    const data = await r.json();
    const items = data.results || [];
    resEl.innerHTML = items.map(m =>
      `<div class="sr-item" onclick="switchSession('${m.session_id}')">
        <div class="sr-head">${esc(m.session_title || m.session_id)}</div>
        <div class="sr-snip">${m.snippet || esc((m.content||'').substring(0,80))}</div>
      </div>`
    ).join('') || '<div class="sitem empty-hint">无匹配结果</div>';
  } catch (e) { resEl.innerHTML = '<div class="sitem" style="color:#f59e0b">搜索失败</div>'; }
}

// ── 上下文压缩 ──────────────────────────────────────────────────────

// 粗略估算消息 token 数 (~4 chars/token)
function estimateTokens(messages) {
  let total = 0;
  for (const m of messages) {
    const c = typeof m.content === 'string' ? m.content : JSON.stringify(m.content);
    total += c.length;
  }
  return Math.ceil(total / 4);
}

// 从 DOM 提取完整对话历史
function collectMessages() {
  const msgs = [];
  const nodes = document.querySelectorAll('#msgs .msg');
  for (const el of nodes) {
    const bubble = el.querySelector('.msg-bubble');
    if (!bubble) continue;
    const role = el.classList.contains('usr') ? 'user' : 'assistant';
    msgs.push({ role, content: bubble.innerText || '' });
  }
  return msgs;
}

// 上下文压缩：当消息超过阈值时调用 /api/compress
// 返回压缩后的 messages 数组，未压缩则返回 null
async function maybeCompress(contextLength) {
  const msgs = collectMessages();
  const est = estimateTokens(msgs);
  const threshold = Math.floor(contextLength * 0.50);

  if (est < threshold) return null; // 未达阈值，无需压缩

  try {
    const provider = backendType;

    const r = await fetch('/api/compress', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: msgs,
        context_length: contextLength,
        threshold_percent: 0.50,
        provider: provider,
      }),
    });
    if (!r.ok) return null;
    const data = await r.json();
    if (data.was_compressed) {
      console.log(`Compressed ${data.original_count}→${data.compressed_count} msgs, saved ~${data.saved_tokens} tokens`);
      return data.compressed;
    }
  } catch (e) {
    console.error('Compression failed:', e);
  }
  return null;
}

// ── 工具调用 ────────────────────────────────────────────────────────

async function loadToolSchemas() {
  try {
    const r = await fetch('/api/tools/list');
    if (!r.ok) return;
    const data = await r.json();
    toolSchemas = data.tools || [];
    console.log(`Loaded ${toolSchemas.length} tool schemas`);
    // 同步复选框状态 — 根据当前后端
    const cbId = isVendorBackend(backendType) ? 'tools-vendor-enable' : 'tools-local-enable';
    const toggle = document.getElementById(cbId);
    if (toggle) toggle.checked = toolsEnabled;
    // 更新状态显示
    const statusId = isVendorBackend(backendType) ? 'tools-status-vendor' : 'tools-status-local';
    const status = document.getElementById(statusId);
    if (status) {
      status.textContent = toolsEnabled
        ? `✅ 已启用 (${toolSchemas.length} 个工具)`
        : `⏸ 已禁用 (${toolSchemas.length} 个工具可用)`;
    }
    return toolSchemas;
  } catch (e) {
    console.error('Failed to load tool schemas:', e);
    ['tools-status-local', 'tools-status-vendor'].forEach(id => {
      const s = document.getElementById(id);
      if (s) s.textContent = '⚠️ 加载工具失败';
    });
    return [];
  }
}

async function executeTool(name, params) {
  try {
    const r = await fetch('/api/tools/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, params })
    });
    if (!r.ok) {
      const errData = await r.json().catch(() => ({}));
      throw new Error(errData.message || errData.error || `HTTP ${r.status}`);
    }
    const data = await r.json();
    return data.result !== undefined ? data.result : data;
  } catch (e) {
    return { error: e.message };
  }
}

function displayToolCall(msgEl, toolName, argsStr) {
  const ct = msgEl.querySelector('.ct');
  if (!ct) return;
  let argsPreview = '';
  try {
    const parsed = typeof argsStr === 'string' ? JSON.parse(argsStr) : argsStr;
    argsPreview = JSON.stringify(parsed).substring(0, 120);
  } catch { argsPreview = (argsStr || '').substring(0, 120); }
  const div = document.createElement('div');
  div.className = 'tool-ind tool-call';
  div.innerHTML = `<span class="tool-icon">🔧</span> <strong>${esc(toolName)}</strong> <span class="tool-args">${esc(argsPreview)}</span>`;
  ct.appendChild(div);
  msgEl.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function displayToolResult(msgEl, toolName, result) {
  const ct = msgEl.querySelector('.ct');
  if (!ct) return;
  const resultText = typeof result === 'string' ? result :
    (result && typeof result === 'object' ? (result.content || result.error || JSON.stringify(result).substring(0, 200)) : String(result));
  const isErr = result && result.error;
  const div = document.createElement('div');
  div.className = 'tool-ind ' + (isErr ? 'tool-err' : 'tool-result');
  div.innerHTML = `<span class="tool-icon">${isErr ? '❌' : '📋'}</span> <strong>${esc(toolName)}</strong> <span class="tool-out">${esc(resultText)}</span>`;
  ct.appendChild(div);
  msgEl.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

// ── Clarify request (interactive question) ─────────────────────────
function displayClarifyRequest(msgEl, data) {
  const ct = msgEl.querySelector('.ct');
  if (!ct) return;
  const div = document.createElement('div');
  div.className = 'tool-ind clarify-req';
  let buttonsHtml = '';
  if (data.choices_offered && data.choices_offered.length > 0) {
    buttonsHtml = '<div class="clarify-choices">';
    data.choices_offered.forEach((choice, i) => {
      buttonsHtml += `<button class="clarify-btn" data-answer="${esc(choice)}" onclick="handleClarifyChoice(this)">${i + 1}. ${esc(choice)}</button>`;
    });
    buttonsHtml += `<button class="clarify-btn" onclick="handleClarifyOther()">${data.choices_offered.length + 1}. Other (type your answer)</button>`;
    buttonsHtml += '</div>';
  } else {
    buttonsHtml = '<div class="clarify-choices"><input class="clarify-input" id="clarify-text-input" placeholder="Type your answer..." /><button class="clarify-btn" onclick="handleClarifyInput()">Submit</button></div>';
  }
  div.innerHTML = `<span class="tool-icon">❓</span><div class="clarify-content"><div class="clarify-question">${esc(data.question)}</div>${buttonsHtml}</div>`;
  ct.appendChild(div);
  msgEl.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function handleClarifyChoice(btn) {
  if (abortCtrl) abortCtrl.abort();
  const answer = btn.dataset.answer || btn.textContent.replace(/^\d+\.\s*/, '').trim();
  // 使用 _fullMessages 续接，包含 clarify 调用的完整上下文
  const fullMessages = [..._fullMessages, { role: 'user', content: answer }];
  const userMsgEl = addMsg('usr', answer);
  const assistantMsgEl = addMsg('ast', '');
  sendVendor(answer, '', [], assistantMsgEl, new AbortController().signal, fullMessages);
}

function handleClarifyOther() {
  const inp = document.getElementById('inp');
  inp.focus();
  showToast('Type your answer in the input box');
}

function handleClarifyInput() {
  const input = document.getElementById('clarify-text-input');
  if (!input || !input.value.trim()) return;
  if (abortCtrl) abortCtrl.abort();
  const inp = document.getElementById('inp');
  inp.value = input.value.trim();
  setTimeout(() => send(), 100);
}

function extractXmlToolCalls(text) {
  const calls = [];
  const regex = /<tool_call\s+name="([^"]+)">\s*([\s\S]*?)\s*<\/tool_call>/gi;
  let m;
  while ((m = regex.exec(text)) !== null) {
    calls.push({ name: m[1], params: m[2].trim() });
  }
  return calls;
}


// ── Plan 模式功能 ──────────────────────────────────────────────────────────

let _pendingPlan = null; // 暂存待审批计划

function renderPlan(msgEl, items, assistantMessage, turn) {
  /**
   * 渲染 Plan 模式返回的计划项，供用户审批执行。
   * items: [{ tool, arguments }, ...]
   */
  console.log('[Plan] renderPlan called', { itemCount: items?.length, assistantMessage: assistantMessage?.substring(0,100) });
  _pendingPlan = { items, assistant_message: assistantMessage, messages: [..._fullMessages] };
  const ct = msgEl.querySelector('.ct');
  if (!ct) {
    console.warn('[Plan] .ct not found in msgEl, retrying...');
    setTimeout(() => renderPlan(msgEl, items, assistantMessage, turn), 200);
    return;
  }

  const card = document.createElement('div');
  card.className = 'plan-card';
  card.id = 'plan-card';

  let html = `<div class="plan-header">📋 <b>执行计划（${items.length} 项）</b></div>`;
  html += '<div class="plan-items">';
  items.forEach((item, i) => {
    const argsStr = typeof item.arguments === 'string'
      ? item.arguments
      : JSON.stringify(item.arguments, null, 2);
    html += `<div class="plan-item">
      <span class="plan-step">步骤 ${i + 1}</span>
      <span class="plan-tool">🔧 ${esc(item.tool)}</span>
      <pre class="plan-args">${esc(argsStr)}</pre>
    </div>`;
  });
  html += '</div>';
  html += `<div class="plan-actions">
    <button class="plan-btn plan-btn-execute" onclick="executePlan()">▶ 执行计划</button>
    <button class="plan-btn plan-btn-cancel" onclick="cancelPlan()">✖ 取消</button>
  </div>`;

  card.innerHTML = html;
  const old = document.getElementById('plan-card');
  if (old) old.remove();
  ct.appendChild(card);
  msgEl.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

async function executePlan() {
  /**
   * 用户点击「执行计划」→ 调用 /api/agent/chat/plan/execute
   * 流式返回各工具执行结果，然后模型继续对话。
   */
  if (!_pendingPlan) { showToast('没有待执行的计划'); return; }
  const { items, assistant_message, messages } = _pendingPlan;
  const card = document.getElementById('plan-card');
  if (card) {
    card.querySelector('.plan-actions').innerHTML = '<span class="plan-status">⏳ 执行中...</span>';
  }

  const creds = vendorCreds[vendorId] || {};
  const body = {
    vendor_id: vendorId,
    model: curM,
    plan_items: items,
    messages: messages,
    assistant_message: assistant_message || '',
    api_key: creds.api_key || '',
    base_url: creds.base_url || '',
  };

  // 复用 addMsg 创建的 bubble，不手动创建第二个
  const msgEl = addMsg('ast', '⏳ 执行计划中...');
  const bubble = msgEl.querySelector('.msg-bubble');
  if (!bubble) return;

  try {
    const res = await fetch('/api/agent/chat/plan/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) { bubble.innerHTML = `<span class="err">计划执行失败: ${res.status}</span>`; return; }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let turnText = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (line === 'data: [DONE]') continue;
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          switch (event.type) {
            case 'plan_result': {
              const r = document.createElement('div');
              r.className = 'plan-exec-result';
              const resultStr = typeof event.result === 'string' ? event.result : JSON.stringify(event.result);
              r.innerHTML = `✅ <b>${esc(event.tool)}</b>`;
              bubble.appendChild(r);
              break;
            }
            case 'token':
              turnText += event.content;
              bubble.innerHTML = renderMarkdown(turnText);
              break;
            case 'tool_call':
              displayToolCall(msgEl, event.name, event.args);
              break;
            case 'tool_result': {
              try {
                const parsed = JSON.parse(event.content);
                displayToolResult(msgEl, event.name, parsed);
              } catch(_) {
                displayToolResult(msgEl, event.name, { content: event.content });
              }
              break;
            }
            case 'plan_execute_done':
              showToast(`计划执行完成（${event.count} 项）`);
              break;
            case 'done':
              // token 已渲染最终文本到 bubble，不重复调 renderFinal
              saveAssistantMsg(msgEl);
              _pendingPlan = null;
              return;
            case 'error':
              bubble.innerHTML += `<div class="err">${esc(event.content)}</div>`;
              return;
          }
        } catch (e) { /* ignore parse errors */ }
      }
    }
  } catch (e) {
    bubble.innerHTML = `<span class="err">执行异常: ${esc(e.message)}</span>`;
  } finally {
    _pendingPlan = null;
  }
}

function cancelPlan() {
  _pendingPlan = null;
  const card = document.getElementById('plan-card');
  if (card) card.remove();
  showToast('计划已取消');
}

function displayPlanResult(msgEl, toolName, result) {
  const ct = msgEl.querySelector('.ct');
  if (!ct) return;
  const div = document.createElement('div');
  div.className = 'plan-result-item';
  const resultStr = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
  div.innerHTML = `<span class="tool-icon">✅</span><div><b>${esc(toolName)}</b>: ${renderMarkdown(resultStr)}</div>`;
  ct.appendChild(div);
}

// ── End Plan 模式功能 ─────────────────────────────────────────────────────

// ═════════════════════════════════════════════════════════════════════════════
// 技能管理
// ═════════════════════════════════════════════════════════════════════════════

async function loadSkillsList() {
  try {
    const res = await fetch('/api/skills');
    const data = await res.json();
    const list = document.getElementById('skills-list');
    if (!list) return;
    if (!data.skills || data.skills.length === 0) {
      list.innerHTML = '<p style="color:var(--muted);padding:20px">暂无技能</p>';
      return;
    }
    list.innerHTML = data.skills.map(s => `
      <div class="skill-item" onclick="viewSkill('${esc(s.name)}')">
        <div class="skill-name">${esc(s.name)}</div>
        <div class="skill-desc">${esc(s.description || '')}</div>
        <div class="skill-meta">优先级: ${s.priority || 0} | 工具: ${(s.tools || []).join(', ') || '无'}</div>
      </div>
    `).join('');
  } catch (e) {
    console.error('Failed to load skills:', e);
  }
}

function newSkill() {
  const ed = document.getElementById('skill-ed');
  if (!ed) return;
  ed.innerHTML = `
    <div class="skill-form">
      <h3>新建技能</h3>
      <div class="s-row"><label>名称</label><input type="text" id="new-skill-name" placeholder="skill-name"></div>
      <div class="s-row"><label>描述</label><input type="text" id="new-skill-desc" placeholder="简短描述"></div>
      <div class="s-row"><label>优先级</label><input type="number" id="new-skill-priority" value="0" min="0" max="100"></div>
      <div class="s-row"><label>工具</label><input type="text" id="new-skill-tools" placeholder="read_file, run_terminal"></div>
      <div class="s-row"><label>内容</label><textarea id="new-skill-content" rows="10" placeholder="# 技能指令\n\n详细说明..."></textarea></div>
      <div class="s-actions">
        <button class="s-save" onclick="createSkill()">💾 创建</button>
        <span class="s-msg" id="skill-msg"></span>
      </div>
    </div>
  `;
}

async function createSkill() {
  const name = document.getElementById('new-skill-name')?.value;
  const description = document.getElementById('new-skill-desc')?.value;
  const priority = parseInt(document.getElementById('new-skill-priority')?.value || '0');
  const toolsStr = document.getElementById('new-skill-tools')?.value || '';
  const content = document.getElementById('new-skill-content')?.value;
  const msg = document.getElementById('skill-msg');

  if (!name || !content) {
    if (msg) msg.textContent = '名称和内容不能为空';
    return;
  }

  const tools = toolsStr.split(',').map(t => t.trim()).filter(t => t);

  try {
    const res = await fetch('/api/skills', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description, content, priority, tools })
    });
    const data = await res.json();
    if (data.status === 'created') {
      if (msg) msg.textContent = '✅ 创建成功';
      loadSkillsList();
    } else {
      if (msg) msg.textContent = '❌ ' + (data.error || '创建失败');
    }
  } catch (e) {
    console.error('Failed to create skill:', e);
    if (msg) msg.textContent = '❌ 网络错误';
  }
}

async function viewSkill(name) {
  try {
    const res = await fetch('/api/skills/' + encodeURIComponent(name));
    const data = await res.json();
    const ed = document.getElementById('skill-ed');
    if (!ed) return;
    if (data.error) {
      ed.innerHTML = `<p style="color:red">${esc(data.error)}</p>`;
      return;
    }
    ed.innerHTML = `
      <div class="skill-form">
        <h3>${esc(data.name)}</h3>
        <div class="skill-meta">优先级: ${data.priority || 0} | 工具: ${(data.tools || []).join(', ') || '无'}</div>
        <div class="skill-content" style="margin-top:16px;padding:16px;background:var(--input);border-radius:8px;white-space:pre-wrap">${esc(data.content)}</div>
        <div class="s-actions" style="margin-top:16px">
          <button class="btn" onclick="deleteSkill('${esc(data.name)}')">🗑️ 删除</button>
        </div>
      </div>
    `;
  } catch (e) {
    console.error('Failed to view skill:', e);
  }
}

async function deleteSkill(name) {
  if (!confirm('确定要删除技能 "' + name + '" 吗？')) return;
  try {
    const res = await fetch('/api/skills/' + encodeURIComponent(name), { method: 'DELETE' });
    const data = await res.json();
    if (data.status === 'deleted') {
      loadSkillsList();
      document.getElementById('skill-ed').innerHTML = '<p style="color:var(--muted);text-align:center;margin-top:80px">选择技能进行编辑，或新建技能</p>';
    } else {
      alert('删除失败: ' + (data.error || '未知错误'));
    }
  } catch (e) {
    console.error('Failed to delete skill:', e);
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// 审批流
// ═════════════════════════════════════════════════════════════════════════════

async function loadPendingApprovals() {
  try {
    const res = await fetch('/api/approval/pending');
    const data = await res.json();
    const list = document.getElementById('approval-list');
    if (!list) return;
    if (!data.approvals || data.approvals.length === 0) {
      list.innerHTML = '<p style="color:var(--muted);padding:20px">暂无待审批请求</p>';
      return;
    }
    list.innerHTML = data.approvals.map(a => `
      <div class="approval-item">
        <div class="approval-cmd">${esc(a.command_preview || a.pattern_key || '未知命令')}</div>
        <div class="approval-meta">会话: ${esc(a.session_key || 'unknown')} | 时间: ${new Date(a.timestamp).toLocaleString()}</div>
        <div class="approval-actions">
          <button onclick="approveRequest('${esc(a.request_key)}')">✅ 批准</button>
          <button onclick="denyRequest('${esc(a.request_key)}')">❌ 拒绝</button>
        </div>
      </div>
    `).join('');
  } catch (e) {
    console.error('Failed to load approvals:', e);
  }
}

async function approveRequest(key) {
  try {
    const res = await fetch('/api/approval/approve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_key: key })
    });
    const data = await res.json();
    if (data.status === 'approved') {
      loadPendingApprovals();
    }
  } catch (e) {
    console.error('Failed to approve:', e);
  }
}

async function denyRequest(key) {
  try {
    const res = await fetch('/api/approval/deny', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_key: key })
    });
    const data = await res.json();
    if (data.status === 'denied') {
      loadPendingApprovals();
    }
  } catch (e) {
    console.error('Failed to deny:', e);
  }
}

async function toggleYOLO() {
  const enabled = document.getElementById('yolo-mode')?.checked || false;
  try {
    const res = await fetch('/api/approval/yolo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled })
    });
    const data = await res.json();
    showToast(data.message || (enabled ? 'YOLO 模式已开启' : 'YOLO 模式已关闭'));
  } catch (e) {
    console.error('Failed to toggle YOLO:', e);
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// 进程管理
// ═════════════════════════════════════════════════════════════════════════════

async function loadProcessList() {
  try {
    const res = await fetch('/api/processes');
    const data = await res.json();
    const list = document.getElementById('process-list');
    if (!list) return;
    if (!data.processes || data.processes.length === 0) {
      list.innerHTML = '<p style="color:var(--muted);padding:20px">暂无后台进程</p>';
      return;
    }
    list.innerHTML = data.processes.map(p => `
      <div class="process-item">
        <div class="process-cmd">${esc(p.command || 'unknown')}</div>
        <div class="process-meta">PID: ${p.pid || 'N/A'} | 状态: ${p.status || 'unknown'} | 会话: ${esc(p.session_id || '')}</div>
        <div class="process-actions">
          <button onclick="killProcess('${esc(p.session_id)}')">🛑 终止</button>
          <button onclick="viewProcessLog('${esc(p.session_id)}')">📋 日志</button>
        </div>
      </div>
    `).join('');
  } catch (e) {
    console.error('Failed to load processes:', e);
  }
}

async function killProcess(sessionId) {
  try {
    const res = await fetch('/api/processes/' + encodeURIComponent(sessionId) + '/kill', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'killed') {
      loadProcessList();
    }
  } catch (e) {
    console.error('Failed to kill process:', e);
  }
}

async function viewProcessLog(sessionId) {
  try {
    const res = await fetch('/api/processes/' + encodeURIComponent(sessionId) + '/log');
    const data = await res.json();
    alert(data.log || '无日志');
  } catch (e) {
    console.error('Failed to get process log:', e);
  }
}

// ═════════════════════════════════════════════════════════════════════════════

function onToolsChange() {
  // 同时更新本地和厂商的工具状态
  const localCb = document.getElementById('tools-local-enable');
  const vendorCb = document.getElementById('tools-vendor-enable');
  // 更新全局 toolsEnabled 为当前后端的值
  toolsEnabled = isVendorBackend(backendType)
    ? (vendorCb?.checked || false)
    : (localCb?.checked || false);
  [
    { el: 'tools-status-local', cb: localCb, prefix: '本地' },
    { el: 'tools-status-vendor', cb: vendorCb, prefix: '厂商' },
  ].forEach(({ el, cb, prefix }) => {
    const s = document.getElementById(el);
    if (s && cb) s.textContent = cb.checked
      ? `${prefix}: ✅ 已启用 (${toolSchemas.length} 个工具)`
      : `${prefix}: ⏸ 已禁用`;
  });
  // 不自动保存，统一由用户点击保存按钮触发 saveSettings()
}

// ═════════════════════════════════════════════════════════════════════════════
// Cron 定时任务
// ═════════════════════════════════════════════════════════════════════════════

async function loadCronJobs() {
  try {
    const r = await fetch('/api/cron/jobs');
    if (!r.ok) throw new Error('加载失败');
    const jobs = await r.json();
    renderCronList(jobs);
  } catch (e) {
    document.getElementById('cron-list').innerHTML = `<p class="err">加载失败: ${esc(e.message)}</p>`;
  }
}

function renderCronList(jobs) {
  const c = document.getElementById('cron-list');
  if (!jobs || jobs.length === 0) {
    c.innerHTML = '<p style="color:var(--muted);text-align:center;margin-top:40px">暂无定时任务</p>';
    return;
  }
  c.innerHTML = jobs.map(j => `
    <div class="cron-item" data-id="${esc(j.id)}">
      <div class="cron-info">
        <strong>${esc(j.name)}</strong>
        <span class="cron-expr">${esc(j.cron_expr)}</span>
        <span class="cron-handler">${esc(j.handler_type)}</span>
        <span class="cron-status ${j.enabled ? 'on' : 'off'}">${j.enabled ? '启用' : '禁用'}</span>
      </div>
      <div class="cron-acts">
        <button onclick="toggleCronJob('${j.id}', ${!j.enabled})">${j.enabled ? '禁用' : '启用'}</button>
        <button onclick="deleteCronJob('${j.id}')" class="danger">删除</button>
      </div>
    </div>
  `).join('');
}

function showCronForm() {
  document.getElementById('cron-form').classList.remove('hid');
}

function hideCronForm() {
  document.getElementById('cron-form').classList.add('hid');
  document.getElementById('cron-name').value = '';
  document.getElementById('cron-expr').value = '';
  document.getElementById('cron-args').value = '';
}

async function createCronJob() {
  const name = document.getElementById('cron-name').value.trim();
  const cronExpr = document.getElementById('cron-expr').value.trim();
  const handlerType = document.getElementById('cron-handler').value;
  const argsText = document.getElementById('cron-args').value.trim();
  const enabled = document.getElementById('cron-enabled').checked;

  if (!name || !cronExpr) {
    alert('名称和 Cron 表达式不能为空');
    return;
  }

  let args = {};
  if (argsText) {
    try { args = JSON.parse(argsText); } catch (e) { alert('参数 JSON 格式错误'); return; }
  }

  try {
    const r = await fetch('/api/cron/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, cron_expr: cronExpr, handler_type: handlerType, args, enabled }),
    });
    if (!r.ok) throw new Error(await r.text());
    hideCronForm();
    loadCronJobs();
  } catch (e) {
    alert('创建失败: ' + e.message);
  }
}

async function toggleCronJob(id, enabled) {
  try {
    const r = await fetch(`/api/cron/jobs/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    if (!r.ok) throw new Error(await r.text());
    loadCronJobs();
  } catch (e) {
    alert('更新失败: ' + e.message);
  }
}

async function deleteCronJob(id) {
  if (!confirm('确定删除此定时任务？')) return;
  try {
    const r = await fetch(`/api/cron/jobs/${id}`, { method: 'DELETE' });
    if (!r.ok) throw new Error(await r.text());
    loadCronJobs();
  } catch (e) {
    alert('删除失败: ' + e.message);
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// 插件管理
// ═════════════════════════════════════════════════════════════════════════════

async function loadPlugins() {
  try {
    const r = await fetch('/api/plugins');
    if (!r.ok) throw new Error('加载失败');
    const plugins = await r.json();
    renderPluginList(plugins);
  } catch (e) {
    document.getElementById('plugin-list').innerHTML = `<p class="err">加载失败: ${esc(e.message)}</p>`;
  }
}

function renderPluginList(plugins) {
  const c = document.getElementById('plugin-list');
  if (!plugins || plugins.length === 0) {
    c.innerHTML = '<p style="color:var(--muted);text-align:center;margin-top:40px">暂无插件</p>';
    return;
  }
  c.innerHTML = plugins.map(p => `
    <div class="plugin-item" data-id="${esc(p.id)}">
      <div class="plugin-info">
        <strong>${esc(p.name)}</strong>
        <span class="plugin-ver">v${esc(p.version || '0.0.1')}</span>
        <span class="plugin-status ${p.enabled ? 'on' : 'off'}">${p.enabled ? '启用' : '禁用'}</span>
        <p class="plugin-desc">${esc(p.description || '无描述')}</p>
      </div>
      <div class="plugin-acts">
        <button onclick="togglePlugin('${p.id}', ${!p.enabled})">${p.enabled ? '禁用' : '启用'}</button>
      </div>
    </div>
  `).join('');
}

async function discoverPlugins() {
  try {
    const r = await fetch('/api/plugins/discover', { method: 'POST' });
    if (!r.ok) throw new Error(await r.text());
    const result = await r.json();
    alert(`发现 ${result.discovered || 0} 个新插件`);
    loadPlugins();
  } catch (e) {
    alert('发现插件失败: ' + e.message);
  }
}

async function togglePlugin(id, enabled) {
  try {
    const r = await fetch(`/api/plugins/${id}/toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    if (!r.ok) throw new Error(await r.text());
    loadPlugins();
  } catch (e) {
    alert('切换失败: ' + e.message);
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// API 文档页面
// ═════════════════════════════════════════════════════════════════════════════

const API_DEFINITIONS = [
  {
    category: 'chat',
    name: '聊天',
    endpoints: [
      { method: 'POST', path: '/api/agent/chat/stream', desc: 'Agent 流式对话（SSE）', params: [
        { name: 'message', type: 'string', req: true, desc: '用户消息' },
        { name: 'session_id', type: 'string', req: false, desc: '会话 ID' },
        { name: 'vendor_id', type: 'string', req: true, desc: '厂商 ID' },
        { name: 'model', type: 'string', req: true, desc: '模型名' },
        { name: 'tools_enabled', type: 'boolean', req: false, desc: '启用工具调用' },
        { name: 'plan_mode', type: 'boolean', req: false, desc: 'Plan 模式' },
      ]},
      { method: 'POST', path: '/api/agent/chat/plan/execute', desc: '执行已审批的计划', params: [
        { name: 'plan_items', type: 'array', req: true, desc: '计划项列表' },
        { name: 'messages', type: 'array', req: true, desc: '对话历史' },
      ]},
    ]
  },
  {
    category: 'skills',
    name: '技能',
    endpoints: [
      { method: 'GET', path: '/api/skills', desc: '列出所有技能' },
      { method: 'POST', path: '/api/skills', desc: '创建技能', params: [
        { name: 'name', type: 'string', req: true, desc: '技能名称' },
        { name: 'description', type: 'string', req: false, desc: '技能描述' },
        { name: 'content', type: 'string', req: true, desc: '技能内容（Markdown）' },
        { name: 'priority', type: 'integer', req: false, desc: '优先级（0-10）' },
        { name: 'tools', type: 'array', req: false, desc: '关联工具列表' },
      ]},
      { method: 'GET', path: '/api/skills/:name', desc: '获取技能详情' },
      { method: 'DELETE', path: '/api/skills/:name', desc: '删除技能' },
    ]
  },
  {
    category: 'processes',
    name: '进程',
    endpoints: [
      { method: 'GET', path: '/api/processes', desc: '列出所有后台进程' },
      { method: 'POST', path: '/api/processes/:session_id/kill', desc: '终止进程' },
      { method: 'GET', path: '/api/processes/:session_id/log', desc: '获取进程日志' },
    ]
  },
  {
    category: 'approval',
    name: '审批',
    endpoints: [
      { method: 'GET', path: '/api/approval/status', desc: '获取审批状态' },
      { method: 'GET', path: '/api/approval/pending', desc: '获取待审批请求' },
      { method: 'POST', path: '/api/approval/approve', desc: '批准请求', params: [
        { name: 'request_key', type: 'string', req: true, desc: '请求标识' },
      ]},
      { method: 'POST', path: '/api/approval/deny', desc: '拒绝请求', params: [
        { name: 'request_key', type: 'string', req: true, desc: '请求标识' },
      ]},
      { method: 'POST', path: '/api/approval/yolo', desc: '切换 YOLO 模式', params: [
        { name: 'enabled', type: 'boolean', req: true, desc: '是否启用' },
      ]},
    ]
  },
  {
    category: 'cron',
    name: '定时任务',
    endpoints: [
      { method: 'GET', path: '/api/cron/jobs', desc: '列出所有定时任务' },
      { method: 'POST', path: '/api/cron/jobs', desc: '创建定时任务', params: [
        { name: 'name', type: 'string', req: true, desc: '任务名称' },
        { name: 'cron_expr', type: 'string', req: true, desc: 'Cron 表达式' },
        { name: 'handler_type', type: 'string', req: true, desc: '处理器类型' },
        { name: 'args', type: 'object', req: false, desc: '处理器参数' },
        { name: 'enabled', type: 'boolean', req: false, desc: '是否启用' },
      ]},
      { method: 'POST', path: '/api/cron/jobs/:id/toggle', desc: '启用/禁用任务' },
      { method: 'DELETE', path: '/api/cron/jobs/:id', desc: '删除任务' },
    ]
  },
  {
    category: 'plugins',
    name: '插件',
    endpoints: [
      { method: 'GET', path: '/api/plugins', desc: '列出所有插件' },
      { method: 'POST', path: '/api/plugins/discover', desc: '发现新插件' },
      { method: 'POST', path: '/api/plugins/:id/toggle', desc: '启用/禁用插件' },
    ]
  },
  {
    category: 'memory',
    name: '记忆',
    endpoints: [
      { method: 'GET', path: '/api/memory', desc: '列出所有记忆' },
      { method: 'POST', path: '/api/memory', desc: '保存记忆', params: [
        { name: 'key', type: 'string', req: true, desc: '键名' },
        { name: 'value', type: 'string', req: true, desc: '值' },
        { name: 'category', type: 'string', req: false, desc: '分类' },
      ]},
      { method: 'GET', path: '/api/memory/search', desc: '搜索记忆' },
      { method: 'DELETE', path: '/api/memory/:key', desc: '删除记忆' },
    ]
  },
  {
    category: 'system',
    name: '系统',
    endpoints: [
      { method: 'GET', path: '/api/health', desc: '健康检查' },
      { method: 'GET', path: '/api/status', desc: '系统状态（后端/模型列表）' },
      { method: 'GET', path: '/api/settings', desc: '获取设置' },
      { method: 'POST', path: '/api/settings', desc: '保存设置' },
      { method: 'GET', path: '/api/tools/list', desc: '列出可用工具' },
      { method: 'POST', path: '/api/tools/execute', desc: '执行工具', params: [
        { name: 'name', type: 'string', req: true, desc: '工具名' },
        { name: 'params', type: 'object', req: true, desc: '工具参数' },
      ]},
    ]
  },
];

function renderAPIDocs() {
  const container = document.getElementById('api-content');
  if (!container) return;

  const search = (document.getElementById('api-search')?.value || '').toLowerCase();
  const catFilter = document.getElementById('api-cat')?.value || '';

  let html = '';
  for (const cat of API_DEFINITIONS) {
    if (catFilter && cat.category !== catFilter) continue;

    const filtered = cat.endpoints.filter(ep =>
      !search || ep.path.toLowerCase().includes(search) || ep.desc.toLowerCase().includes(search)
    );
    if (!filtered.length) continue;

    html += `<div class="api-cat" data-cat="${esc(cat.category)}">
      <h3>${esc(cat.name)}</h3>`;
    for (const ep of filtered) {
      const epId = 'ep-' + Math.random().toString(36).substr(2, 8);
      html += `
      <div class="api-endpoint">
        <div class="api-endpoint-header" onclick="toggleEndpoint('${epId}')">
          <span class="api-method ${ep.method.toLowerCase()}">${ep.method}</span>
          <span class="api-path">${esc(ep.path)}</span>
          <span class="api-toggle">▼</span>
        </div>
        <div class="api-endpoint-body" id="${epId}">
          <p>${esc(ep.desc)}</p>`;
      if (ep.params && ep.params.length) {
        html += `
          <h4>参数</h4>
          <table class="api-params-table">
            <tr><th>名称</th><th>类型</th><th>必填</th><th>说明</th></tr>`;
        for (const p of ep.params) {
          html += `<tr><td>${esc(p.name)}</td><td>${esc(p.type)}</td><td>${p.req ? '是' : '否'}</td><td>${esc(p.desc)}</td></tr>`;
        }
        html += '</table>';
      }
      html += `
          <h4>示例请求</h4>
          <pre><code>${esc(ep.method)} ${esc(ep.path)}
Content-Type: application/json

${ep.params ? JSON.stringify(Object.fromEntries(ep.params.map(p => [p.name, p.type === 'boolean' ? false : p.type === 'integer' ? 0 : p.type === 'array' ? [] : p.type === 'object' ? {} : 'string'])), null, 2) : ''}</code></pre>
        </div>
      </div>`;
    }
    html += '</div>';
  }

  if (!html) html = '<p style="color:var(--muted);text-align:center;margin-top:40px">无匹配结果</p>';
  container.innerHTML = html;
}

function toggleEndpoint(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle('open');
  const header = el.previousElementSibling;
  if (header) {
    const toggle = header.querySelector('.api-toggle');
    if (toggle) toggle.textContent = el.classList.contains('open') ? '▲' : '▼';
  }
}

function filterAPI(query) {
  renderAPIDocs();
}

function filterAPICategory(cat) {
  renderAPIDocs();
}

// ═════════════════════════════════════════════════════════════════════════════
// 状态监控面板
// ═════════════════════════════════════════════════════════════════════════════

let _monitorInterval = null;

async function loadMonitorData() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) return;
    const data = await res.json();
    renderMonitor(data);
  } catch (e) {
    console.error('Failed to load metrics:', e);
  }
}

function renderMonitor(data) {
  const container = document.getElementById('monitor-content');
  if (!container) return;

  const metrics = data.metrics || {};
  const latency = metrics.latency || {};
  const throughput = metrics.throughput || {};
  const errors = metrics.errors || {};
  const cache = metrics.cache || {};

  // 生成随机柱状图数据（实际应从后端获取时间序列）
  const bars = Array(20).fill(0).map(() => Math.random() * 100);
  const barHtml = bars.map(h => `<div class="monitor-bar" style="height:${h}%"></div>`).join('');

  container.innerHTML = `
    <div class="monitor-grid">
      <div class="monitor-card">
        <h4><span class="icon">⏱️</span> 延迟</h4>
        <div class="monitor-metric">
          <span class="monitor-metric-label">P50</span>
          <span class="monitor-metric-value ${latency.p50 > 1000 ? 'warn' : 'good'}">${(latency.p50 || 0).toFixed(0)} ms</span>
        </div>
        <div class="monitor-metric">
          <span class="monitor-metric-label">P95</span>
          <span class="monitor-metric-value ${latency.p95 > 3000 ? 'warn' : 'good'}">${(latency.p95 || 0).toFixed(0)} ms</span>
        </div>
        <div class="monitor-metric">
          <span class="monitor-metric-label">P99</span>
          <span class="monitor-metric-value ${latency.p99 > 5000 ? 'bad' : 'good'}">${(latency.p99 || 0).toFixed(0)} ms</span>
        </div>
        <div class="monitor-chart">${barHtml}</div>
      </div>
      <div class="monitor-card">
        <h4><span class="icon">📊</span> 吞吐量</h4>
        <div class="monitor-metric">
          <span class="monitor-metric-label">请求/秒</span>
          <span class="monitor-metric-value good">${(throughput.rps || 0).toFixed(1)}</span>
        </div>
        <div class="monitor-metric">
          <span class="monitor-metric-label">Token/秒</span>
          <span class="monitor-metric-value good">${(throughput.tps || 0).toFixed(1)}</span>
        </div>
        <div class="monitor-metric">
          <span class="monitor-metric-label">总请求数</span>
          <span class="monitor-metric-value">${throughput.total_requests || 0}</span>
        </div>
      </div>
      <div class="monitor-card">
        <h4><span class="icon">❌</span> 错误率</h4>
        <div class="monitor-metric">
          <span class="monitor-metric-label">错误率</span>
          <span class="monitor-metric-value ${errors.rate > 0.05 ? 'bad' : 'good'}">${((errors.rate || 0) * 100).toFixed(2)}%</span>
        </div>
        <div class="monitor-metric">
          <span class="monitor-metric-label">总错误</span>
          <span class="monitor-metric-value">${errors.total || 0}</span>
        </div>
        <div class="monitor-metric">
          <span class="monitor-metric-label">最近错误</span>
          <span class="monitor-metric-value ${errors.recent > 0 ? 'warn' : 'good'}">${errors.recent || 0}</span>
        </div>
      </div>
      <div class="monitor-card">
        <h4><span class="icon">💾</span> 缓存</h4>
        <div class="monitor-metric">
          <span class="monitor-metric-label">命中率</span>
          <span class="monitor-metric-value good">${((cache.hit_rate || 0) * 100).toFixed(1)}%</span>
        </div>
        <div class="monitor-metric">
          <span class="monitor-metric-label">内存条目</span>
          <span class="monitor-metric-value">${cache.memory_items || 0}</span>
        </div>
        <div class="monitor-metric">
          <span class="monitor-metric-label">磁盘条目</span>
          <span class="monitor-metric-value">${cache.disk_items || 0}</span>
        </div>
      </div>
      <div class="monitor-card">
        <h4><span class="icon">🔧</span> 工具</h4>
        <div class="monitor-metric">
          <span class="monitor-metric-label">已注册</span>
          <span class="monitor-metric-value">${metrics.tools?.registered || 0}</span>
        </div>
        <div class="monitor-metric">
          <span class="monitor-metric-label">今日调用</span>
          <span class="monitor-metric-value">${metrics.tools?.calls_today || 0}</span>
        </div>
        <div class="monitor-metric">
          <span class="monitor-metric-label">成功率</span>
          <span class="monitor-metric-value good">${((metrics.tools?.success_rate || 0) * 100).toFixed(1)}%</span>
        </div>
      </div>
      <div class="monitor-card">
        <h4><span class="icon">🖥️</span> 系统</h4>
        <div class="monitor-metric">
          <span class="monitor-metric-label">后端</span>
          <span class="monitor-metric-value">${esc(metrics.system?.backend || 'unknown')}</span>
        </div>
        <div class="monitor-metric">
          <span class="monitor-metric-label">运行时间</span>
          <span class="monitor-metric-value">${formatUptime(metrics.system?.uptime || 0)}</span>
        </div>
        <div class="monitor-metric">
          <span class="monitor-metric-label">版本</span>
          <span class="monitor-metric-value">${esc(metrics.system?.version || 'unknown')}</span>
        </div>
      </div>
    </div>
  `;
}

function formatUptime(seconds) {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function startMonitor() {
  if (_monitorInterval) clearInterval(_monitorInterval);
  loadMonitorData();
  _monitorInterval = setInterval(loadMonitorData, 5000);
}

function stopMonitor() {
  if (_monitorInterval) {
    clearInterval(_monitorInterval);
    _monitorInterval = null;
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// 审批流独立页面
// ═════════════════════════════════════════════════════════════════════════════

async function loadApprovalPage() {
  try {
    const [statusRes, pendingRes] = await Promise.all([
      fetch('/api/approval/status'),
      fetch('/api/approval/pending'),
    ]);
    const status = await statusRes.json();
    const pending = await pendingRes.json();
    renderApprovalPage(status, pending);
  } catch (e) {
    console.error('Failed to load approval page:', e);
  }
}

function renderApprovalPage(status, pending) {
  const container = document.getElementById('approval-page-content');
  if (!container) return;

  const stats = status.stats || { pending: 0, approved: 0, denied: 0 };
  const list = pending.approvals || pending.pending || [];

  container.innerHTML = `
    <div class="approval-stats">
      <div class="approval-stat">
        <div class="approval-stat-num pending">${stats.pending || list.length}</div>
        <div class="approval-stat-label">待审批</div>
      </div>
      <div class="approval-stat">
        <div class="approval-stat-num approved">${stats.approved || 0}</div>
        <div class="approval-stat-label">已批准</div>
      </div>
      <div class="approval-stat">
        <div class="approval-stat-num denied">${stats.denied || 0}</div>
        <div class="approval-stat-label">已拒绝</div>
      </div>
    </div>
    <div class="approval-list">
      ${list.length === 0 ? '<p style="color:var(--muted);padding:20px;text-align:center">暂无待审批请求</p>' : list.map(a => `
        <div class="approval-item">
          <div class="approval-item-header">
            <span class="approval-item-title">🔐 ${esc(a.command_preview || a.pattern_key || '未知命令')}</span>
            <span class="approval-item-time">${new Date(a.timestamp).toLocaleString()}</span>
          </div>
          <div class="approval-item-body">${esc(a.command_preview || '')}</div>
          <div class="approval-item-meta">
            <span>📍 ${esc(a.session_key || 'unknown')}</span>
            <span>🔑 ${esc(a.pattern_key || 'N/A')}</span>
          </div>
          <div class="approval-item-actions">
            <button class="btn-approve" onclick="approveRequest('${esc(a.request_key)}');loadApprovalPage();">✅ 批准</button>
            <button class="btn-deny" onclick="denyRequest('${esc(a.request_key)}');loadApprovalPage();">❌ 拒绝</button>
            <button class="btn-details" onclick="showApprovalDetails('${esc(a.request_key)}')">详情</button>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function showApprovalDetails(key) {
  showToast('审批详情: ' + key);
}

// ═════════════════════════════════════════════════════════════════════════════
// 日志查看器
// ═════════════════════════════════════════════════════════════════════════════

let _logInterval = null;
let _logLevelFilter = 'all';
let _logSourceFilter = 'all';

async function loadLogs() {
  try {
    const res = await fetch('/api/logs?limit=200&level=' + _logLevelFilter);
    if (!res.ok) return;
    const data = await res.json();
    renderLogs(data.logs || []);
  } catch (e) {
    console.error('Failed to load logs:', e);
  }
}

function renderLogs(logs) {
  const container = document.getElementById('log-content');
  if (!container) return;

  if (!logs.length) {
    container.innerHTML = '<div style="color:var(--muted);text-align:center;padding:40px">暂无日志</div>';
    return;
  }

  container.innerHTML = logs.map(l => {
    const level = (l.level || 'info').toLowerCase();
    const time = l.timestamp ? new Date(l.timestamp).toLocaleTimeString() : '--:--:--';
    return `<div class="log-line">
      <span class="log-time">${esc(time)}</span>
      <span class="log-level ${level}">${esc(level.toUpperCase())}</span>
      <span class="log-logger">${esc(l.logger || 'app')}</span>
      <span class="log-msg">${esc(l.message || '')}</span>
    </div>`;
  }).join('');

  // 自动滚动到底部
  container.scrollTop = container.scrollHeight;
}

function setLogLevel(level) {
  _logLevelFilter = level;
  // 更新按钮状态
  document.querySelectorAll('.log-level-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.level === level);
  });
  loadLogs();
}

function startLogStream() {
  if (_logInterval) clearInterval(_logInterval);
  loadLogs();
  _logInterval = setInterval(loadLogs, 3000);
}

function stopLogStream() {
  if (_logInterval) {
    clearInterval(_logInterval);
    _logInterval = null;
  }
}

function clearLogs() {
  const container = document.getElementById('log-content');
  if (container) container.innerHTML = '';
}
function renderMarkdown(text) {
  if (!text) return '';
  var html = text;
  // 代码块 (```...```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, function(_, lang, code) {
    return '<pre><code class="language-' + lang + '">' + escHtml(code.trim()) + '</code></pre>';
  });
  // 行内代码
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // 粗体
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // 斜体
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  // 标题
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  // 无序列表
  html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/<li>/g, '<ul><li>');
  html = html.replace(/<\/li>/g, '</li></ul>');
  // 段落（双换行分割）
  html = '<p>' + html.replace(/\n\n/g, '</p><p>') + '</p>';
  html = html.replace(/\n/g, '<br>');
  html = html.replace(/<p><\/p>/g, '');
  return html;
}

function escHtml(text) {
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function restoreAttachments(bubble) {
  if (!bubble) return;
  try {
    var raw = bubble.dataset.attachments;
    if (!raw) return;
    var data = JSON.parse(raw);
    attI = data.images || [];
    attF = data.files || [];
    if (typeof renderAttachments === 'function') {
      renderAttachments();
    }
  } catch(e) {
    console.error('restoreAttachments:', e);
  }
}


// ═════════════════════════════════════════════════════════════════════════════
// TPS (Tokens Per Second) 实时显示
// ═════════════════════════════════════════════════════════════════════════════

function updateMsgTps(msgEl, tokenCount, startTime, isFinal) {
  if (!msgEl) return;
  var tpsEl = msgEl.querySelector('.msg-tps');
  if (!tpsEl) {
    var actions = msgEl.querySelector('.msg-actions');
    if (!actions) return;
    tpsEl = document.createElement('span');
    tpsEl.className = 'msg-tps';
    actions.appendChild(tpsEl);
  }
  var elapsed = (Date.now() - startTime) / 1000;
  var tps = elapsed > 0 ? (tokenCount / elapsed) : 0;
  tpsEl.textContent = tokenCount + ' tok ' + String.fromCharCode(183) + ' ' + tps.toFixed(1) + ' t/s';
  tpsEl.style.opacity = isFinal ? '0.6' : '0.9';
}

// ═════════════════════════════════════════════════════════════════════════════
// 思维链 (Reasoning / Thinking) 折叠块
// ═════════════════════════════════════════════════════════════════════════════

function _ensureReasoningBlock(msgEl, text) {
  if (!msgEl) return;
  var ct = msgEl.querySelector('.ct');
  if (!ct) return;
  var existing = ct.querySelector('.think-block');
  if (!existing) {
    existing = document.createElement('details');
    existing.className = 'think-block';
    existing.setAttribute('open', '');
    var summary = document.createElement('summary');
    summary.className = 'think-hd';
    summary.textContent = '\u{1f4ad} \u601d\u8003\u4e2d...';
    existing.appendChild(summary);
    var contentDiv = document.createElement('div');
    contentDiv.className = 'think-content';
    existing.appendChild(contentDiv);
    var bubble = ct.querySelector('.msg-bubble');
    if (bubble) ct.insertBefore(existing, bubble);
    else ct.appendChild(existing);
  }
  var content = existing.querySelector('.think-content');
  if (content) content.textContent += text;
  var bubble = msgEl.querySelector('.msg-bubble');
  if (bubble) bubble.dataset.reasoningText = (bubble.dataset.reasoningText || '') + text;
}

// ═════════════════════════════════════════════════════════════════════════════
// Toast 提示
// ═════════════════════════════════════════════════════════════════════════════

function showToast(msg) {
  const toast = document.createElement('div');
  toast.style.cssText = 'position:fixed;bottom:20px;right:20px;padding:10px 16px;background:var(--card);color:var(--text);border-radius:8px;border:1px solid var(--border);font-size:13px;z-index:9999;animation:fadeIn .3s';
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity .3s';
    setTimeout(() => toast.remove(), 300);
  }, 2500);
}

// 覆盖 nav 函数以处理新页面
const _origNav2 = nav;
nav = function(n) {
  _origNav2(n);
  if (n === 'api-docs') {
    renderAPIDocs();
    document.getElementById('pg-ttl').textContent = 'API 文档';
  } else if (n === 'help') {
    document.getElementById('pg-ttl').textContent = '帮助';
  } else if (n === 'monitor') {
    startMonitor();
    document.getElementById('pg-ttl').textContent = '状态监控';
  } else if (n === 'approval-page') {
    loadApprovalPage();
    document.getElementById('pg-ttl').textContent = '审批管理';
  } else if (n === 'logs') {
    startLogStream();
    document.getElementById('pg-ttl').textContent = '日志查看器';
  } else {
    stopMonitor();
    stopLogStream();
  }
};
