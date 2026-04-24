let curM=null, models=[], tpls=[], curTpl=null, attF=[], attI=[];
let backendType = 'none';  // 'llama-cpp', 'ollama'

// ──────────────────────────────────────────────────────────────────────────────
// 初始化
// ──────────────────────────────────────────────────────────────────────────────
async function init() {
  await detectBackend();
  await loadT();
  
  // 根据后端类型初始化模型列表
  if (backendType === 'llama-cpp') {
    await loadLlamaModels();
  } else if (backendType === 'ollama') {
    await loadOllamaModels();
    setInterval(loadOllamaModels, 30000);
  }
  
  // 更新 UI 状态
  updateBackendStatus();
}

async function detectBackend() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    backendType = data.backend;
    console.log('[init] 后端:', backendType, data);
    
    // 更新侧边栏状态显示
    const dot = document.getElementById('st-dot');
    const txt = document.getElementById('st-txt');
    if (backendType === 'llama-cpp') {
      dot.classList.add('on');
      txt.textContent = data.gpu_available ? 'llama-cpp (GPU)' : 'llama-cpp (CPU)';
    } else if (backendType === 'ollama') {
      dot.classList.add('on');
      txt.textContent = data.running ? 'Ollama 运行中' : 'Ollama 未连接';
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
    // 可以显示 GPU/CPU 信息
    fetch('/api/status').then(r=>r.json()).then(data => {
      if (data.cpu_mode) badge.title = 'CPU 模式';
      else if (data.gpu_available) badge.title = 'GPU 模式';
    }).catch(()=>{});
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// llama-cpp 模型管理
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

// ──────────────────────────────────────────────────────────────────────────────
// Ollama 模型管理（原有）
// ──────────────────────────────────────────────────────────────────────────────
async function loadOllamaModels() {
  try {
    const res = await fetch('/api/ollama_status');
    const data = await res.json();
    const dot = document.getElementById('st-dot');
    const txt = document.getElementById('st-txt');
    if (data.running) {
      dot.classList.add('on');
      txt.textContent = 'Ollama 运行中';
      models = data.models || [];
      renderModelSelect();
      if (!curM && models.length) selModel(models[0]);
    } else {
      dot.classList.remove('on');
      txt.textContent = 'Ollama 未连接';
    }
  } catch (e) {
    console.error('加载 Ollama 模型失败:', e);
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
  // n 可能是 {path, mmproj, has_vision} 对象或纯字符串
  const modelPath = (typeof n === 'object') ? n.path : n;
  const modelObj = (typeof n === 'object') ? n : models.find(m => (m.path || m) === n);
  curM = modelPath;
  document.getElementById('m-sel').value = modelPath;
  const visionTag = (modelObj && modelObj.has_vision) ? ' 👁' : '';
  document.getElementById('cur-m').textContent = modelPath || '未选择';
  
  // llama-cpp 需要主动加载模型
  if (backendType === 'llama-cpp' && modelPath) {
    loadLlamaModel(modelPath, modelObj);
  }
}

async function loadLlamaModel(modelName, modelObj) {
  const btn = document.getElementById('cur-m');
  const originalText = btn.textContent;
  btn.textContent = '加载中...';
  
  try {
    const body = {
      model: modelName,
      chat_handler: 'auto'  // 自动检测 mmproj
    };
    
    const res = await fetch('/api/llama/load_model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (res.ok) {
      console.log('模型加载成功:', data);
      const mmprojInfo = data.config?.mmproj_loaded ? ' (视觉已启用)' : '';
      btn.textContent = modelName + mmprojInfo;
    } else {
      console.error('模型加载失败:', data.error);
      btn.textContent = originalText;
      alert('加载失败: ' + (data.error || '未知错误'));
    }
  } catch (e) {
    console.error('加载模型请求失败:', e);
    btn.textContent = originalText;
    alert('加载失败: ' + e.message);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// 对话发送（统一后端）
// ──────────────────────────────────────────────────────────────────────────────
async function send() {
  const inp = document.getElementById('inp');
  const txt = inp.value.trim();
  if (!txt && !attF.length && !attI.length) return;
  if (!curM) {
    alert('请先选择模型');
    return;
  }
  document.getElementById('empty')?.classList.add('hid');

  // 构造消息内容
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

  // 添加用户消息到 UI
  addMsg('usr', txt, [...attI, ...attF]);
  
  // 清空输入
  inp.value = '';
  inp.style.height = 'auto';
  const savedImages = [...attI];
  const savedFiles = [...attF];
  attF = [];
  attI = [];
  document.getElementById('atchs').classList.add('hid');
  document.getElementById('atchs').innerHTML = '';

  const assistantMsg = addMsg('ast', '');

  try {
    if (backendType === 'llama-cpp') {
      await sendLlama(content, systemPrompt, savedImages, assistantMsg);
    } else if (backendType === 'ollama') {
      await sendOllama(content, systemPrompt, assistantMsg);
    } else {
      throw new Error('无可用后端');
    }
  } catch (e) {
    assistantMsg.querySelector('.ct').innerHTML = `<span class="err">请求失败: ${esc(e.message)}</span>`;
  }
}

async function sendLlama(content, systemPrompt, images, msgEl) {
  const body = {
    prompt: content,
    system_prompt: systemPrompt || undefined,
    max_tokens: parseInt(document.getElementById('s-max').value),
    temperature: parseFloat(document.getElementById('s-temp').value),
    top_p: parseFloat(document.getElementById('s-topp').value),
    top_k: 40,
    repeat_penalty: 1.0,
    images: images.map(img => img.base64),
    stream: true  // 启用流式输出
  };

  const res = await fetch('/api/llama/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HTTP ${res.status}: ${err.slice(0, 200)}`);
  }

  // 处理流式响应
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullText = '';
  const ctElement = msgEl.querySelector('.ct');
  ctElement.textContent = '';

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
        if (data.content) {
          fullText += data.content;
          ctElement.textContent = fullText;
        }
        if (data.error) {
          throw new Error(data.error);
        }
      } catch (e) {
        if (e.message) throw e;
      }
    }
  }
  
  if (!fullText) ctElement.textContent = '(空响应)';
}

async function sendOllama(content, systemPrompt, msgEl) {
  const messages = [];
  if (systemPrompt) {
    messages.push({ role: 'system', content: systemPrompt });
  }
  messages.push({ role: 'user', content: content });

  const body = {
    model: curM,
    messages: messages,
    max_tokens: parseInt(document.getElementById('s-max').value),
    temperature: parseFloat(document.getElementById('s-temp').value),
    top_p: parseFloat(document.getElementById('s-topp').value),
  };

  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HTTP ${res.status}: ${err.slice(0, 200)}`);
  }

  let full = '';
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

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
        if (data.message?.content) {
          full += data.message.content;
          msgEl.querySelector('.ct').textContent = full;
        }
        if (data.error) {
          throw new Error(data.error);
        }
      } catch {}
    }
  }
  if (!full) msgEl.querySelector('.ct').textContent = '(空响应)';
}

// ──────────────────────────────────────────────────────────────────────────────
// Web 搜索功能
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
    
    // 显示搜索结果
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
  
  // 找到消息区域并插入
  const msgsDiv = document.getElementById('msgs');
  msgsDiv.appendChild(resultsDiv);
  msgsDiv.scrollTop = msgsDiv.scrollHeight;
  
  // 绑定插入按钮事件
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
// UI 辅助函数（保留原有）
// ──────────────────────────────────────────────────────────────────────────────
function addMsg(role, txt, att = []) {
  const c = document.getElementById('msgs');
  const d = document.createElement('div');
  d.className = 'msg ' + (role === 'usr' ? 'usr' : 'ast');
  let h = '';
  if (att.length) {
    const imgs = att.filter(a => a.type === 'image');
    const files = att.filter(a => a.type === 'file');
    h += imgs.map(a => `<img src="${esc(a.preview)}" style="max-width:200px;border-radius:8px;margin:4px">`).join('');
    h += files.map(a => `<div style="background:var(--input);padding:4px 8px;border-radius:4px;font-size:12px;margin:4px;color:var(--text2)">📄 ${esc(a.name)}</div>`).join('');
  }
  if (txt) h += esc(txt);
  d.innerHTML = `<div class="ct">${h || '...'}</div>`;
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
  return d;
}

function nav(n) {
  document.querySelectorAll('.nav').forEach(e => e.classList.toggle('act', e.dataset.n === n));
  document.querySelectorAll('.panel').forEach(e => e.classList.toggle('act', e.id === 'p-' + n));
  const titles = { chat: '对话', translate: '翻译', templates: '模板', settings: '设置' };
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

// 文件上传处理
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
(function() {
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

// 模板（保留原有）
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
  if (!curTpl || !confirm('删除此模板？')) return;
  await fetch('/api/prompt_templates/' + encodeURIComponent(curTpl), { method: 'DELETE' });
  curTpl = null;
  await loadT();
}

// 翻译（保留原有）
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
        } catch {}
      }
    }
  }
}

function clrTr() {
  document.getElementById('st').value = '';
  document.getElementById('to').value = '';
}

// 设置
function us(k) {
  const ids = { temp: 's-temp', max: 's-max', topp: 's-topp' };
  const vals = { temp: 'v-temp', max: 'v-max', topp: 'v-topp' };
  document.getElementById(vals[k]).textContent = document.getElementById(ids[k]).value;
}

// 启动
init();
