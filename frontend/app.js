const fileInput = document.getElementById('file');
const preview = document.getElementById('preview');
const btnOCR = document.getElementById('btn-ocr');
const btnAsk = document.getElementById('btn-ask');
const btnPDF = document.getElementById('btn-pdf');
const extractedEl = document.getElementById('extracted');
const pagesEl = document.getElementById('pages');
const questionEl = document.getElementById('question');
const statusEl = document.getElementById('status');
const useFullEl = document.getElementById('use-full');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const btnProcess = document.getElementById('btn-process');
const structuredCard = document.getElementById('structured-card');
const structuredFields = document.getElementById('structured-fields');
const btnExportCsv = document.getElementById('btn-export-csv');
const btnExportXlsx = document.getElementById('btn-export-xlsx');
const btnExportDocx = document.getElementById('btn-export-docx');
const btnVisualize = document.getElementById('btn-visualize');
const vizCanvas = document.getElementById('viz-canvas');
console.log("btnProcess = ", document.getElementById("btn-process"));

let progressInterval = null;

function startProgress(){
  if (progressInterval) return;
  progressContainer.setAttribute('aria-hidden','false');
  progressBar.style.width = '2%';
  let val = 2;
  progressInterval = setInterval(()=>{
    // increment quickly at first, then slow down as it approaches 90%
    const step = val < 60 ? (Math.random()*6 + 4) : (Math.random()*2 + 0.5);
    val = Math.min(90, val + step);
    progressBar.style.width = val + '%';
  }, 250);
}

function stopProgress(){
  if (progressInterval){
    clearInterval(progressInterval);
    progressInterval = null;
  }
  progressBar.style.width = '100%';
  setTimeout(()=>{
    progressContainer.setAttribute('aria-hidden','true');
    progressBar.style.width = '0%';
  }, 400);
}

let lastText = '';
let lastPages = 0;
let lastFile = null;
const sessionsKey = 'ocr_sessions_v1';
let currentSessionId = null;
let currentServerSessionId = null;

function loadSessions(){
  try{
    const raw = localStorage.getItem(sessionsKey);
    return raw ? JSON.parse(raw) : [];
  }catch(e){ console.warn('Failed to load sessions', e); return [] }
}

function saveSessions(arr){
  try{ localStorage.setItem(sessionsKey, JSON.stringify(arr)) }catch(e){ console.warn('Failed to save sessions', e) }
}

function renderSidebar(){
  const list = document.getElementById('sessions-list');
  list.innerHTML = '';
  const sessions = loadSessions();
  if (!sessions.length){ list.innerHTML = '<div style="color:#666;padding:8px">No sessions yet</div>'; return }
  sessions.slice().reverse().forEach(sess=>{
    const el = document.createElement('div'); el.className = 'session-item';
    const thumb = document.createElement('img');
    thumb.src = sess.preview || '/static/placeholder.png';
    const meta = document.createElement('div'); meta.className='session-meta';
    const title = document.createElement('div'); title.className='title'; title.textContent = sess.name || ('Session ' + sess.id);
    const sub = document.createElement('div'); sub.className='sub'; sub.textContent = `${new Date(sess.ts).toLocaleString()} · ${sess.pages||1} page(s)`;
    meta.appendChild(title); meta.appendChild(sub);
    const actions = document.createElement('div'); actions.className='session-actions';
    const loadBtn = document.createElement('button'); loadBtn.textContent='Open'; loadBtn.onclick = ()=>{ selectSession(sess.id) };
    const delBtn = document.createElement('button'); delBtn.textContent='✕'; delBtn.title='Delete'; delBtn.onclick = (e)=>{ e.stopPropagation(); deleteSession(sess.id) };
    actions.appendChild(loadBtn); actions.appendChild(delBtn);
    el.appendChild(thumb); el.appendChild(meta); el.appendChild(actions);
    list.appendChild(el);
  });
}

async function refreshServerSessions(){
  const list = document.getElementById('server-sessions-list');
  list.innerHTML = '';
  try{
    const res = await fetch('/sessions');
    if (!res.ok) return;
    const j = await res.json();
    const sessions = j.sessions || [];
    if (!sessions.length){ list.innerHTML = '<div style="color:#666;padding:8px">No server sessions</div>'; return }
    sessions.slice().reverse().forEach(id=>{
      const el = document.createElement('div'); el.className='session-item';
      const metaDiv = document.createElement('div'); metaDiv.className='session-meta';
      const title = document.createElement('div'); title.className='title'; title.textContent = id;
      const loadBtn = document.createElement('button'); loadBtn.textContent='Open'; loadBtn.onclick = ()=> loadServerSession(id);
      metaDiv.appendChild(title);
      el.appendChild(metaDiv);
      el.appendChild(loadBtn);
      list.appendChild(el);
    });
  }catch(e){ console.warn('refreshServerSessions', e); }
}

async function loadServerSession(session_id){
  try{
    const res = await fetch(`/session/${session_id}`);
    if (!res.ok){ status('Failed to load session'); return }
    const j = await res.json();
    const meta = j.meta || {};
    // show structured fields
    structuredCard.style.display = 'block';
    structuredFields.innerHTML = '';
    const s = meta.structured || {};
    const fields = ['title','author','date','institution','keywords','full_text'];
    fields.forEach(k=>{
      const row = document.createElement('div'); row.className='field';
      const lbl = document.createElement('div'); lbl.className='label'; lbl.textContent = k;
      const val = document.createElement('div'); val.className='value'; val.textContent = Array.isArray(s[k]) ? s[k].join(', ') : (s[k] || '');
      row.appendChild(lbl); row.appendChild(val); structuredFields.appendChild(row);
    });
    // also populate extracted text from full_text and set internal state
    if (s.full_text) {
      lastText = s.full_text;
      extractedEl.value = s.full_text;
      // prefer full-text retrieval for server sessions
      useFullEl.checked = true;
      useFullEl.disabled = true;
    } else {
      useFullEl.disabled = false;
    }
    // set as current server session
    currentServerSessionId = session_id;
    // also populate extracted text and pages count (fetch chunks count)
    lastPages = j.chunks_count || 0;
    pagesEl.textContent = `Pages (chunks): ${lastPages}`;
    // fetch and render chat history if present in meta
    if (meta.messages && meta.messages.length){
      // replace local chat rendering
      const chat = document.getElementById('chat-history'); chat.innerHTML = '';
      meta.messages.forEach(m=>{
        const el = document.createElement('div'); el.className = 'msg ' + (m.role==='user'?'user':'assistant');
        el.innerHTML = `<div class="body">${escapeHtml(m.text)}</div><div class="meta">${new Date(m.ts*1000).toLocaleString()}</div>`;
        chat.appendChild(el);
      });
    }
    status('Loaded server session');
  }catch(e){ console.error(e); status('Load server session failed') }
}

function addSession(session){
  // local session storage has been removed; server sessions are authoritative
  // keep no-op to avoid errors from older code paths
  return;
}

function deleteSession(id){
  let sessions = loadSessions();
  sessions = sessions.filter(s=>s.id !== id);
  saveSessions(sessions);
  renderSidebar();
}

function selectSession(id){
  const sessions = loadSessions();
  const s = sessions.find(x=>x.id===id);
  if (!s){ status('Session not found'); return }
  lastText = s.text || '';
  lastPages = s.pages || 0;
  extractedEl.value = lastText;
  pagesEl.textContent = `Pages: ${lastPages}`;
  // preview
  if (s.preview){ preview.innerHTML = `<img src="${s.preview}"/>` }
  // set as current session and render chat history
  currentSessionId = s.id;
  renderChatHistoryForSession(s);
  status('Loaded session');
}

document.getElementById('clear-sessions').addEventListener('click', ()=>{
  // local session clearing removed; refresh server sessions instead
  if (!confirm('Refresh server sessions list?')) return;
  refreshServerSessions();
});

// generate thumbnail for image files (returns dataURL) - null for non-image
function createThumbnail(file){
  return new Promise((resolve)=>{
    if (!file || !file.type.startsWith('image/')) return resolve(null);
    const fr = new FileReader();
    fr.onload = ()=>{
      const img = new Image(); img.onload = ()=>{
        const canvas = document.createElement('canvas');
        const maxW = 200; const scale = Math.min(1, maxW / img.width);
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img,0,0,canvas.width,canvas.height);
        resolve(canvas.toDataURL('image/jpeg',0.8));
      };
      img.src = fr.result;
    };
    fr.onerror = ()=>resolve(null);
    fr.readAsDataURL(file);
  });
}

// initialize sidebar
refreshServerSessions();

function getSessionById(id){
  const sessions = loadSessions();
  return sessions.find(s=>s.id===id);
}

function renderChatHistoryForSession(session){
  const chat = document.getElementById('chat-history');
  if (!chat) return;
  chat.innerHTML = '';
  if (!session || !session.messages || !session.messages.length){
    chat.innerHTML = '<div style="color:#666">No chat history yet</div>';
    return;
  }
  session.messages.forEach(m=>{
    const el = document.createElement('div');
    el.className = 'msg ' + (m.role === 'user' ? 'user' : 'assistant');
    el.innerHTML = `<div class="body">${escapeHtml(m.text)}</div><div class="meta">${new Date(m.ts).toLocaleString()}</div>`;
    chat.appendChild(el);
  });
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') }

function saveMessageToSession(sessionId, msg){
  const sessions = loadSessions();
  const idx = sessions.findIndex(s=>s.id===sessionId);
  if (idx === -1) return;
  sessions[idx].messages = sessions[idx].messages || [];
  sessions[idx].messages.push(msg);
  saveSessions(sessions);
}

fileInput.addEventListener('change', () => {
  extractedEl.value = '';
  pagesEl.textContent = '';
  status('Ready');
  const f = fileInput.files[0];
  lastFile = f;
  if (!f) return;
  if (f.type.startsWith('image/')){
    const url = URL.createObjectURL(f);
    preview.innerHTML = `<img src="${url}" />`;
  } else {
    preview.innerHTML = `<div>Uploaded: ${f.name}</div>`;
  }
});

async function uploadFile(){
  if (!lastFile) { status('Select a file first'); return null }
  const fd = new FormData();
  fd.append('file', lastFile);
  status('Uploading...');
  startProgress();
  const res = await fetch('/ocr', { method:'POST', body: fd });
  if (!res.ok){ const t = await res.text(); status('Error: '+t); throw new Error(t) }
  const j = await res.json();
  stopProgress();
  return j;
}

// Run OCR now creates a server session by invoking the same server process flow
btnOCR.addEventListener('click', async ()=>{
  if (!lastFile){ status('Select a file first'); return }
  // trigger the server process flow which will create and load a session
  btnProcess.click();
});

// Server-side full processing: /process endpoint
btnProcess.addEventListener('click', async ()=>{
  if (!lastFile){ status('Select a file first'); return }
  const fd = new FormData(); fd.append('file', lastFile);
  // immediate UI feedback: disable button, mark busy and show progress
  const origText = btnProcess.innerText;
  try{
    btnProcess.disabled = true;
    btnProcess.setAttribute('aria-busy','true');
    btnProcess.innerText = 'Processing...';
  }catch(err){ console.warn('btnProcess update failed', err) }
  // immediate status and console trace for debugging
  status('Processing (starting upload)...');
  console.log('Process button clicked, starting server process...');
  // start visual progress immediately and force a repaint using two RAF ticks
  startProgress();
  await new Promise(r => requestAnimationFrame(()=>requestAnimationFrame(r)));
  try{
    const res = await fetch('/process', { method:'POST', body: fd });
    if (!res.ok){ const t = await res.text(); stopProgress(); status('Server process failed: '+t); return }
    const j = await res.json();
    stopProgress();
    status('Server processed — session created');
    // show structured fields
    await refreshServerSessions();
    if (j.session_id) {
      // automatically load the session so RAG is ready
      await loadServerSession(j.session_id);
      status('Session ready — use Ask button to query with RAG');
    }
  }catch(e){ console.error('Process error', e); stopProgress(); status('Server process error') }
  finally{
    // restore button state
    btnProcess.disabled = false;
    btnProcess.removeAttribute('aria-busy');
    btnProcess.innerText = origText;
  }
});

btnAsk.addEventListener('click', async ()=>{
  const q = questionEl.value.trim();
  if (!q) { status('Type a question'); return }
  if (!lastText){ status('Run OCR first'); return }
  status('Querying model...');
  try{
    // append user message locally (do not persist to local sessions when using server sessions)
    const userMsg = { role: 'user', text: q, ts: Date.now() };
    const chat = document.getElementById('chat-history');
    const uel = document.createElement('div'); uel.className = 'msg user'; uel.innerHTML = `<div class="body">${escapeHtml(q)}</div><div class="meta">${new Date(userMsg.ts).toLocaleString()}</div>`; chat.appendChild(uel); chat.scrollTop = chat.scrollHeight;
    if (currentSessionId){
      saveMessageToSession(currentSessionId, userMsg);
    }

    // If a server session is active, use /query (retrieval+LLM); otherwise fallback to /rag
    if (currentServerSessionId){
      // for server sessions always prefer using the session full_text as context
      const payload = { session_id: currentServerSessionId, question: q, top_k: 5, use_full: true };
      startProgress();
      const res = await fetch('/query', { method:'POST', headers:{ 'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      if (!res.ok){ const t=await res.text(); status('Error: '+t); stopProgress(); return }
      const j = await res.json();
      const ans = j.answer || 'No answer';
      const assistantMsg = { role: 'assistant', text: ans, ts: Date.now() };
      const ael = document.createElement('div'); ael.className = 'msg assistant'; ael.innerHTML = `<div class="body">${escapeHtml(ans)}</div><div class="meta">${new Date(assistantMsg.ts).toLocaleString()}</div>`; chat.appendChild(ael); chat.scrollTop = chat.scrollHeight;
      stopProgress();
      status('Done');
    }else{
      const payload = { text: lastText, question: q, use_full: useFullEl.checked };
      startProgress();
      const res = await fetch('/rag', { method:'POST', headers:{ 'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      if (!res.ok){ const t=await res.text(); status('Error: '+t); stopProgress(); return }
      const j = await res.json();
      const ans = j.answer || 'No answer';
      const assistantMsg = { role: 'assistant', text: ans, ts: Date.now() };
      const ael = document.createElement('div'); ael.className = 'msg assistant'; ael.innerHTML = `<div class="body">${escapeHtml(ans)}</div><div class="meta">${new Date(assistantMsg.ts).toLocaleString()}</div>`; chat.appendChild(ael); chat.scrollTop = chat.scrollHeight;
      stopProgress();
      status('Done');
    }
    questionEl.value = '';
  }catch(e){ console.error(e); stopProgress(); status('RAG failed') }
});

btnPDF.addEventListener('click', async ()=>{
  if (!lastText) { status('Run OCR first'); return }
  status('Generating PDF...');
  try{
    const res = await fetch('/generate_pdf', { method:'POST', headers:{ 'Content-Type':'application/json'}, body: JSON.stringify({ text: lastText }) });
    if (!res.ok){ const t=await res.text(); status('Error: '+t); return }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'notes.pdf'; document.body.appendChild(a); a.click(); a.remove();
    status('PDF downloaded');
  }catch(e){ console.error(e); status('PDF generation failed') }
});

btnExportCsv.addEventListener('click', ()=> exportSession('csv'));
btnExportXlsx.addEventListener('click', ()=> exportSession('xlsx'));
btnExportDocx.addEventListener('click', ()=> exportSession('docx'));

async function exportSession(fmt){
  if (!currentServerSessionId){ status('Load a server session first'); return }
  try{
    const res = await fetch('/export', { method:'POST', headers:{ 'Content-Type':'application/json'}, body: JSON.stringify({ session_id: currentServerSessionId, format: fmt })});
    if (!res.ok){ const t = await res.text(); status('Export failed: '+t); return }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = `export.${fmt === 'xlsx' ? 'xlsx' : fmt}`; document.body.appendChild(a); a.click(); a.remove();
    status('Export downloaded');
  }catch(e){ console.error(e); status('Export error') }
}

btnVisualize.addEventListener('click', async ()=>{
  if (!currentServerSessionId){ status('Load a server session first'); return }
  try{
    startProgress();
    const res = await fetch(`/visualize/${currentServerSessionId}?top_k=12`);
    if (!res.ok){ stopProgress(); status('Visualization failed'); return }
    const j = await res.json();
    const items = j.top_terms || [];
    if (!items.length){ stopProgress(); status('No terms to visualize'); return }
    vizCanvas.style.display = 'block';
    const labels = items.map(x=>x[0]); const data = items.map(x=>x[1]);
    // draw chart using Chart.js
    if (window._chart) window._chart.destroy();
    const ctx = vizCanvas.getContext('2d');
    window._chart = new Chart(ctx, { type:'bar', data:{ labels, datasets:[{ label:'term frequency', data, backgroundColor:'#60a5fa' }] }, options:{ responsive:true, maintainAspectRatio:false } });
    stopProgress();
    status('Visualization ready');
  }catch(e){ console.error(e); stopProgress(); status('Visualization error') }
});

function status(s){ statusEl.textContent = s }
status('Ready');
