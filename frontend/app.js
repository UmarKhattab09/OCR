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

function addSession(session){
  const sessions = loadSessions();
  sessions.push(session);
  // limit to 200 sessions
  if (sessions.length > 200) sessions.splice(0, sessions.length-200);
  saveSessions(sessions);
  renderSidebar();
  // set created session as current and render its chat history
  currentSessionId = session.id;
  renderChatHistoryForSession(session);
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
  if (!confirm('Clear all saved sessions?')) return;
  localStorage.removeItem(sessionsKey);
  renderSidebar();
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
renderSidebar();

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

btnOCR.addEventListener('click', async ()=>{
  try{
    const j = await uploadFile();
    lastText = j.text || '';
    lastPages = j.pages || 0;
    extractedEl.value = lastText;
    pagesEl.textContent = `Pages: ${lastPages}`;
    status('OCR complete');
    // save session with thumbnail (if image)
    const thumb = await createThumbnail(lastFile);
    const sess = { id: Date.now().toString(), name: lastFile ? lastFile.name : 'upload', ts: Date.now(), text: lastText, pages: lastPages, preview: thumb, messages: [] };
    addSession(sess);
  }catch(e){
    console.error(e);
    status('OCR failed');
  }
});

btnAsk.addEventListener('click', async ()=>{
  const q = questionEl.value.trim();
  if (!q) { status('Type a question'); return }
  if (!lastText){ status('Run OCR first'); return }
  status('Querying model...');
  try{
    // ensure we have a session to attach messages to
    if (!currentSessionId){
      const sess = { id: Date.now().toString(), name: lastFile ? lastFile.name : 'unsaved', ts: Date.now(), text: lastText, pages: lastPages, preview: null, messages: [] };
      addSession(sess);
    }
    const userMsg = { role: 'user', text: q, ts: Date.now() };
    // append user message locally
    const chat = document.getElementById('chat-history');
    const uel = document.createElement('div'); uel.className = 'msg user'; uel.innerHTML = `<div class="body">${escapeHtml(q)}</div><div class="meta">${new Date(userMsg.ts).toLocaleString()}</div>`; chat.appendChild(uel); chat.scrollTop = chat.scrollHeight;
    saveMessageToSession(currentSessionId, userMsg);

    const payload = { text: lastText, question: q, use_full: useFullEl.checked };
    startProgress();
    const res = await fetch('/rag', { method:'POST', headers:{ 'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    if (!res.ok){ const t=await res.text(); status('Error: '+t); stopProgress(); return }
    const j = await res.json();
    const ans = j.answer || 'No answer';
    const assistantMsg = { role: 'assistant', text: ans, ts: Date.now() };
    const ael = document.createElement('div'); ael.className = 'msg assistant'; ael.innerHTML = `<div class="body">${escapeHtml(ans)}</div><div class="meta">${new Date(assistantMsg.ts).toLocaleString()}</div>`; chat.appendChild(ael); chat.scrollTop = chat.scrollHeight;
    saveMessageToSession(currentSessionId, assistantMsg);
    stopProgress();
    status('Done');
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

function status(s){ statusEl.textContent = s }
status('Ready');
