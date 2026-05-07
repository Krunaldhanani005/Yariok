import os

def patch_index_html():
    with open('templates/index.html', 'r') as f:
        content = f.read()
    
    # Add Navbar styling
    nav_css = """
/* ═══════════ NAVBAR ═══════════════════════════════════════════════════════ */
.navbar {
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  display: flex;
  gap: 20px;
  padding: 0 20px;
  height: 40px;
  align-items: center;
}
.nav-link {
  color: var(--muted);
  text-decoration: none;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  transition: color 0.2s;
}
.nav-link:hover { color: var(--text); }
.nav-link.active { color: var(--blue); border-bottom: 2px solid var(--blue); padding-bottom: 8px; }
"""
    content = content.replace("/* ═══════════ HEADER ═══════════════════════════════════════════════════════ */", nav_css + "\n/* ═══════════ HEADER ═══════════════════════════════════════════════════════ */")

    # Update body grid template
    content = content.replace("grid-template-rows:56px 1fr 130px;", "grid-template-rows:56px 40px 1fr 130px;")

    # Add Navbar HTML
    nav_html = """
<div class="navbar">
  <a href="/" class="nav-link active">Dashboard</a>
  <a href="/snapshots" class="nav-link">Snapshots</a>
  <a href="#" class="nav-link">Logs</a>
  <a href="#" class="nav-link">Settings</a>
</div>
"""
    content = content.replace("</header>", "</header>\n" + nav_html)

    # Add System Status and Control Card HTML
    sys_card_html = """
    <!-- System Controls -->
    <div class="sec">
      <div class="sec-title">System Controls</div>
      <div class="action-grid" style="margin-bottom:8px">
        <button class="abtn" id="btn-sys-start" onclick="ctrlSys('start')"><span class="ai">▶</span>START SYSTEM</button>
        <button class="abtn" id="btn-sys-stop" onclick="ctrlSys('stop')"><span class="ai">⏹</span>STOP SYSTEM</button>
      </div>
      <div class="action-grid">
        <button class="abtn" id="btn-voice-on" onclick="ctrlVoice('on')"><span class="ai">🎙</span>VOICE ON</button>
        <button class="abtn" id="btn-voice-off" onclick="ctrlVoice('off')" style="border-color:var(--blue);color:var(--blue)"><span class="ai">🔇</span>VOICE OFF</button>
      </div>
    </div>
    
    <!-- System Status -->
    <div class="sec">
      <div class="sec-title">System Status</div>
      <div style="font-size:11px; color:var(--muted); line-height:1.8">
        <div>System: <span id="st-sys" style="color:var(--text)">Stopped</span></div>
        <div>Camera: <span id="st-cam" style="color:var(--text)">Off</span></div>
        <div>Voice: <span id="st-voice" style="color:var(--text)">Off</span></div>
        <div>Detection: <span id="st-det" style="color:var(--text)">Inactive</span></div>
      </div>
    </div>
"""
    # Insert sys_card_html before Live Stats
    content = content.replace('<!-- Live Stats -->', sys_card_html + '\n    <!-- Live Stats -->')

    # Update Camera Placeholder
    cam_placeholder = """
  <div class="cam-wrap" id="cam-wrap">
    <div id="cam-placeholder" style="color:var(--muted); font-size:18px; text-transform:uppercase; letter-spacing:2px">System Not Started</div>
    <img id="cam-img" src="" alt="Live Camera" style="display:none">
"""
    content = content.replace('<div class="cam-wrap" id="cam-wrap">\n    <img src="/video_feed" alt="Live Camera">', cam_placeholder)

    # Add JS logic for new controls
    new_js = """
async function ctrlSys(action) {
  await fetch('/api/system/' + action, { method: 'POST' });
  refresh();
}
async function ctrlVoice(action) {
  await fetch('/api/voice/' + action, { method: 'POST' });
  refresh();
}
"""
    content = content.replace("async function ctrl(action,btn){", new_js + "\nasync function ctrl(action,btn){")

    # Update JS refresh logic to update status labels and cam visibility
    refresh_patch = """
    // Update System Status
    const sysRun = d.system_running || false;
    const voiceRun = d.voice_running || false;
    
    document.getElementById('st-sys').textContent = sysRun ? 'Running' : 'Stopped';
    document.getElementById('st-sys').style.color = sysRun ? 'var(--green)' : 'var(--text)';
    
    document.getElementById('st-cam').textContent = sysRun ? 'On' : 'Off';
    document.getElementById('st-cam').style.color = sysRun ? 'var(--green)' : 'var(--text)';
    
    document.getElementById('st-det').textContent = sysRun ? 'Active' : 'Inactive';
    document.getElementById('st-det').style.color = sysRun ? 'var(--green)' : 'var(--text)';
    
    document.getElementById('st-voice').textContent = voiceRun ? 'On' : 'Off';
    document.getElementById('st-voice').style.color = voiceRun ? 'var(--green)' : 'var(--text)';
    
    // Toggle Camera View
    const camImg = document.getElementById('cam-img');
    const camPlace = document.getElementById('cam-placeholder');
    if(sysRun) {
      if(camImg.style.display === 'none') {
         camImg.src = '/video_feed';
         camImg.style.display = 'block';
         camPlace.style.display = 'none';
      }
    } else {
      camImg.style.display = 'none';
      camImg.src = '';
      camPlace.style.display = 'block';
    }
    
    // Toggle Buttons
    document.getElementById('btn-sys-start').style.borderColor = sysRun ? 'var(--green)' : 'var(--border)';
    document.getElementById('btn-sys-start').style.color = sysRun ? 'var(--green)' : 'var(--muted)';
    document.getElementById('btn-sys-stop').style.borderColor = !sysRun ? 'var(--blue)' : 'var(--border)';
    document.getElementById('btn-sys-stop').style.color = !sysRun ? 'var(--blue)' : 'var(--muted)';
    
    document.getElementById('btn-voice-on').style.borderColor = voiceRun ? 'var(--green)' : 'var(--border)';
    document.getElementById('btn-voice-on').style.color = voiceRun ? 'var(--green)' : 'var(--muted)';
    document.getElementById('btn-voice-off').style.borderColor = !voiceRun ? 'var(--blue)' : 'var(--border)';
    document.getElementById('btn-voice-off').style.color = !voiceRun ? 'var(--blue)' : 'var(--muted)';
"""
    content = content.replace("document.getElementById('s-ppl').textContent=d.people_now??0;", refresh_patch + "\n    document.getElementById('s-ppl').textContent=d.people_now??0;")

    with open('templates/index.html', 'w') as f:
        f.write(content)

def create_snapshots_html():
    content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yaariok Snapshots</title>
<style>
:root {
  --bg:      #06090f;
  --panel:   #0b1120;
  --card:    #0f1929;
  --border:  #1a2540;
  --border2: #243050;
  --text:    #e2e8f0;
  --muted:   #4b6080;
  --green:   #10b981;
  --red:     #ef4444;
  --amber:   #f59e0b;
  --blue:    #3b82f6;
  --purple:  #8b5cf6;
  --cyan:    #06b6d4;
}
body{
  background:var(--bg);
  color:var(--text);
  font-family:'Segoe UI',system-ui,sans-serif;
  margin:0;
}
.navbar {
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  display: flex;
  gap: 20px;
  padding: 0 20px;
  height: 60px;
  align-items: center;
}
.nav-link {
  color: var(--muted);
  text-decoration: none;
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  transition: color 0.2s;
}
.nav-link:hover { color: var(--text); }
.nav-link.active { color: var(--blue); border-bottom: 2px solid var(--blue); padding-bottom: 18px; }

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
  padding: 30px;
}
.card {
  background: var(--card);
  border: 1px solid var(--border2);
  border-radius: 12px;
  overflow: hidden;
}
.card img {
  width: 100%;
  height: 180px;
  object-fit: cover;
}
.card-body {
  padding: 15px;
}
.card-title {
  font-size: 14px;
  font-weight: bold;
  margin-bottom: 5px;
}
.card-sub {
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 15px;
}
.actions {
  display: flex;
  gap: 10px;
}
.btn {
  flex: 1;
  padding: 8px;
  border-radius: 6px;
  border: 1px solid var(--border2);
  background: transparent;
  color: var(--text);
  font-size: 12px;
  cursor: pointer;
  text-align: center;
  text-decoration: none;
}
.btn:hover { background: var(--border); }
.btn.danger { color: var(--red); border-color: rgba(239,68,68,0.3); }
.btn.danger:hover { background: rgba(239,68,68,0.1); }
</style>
</head>
<body>
<div class="navbar">
  <a href="/" class="nav-link">Dashboard</a>
  <a href="/snapshots" class="nav-link active">Snapshots</a>
  <a href="#" class="nav-link">Logs</a>
  <a href="#" class="nav-link">Settings</a>
</div>

<div class="grid" id="grid">
  <!-- Dynamic -->
</div>

<script>
async function loadSnaps() {
  const res = await fetch('/api/snapshots').then(r=>r.json());
  const grid = document.getElementById('grid');
  grid.innerHTML = res.snapshots.map(s => `
    <div class="card">
      <img src="/snapshots/${s.filename}" alt="${s.filename}">
      <div class="card-body">
        <div class="card-title">${s.visitor_type || 'Unknown'} Detection</div>
        <div class="card-sub">${s.timestamp}</div>
        <div class="actions">
          <a class="btn" href="/snapshots/${s.filename}" target="_blank">View</a>
          <button class="btn danger" onclick="delSnap(${s.id})">Delete</button>
        </div>
      </div>
    </div>
  `).join('');
}
async function delSnap(id) {
  if(!confirm("Delete snapshot?")) return;
  await fetch('/api/snapshot/' + id, { method: 'DELETE' });
  loadSnaps();
}
loadSnaps();
</script>
</body>
</html>"""
    with open('templates/snapshots.html', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    patch_index_html()
    create_snapshots_html()
