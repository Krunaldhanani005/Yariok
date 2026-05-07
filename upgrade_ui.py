import os

def upgrade_index_html():
    with open('templates/index.html', 'r') as f:
        content = f.read()
    
    # 1. Update Navbar
    navbar = """<div class="navbar">
  <a href="/" class="nav-link active">Dashboard</a>
  <a href="/snapshots" class="nav-link">Snapshots</a>
  <a href="/alerts" class="nav-link">Alerts</a>
  <a href="#" class="nav-link">Logs</a>
  <a href="#" class="nav-link">Settings</a>
</div>"""
    content = content.replace("""<div class="navbar">
  <a href="/" class="nav-link active">Dashboard</a>
  <a href="/snapshots" class="nav-link">Snapshots</a>
  <a href="#" class="nav-link">Logs</a>
  <a href="#" class="nav-link">Settings</a>
</div>""", navbar)

    # 2. Add Alert Controls in system controls
    alert_ctrl = """
      <div class="action-grid" style="margin-top:8px">
        <button class="abtn" id="btn-alert-on" onclick="ctrlAlert('on')"><span class="ai">🚨</span>ALERT MODE ON</button>
        <button class="abtn" id="btn-alert-off" onclick="ctrlAlert('off')" style="border-color:var(--border);color:var(--muted)"><span class="ai">🛡</span>ALERT MODE OFF</button>
      </div>
"""
    content = content.replace('      </div>\n    </div>\n    \n    <!-- System Status -->', alert_ctrl + '      </div>\n    </div>\n    \n    <!-- System Status -->')

    # 3. Add Live Alert Popup Notification
    live_alert = """
    <!-- Live Alert Popup -->
    <div id="live-alert-popup" style="display:none; position:fixed; bottom:20px; right:20px; background:var(--red); padding:15px; border-radius:10px; z-index:9999; box-shadow:0 10px 25px rgba(255,0,0,0.3); border:2px solid #ff8888;">
      <div style="font-weight:bold; font-size:16px; margin-bottom:5px; color:white;">🚨 ALERT DETECTED</div>
      <div id="live-alert-time" style="font-size:12px; color:rgba(255,255,255,0.8); margin-bottom:10px;"></div>
      <img id="live-alert-img" src="" style="width:200px; border-radius:6px; display:block;">
    </div>
"""
    content = content.replace("</body>", live_alert + "\n</body>")

    # 4. JS updates for new controls and alert popup
    js_update = """
async function ctrlAlert(action) {
  await fetch('/api/alert/' + action, { method: 'POST' });
  refresh();
}

let lastAlertTime = "";
"""
    content = content.replace("async function ctrlSys(action) {", js_update + "async function ctrlSys(action) {")

    js_refresh = """
    // Alert Mode buttons
    const secMode = d.security_mode || false;
    document.getElementById('btn-alert-on').style.borderColor = secMode ? 'var(--red)' : 'var(--border)';
    document.getElementById('btn-alert-on').style.color = secMode ? 'var(--red)' : 'var(--muted)';
    document.getElementById('btn-alert-off').style.borderColor = !secMode ? 'var(--blue)' : 'var(--border)';
    document.getElementById('btn-alert-off').style.color = !secMode ? 'var(--blue)' : 'var(--muted)';
    
    // Check for new live alerts
    if (d.last_live_alert_time && d.last_live_alert_time !== lastAlertTime) {
       lastAlertTime = d.last_live_alert_time;
       document.getElementById('live-alert-time').textContent = d.last_live_alert_time;
       document.getElementById('live-alert-img').src = '/alerts/' + d.last_live_alert;
       document.getElementById('live-alert-popup').style.display = 'block';
       setTimeout(() => { document.getElementById('live-alert-popup').style.display = 'none'; }, 8000);
    }
"""
    content = content.replace("// Toggle Camera View", js_refresh + "\n    // Toggle Camera View")

    with open('templates/index.html', 'w') as f:
        f.write(content)

def upgrade_snapshots_html():
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

.container { padding: 30px; }
.date-header {
  font-size: 20px;
  font-weight: bold;
  color: var(--blue);
  margin: 30px 0 15px 0;
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
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
  <a href="/alerts" class="nav-link">Alerts</a>
  <a href="#" class="nav-link">Logs</a>
  <a href="#" class="nav-link">Settings</a>
</div>

<div class="container" id="content">
  <!-- Dynamic -->
</div>

<script>
function isToday(dateString) {
  const today = new Date();
  const d = new Date(dateString);
  return d.getDate() === today.getDate() &&
         d.getMonth() === today.getMonth() &&
         d.getFullYear() === today.getFullYear();
}
function isYesterday(dateString) {
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const d = new Date(dateString);
  return d.getDate() === yesterday.getDate() &&
         d.getMonth() === yesterday.getMonth() &&
         d.getFullYear() === yesterday.getFullYear();
}
function getFriendlyDate(dateStr) {
    if(isToday(dateStr)) return "Today";
    if(isYesterday(dateStr)) return "Yesterday";
    return dateStr;
}

async function loadSnaps() {
  const res = await fetch('/api/snapshots/daywise').then(r=>r.json());
  const content = document.getElementById('content');
  let html = '';
  
  for(const [date, snaps] of Object.entries(res.grouped)) {
      html += `<div class="date-header">${getFriendlyDate(date)}</div><div class="grid">`;
      html += snaps.map(s => `
        <div class="card">
          <img src="/snapshots/${s.filename}" alt="${s.filename}">
          <div class="card-body">
            <div class="card-title">${s.visitor_type || 'Unknown'} Detection</div>
            <div class="card-sub">${s.timestamp}</div>
            <div class="actions">
              <a class="btn" href="/snapshots/${s.filename}" download>Download</a>
              <a class="btn" href="/snapshots/${s.filename}" target="_blank">View</a>
              <button class="btn danger" onclick="delSnap(${s.id})">Delete</button>
            </div>
          </div>
        </div>
      `).join('');
      html += `</div>`;
  }
  content.innerHTML = html || "<div style='color:var(--muted)'>No snapshots found.</div>";
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

def create_alerts_html():
    content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yaariok Alerts</title>
<style>
:root {
  --bg:      #06090f;
  --panel:   #0b1120;
  --card:    #150e10;
  --border:  #1a2540;
  --border2: #35151a;
  --text:    #e2e8f0;
  --muted:   #4b6080;
  --red:     #ef4444;
  --blue:    #3b82f6;
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
.nav-link.active { color: var(--red); border-bottom: 2px solid var(--red); padding-bottom: 18px; }

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
  box-shadow: 0 4px 15px rgba(239, 68, 68, 0.05);
}
.card img {
  width: 100%;
  height: 180px;
  object-fit: cover;
  border-bottom: 2px solid var(--red);
}
.card-body {
  padding: 15px;
}
.card-title {
  font-size: 14px;
  font-weight: bold;
  margin-bottom: 5px;
  color: #ffcccc;
}
.card-sub {
  font-size: 11px;
  color: #ff8888;
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
.btn:hover { background: rgba(255,255,255,0.05); }
.btn.danger { color: var(--red); border-color: rgba(239,68,68,0.4); }
.btn.danger:hover { background: rgba(239,68,68,0.2); }
</style>
</head>
<body>
<div class="navbar">
  <a href="/" class="nav-link">Dashboard</a>
  <a href="/snapshots" class="nav-link">Snapshots</a>
  <a href="/alerts" class="nav-link active">Alerts</a>
  <a href="#" class="nav-link">Logs</a>
  <a href="#" class="nav-link">Settings</a>
</div>

<div class="grid" id="grid">
  <!-- Dynamic -->
</div>

<script>
async function loadAlerts() {
  const res = await fetch('/api/alerts').then(r=>r.json());
  const grid = document.getElementById('grid');
  grid.innerHTML = res.alerts.map(s => `
    <div class="card">
      <img src="/alerts/${s.image_path}" alt="${s.image_path}">
      <div class="card-body">
        <div class="card-title">🚨 ${s.alert_type || 'Motion Detected'}</div>
        <div class="card-sub">${s.timestamp} | Status: ${s.status}</div>
        <div class="actions">
          <a class="btn" href="/alerts/${s.image_path}" download>Download</a>
          <a class="btn" href="/alerts/${s.image_path}" target="_blank">View</a>
          <button class="btn danger" onclick="delAlert(${s.id})">Delete</button>
        </div>
      </div>
    </div>
  `).join('');
  if(!res.alerts.length) grid.innerHTML = "<div style='color:var(--muted)'>No alerts recorded.</div>";
}
async function delAlert(id) {
  if(!confirm("Delete alert record?")) return;
  await fetch('/api/alert/' + id, { method: 'DELETE' });
  loadAlerts();
}
loadAlerts();

// Auto refresh every 5 seconds
setInterval(loadAlerts, 5000);
</script>
</body>
</html>"""
    with open('templates/alerts.html', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    upgrade_index_html()
    upgrade_snapshots_html()
    create_alerts_html()
