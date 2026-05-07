import os
import re

def fix_ui():
    with open('templates/index.html', 'r') as f:
        content = f.read()
    
    # 1. Replace the buttons
    old_buttons = """      <div class="action-grid" style="margin-bottom:8px">
        <button class="abtn" id="btn-sys-start" onclick="ctrlSys('start')"><span class="ai">▶</span>START SYSTEM</button>
        <button class="abtn" id="btn-sys-stop" onclick="ctrlSys('stop')"><span class="ai">⏹</span>STOP SYSTEM</button>
      </div>
      <div class="action-grid">
        <button class="abtn" id="btn-voice-on" onclick="ctrlVoice('on')"><span class="ai">🎙</span>VOICE ON</button>
        <button class="abtn" id="btn-voice-off" onclick="ctrlVoice('off')" style="border-color:var(--blue);color:var(--blue)"><span class="ai">🔇</span>VOICE OFF</button>"""
    new_buttons = """      <div class="action-grid" style="margin-bottom:8px">
        <button class="abtn" id="btn-sys-toggle" onclick="toggleSys()"><span class="ai" id="sys-icon">▶</span><span id="sys-text">START SYSTEM</span></button>
      </div>
      <div class="action-grid" style="margin-bottom:8px">
        <button class="abtn" id="btn-voice-toggle" onclick="toggleVoice()"><span class="ai" id="voice-icon">🎙</span><span id="voice-text">VOICE ON</span></button>
      </div>"""
    content = content.replace(old_buttons, new_buttons)
    
    # 2. Add toggle logic functions
    toggle_logic = """
async function toggleSys() {
  console.log("Button clicked: System Toggle");
  const isRun = document.getElementById('st-sys').textContent === 'Running';
  await fetch('/api/system/' + (isRun ? 'stop' : 'start'), { method: 'POST' });
  refresh();
}

async function toggleVoice() {
  console.log("Button clicked: Voice Toggle");
  const isRun = document.getElementById('st-voice').textContent === 'On';
  await fetch('/api/voice/' + (isRun ? 'off' : 'on'), { method: 'POST' });
  refresh();
}
"""
    content = content.replace("async function ctrlSys(action) {", toggle_logic + "\nasync function ctrlSys(action) {")
    
    # 3. Update refresh logic
    old_refresh = """    // Toggle Buttons
    document.getElementById('btn-sys-start').style.borderColor = sysRun ? 'var(--green)' : 'var(--border)';
    document.getElementById('btn-sys-start').style.color = sysRun ? 'var(--green)' : 'var(--muted)';
    document.getElementById('btn-sys-stop').style.borderColor = !sysRun ? 'var(--blue)' : 'var(--border)';
    document.getElementById('btn-sys-stop').style.color = !sysRun ? 'var(--blue)' : 'var(--muted)';
    
    document.getElementById('btn-voice-on').style.borderColor = voiceRun ? 'var(--green)' : 'var(--border)';
    document.getElementById('btn-voice-on').style.color = voiceRun ? 'var(--green)' : 'var(--muted)';
    document.getElementById('btn-voice-off').style.borderColor = !voiceRun ? 'var(--blue)' : 'var(--border)';
    document.getElementById('btn-voice-off').style.color = !voiceRun ? 'var(--blue)' : 'var(--muted)';"""
    
    new_refresh = """    // Toggle Buttons
    const sysBtn = document.getElementById('btn-sys-toggle');
    if(sysBtn) {
       sysBtn.style.borderColor = sysRun ? 'var(--red)' : 'var(--green)';
       sysBtn.style.color = sysRun ? 'var(--red)' : 'var(--green)';
       document.getElementById('sys-icon').textContent = sysRun ? '⏹' : '▶';
       document.getElementById('sys-text').textContent = sysRun ? 'STOP SYSTEM' : 'START SYSTEM';
    }

    const voiceBtn = document.getElementById('btn-voice-toggle');
    if(voiceBtn) {
       voiceBtn.style.borderColor = voiceRun ? 'var(--red)' : 'var(--green)';
       voiceBtn.style.color = voiceRun ? 'var(--red)' : 'var(--green)';
       document.getElementById('voice-icon').textContent = voiceRun ? '🔇' : '🎙';
       document.getElementById('voice-text').textContent = voiceRun ? 'VOICE OFF' : 'VOICE ON';
    }"""
    content = content.replace(old_refresh, new_refresh)

    with open('templates/index.html', 'w') as f:
        f.write(content)

def fix_backend():
    with open('smart_system.py', 'r') as f:
        content = f.read()

    # 1. Update start/stop system functions with logging
    old_start_sys = """def start_system():
    global _camera_t
    if not state["system_running"]:
        S(system_running=True)
        _camera_t = threading.Thread(target=camera_thread, daemon=True)
        _camera_t.start()
        log("System started")"""
    new_start_sys = """def start_system():
    global _camera_t
    if not state["system_running"]:
        print("  [System Debug] System started")
        S(system_running=True)
        _camera_t = threading.Thread(target=camera_thread, daemon=True)
        _camera_t.start()
        log("System started")"""
    content = content.replace(old_start_sys, new_start_sys)

    old_stop_sys = """def stop_system():
    if state["system_running"]:
        S(system_running=False, show_camera=False)
        log("System stopped")"""
    new_stop_sys = """def stop_system():
    if state["system_running"]:
        print("  [System Debug] System stopped")
        S(system_running=False, show_camera=False)
        log("System stopped")"""
    content = content.replace(old_stop_sys, new_stop_sys)
    
    # 2. Add adjust_for_ambient_noise to microphone initialization
    old_listen = """def listen(label="Listening", timeout=8) -> str:
    S(robot_status=f"{label}...")
    try:
        with sr.Microphone() as src:
            try:"""
    new_listen = """def listen(label="Listening", timeout=8) -> str:
    S(robot_status=f"{label}...")
    try:
        with sr.Microphone() as src:
            _rec.adjust_for_ambient_noise(src, duration=0.5)
            try:"""
    content = content.replace(old_listen, new_listen)

    with open('smart_system.py', 'w') as f:
        f.write(content)

def fix_dashboard():
    with open('dashboard.py', 'r') as f:
        content = f.read()
    
    # Update status api log
    old_status = """@app.route("/api/status")
def api_status():
    return jsonify(dict(_s))"""
    new_status = """@app.route("/api/status")
def api_status():
    return jsonify({
        "system_running": _s.get("system_running", False),
        "voice_running": _s.get("voice_running", False),
        "camera_status": _s.get("camera_status", "Off"),
        "security_mode": _s.get("security_mode", False),
        "people_now": _s.get("people_now", 0),
        "visitors_today": _s.get("visitors_today", 0),
        "alerts_today": _s.get("alerts_today", 0),
        "snapshots_taken": _s.get("snapshots_taken", 0),
        "last_live_alert": _s.get("last_live_alert", ""),
        "last_live_alert_time": _s.get("last_live_alert_time", ""),
        "robot_status": _s.get("robot_status", ""),
        "robot_speech": _s.get("robot_speech", "")
    })"""
    content = content.replace(old_status, new_status)
    
    # Update API logs
    old_start = """@app.route("/api/system/start", methods=["POST"])
def api_sys_start():
    if hasattr(app, "smart_funcs"):
        app.smart_funcs["start_system"]()
    return jsonify({"ok": True})"""
    new_start = """@app.route("/api/system/start", methods=["POST"])
def api_sys_start():
    print("  [API Debug] API called: /api/system/start")
    if hasattr(app, "smart_funcs"):
        app.smart_funcs["start_system"]()
    return jsonify({"ok": True})"""
    content = content.replace(old_start, new_start)

    old_stop = """@app.route("/api/system/stop", methods=["POST"])
def api_sys_stop():
    if hasattr(app, "smart_funcs"):
        app.smart_funcs["stop_system"]()
    return jsonify({"ok": True})"""
    new_stop = """@app.route("/api/system/stop", methods=["POST"])
def api_sys_stop():
    print("  [API Debug] API called: /api/system/stop")
    if hasattr(app, "smart_funcs"):
        app.smart_funcs["stop_system"]()
    return jsonify({"ok": True})"""
    content = content.replace(old_stop, new_stop)

    with open('dashboard.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    fix_ui()
    fix_backend()
    fix_dashboard()
