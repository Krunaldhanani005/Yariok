import os
import sqlite3
import re

def upgrade_smart_system():
    with open('smart_system.py', 'r') as f:
        content = f.read()

    # 1. Add Alert DB logic to init_db
    db_patch = """
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_path TEXT,
                  timestamp TEXT,
                  alert_type TEXT,
                  status TEXT)''')
    conn.commit()
"""
    content = content.replace("conn.commit()\n    conn.close()", db_patch + "    conn.close()")

    alert_funcs = """
def save_alert(filename, alert_type="Person Detected", status="Active"):
    conn = sqlite3.connect("snapshots.db")
    c = conn.cursor()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO alerts (image_path, timestamp, alert_type, status) VALUES (?, ?, ?, ?)",
              (filename, ts, alert_type, status))
    conn.commit()
    conn.close()

def play_alert_sound():
    try:
        if _neural_ok:
            # We already have pygame initialized
            # Make a simple beep sound
            arr = np.array([4096 * np.sin(2.0 * np.pi * 1000 * x / 44100) for x in range(0, 44100//4)]).astype(np.int16)
            arr = np.column_stack((arr, arr))
            snd = _pygame.sndarray.make_sound(arr)
            snd.play()
    except:
        print("\\a", end="", flush=True)

def trigger_alert(frame, num_people):
    fname = datetime.datetime.now().strftime("alert_%H%M%S.jpg")
    os.makedirs("./alerts/", exist_ok=True)
    snap = "./alerts/" + fname
    cv2.imwrite(snap, frame)
    save_alert(fname, alert_type=f"{num_people} Person(s) Detected")
    play_alert_sound()
    S(last_live_alert=fname, last_live_alert_time=datetime.datetime.now().strftime("%H:%M:%S"))
"""
    content = content.replace("def save_snapshot_db", alert_funcs + "\ndef save_snapshot_db")

    # 2. Refactor Voice commands & system commands
    voice_loop_patch = """
        cmd = cmd.lower()
        if "start system" in cmd:
            start_system()
        elif "stop system" in cmd:
            stop_system()
        elif "alert mode on" in cmd:
            S(security_mode=True, karaoke_mode=False)
            speak("Alert mode on")
        elif "alert mode off" in cmd:
            S(security_mode=False)
            speak("Alert mode off")
        elif "voice off" in cmd:
            speak("Voice off")
            stop_voice()
        elif "voice on" in cmd:
            pass # handled externally since listener is off
        else:
            if not handle(cmd):
                break
"""
    content = content.replace("if not handle(cmd):\n            break", voice_loop_patch)

    # 3. Rename functions as requested
    content = content.replace("def voice_loop():", "def process_voice_command():\n    pass\n\ndef voice_loop():")
    content = content.replace("def start_voice():", "def start_voice_listener():")
    content = content.replace("def stop_voice():", "def stop_voice_listener():")
    content = content.replace("start_voice,", "start_voice_listener,")
    content = content.replace("stop_voice,", "stop_voice_listener,")

    # 4. Integrate trigger_alert into the main loop
    alert_logic_patch = """
                    if state["security_mode"]:
                        trigger_alert(frame, people)
                        def _sec_greet(f=frame.copy(), v=vnum):
                            speak("Security alert! Unauthorized person detected. Analyzing now.")
"""
    content = content.replace("""                    if state["security_mode"]:
                        def _sec_greet""", alert_logic_patch)

    # 5. Start voice listener automatically in main
    main_patch = """
    _init_llm()
    # Start voice by default
    start_voice_listener()
    
    # Start web dashboard
"""
    content = content.replace("    _init_llm()\n    # Do NOT start camera or voice by default\n    # Start web dashboard", main_patch)

    with open('smart_system.py', 'w') as f:
        f.write(content)

def upgrade_dashboard():
    with open('dashboard.py', 'r') as f:
        content = f.read()

    new_routes = """
@app.route("/api/alert/on", methods=["POST"])
def api_alert_on():
    _s["security_mode"] = True
    return jsonify({"ok": True})

@app.route("/api/alert/off", methods=["POST"])
def api_alert_off():
    _s["security_mode"] = False
    return jsonify({"ok": True})

@app.route("/alerts")
def page_alerts():
    return render_template("alerts.html")

@app.route("/api/alerts")
def api_alerts():
    try:
        conn = sqlite3.connect("snapshots.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM alerts ORDER BY id DESC")
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return jsonify({"alerts": rows})
    except Exception as e:
        return jsonify({"alerts": [], "error": str(e)})

@app.route("/api/alert/<int:sid>", methods=["DELETE"])
def api_del_alert(sid):
    try:
        conn = sqlite3.connect("snapshots.db")
        c = conn.cursor()
        c.execute("SELECT image_path FROM alerts WHERE id=?", (sid,))
        row = c.fetchone()
        if row:
            fname = row[0]
            fpath = os.path.join("./alerts/", fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        c.execute("DELETE FROM alerts WHERE id=?", (sid,))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False})

@app.route('/alerts/<path:filename>')
def serve_alert(filename):
    return send_from_directory('./alerts', filename)

@app.route("/api/snapshots/daywise")
def api_snapshots_daywise():
    try:
        conn = sqlite3.connect("snapshots.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM snapshots ORDER BY timestamp DESC")
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        
        # Group by day
        grouped = {}
        for r in rows:
            date = r["timestamp"].split(" ")[0]
            if date not in grouped:
                grouped[date] = []
            grouped[date].append(r)
        
        return jsonify({"grouped": grouped})
    except Exception as e:
        return jsonify({"grouped": {}, "error": str(e)})
"""
    content = content.replace('@app.route("/snapshots")', new_routes + '\n@app.route("/snapshots")')

    with open('dashboard.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    upgrade_smart_system()
    upgrade_dashboard()
