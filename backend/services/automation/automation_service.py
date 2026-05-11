"""AutomationService — unified command handler.

Subscribes to voice_command events and executes them locally.
Unrecognised commands are forwarded as ai_query events for the AI service.

Also exposes set_mode / start_system / stop_system / take_snapshot as
public methods so that the dashboard routes can call them directly via
the service registry.
"""

import datetime
import logging
import re
import threading
import time
import urllib.parse
import webbrowser

import requests

from backend.config.settings import LOG_FILE
from backend.core.constants import (
    COMMAND_FILLERS,
    EV_AI_QUERY,
    EV_SPEAK_TEXT,
    EV_STOP_VOICE,
    EV_SYSTEM_STARTED,
    EV_SYSTEM_STOPPED,
    EV_VOICE_COMMAND,
    MODE_KARAOKE,
    MODE_NORMAL,
    MODE_SECURITY,
)

logger = logging.getLogger(__name__)


class AutomationService:

    def __init__(self, state, bus, db, registry) -> None:
        self._state = state
        self._bus = bus
        self._db = db
        self._registry = registry  # used for lazy VisionService lookup (calibration)

        bus.subscribe(EV_VOICE_COMMAND, self._on_voice_command)

    # ── Event handler ────────────────────────────────────────────────────────────

    def _on_voice_command(self, command: str) -> None:
        self._state.update(last_command=command[:48])
        self._db.save_log("voice", f"Command: {command}")
        if not self._dispatch(command):
            self._bus.publish(EV_AI_QUERY, query=command)

    # ── Command dispatch ─────────────────────────────────────────────────────────

    def _dispatch(self, raw: str) -> bool:
        """Pattern-match the command. Returns True if handled locally."""
        c = self._clean(raw)

        # System control
        if "start system" in c:
            self.start_system()
            self._bus.publish(EV_SPEAK_TEXT, text="System started and camera is now active.")

        elif "stop system" in c:
            self.stop_system()
            self._bus.publish(EV_SPEAK_TEXT, text="System stopped.")

        # Mode switching
        elif any(p in c for p in [
            "security mode", "start security", "enable security",
            "activate security", "security on", "lock down", "lockdown",
        ]):
            self.set_mode(MODE_SECURITY)

        elif any(p in c for p in [
            "karaoke mode", "start karaoke", "enable karaoke",
            "karaoke on", "party mode", "start party",
        ]):
            self.set_mode(MODE_KARAOKE)

        elif any(p in c for p in [
            "normal mode", "reset mode", "back to normal", "disable security",
            "security off", "deactivate security", "disable karaoke",
            "stop karaoke", "karaoke off", "default mode",
        ]):
            self.set_mode(MODE_NORMAL)

        # Camera
        elif any(p in c for p in ["take snapshot", "snapshot", "take photo", "take picture", "capture", "click photo"]):
            self.take_snapshot()

        elif any(p in c for p in ["show camera", "open camera", "camera on", "show feed"]):
            self._state.update(show_camera=True)
            self._bus.publish(EV_SPEAK_TEXT, text="Opening live camera feed. Press Q on the window to close.")

        elif any(p in c for p in ["hide camera", "close camera", "camera off"]):
            self._state.update(show_camera=False)
            self._bus.publish(EV_SPEAK_TEXT, text="Camera window closed.")

        # Calibration — delegates to VisionService via registry
        elif "calibrate" in c or "run calibration" in c:
            vision = self._registry.get("vision")
            if vision:
                vision.run_calibration()
            self._bus.publish(EV_SPEAK_TEXT, text="Initializing vision calibration sequence.")

        # Snapshot cleanup
        elif any(p in c for p in ["delete today snapshots", "delete snapshots", "clear snapshots", "clear today snapshots"]):
            n = self._db.delete_today_snapshots()
            self._state.update(snapshots_taken=0)
            self._db.log_event(f"Deleted {n} today snapshots via voice", "activity")
            self._bus.publish(EV_SPEAK_TEXT, text=f"Deleted {n} snapshots from today.")

        # Dashboard
        elif any(p in c for p in ["open dashboard", "show dashboard", "dashboard"]):
            webbrowser.open("http://localhost:5000")
            self._db.log_event("Dashboard opened via voice", "activity")
            self._bus.publish(EV_SPEAK_TEXT, text="Opening the dashboard now.")

        # YouTube / music
        elif "play " in c:
            song = c.split("play ", 1)[-1].strip()
            if song:
                threading.Thread(target=self._play_youtube, args=(song,), daemon=True).start()
            else:
                self._bus.publish(EV_SPEAK_TEXT, text="What song should I play?")

        elif "youtube" in c:
            webbrowser.open("https://www.youtube.com")
            self._bus.publish(EV_SPEAK_TEXT, text="Opening YouTube.")

        # Search
        elif c.startswith("search "):
            query = c.replace("search", "", 1).strip()
            if query:
                webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(query)}")
                self._db.log_event(f"Google search: {query}", "activity")
                self._bus.publish(EV_SPEAK_TEXT, text=f"Searching Google for {query}.")
            else:
                return False

        # Voice control
        elif "voice off" in c:
            self._bus.publish(EV_SPEAK_TEXT, text="Turning off voice listener.")
            self._bus.publish(EV_STOP_VOICE)

        # Queries answered locally (no LLM needed)
        elif any(p in c for p in ["how many people", "who is in", "anyone in", "count people", "anyone there"]):
            n = self._state.get("people_now")
            v = self._state.get("visitors_today")
            word = "person" if n == 1 else "people"
            self._bus.publish(
                EV_SPEAK_TEXT,
                text=f"I can see {n} {word} in reception right now. Total visitors today: {v}.",
            )

        elif "time" in c:
            now = datetime.datetime.now().strftime("%I:%M %p")
            self._bus.publish(EV_SPEAK_TEXT, text=f"It is {now}.")

        elif any(p in c for p in ["status", "report", "system status", "what is happening", "give update"]):
            self._speak_status()

        elif any(p in c for p in ["show log", "visitor log", "read log", "today visitors"]):
            self._speak_log()

        elif any(p in c for p in ["nantatech", "nanta tech", "open nanta", "nanta website"]):
            webbrowser.open("https://www.nantatech.com")
            self._db.log_event("Opened nantatech.com", "activity")
            self._bus.publish(EV_SPEAK_TEXT, text="Opening Nanta Tech Limited website.")

        elif any(p in c for p in ["help", "what can you do", "commands"]):
            self._bus.publish(
                EV_SPEAK_TEXT,
                text=(
                    "I can play music on YouTube, control the camera, count people, "
                    "take snapshots, enable security or karaoke mode, give a status "
                    "report, and answer your questions."
                ),
            )

        elif any(p in c for p in ["stop", "exit", "quit", "bye", "goodbye", "shut down", "shutdown"]):
            self._bus.publish(EV_SPEAK_TEXT, text="Goodbye! Shutting down Yaariok.")
            self._state.update(running=False, show_camera=False)

        else:
            return False  # Forward to AI

        return True

    # ── Public service methods (called by dashboard routes) ──────────────────────

    def set_mode(self, mode: str) -> None:
        if mode == MODE_SECURITY:
            self._state.update(security_mode=True, karaoke_mode=False)
            self._notify("Security Mode ON", "Every visitor will trigger an alert")
            self._beep(2)
            self._db.log_event("Security mode ON", "activity")
            self._bus.publish(EV_SPEAK_TEXT, text="Security mode activated. Monitoring all visitors.")
        elif mode == MODE_KARAOKE:
            self._state.update(karaoke_mode=True, security_mode=False)
            self._notify("Karaoke Mode ON", "Visitors welcomed with music!")
            self._beep(1)
            self._db.log_event("Karaoke mode ON", "activity")
            self._bus.publish(EV_SPEAK_TEXT, text="Karaoke mode activated. Get ready to party!")
        else:
            self._state.update(security_mode=False, karaoke_mode=False)
            self._notify("Normal Mode", "Back to normal")
            self._db.log_event("Normal mode activated", "activity")
            self._bus.publish(EV_SPEAK_TEXT, text="Normal mode activated.")

    def start_system(self) -> None:
        if not self._state.get("system_running"):
            self._state.update(system_running=True)
            self._db.log_event("System started", "system")
            self._bus.publish(EV_SYSTEM_STARTED)

    def stop_system(self) -> None:
        if self._state.get("system_running"):
            self._state.update(system_running=False, show_camera=False)
            self._db.log_event("System stopped", "system")
            self._bus.publish(EV_SYSTEM_STOPPED)

    def take_snapshot(self) -> None:
        self._state.update(take_snapshot=True)
        self._notify("Snapshot", "Photo saved")
        self._bus.publish(EV_SPEAK_TEXT, text="Snapshot taken.")

    def reset_stats(self) -> None:
        self._state.update(visitors_today=0, alerts_today=0, snapshots_taken=0)
        try:
            open(LOG_FILE, "w").close()
        except Exception:
            pass

    # ── Private helpers ──────────────────────────────────────────────────────────

    def _clean(self, cmd: str) -> str:
        for f in COMMAND_FILLERS:
            cmd = re.sub(r"\b" + re.escape(f) + r"\b", " ", cmd)
        return " ".join(cmd.split())

    def _speak_status(self) -> None:
        n = self._state.get("people_now")
        v = self._state.get("visitors_today")
        a = self._state.get("alerts_today")
        sn = self._state.get("snapshots_taken")
        cam = self._state.get("camera_status")
        mode = (
            MODE_SECURITY if self._state.get("security_mode")
            else MODE_KARAOKE if self._state.get("karaoke_mode")
            else MODE_NORMAL
        )
        self._bus.publish(
            EV_SPEAK_TEXT,
            text=(
                f"System status. Camera is {cam}. "
                f"Currently {n} {'person' if n == 1 else 'people'} in reception. "
                f"{v} visitors today, {a} alerts, {sn} snapshots. "
                f"Mode is {mode}."
            ),
        )

    def _speak_log(self) -> None:
        try:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            lines = [l for l in open(LOG_FILE) if today in l]
            v = self._state.get("visitors_today")
            a = self._state.get("alerts_today")
            self._bus.publish(
                EV_SPEAK_TEXT,
                text=f"Today's log has {len(lines)} entries. {v} visitors, {a} alerts.",
            )
        except Exception:
            self._bus.publish(EV_SPEAK_TEXT, text="No log file found yet.")

    def _play_youtube(self, query: str) -> None:
        try:
            html = requests.get(
                f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=6,
            ).text
            ids = re.findall(r"watch\?v=([a-zA-Z0-9_-]{11})", html)
            if ids:
                webbrowser.open(f"https://www.youtube.com/watch?v={ids[0]}&autoplay=1")
                self._db.log_event(f"YouTube: {query}", "activity")
                return
        except Exception:
            pass
        webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}")

    def _notify(self, title: str, msg: str) -> None:
        def _n() -> None:
            try:
                from plyer import notification
                notification.notify(title=title, message=msg, app_name="Yaariok Smart System", timeout=4)
            except Exception:
                pass
        threading.Thread(target=_n, daemon=True).start()

    def _beep(self, times: int = 1) -> None:
        def _b() -> None:
            for _ in range(times):
                print("\a", end="", flush=True)
                time.sleep(0.6)
        threading.Thread(target=_b, daemon=True).start()
