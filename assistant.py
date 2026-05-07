import speech_recognition as sr
import pyttsx3
import webbrowser
import urllib.parse
import sys
import time
import re
import requests

# ── TTS Setup ──────────────────────────────────────────────────────────────────
_engine = pyttsx3.init()
_engine.setProperty("rate", 160)
_engine.setProperty("volume", 1.0)

# Pick a better voice if available
voices = _engine.getProperty("voices")
for v in voices:
    if "zira" in v.name.lower() or "female" in v.name.lower():
        _engine.setProperty("voice", v.id)
        break


def speak(text: str):
    print(f"[Robot] {text}")
    _engine.say(text)
    _engine.runAndWait()


# ── STT Helper ─────────────────────────────────────────────────────────────────
_recognizer = sr.Recognizer()
_recognizer.pause_threshold = 0.8
_recognizer.energy_threshold = 300


def listen(prompt_label: str = "Listening", timeout: int = 8) -> str:
    with sr.Microphone() as source:
        _recognizer.adjust_for_ambient_noise(source, duration=0.4)
        print(f"\n[{prompt_label}...]")
        try:
            audio = _recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            text = _recognizer.recognize_google(audio).lower().strip()
            print(f"[You said] {text}")
            return text
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            speak("Speech service is unavailable. Check your internet.")
            return ""


# ── Light simulation (replace with real API later) ─────────────────────────────
_lights_on = False


def lights_control(state: bool):
    global _lights_on
    _lights_on = state
    bar = "█" * 40
    if state:
        print(f"\n\033[93m{'─'*45}\n  💡  LIGHTS ON   {bar}\n{'─'*45}\033[0m")
        speak("Lights are now on.")
    else:
        print(f"\n\033[90m{'─'*45}\n  🌑  LIGHTS OFF  {'░'*40}\n{'─'*45}\033[0m")
        speak("Lights are now off.")


# ── YouTube ────────────────────────────────────────────────────────────────────
_YT_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def _get_first_video_id(query: str) -> str:
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
    try:
        html = requests.get(url, headers=_YT_HEADERS, timeout=6).text
        ids = re.findall(r'videoId.*?([a-zA-Z0-9_-]{11})', html)
        return ids[0] if ids else ""
    except Exception:
        return ""

def play_on_youtube(query: str):
    if not query:
        speak("What song should I play?")
        return
    speak(f"Playing {query} on YouTube.")
    video_id = _get_first_video_id(query)
    if video_id:
        webbrowser.open(f"https://www.youtube.com/watch?v={video_id}&autoplay=1")
    else:
        # Fallback to search page
        webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}")


# ── Command Parser ─────────────────────────────────────────────────────────────
FILLER_WORDS = ["please", "can you", "could you", "would you", "now", "just"]


def clean(command: str) -> str:
    """Strip filler words so 'please open youtube' → 'open youtube'."""
    for filler in FILLER_WORDS:
        command = command.replace(filler, "")
    return " ".join(command.split())  # collapse extra spaces


def handle_command(command: str) -> bool:
    """Returns False to quit, True to keep running."""

    if not command:
        speak("I didn't catch that. Try again.")
        return True

    cmd = clean(command)

    # ── YouTube / play song ──
    if "youtube" in cmd or "play" in cmd:
        song = ""
        # Try to extract what comes after "play" keyword
        for keyword in ["open youtube play", "youtube play", "play"]:
            if keyword in cmd:
                song = cmd.split(keyword, 1)[-1].strip()
                break

        if song:
            play_on_youtube(song)
        else:
            # No song mentioned — just open YouTube
            speak("Opening YouTube.")
            webbrowser.open("https://www.youtube.com")

    # ── Lights on ──
    elif any(p in cmd for p in ["lights on", "turn on lights", "turn on the lights",
                                  "switch on lights", "light on", "on the lights",
                                  "on lights"]):
        lights_control(True)

    # ── Lights off ──
    elif any(p in cmd for p in ["lights off", "turn off lights", "turn off the lights",
                                  "switch off lights", "light off", "off the lights",
                                  "off lights"]):
        lights_control(False)

    # ── What time / date ──
    elif "time" in cmd:
        from datetime import datetime
        now = datetime.now().strftime("%I:%M %p")
        speak(f"It is {now}.")

    # ── Volume ──
    elif "volume up" in cmd:
        speak("Increasing volume.")

    elif "volume down" in cmd:
        speak("Decreasing volume.")

    # ── Stop / exit ──
    elif any(p in cmd for p in ["stop", "exit", "quit", "bye", "goodbye", "shut down"]):
        speak("Goodbye! Have a great time.")
        return False

    else:
        speak(f"Sorry, I don't know how to do that yet.")

    return True


# ── Main Loop ──────────────────────────────────────────────────────────────────
# Includes common mishearings of "hey robot" by Google STT
WAKE_WORDS = [
    "hey robot", "hey, robot", "hi robot", "okay robot", "ok robot",
    "a robot", "the robot", "hey robots", "hi robots", "robot",
    "hey robert", "a robert", "hey rob"
]


def run():
    speak("Hey Robot is ready. Say 'Hey Robot' to wake me up.")

    while True:
        raw = listen(prompt_label="Waiting for wake word", timeout=15)

        if not raw:
            continue

        # Check if wake word is present
        triggered = any(w in raw for w in WAKE_WORDS)

        if not triggered:
            print(f'  [Hint] I heard "{raw}" — say "Hey Robot" first to wake me!')
            continue

        # Extract inline command BEFORE speaking "Yes?" so we don't lose it
        inline_command = raw
        for w in sorted(WAKE_WORDS, key=len, reverse=True):  # longest first
            inline_command = inline_command.replace(w, "")
        inline_command = " ".join(inline_command.split()).strip()

        # Acknowledge
        speak("Yes?")

        if inline_command:
            command = inline_command
        else:
            # Wait for the command separately
            command = listen(prompt_label="Command", timeout=8)

        keep_running = handle_command(command)
        if not keep_running:
            break


if __name__ == "__main__":
    try:
        from camera_monitor import CameraMonitor
        cam = CameraMonitor(on_alert=speak, sensitivity=1500)
        cam.start()
    except Exception as e:
        print(f"[Camera] Could not start monitor: {e}")
        cam = None

    try:
        run()
    except KeyboardInterrupt:
        print("\n[Stopped by user]")
        if cam:
            cam.stop()
        sys.exit(0)
