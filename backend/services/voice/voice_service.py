"""VoiceService — speech recognition, wake-word detection, and TTS output.

Subscribes to:
  speak_text        → synthesise speech (blocking per call, serialised)
  stop_voice_request → shut down the voice loop

Publishes:
  voice_command     → when a recognised wake-word command is ready
"""

import logging
import threading
import time

import speech_recognition as sr

from backend.config.settings import EDGE_VOICE
from backend.core.constants import (
    EV_SPEAK_TEXT,
    EV_STOP_VOICE,
    EV_VOICE_COMMAND,
    WAKE_WORDS,
)
from backend.utils.tts import init_tts, speak_neural, speak_pyttsx3

logger = logging.getLogger(__name__)


class VoiceService:

    def __init__(self, state, bus) -> None:
        self._state = state
        self._bus = bus
        self._tts_lock = threading.Lock()
        self._recent_tts: dict = {}  # hash → timestamp for deduplication

        bus.subscribe(EV_SPEAK_TEXT, self._on_speak_text)
        bus.subscribe(EV_STOP_VOICE, self._on_stop_voice)

    # ── Public API ────────────────────────────────────────────────────────────────

    def init(self) -> None:
        """Initialise TTS backends. Call once at startup."""
        init_tts(edge_voice=EDGE_VOICE)

    def start(self) -> None:
        if not self._state.get("voice_running"):
            self._state.update(voice_running=True)
            threading.Thread(target=self._voice_loop, daemon=True).start()
            logger.info("Voice listener started")

    def stop(self) -> None:
        if self._state.get("voice_running"):
            self._state.update(voice_running=False)
            logger.info("Voice listener stopped")

    def speak(self, text: str) -> None:
        """Synthesise text to speech, blocking until playback completes."""
        if not text or not text.strip():
            return

        # Deduplicate: skip if the same text was spoken in the last 10 s
        rid = hash(text)
        now = time.monotonic()
        if rid in self._recent_tts and (now - self._recent_tts[rid] < 10.0):
            logger.debug("[TTS] Blocked duplicate: %s", text[:40])
            return
        self._recent_tts[rid] = now

        self._state.update(
            voice_state="lock",
            robot_status=f"Speaking: {text[:45]}",
            robot_speech=text,
        )
        logger.info("[Robot] %s", text)

        with self._tts_lock:
            if not speak_neural(text):
                speak_pyttsx3(text)

        self._state.update(voice_state="listen", robot_status="Waiting for wake word...")

    # ── Event handlers ────────────────────────────────────────────────────────────

    def _on_speak_text(self, text: str) -> None:
        self.speak(text)

    def _on_stop_voice(self) -> None:
        self.stop()

    # ── Voice recognition loop ────────────────────────────────────────────────────

    def _voice_loop(self) -> None:
        logger.info("Voice loop started")
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 0.4
        recognizer.non_speaking_duration = 0.3
        recognizer.dynamic_energy_threshold = False
        recognizer.energy_threshold = 300

        clarification_asked = False

        while self._state.get("voice_running") and self._state.get("running"):
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)

                    while self._state.get("voice_running") and self._state.get("running"):
                        # Do not listen while TTS is playing (prevents echo re-trigger)
                        if self._state.get("voice_state") == "lock":
                            time.sleep(0.1)
                            continue

                        self._state.update(voice_state="listen", robot_status="Listening...")

                        try:
                            audio = recognizer.listen(source, timeout=2, phrase_time_limit=8)
                        except sr.WaitTimeoutError:
                            continue

                        # Drop captured audio if TTS started while we were listening
                        if self._state.get("voice_state") == "lock":
                            continue

                        self._state.update(voice_state="process")

                        try:
                            text = recognizer.recognize_google(audio).lower().strip()
                            logger.debug("[STT] Heard: %s", text)
                            clarification_asked = False
                        except sr.UnknownValueError:
                            if not clarification_asked:
                                self.speak("I didn't quite catch that, could you repeat?")
                                clarification_asked = True
                            continue
                        except sr.RequestError as e:
                            logger.error("[STT] Google STT unavailable: %s", e)
                            continue

                        cmd = self._extract_command(text)
                        if cmd is not None:
                            self._state.update(voice_state="respond")
                            self._bus.publish(EV_VOICE_COMMAND, command=cmd)

            except Exception as e:
                logger.error("[Voice loop] Restarting mic: %s", e)
                time.sleep(2)

    def _extract_command(self, text: str) -> str | None:
        """Return the command string if a wake word is detected, else None."""
        has_wake = any(w in text for w in WAKE_WORDS)
        if not has_wake:
            return None

        cmd = text
        for w in sorted(WAKE_WORDS, key=len, reverse=True):
            cmd = cmd.replace(w, "")
        cmd = " ".join(cmd.split()).strip()

        if not cmd:
            self.speak("Yes?")
            return None

        return cmd
