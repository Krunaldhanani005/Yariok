"""Text-to-speech utilities.

Pure I/O — no state management, no event bus. Callers are responsible
for any locking or deduplication they need.
"""

import logging
import os
import tempfile
import threading
import time

logger = logging.getLogger(__name__)

_EDGE_VOICE: str = "en-US-AriaNeural"
_neural_ok: bool = False
_pygame = None
_engine = None  # pyttsx3 fallback


def init_tts(edge_voice: str = "en-US-AriaNeural") -> None:
    """Initialise both TTS backends. Call once at startup."""
    global _EDGE_VOICE, _neural_ok, _pygame, _engine
    _EDGE_VOICE = edge_voice

    # pyttsx3 fallback (offline, lower quality)
    try:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 165)
        _engine.setProperty("volume", 1.0)
        for v in _engine.getProperty("voices"):
            if "zira" in v.name.lower():
                _engine.setProperty("voice", v.id)
                break
        logger.info("TTS: pyttsx3 fallback ready")
    except Exception as e:
        logger.warning("TTS: pyttsx3 unavailable — %s", e)

    # Neural TTS via edge-tts + pygame
    try:
        import pygame as pg
        pg.mixer.pre_init(44100, -16, 2, 512)
        pg.mixer.init()
        pg.mixer.music.set_volume(1.0)
        _pygame = pg
        _neural_ok = True
        logger.info("TTS: neural (edge-tts + pygame) ready")
    except Exception as e:
        logger.warning("TTS: neural backend unavailable — %s", e)


def speak_neural(text: str) -> bool:
    """Synthesise text with Edge neural TTS and play via pygame. Blocking."""
    if not _neural_ok or _pygame is None:
        return False
    try:
        import asyncio
        import edge_tts

        tmp = tempfile.mktemp(suffix=".mp3")

        def _generate() -> None:
            async def _run() -> None:
                tts = edge_tts.Communicate(text, voice=_EDGE_VOICE, rate="+8%", volume="+0%")
                await tts.save(tmp)
            asyncio.run(_run())

        gen = threading.Thread(target=_generate, daemon=True)
        gen.start()
        gen.join(timeout=12)

        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            return False

        _pygame.mixer.music.load(tmp)
        _pygame.mixer.music.play()
        while _pygame.mixer.music.get_busy():
            time.sleep(0.05)

        try:
            os.unlink(tmp)
        except Exception:
            pass
        return True
    except Exception as e:
        logger.error("Neural TTS error: %s", e)
        return False


def speak_pyttsx3(text: str) -> bool:
    """Speak text using pyttsx3 offline engine. Blocking."""
    if _engine is None:
        return False
    try:
        _engine.say(text)
        _engine.runAndWait()
        return True
    except Exception as e:
        logger.error("pyttsx3 TTS error: %s", e)
        return False


def play_alert_sound() -> None:
    """Play a short alert beep via pygame or terminal bell."""
    if _neural_ok and _pygame:
        try:
            import numpy as np
            frames = 44100 // 4
            arr = (4096 * np.sin(2.0 * np.pi * 1000 * np.arange(frames) / 44100)).astype(np.int16)
            arr = np.column_stack((arr, arr))
            snd = _pygame.sndarray.make_sound(arr)
            snd.play()
            return
        except Exception:
            pass
    print("\a", end="", flush=True)
