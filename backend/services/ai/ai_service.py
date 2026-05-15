"""AIService — all LLM calls live here.

Subscribes to:
  visitor_detected  → generate greeting → publish speak_text
  security_alert    → generate analysis → publish speak_text
  ai_query          → answer question   → publish speak_text

Never imports VisionService, VoiceService, or AutomationService.
Cross-service effects go only through the EventBus.
"""

import base64
import json
import logging
import threading
import webbrowser
import urllib.parse
import re
import datetime

import requests

from backend.config.settings import OPENROUTER_API_KEY, OR_URL, OR_MODEL, LOG_FILE
from backend.core.constants import (
    EV_VISITOR_DETECTED, EV_SECURITY_ALERT, EV_AI_QUERY, EV_SPEAK_TEXT,
    MODE_SECURITY, MODE_KARAOKE,
)

logger = logging.getLogger(__name__)

# LLM tool definitions exposed to Claude
_LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "take_snapshot",
            "description": "Take a photo snapshot from the reception camera.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_nantatech",
            "description": "Open the Nanta Tech Limited company website in the browser.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_youtube",
            "description": "Open YouTube and play a song or video by search query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term for YouTube"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "camera_status",
            "description": "Get live camera and detection status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_mode",
            "description": "Change operating mode: normal, security, or karaoke.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["normal", "security", "karaoke"],
                    }
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_camera",
            "description": "Show or hide the live camera window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "visible": {"type": "boolean"}
                },
                "required": ["visible"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visitor_log",
            "description": "Get a summary of today's visitor log.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class AIService:

    def __init__(self, state, bus) -> None:
        self._state = state
        self._bus = bus
        self._ready = False
        self._conv_history: list = []

        bus.subscribe(EV_VISITOR_DETECTED, self._on_visitor_detected)
        bus.subscribe(EV_SECURITY_ALERT, self._on_security_alert)
        bus.subscribe(EV_AI_QUERY, self._on_ai_query)

    # ── Initialisation ──────────────────────────────────────────────────────────

    def init(self) -> None:
        """Test OpenRouter connectivity. Call once at startup."""
        if not OPENROUTER_API_KEY:
            logger.warning("[LLM] No OPENROUTER_API_KEY — add it to .env")
            return
        try:
            data = self._or_call(
                [{"role": "user", "content": "Say OK"}],
                use_tools=False,
                max_tokens=5,
            )
            _ = data["choices"][0]["message"]["content"]
            self._ready = True
            logger.info("[LLM] Claude Sonnet ready via OpenRouter")
        except Exception as e:
            logger.error("[LLM] OpenRouter init failed: %s", e)

    # ── Event handlers ──────────────────────────────────────────────────────────

    def _on_visitor_detected(
        self,
        frame,
        visitor_num: int,
        mode: str,
        in_zone: bool = True,
    ) -> None:
        # Phase C entry-zone gate: visitors outside the configured polygon
        # still get a snapshot + DB row + counter bump (handled upstream in
        # MqttStateBridge), but the VOICE stays silent. Default-allow when
        # the publisher didn't tag the event so we never silently regress
        # callers that haven't been updated.
        if not in_zone:
            # Bumped from DEBUG → INFO so this is visible in production
            # logs when the user is wondering "why didn't the system
            # greet that person?". The answer: their feet weren't inside
            # the configured entry polygon.
            logger.info(
                "Visitor #%s outside entry zone — voice greeting silenced "
                "(snapshot + counter still recorded).",
                visitor_num,
            )
            return

        greeting = self.llm_greet_vision(visitor_num, frame)
        self._bus.publish(EV_SPEAK_TEXT, text=greeting)
        if mode == MODE_KARAOKE:
            threading.Thread(
                target=self._play_youtube,
                args=("party welcome music upbeat",),
                daemon=True,
            ).start()

    def _on_security_alert(self, frame, num_people: int) -> None:
        analysis = self.llm_security_analysis(frame)
        parts = ["Security alert! Unauthorized person detected."]
        if analysis:
            parts.append(analysis)
            self._state.update(ai_alert=analysis[:80])
        self._bus.publish(EV_SPEAK_TEXT, text=" ".join(parts))

    def _on_ai_query(self, query: str) -> None:
        reply = self.chat(query)
        self._bus.publish(EV_SPEAK_TEXT, text=reply)

    # ── Public LLM helpers ──────────────────────────────────────────────────────

    def llm_greet_vision(self, visitor_num: int, frame) -> str:
        """Vision-aware greeting: Claude sees the camera frame."""
        if not self._ready or frame is None:
            return self._llm_greet_text(visitor_num)
        mode = self._current_mode()
        try:
            data = self._or_call(
                self._vision_messages(
                    frame,
                    system_text=(
                        "You are Yaariok, a friendly AI reception robot with a live camera. "
                        "Look at the visitor and generate ONE warm, personalized welcome. "
                        "Mention something you can see — clothing color, if they are smiling, etc. "
                        "No markdown. Max 2 short sentences. Spoken aloud via TTS."
                    ),
                    user_text=(
                        f"Visitor number {visitor_num} just walked in. Mode: {mode}. "
                        "Welcome them warmly based on what you see in the camera!"
                    ),
                ),
                use_tools=False,
                max_tokens=100,
            )
            greeting = (data["choices"][0]["message"].get("content") or "").strip()
            logger.info("[VISION] Greeting: %s", greeting)
            return greeting or self._llm_greet_text(visitor_num)
        except Exception as e:
            logger.warning("[VISION] %s", e)
            return self._llm_greet_text(visitor_num)

    def llm_security_analysis(self, frame) -> str:
        """Claude analyses the security camera frame and returns a threat description."""
        if not self._ready or frame is None:
            return ""
        try:
            data = self._or_call(
                self._vision_messages(
                    frame,
                    system_text=(
                        "You are a security AI monitoring a reception camera. "
                        "Describe who you see: appearance, behavior, concern level. "
                        "Be direct and brief. 1-2 sentences. No markdown. Spoken aloud via TTS."
                    ),
                    user_text="Security alert triggered. Analyse this camera frame.",
                ),
                use_tools=False,
                max_tokens=80,
            )
            result = (data["choices"][0]["message"].get("content") or "").strip()
            logger.info("[SECURITY-AI] %s", result)
            return result
        except Exception as e:
            logger.warning("[SECURITY-AI] %s", e)
            return ""

    def chat(self, user_msg: str) -> str:
        """Full conversational turn with tool calling and history."""
        if not self._ready:
            return "My AI brain is not connected yet. Please check the OpenRouter key in dot env."

        self._conv_history.append({"role": "user", "content": user_msg})
        if len(self._conv_history) > 14:
            self._conv_history = self._conv_history[-14:]

        messages = [{"role": "system", "content": self._sys_prompt()}] + self._conv_history

        try:
            data = self._or_call(messages, use_tools=True, max_tokens=512)
            choice = data["choices"][0]
            msg = choice["message"]
            content = (msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                tool_results = []
                assistant_msg = {"role": "assistant", "content": content, "tool_calls": tool_calls}
                for tc in tool_calls:
                    fn = tc["function"]
                    args = json.loads(fn.get("arguments") or "{}")
                    result = self._exec_tool(fn["name"], args)
                    logger.info("[TOOL] %s(%s) → %s", fn["name"], args, result)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                data2 = self._or_call(
                    messages + [assistant_msg] + tool_results,
                    use_tools=False,
                    max_tokens=150,
                )
                reply = (data2["choices"][0]["message"].get("content") or "").strip()
            else:
                reply = content

            reply = reply or "I'm here! Go ahead."
            self._conv_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            logger.error("[LLM] chat error: %s", e)
            return "I had trouble thinking just now — ask me again."

    def ask(self, user_text: str) -> str:
        """Simple one-shot question, no history, no tools."""
        if not self._ready:
            return user_text
        try:
            resp = self._or_call(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are YaariOK, a friendly AI receptionist. "
                            "Be warm and brief — under 2 sentences. "
                            "No markdown, no bullet points — output is spoken aloud via TTS."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ],
                use_tools=False,
                max_tokens=120,
            )
            return (resp["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            logger.error("[LLM] ask error: %s", e)
            return "I had trouble connecting. Please try again."

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _llm_greet_text(self, visitor_num: int) -> str:
        """Text-only greeting when vision is unavailable."""
        if not self._ready:
            return f"Welcome to Yaariok! You are visitor number {visitor_num} today."
        mode = self._current_mode()
        try:
            data = self._or_call(
                messages=[
                    {"role": "system", "content": (
                        "You are Yaariok, a friendly AI reception robot. "
                        "Generate one short, warm, unique welcome greeting. "
                        "No markdown. Max 1 sentence. Spoken aloud via TTS."
                    )},
                    {"role": "user", "content": (
                        f"Visitor number {visitor_num} just walked in. Mode: {mode}. Greet them!"
                    )},
                ],
                use_tools=False,
                max_tokens=80,
            )
            return (data["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            return f"Welcome to Yaariok! You are visitor number {visitor_num} today."

    def _exec_tool(self, name: str, args: dict) -> str:
        if name == "take_snapshot":
            self._state.update(take_snapshot=True)
            return "Snapshot taken and saved."

        if name == "open_nantatech":
            webbrowser.open("https://www.nantatech.com")
            return "Opened Nanta Tech Limited website."

        if name == "play_youtube":
            query = args.get("query", "music")
            threading.Thread(target=self._play_youtube, args=(query,), daemon=True).start()
            return f"Opening YouTube for: {query}"

        if name == "camera_status":
            n = self._state.get("people_now")
            v = self._state.get("visitors_today")
            a = self._state.get("alerts_today")
            sn = self._state.get("snapshots_taken")
            cam = self._state.get("camera_status")
            mode = self._current_mode()
            return (
                f"Camera: {cam}. People in reception: {n}. "
                f"Visitors today: {v}. Alerts: {a}. Snapshots: {sn}. Mode: {mode}."
            )

        if name == "set_mode":
            mode = args.get("mode", "normal")
            if mode == "security":
                self._state.update(security_mode=True, karaoke_mode=False)
                return "Security mode activated."
            elif mode == "karaoke":
                self._state.update(karaoke_mode=True, security_mode=False)
                return "Karaoke mode activated."
            else:
                self._state.update(security_mode=False, karaoke_mode=False)
                return "Normal mode activated."

        if name == "show_camera":
            visible = args.get("visible", True)
            self._state.update(show_camera=visible)
            return "Camera window opened." if visible else "Camera window closed."

        if name == "visitor_log":
            try:
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                lines = [l for l in open(LOG_FILE) if today in l]
                v = self._state.get("visitors_today")
                a = self._state.get("alerts_today")
                return f"Today's log: {len(lines)} entries. {v} visitors, {a} alerts."
            except Exception:
                return "No log entries yet today."

        return f"Unknown tool: {name}"

    def _or_call(self, messages: list, use_tools: bool = True, max_tokens: int = 256) -> dict:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Yaariok Smart System",
        }
        body: dict = {
            "model": OR_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        if use_tools:
            body["tools"] = _LLM_TOOLS
            body["tool_choice"] = "auto"
        resp = requests.post(OR_URL, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _frame_to_b64(self, frame, width: int = 480) -> str:
        import cv2
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (width, int(h * width / w)))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _vision_messages(self, frame, system_text: str, user_text: str) -> list:
        b64 = self._frame_to_b64(frame)
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": user_text},
            ]},
        ]

    def _sys_prompt(self) -> str:
        n = self._state.get("people_now")
        mode = self._current_mode()
        return (
            "You are Yaariok, a smart and friendly AI reception robot at the Yaariok office. "
            "Speak naturally — warm, a little witty, conversational. "
            "Keep every spoken reply under 2 short sentences. "
            "No bullet points, no markdown — output is read aloud via text-to-speech. "
            "When the user asks you to DO something, call the appropriate tool. "
            f"Live context: {n} {'person' if n == 1 else 'people'} in reception, "
            f"mode is {mode}, {self._state.get('visitors_today')} visitors today."
        )

    def _current_mode(self) -> str:
        if self._state.get("security_mode"):
            return MODE_SECURITY
        if self._state.get("karaoke_mode"):
            return MODE_KARAOKE
        return "normal"

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
                return
        except Exception:
            pass
        webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}")
