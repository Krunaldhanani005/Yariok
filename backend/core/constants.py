# System mode identifiers
MODE_NORMAL = "normal"
MODE_SECURITY = "security"
MODE_KARAOKE = "karaoke"

# Event bus event names
EV_VISITOR_DETECTED = "visitor_detected"
EV_VISITOR_LEFT = "visitor_left"
EV_SECURITY_ALERT = "security_alert"
EV_VOICE_COMMAND = "voice_command"
EV_AI_QUERY = "ai_query"
EV_SPEAK_TEXT = "speak_text"
EV_MODE_CHANGED = "mode_changed"
EV_SNAPSHOT_TAKEN = "snapshot_taken"
EV_SYSTEM_STARTED = "system_started"
EV_SYSTEM_STOPPED = "system_stopped"
EV_CAMERA_STATUS = "camera_status_changed"
EV_STOP_VOICE = "stop_voice_request"

# Wake word variants (STT mis-hearings included)
WAKE_WORDS = [
    "hey robot", "hi robot", "okay robot", "ok robot",
    "a robot", "the robot", "hey robots", "robot",
    "hey robert", "hey rob",
    "hey ravan", "hey rabot", "hey rubot", "hey robo",
    "hey ribbon", "hey robin", "air robot",
    "ravan", "rubot", "rabot",
]

# Filler words stripped from commands before pattern matching
COMMAND_FILLERS = [
    "please", "can you", "could you", "would you", "just",
    "some music", "a song", "for me",
]
