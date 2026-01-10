"""
Ramblebot Configuration

Central configuration file for all Ramblebot modules.
Settings can be overridden via environment variables.

Environment Variables:
    RAMBLEBOT_GATEWAY_IP    - Android phone IP (default: 192.168.86.35)
    RAMBLEBOT_GATEWAY_PORT  - Gateway port (default: 8765)
    RAMBLEBOT_OLLAMA_IP     - Ollama server IP (default: 192.168.86.250)
    RAMBLEBOT_OLLAMA_PORT   - Ollama port (default: 11434)
    RAMBLEBOT_MODEL         - LLM model name (default: devstral-small-2:latest)
    RAMBLEBOT_USE_DEPTH     - Enable MiDaS depth (default: true)
    RAMBLEBOT_USE_SPEECH    - Enable Finnish TTS (default: true)
    RAMBLEBOT_SPEECH_RATE   - How often robot talks 0-1 (default: 0.6)

Usage:
    from config import Config

    # Access settings
    print(Config.GATEWAY_BASE)
    print(Config.OLLAMA_BASE)

    # Or use environment variables before importing:
    # export RAMBLEBOT_GATEWAY_IP=192.168.1.100
    # export RAMBLEBOT_OLLAMA_IP=localhost
"""

import os


def _get_env(key: str, default: str) -> str:
    """Get environment variable with prefix."""
    return os.environ.get(f"RAMBLEBOT_{key}", default)


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable."""
    val = os.environ.get(f"RAMBLEBOT_{key}", str(default)).lower()
    return val in ("true", "1", "yes", "on")


def _get_env_float(key: str, default: float) -> float:
    """Get float environment variable."""
    try:
        return float(os.environ.get(f"RAMBLEBOT_{key}", str(default)))
    except ValueError:
        return default


class Config:
    """Central configuration for Ramblebot."""

    # ========================
    # Network Configuration
    # ========================

    # Android Phone Gateway (camera, sensors, robot control, TTS)
    GATEWAY_IP = _get_env("GATEWAY_IP", "192.168.101.102")
    GATEWAY_PORT = int(_get_env("GATEWAY_PORT", "8765"))
    GATEWAY_BASE = f"http://{GATEWAY_IP}:{GATEWAY_PORT}"

    # Ollama LLM Server
    OLLAMA_IP = _get_env("OLLAMA_IP", "192.168.101.100")
    OLLAMA_PORT = int(_get_env("OLLAMA_PORT", "11434"))
    OLLAMA_BASE = f"http://{OLLAMA_IP}:{OLLAMA_PORT}"

    # ========================
    # LLM Configuration
    # ========================

    # Model for planning and speech generation
    MODEL = _get_env("MODEL", "devstral-small-2:latest")

    # Request timeout (seconds)
    TIMEOUT = int(_get_env("TIMEOUT", "30"))

    # ========================
    # Feature Toggles
    # ========================

    # MiDaS depth estimation (slower without GPU)
    USE_MIDAS_DEPTH = _get_env_bool("USE_DEPTH", True)

    # Finnish speech commentary
    USE_SPEECH = _get_env_bool("USE_SPEECH", True)

    # How often robot comments (0-1)
    SPEECH_CHATTINESS = _get_env_float("SPEECH_RATE", 0.6)

    # ========================
    # Robot Personality
    # ========================

    # Personality type: "pet", "doctor", "assistant", "explorer"
    PERSONALITY = _get_env("PERSONALITY", "doctor")

    # Robot name
    ROBOT_NAME = _get_env("ROBOT_NAME", "Tohtori Ramble")

    # Voice: "male" or "female"
    VOICE_GENDER = _get_env("VOICE_GENDER", "male")

    # ========================
    # Robot Safety Limits
    # ========================

    MAX_SPEED = int(_get_env("MAX_SPEED", "170"))
    MAX_MS = int(_get_env("MAX_MS", "1200"))
    MAX_STEPS = int(_get_env("MAX_STEPS", "100"))

    # ========================
    # Vision Configuration
    # ========================

    # YOLO person detection confidence threshold
    PERSON_CONF_MIN = _get_env_float("PERSON_CONF", 0.35)

    # Doorway detection thresholds
    OPENING_MIN_SCORE = _get_env_float("OPENING_MIN", 0.55)
    OPENING_APPROACH_SCORE = _get_env_float("OPENING_APPROACH", 0.70)

    # ========================
    # Derived URLs
    # ========================

    @classmethod
    def stream_url(cls) -> str:
        return f"{cls.GATEWAY_BASE}/stream.mjpeg"

    @classmethod
    def sensors_url(cls) -> str:
        return f"{cls.GATEWAY_BASE}/sensors.json"

    @classmethod
    def arcore_url(cls) -> str:
        return f"{cls.GATEWAY_BASE}/arcore.json"

    @classmethod
    def speak_url(cls) -> str:
        return f"{cls.GATEWAY_BASE}/speak"

    @classmethod
    def cmd_url(cls) -> str:
        return f"{cls.GATEWAY_BASE}/cmd"

    # ========================
    # Info
    # ========================

    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "="*50)
        print("  RAMBLEBOT CONFIGURATION")
        print("="*50)
        print(f"\nNetwork:")
        print(f"  Gateway:  {cls.GATEWAY_BASE}")
        print(f"  Ollama:   {cls.OLLAMA_BASE}")
        print(f"  Model:    {cls.MODEL}")
        print(f"\nFeatures:")
        print(f"  MiDaS Depth:  {'Enabled' if cls.USE_MIDAS_DEPTH else 'Disabled'}")
        print(f"  Finnish TTS:  {'Enabled' if cls.USE_SPEECH else 'Disabled'}")
        print(f"  Chattiness:   {cls.SPEECH_CHATTINESS}")
        print(f"\nSafety:")
        print(f"  Max Speed:    {cls.MAX_SPEED}")
        print(f"  Max Duration: {cls.MAX_MS}ms")
        print(f"  Max Steps:    {cls.MAX_STEPS}")
        print()

    @classmethod
    def to_dict(cls) -> dict:
        """Export config as dictionary."""
        return {
            "gateway_base": cls.GATEWAY_BASE,
            "ollama_base": cls.OLLAMA_BASE,
            "model": cls.MODEL,
            "use_depth": cls.USE_MIDAS_DEPTH,
            "use_speech": cls.USE_SPEECH,
            "speech_chattiness": cls.SPEECH_CHATTINESS,
            "max_speed": cls.MAX_SPEED,
            "max_ms": cls.MAX_MS,
            "max_steps": cls.MAX_STEPS,
        }


# Convenience exports
GATEWAY_BASE = Config.GATEWAY_BASE
OLLAMA_BASE = Config.OLLAMA_BASE
MODEL = Config.MODEL


if __name__ == "__main__":
    Config.print_config()

    print("Environment variable examples:")
    print("  export RAMBLEBOT_GATEWAY_IP=192.168.1.100")
    print("  export RAMBLEBOT_OLLAMA_IP=localhost")
    print("  export RAMBLEBOT_MODEL=llama3:8b")
    print("  export RAMBLEBOT_USE_DEPTH=false")
    print("  export RAMBLEBOT_USE_SPEECH=true")
