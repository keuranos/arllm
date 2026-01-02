import json
import time
import requests

OLLAMA_BASE = "http://192.168.86.250:11434"
MODEL = "devstral-small-2:latest"  # löytyy /api/tags listasta

# Puhelimen RamblebotGateway
GATEWAY_BASE = "http://192.168.86.35:8765"

SESSION = requests.Session()
TIMEOUT = 30

# Turvarajat
MAX_SPEED = 180
MAX_MS = 1200


SYSTEM = """You control a telepresence robot.

Return ONLY a single JSON object, no extra text.

Allowed actions:
1) {"action":"drive","l":-255..255,"r":-255..255,"ms":50..4000}
2) {"action":"stop"}
3) {"action":"head","pos":0..180,"speed":0..9}

Drive convention (IMPORTANT):
- Positive speed = forward, negative = backward.
- Turning RIGHT in place: left wheel forward, right wheel backward.
  Example: {"action":"drive","l":160,"r":-160,"ms":400}
- Turning LEFT in place: left wheel backward, right wheel forward.
  Example: {"action":"drive","l":-160,"r":160,"ms":400}
- Forward: l and r both positive. Backward: l and r both negative.

Rules:
- Prefer short movements (200-800ms) then stop.
- If unsure: stop.
- Never output anything except JSON.
"""


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
        s = s.replace("json", "", 1).strip()
    return s.strip()


def _is_spin(cmd: dict) -> bool:
    """
    Spin in place = wheels opposite directions (l>0,r<0) or (l<0,r>0).
    Lisäksi tarkistetaan että magnitudit on suunnilleen samat.
    """
    try:
        l = int(cmd.get("l", 0))
        r = int(cmd.get("r", 0))
        if l == 0 or r == 0:
            return False
        opposite = (l > 0 > r) or (l < 0 < r)
        if not opposite:
            return False
        return abs(abs(l) - abs(r)) <= 5
    except Exception:
        return False


def _fix_left_right(user_text: str, cmd: dict) -> dict:
    """
    Jos käyttäjä pyytää 'right/oikealle' ja LLM tuottaa selkeästi vasemman spinin,
    tai päinvastoin, käännetään spinin suunta vaihtamalla molempien merkit.
    """
    if cmd.get("action") != "drive" or not _is_spin(cmd):
        return cmd

    u = user_text.lower()
    l = int(cmd.get("l", 0))
    r = int(cmd.get("r", 0))

    # Oikea spin: l>0, r<0
    # Vasen spin: l<0, r>0
    asked_right = ("right" in u) or ("oikea" in u) or ("oikealle" in u)
    asked_left = ("left" in u) or ("vasen" in u) or ("vasemmalle" in u)

    if asked_right and (l < 0 < r):  # LLM teki vasemman
        cmd["l"], cmd["r"] = -l, -r

    if asked_left and (l > 0 > r):  # LLM teki oikean
        cmd["l"], cmd["r"] = -l, -r

    return cmd


def ollama_command(user_text: str) -> dict:
    prompt = SYSTEM + "\n\nUser: " + user_text + "\nAssistant:"
    payload = {"model": MODEL, "prompt": prompt, "stream": False}

    r = SESSION.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=TIMEOUT)
    r.raise_for_status()

    content = r.json().get("response", "")
    content = _strip_fences(content)

    cmd = json.loads(content)
    cmd = _fix_left_right(user_text, cmd)
    return cmd


def do_gateway(cmd: dict):
    a = cmd.get("action")

    if a == "drive":
        l = clamp(int(cmd.get("l", 0)), -MAX_SPEED, MAX_SPEED)
        r = clamp(int(cmd.get("r", 0)), -MAX_SPEED, MAX_SPEED)
        ms = clamp(int(cmd.get("ms", 400)), 50, MAX_MS)

        SESSION.get(
            f"{GATEWAY_BASE}/cmd",
            params={"do": "drive", "l": l, "r": r},
            timeout=TIMEOUT,
        )
        time.sleep(ms / 1000.0)
        SESSION.get(
            f"{GATEWAY_BASE}/cmd",
            params={"do": "stop"},
            timeout=TIMEOUT,
        )
        return

    if a == "head":
        pos = clamp(int(cmd.get("pos", 90)), 0, 180)
        speed = clamp(int(cmd.get("speed", 0)), 0, 9)
        SESSION.get(
            f"{GATEWAY_BASE}/cmd",
            params={"do": "head", "pos": pos, "speed": speed},
            timeout=TIMEOUT,
        )
        return

    SESSION.get(
        f"{GATEWAY_BASE}/cmd",
        params={"do": "stop"},
        timeout=TIMEOUT,
    )


def main():
    print("Ramblebot agent ready. Type commands. Ctrl+C to exit.")
    while True:
        user = input("> ").strip()
        if not user:
            continue
        try:
            cmd = ollama_command(user)
            print("LLM:", cmd)
            do_gateway(cmd)
        except Exception as e:
            print("ERR:", e)
            try:
                SESSION.get(f"{GATEWAY_BASE}/cmd", params={"do": "stop"}, timeout=TIMEOUT)
            except Exception:
                pass


if __name__ == "__main__":
    main()
