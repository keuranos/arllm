import base64
import json
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import Config

# ---- CONFIG ----
OLLAMA_BASE = Config.OLLAMA_BASE
MODEL = Config.MODEL  # varmista että mallisi tukee kuvia Ollamassa
GATEWAY_BASE = Config.GATEWAY_BASE

TIMEOUT = Config.TIMEOUT
SESSION = requests.Session()

# Safety limits
MAX_SPEED = Config.MAX_SPEED
MAX_MS = Config.MAX_MS
MAX_STEPS = Config.MAX_STEPS
STEP_DELAY = 0.15

# MJPEG stream
STREAM_URL = f"{GATEWAY_BASE}/stream.mjpeg"
FRAME_READ_TIMEOUT = 8.0

# Hybrid patrol defaults (door search)
PATROL_FORWARD_L = 150
PATROL_FORWARD_R = 150
PATROL_FORWARD_MS = 550

PATROL_TURN_MS = 420
PATROL_TURN_SPEED = 150

# If we suspect doorway ahead, approach slowly
APPROACH_L = 120
APPROACH_R = 120
APPROACH_MS = 450

# ---- SYSTEM PROMPT ----
SYSTEM = """You are an autonomous robot planner controlling a telepresence robot in a house.

You receive:
- task: a high-level goal
- image: the latest camera image
- telemetry: simple signals from a reflex controller (movement delta, doorway heuristic)

You must respond ONLY with JSON.

You may output:
A) {"action":"drive","l":..,"r":..,"ms":..,"note":"..."}
B) {"action":"head","pos":..,"speed":..,"note":"..."}
C) {"action":"stop","note":"..."}
D) {"action":"done","result":"..."}  // when task is complete
E) {"actions":[...], "note":"..."}   // multi-step batch

Drive convention:
- Positive speed = forward, negative = backward.
- RIGHT spin in place: l>0, r<0 (e.g. l=150,r=-150)
- LEFT  spin in place: l<0, r>0 (e.g. l=-150,r=150)

Policy for task "go to another room" (IMPORTANT):
- You MUST attempt motion, not only scanning.
- Use a loop: small forward -> stop -> scan -> small turn -> repeat.
- When doorway_heuristic is high, approach forward and keep centered.

Safety:
- Prefer short motions (200-800ms) then stop.
- If uncertain, do a small turn + scan rather than doing nothing.
- Output only JSON.
"""


@dataclass
class Safety:
    max_speed: int = MAX_SPEED
    max_ms: int = MAX_MS


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
        s = s.replace("json", "", 1).strip()
    return s.strip()


# -------- MJPEG frame grab --------
def get_mjpeg_frame(url: str) -> bytes:
    r = SESSION.get(url, stream=True, timeout=FRAME_READ_TIMEOUT)
    r.raise_for_status()

    buf = bytearray()
    soi = b"\xff\xd8"
    eoi = b"\xff\xd9"
    start = None

    t0 = time.time()
    for chunk in r.iter_content(chunk_size=4096):
        if not chunk:
            continue
        buf.extend(chunk)

        if start is None:
            idx = buf.find(soi)
            if idx != -1:
                start = idx

        if start is not None:
            idx2 = buf.find(eoi, start + 2)
            if idx2 != -1:
                frame = bytes(buf[start:idx2 + 2])
                r.close()
                return frame

        if time.time() - t0 > FRAME_READ_TIMEOUT:
            r.close()
            raise TimeoutError("Timed out reading MJPEG frame")


def jpeg_to_base64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("ascii")


# -------- Simple telemetry from images (no OpenCV) --------
def quick_image_fingerprint(jpeg_bytes: bytes) -> str:
    # fast stable hash for "did image change"
    return hashlib.sha256(jpeg_bytes[:20000]).hexdigest()  # enough for signal


def image_delta(prev_fp: Optional[str], fp: str) -> float:
    if prev_fp is None:
        return 1.0
    # crude: fraction of differing hex chars
    dif = sum(1 for a, b in zip(prev_fp, fp) if a != b)
    return dif / max(1, len(fp))


def doorway_heuristic(jpeg_bytes: bytes) -> float:
    """
    Super lightweight heuristic (0..1) without decoding image:
    uses jpeg byte distribution proxies.
    This is not "true doorway detection" but gives a weak bias.
    We'll replace with OpenCV/YOLO later.
    """
    sample = jpeg_bytes[0:60000]
    if not sample:
        return 0.0
    # Heuristic: more edges/contrast often compresses worse -> higher entropy-ish
    # Approximate by unique byte ratio.
    uniq = len(set(sample))
    score = uniq / 256.0
    # map to 0..1 with mild scaling
    return max(0.0, min(1.0, (score - 0.35) / 0.35))


# -------- Ollama multimodal step --------
def ollama_step(task: str, image_b64: str, telemetry: Dict[str, Any], history: List[str]) -> Dict[str, Any]:
    short_hist = history[-10:]
    prompt = (
        SYSTEM
        + "\n\nTASK:\n" + task
        + "\n\nTELEMETRY (from reflex controller):\n" + json.dumps(telemetry, ensure_ascii=False)
        + "\n\nHISTORY (most recent last):\n" + "\n".join(short_hist)
        + "\n\nReturn next action(s) JSON now."
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }

    resp = SESSION.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    content = resp.json().get("response", "")
    content = strip_fences(content)
    return json.loads(content)


# -------- Gateway I/O --------
def gw_get(params: Dict[str, Any]) -> None:
    SESSION.get(f"{GATEWAY_BASE}/cmd", params=params, timeout=TIMEOUT).raise_for_status()


def do_stop() -> None:
    gw_get({"do": "stop"})


def do_head(pos: int, speed: int) -> None:
    pos = clamp(int(pos), 0, 180)
    speed = clamp(int(speed), 0, 9)
    gw_get({"do": "head", "pos": pos, "speed": speed})


def do_drive(l: int, r: int, ms: int, safety: Safety) -> None:
    l = clamp(int(l), -safety.max_speed, safety.max_speed)
    r = clamp(int(r), -safety.max_speed, safety.max_speed)
    ms = clamp(int(ms), 50, safety.max_ms)

    gw_get({"do": "drive", "l": l, "r": r})
    time.sleep(ms / 1000.0)
    do_stop()


# -------- Action parsing/execution --------
def normalize_actions(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(obj.get("actions"), list):
        return [a for a in obj["actions"] if isinstance(a, dict)]
    if "action" in obj:
        return [obj]
    return []


def execute_actions(actions: List[Dict[str, Any]], safety: Safety, history: List[str]) -> Tuple[bool, Optional[str]]:
    for a in actions:
        kind = a.get("action")
        note = a.get("note")

        if kind == "done":
            res = str(a.get("result", "done"))
            history.append(f"EXEC: done -> {res}")
            do_stop()
            return True, res

        if kind == "stop":
            history.append(f"EXEC: stop {note or ''}".strip())
            do_stop()
            time.sleep(STEP_DELAY)
            continue

        if kind == "head":
            pos = a.get("pos", 90)
            speed = a.get("speed", 0)
            history.append(f"EXEC: head pos={pos} speed={speed} {note or ''}".strip())
            do_head(pos, speed)
            time.sleep(STEP_DELAY)
            continue

        if kind == "drive":
            l = a.get("l", 0)
            r = a.get("r", 0)
            ms = a.get("ms", 400)
            history.append(f"EXEC: drive l={l} r={r} ms={ms} {note or ''}".strip())
            do_drive(l, r, ms, safety)
            time.sleep(STEP_DELAY)
            continue

        history.append(f"EXEC: unknown action -> stop {a}")
        do_stop()
        time.sleep(STEP_DELAY)

    return False, None


# -------- Reflex controller: if LLM stalls, patrol automatically --------
def reflex_patrol(step_index: int, telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a small action batch to keep moving.
    Uses simple alternating pattern: forward -> stop -> small scan -> turn -> stop.
    If doorway_heuristic high, approach forward more often.
    """
    door = float(telemetry.get("doorway_score", 0.0))

    actions: List[Dict[str, Any]] = []

    # Always a tiny scan sometimes
    if step_index % 4 == 1:
        actions.append({"action": "head", "pos": 70, "speed": 0, "note": "scan up"})
    elif step_index % 4 == 2:
        actions.append({"action": "head", "pos": 110, "speed": 0, "note": "scan down"})
    else:
        actions.append({"action": "head", "pos": 90, "speed": 0, "note": "center"})

    # Movement decision
    if door > 0.6:
        actions.append({"action": "drive", "l": APPROACH_L, "r": APPROACH_R, "ms": APPROACH_MS, "note": "approach suspected doorway"})
        actions.append({"action": "stop"})
        return actions

    # Normal patrol
    actions.append({"action": "drive", "l": PATROL_FORWARD_L, "r": PATROL_FORWARD_R, "ms": PATROL_FORWARD_MS, "note": "patrol forward"})
    actions.append({"action": "stop"})

    # Turn alternates to explore
    if step_index % 2 == 0:
        actions.append({"action": "drive", "l": PATROL_TURN_SPEED, "r": -PATROL_TURN_SPEED, "ms": PATROL_TURN_MS, "note": "turn right to search"})
    else:
        actions.append({"action": "drive", "l": -PATROL_TURN_SPEED, "r": PATROL_TURN_SPEED, "ms": PATROL_TURN_MS, "note": "turn left to search"})
    actions.append({"action": "stop"})

    return actions


def run_task(task: str) -> None:
    safety = Safety()
    history: List[str] = []
    print(f"\nTASK START: {task}\n")

    # Init
    try:
        do_stop()
        do_head(90, 0)
    except Exception:
        pass

    prev_fp: Optional[str] = None
    stall_count = 0

    for step in range(1, MAX_STEPS + 1):
        try:
            jpeg = get_mjpeg_frame(STREAM_URL)
            fp = quick_image_fingerprint(jpeg)
            delta = image_delta(prev_fp, fp)
            prev_fp = fp
            door_score = doorway_heuristic(jpeg)

            telemetry = {
                "step": step,
                "image_delta": round(delta, 3),
                "doorway_score": round(door_score, 3),
                "stall_count": stall_count,
            }

            img_b64 = jpeg_to_base64(jpeg)

            # Ask LLM
            model_out = ollama_step(task, img_b64, telemetry, history)
            history.append("MODEL: " + json.dumps(model_out, ensure_ascii=False))

            actions = normalize_actions(model_out)

            # Debug output (show more than "[step] head (batch)")
            print(f"\n[step {step}] TELEMETRY: {telemetry}")
            print(f"[step {step}] MODEL JSON: {json.dumps(model_out, ensure_ascii=False)}")

            # If model returns nothing useful, fallback
            if not actions:
                stall_count += 1
                fb = reflex_patrol(step, telemetry)
                print(f"[step {step}] REFLEX: fallback patrol -> {fb}")
                done, result = execute_actions(fb, safety, history)
                if done:
                    print("\nTASK DONE:", result)
                    return
                continue

            # Detect "stall": if only head actions repeatedly, enforce patrol after a few
            only_head = all(a.get("action") == "head" for a in actions)
            if only_head:
                stall_count += 1
            else:
                stall_count = 0

            if stall_count >= 3:
                fb = reflex_patrol(step, telemetry)
                print(f"[step {step}] REFLEX: model stalling (head-only). Forcing patrol -> {fb}")
                done, result = execute_actions(fb, safety, history)
                if done:
                    print("\nTASK DONE:", result)
                    return
                stall_count = 0
                continue

            # Execute model actions
            done, result = execute_actions(actions, safety, history)
            print(f"[step {step}] EXECUTED: {len(actions)} action(s)")
            if done:
                print("\nTASK DONE:", result)
                return

        except KeyboardInterrupt:
            print("\nInterrupted. Stopping robot.")
            try:
                do_stop()
            except Exception:
                pass
            return
        except Exception as e:
            print(f"[step {step}] ERR: {e}")
            try:
                do_stop()
            except Exception:
                pass
            history.append(f"ERR: {repr(e)}")
            time.sleep(0.3)

    print("\nTASK END: reached MAX_STEPS, stopping.")
    try:
        do_stop()
    except Exception:
        pass


def main():
    print("Ramblebot Hybrid Agent (multimodal + reflex patrol)")
    print(f"- Ollama: {OLLAMA_BASE}")
    print(f"- Model:  {MODEL}")
    print(f"- Robot:  {GATEWAY_BASE}")
    print("Ctrl+C to stop.\n")

    while True:
        task = input("> ").strip()
        if not task:
            continue
        run_task(task)


if __name__ == "__main__":
    main()
