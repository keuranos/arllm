import base64
import json
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests
import numpy as np
import cv2

# Person detector (YOLO)
from ultralytics import YOLO


# ---------------- CONFIG ----------------
OLLAMA_BASE = "http://192.168.86.250:11434"
MODEL = "devstral-small-2:latest"           # LLM (planner)
GATEWAY_BASE = "http://192.168.86.35:8765"

TIMEOUT = 30
SESSION = requests.Session()

STREAM_URL = f"{GATEWAY_BASE}/stream.mjpeg"
FRAME_READ_TIMEOUT = 8.0
SENSORS_URL = f"{GATEWAY_BASE}/sensors.json"

# Safety limits
MAX_SPEED = 170
MAX_MS = 1200
MAX_STEPS = 80
STEP_DELAY = 0.12
MAX_ACTIONS_PER_STEP = 6

# Movement defaults
FWD_L, FWD_R, FWD_MS = 145, 145, 520
TURN_SPEED, TURN_MS = 150, 420
APPROACH_L, APPROACH_R, APPROACH_MS = 120, 120, 450

# Vision tuning
PERSON_CONF_MIN = 0.35

# "Door/opening" heuristic tuning
OPENING_MIN_SCORE = 0.55      # if above -> we believe there's an opening
OPENING_APPROACH_SCORE = 0.70 # if above -> approach more confidently


# ---------------- SYSTEM PROMPT ----------------
SYSTEM = """You are an autonomous robot planner controlling a telepresence robot.

You will receive:
- task: a high-level goal in Finnish or English
- telemetry: includes detections from vision:
  - persons[]: normalized bbox and center coordinates
  - opening: {score, center_x, center_y, bbox_norm} for a doorway-like opening (heuristic)
  - sensors: summarized phone sensor data (motion, orientation, proximity, light)
- history: what was executed recently
- memory_summary: a short summary of recent behavior
- state: current high-level mode (observe/plan/act/recover)

You MUST respond ONLY with JSON.

Allowed actions:
A) {"action":"drive","l":-255..255,"r":-255..255,"ms":50..4000,"note":"..."}
B) {"action":"head","pos":0..180,"speed":0..9,"note":"..."}
C) {"action":"stop","note":"..."}
D) {"action":"done","result":"..."}
E) {"actions":[...], "note":"..."}   // batch

Drive convention:
- Positive = forward, negative = backward
- RIGHT spin in place: l>0, r<0
- LEFT  spin in place: l<0, r>0

How to use detections:
- If searching for people: scan/turn to maximize person detections; if a person is detected, center them (use person_center_x near 0.5).
- If going to another room: prioritize the opening signal; turn to center opening_center_x near 0.5 then approach forward in short steps.

Policy (important):
- Do not get stuck doing head-only moves. If uncertain, do a small turn and/or a short forward step.
- Prefer short motions (200-800ms) then stop.
- Avoid repeating the same action if the image is not changing.
- Output only JSON.
"""


# ---------------- UTIL ----------------
@dataclass
class Safety:
    max_speed: int = MAX_SPEED
    max_ms: int = MAX_MS


class Mode(Enum):
    OBSERVE = "observe"
    PLAN = "plan"
    ACT = "act"
    RECOVER = "recover"


@dataclass
class AgentState:
    mode: Mode = Mode.OBSERVE
    stall_count: int = 0
    error_count: int = 0
    last_action_fp: Optional[str] = None
    memory_summary: str = ""
    recent_actions: List[str] = field(default_factory=list)


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


def jpeg_to_b64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("ascii")


def quick_fp(b: bytes) -> str:
    return hashlib.sha256(b[:20000]).hexdigest()


def fp_delta(prev: Optional[str], cur: str) -> float:
    if prev is None:
        return 1.0
    dif = sum(1 for a, b in zip(prev, cur) if a != b)
    return dif / max(1, len(cur))


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = strip_fences(text)
        return json.loads(cleaned)


# ---------------- MJPEG GRAB ----------------
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


def fetch_sensors() -> Optional[Dict[str, Any]]:
    try:
        r = SESSION.get(SENSORS_URL, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def summarize_sensors(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not raw:
        return {"available": False}

    sensors = raw.get("sensors", [])
    summary = {
        "available": True,
        "sensor_count": raw.get("sensorCount", len(sensors)),
        "motion": {},
        "environment": {},
    }

    def find_by_type(substr: str) -> Optional[Dict[str, Any]]:
        for item in sensors:
            st = str(item.get("stringType", "")).lower()
            name = str(item.get("name", "")).lower()
            if substr in st or substr in name:
                return item
        return None

    def extract_values(item: Optional[Dict[str, Any]]) -> Optional[List[float]]:
        if not item or not item.get("hasReading"):
            return None
        return item.get("values")

    summary["motion"]["accelerometer"] = extract_values(find_by_type("accelerometer"))
    summary["motion"]["gyroscope"] = extract_values(find_by_type("gyroscope"))
    summary["motion"]["gravity"] = extract_values(find_by_type("gravity"))
    summary["motion"]["rotation_vector"] = extract_values(find_by_type("rotation_vector"))
    summary["environment"]["light"] = extract_values(find_by_type("light"))
    summary["environment"]["proximity"] = extract_values(find_by_type("proximity"))

    return summary


# ---------------- IMAGE DECODE ----------------
def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return img


# ---------------- PERSON DETECTOR ----------------
class PersonDetector:
    def __init__(self):
        # Tämä hakee automaattisesti sopivat painot jos puuttuu.
        # Jos haluat nopeamman: yolov8n.pt (default). Tarkempi: yolov8s.pt.
        self.model = YOLO("yolov8n.pt")

    def detect(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns list of persons with normalized bbox and centers.
        """
        h, w = img_bgr.shape[:2]
        res = self.model.predict(img_bgr, verbose=False, conf=PERSON_CONF_MIN)
        out: List[Dict[str, Any]] = []

        if not res:
            return out

        r0 = res[0]
        if r0.boxes is None:
            return out

        boxes = r0.boxes
        # COCO: class 0 = person
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            if cls != 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            out.append({
                "conf": round(conf, 3),
                "bbox_norm": {
                    "x1": round(x1 / w, 4),
                    "y1": round(y1 / h, 4),
                    "x2": round(x2 / w, 4),
                    "y2": round(y2 / h, 4),
                },
                "center_x": round(cx / w, 4),
                "center_y": round(cy / h, 4),
                "area_norm": round(((x2 - x1) * (y2 - y1)) / (w * h), 5),
            })

        # isoimmat ensin (usein lähin)
        out.sort(key=lambda p: p["area_norm"], reverse=True)
        return out


# ---------------- OPENING / DOORWAY HEURISTIC ----------------
def detect_opening(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Heuristinen "door/opening" tunnistus.
    Idea: etsitään kuvasta pystysuuntaisia reunoja (Canny) ja etsitään niistä
    kaksi vahvaa pystylinjaa, joiden välissä on "aukkomainen" alue.
    Palauttaa:
      {"score":0..1, "center_x":.., "center_y":.., "bbox_norm":{...}} tai score=0.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 160)

    # Korostetaan pystysuoria piirteitä morfologialla
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    vert = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Hough-lines
    lines = cv2.HoughLinesP(
        vert,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(h * 0.25),
        maxLineGap=15,
    )

    if lines is None:
        return {"score": 0.0}

    # Kerää pystylinjat: |dx| pieni, pituus iso
    candidates = []
    for (x1, y1, x2, y2) in lines[:, 0, :]:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy < int(h * 0.25):
            continue
        if dx > 10:
            continue
        x = int((x1 + x2) / 2)
        length = dy
        y_top = min(y1, y2)
        y_bot = max(y1, y2)
        candidates.append((x, length, y_top, y_bot))

    if len(candidates) < 2:
        return {"score": 0.0}

    # Ryhmittele x-koordinaatin mukaan (poista duplikaatit)
    candidates.sort(key=lambda t: t[0])
    merged = []
    for c in candidates:
        if not merged or abs(c[0] - merged[-1][0]) > 12:
            merged.append(c)
        else:
            # pidempi voittaa
            if c[1] > merged[-1][1]:
                merged[-1] = c

    if len(merged) < 2:
        return {"score": 0.0}

    # Etsi paras pari (vasen, oikea) järkevällä leveydellä
    best = None
    best_score = 0.0

    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            xL, lenL, topL, botL = merged[i]
            xR, lenR, topR, botR = merged[j]
            width = xR - xL
            if width < int(w * 0.15) or width > int(w * 0.75):
                continue

            y_top = min(topL, topR)
            y_bot = max(botL, botR)
            if (y_bot - y_top) < int(h * 0.35):
                continue

            # Arvioi “aukkoisuus”: keskikaistasta vähän reunoja
            mid_x1 = int(xL + width * 0.25)
            mid_x2 = int(xR - width * 0.25)
            mid_y1 = int(y_top + (y_bot - y_top) * 0.15)
            mid_y2 = int(y_bot - (y_bot - y_top) * 0.15)

            mid_x1 = clamp(mid_x1, 0, w - 1)
            mid_x2 = clamp(mid_x2, 0, w - 1)
            mid_y1 = clamp(mid_y1, 0, h - 1)
            mid_y2 = clamp(mid_y2, 0, h - 1)
            if mid_x2 <= mid_x1 or mid_y2 <= mid_y1:
                continue

            roi = edges[mid_y1:mid_y2, mid_x1:mid_x2]
            edge_density = float(np.mean(roi > 0))  # 0..1
            # Aukko: edge_density usein pienempi kuin seinä/kaluste
            openness = 1.0 - edge_density

            # Vahvuus: linjojen pituus + openness + leveys
            strength = (min(lenL, lenR) / h) * 0.45 + openness * 0.45 + (width / w) * 0.10

            if strength > best_score:
                best_score = strength
                best = (xL, y_top, xR, y_bot)

    if best is None:
        return {"score": 0.0}

    x1, y1, x2, y2 = best
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    score = float(max(0.0, min(1.0, best_score)))

    return {
        "score": round(score, 3),
        "center_x": round(cx / w, 4),
        "center_y": round(cy / h, 4),
        "bbox_norm": {
            "x1": round(x1 / w, 4),
            "y1": round(y1 / h, 4),
            "x2": round(x2 / w, 4),
            "y2": round(y2 / h, 4),
        },
    }


# ---------------- OLLAMA STEP ----------------
def build_prompt(
    task: str,
    telemetry: Dict[str, Any],
    history: List[str],
    memory_summary: str,
    state: AgentState,
) -> str:
    short_hist = history[-10:]
    return (
        SYSTEM
        + "\n\nTASK:\n" + task
        + "\n\nSTATE:\n" + state.mode.value
        + "\n\nMEMORY SUMMARY:\n" + (memory_summary or "(none)")
        + "\n\nTELEMETRY:\n" + json.dumps(telemetry, ensure_ascii=False)
        + "\n\nHISTORY (most recent last):\n" + "\n".join(short_hist)
        + "\n\nReturn next action(s) JSON now."
    )


def ollama_plan(prompt: str, image_b64: str) -> Dict[str, Any]:
    payload = {"model": MODEL, "prompt": prompt, "images": [image_b64], "stream": False}
    r = SESSION.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    content = r.json().get("response", "")
    return safe_json_loads(content)


# ---------------- GATEWAY ACTIONS ----------------
def gw_get(params: Dict[str, Any]) -> None:
    SESSION.get(f"{GATEWAY_BASE}/cmd", params=params, timeout=TIMEOUT).raise_for_status()


def do_stop():
    gw_get({"do": "stop"})


def do_head(pos: int, speed: int):
    pos = clamp(int(pos), 0, 180)
    speed = clamp(int(speed), 0, 9)
    gw_get({"do": "head", "pos": pos, "speed": speed})


def do_drive(l: int, r: int, ms: int, safety: Safety):
    l = clamp(int(l), -safety.max_speed, safety.max_speed)
    r = clamp(int(r), -safety.max_speed, safety.max_speed)
    ms = clamp(int(ms), 50, safety.max_ms)
    gw_get({"do": "drive", "l": l, "r": r})
    time.sleep(ms / 1000.0)
    do_stop()


def normalize_actions(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(obj.get("actions"), list):
        return [a for a in obj["actions"] if isinstance(a, dict)]
    if "action" in obj:
        return [obj]
    return []


def validate_actions(actions: List[Dict[str, Any]], safety: Safety) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for a in actions:
        kind = a.get("action")
        if kind not in {"drive", "head", "stop", "done"}:
            continue
        if kind == "drive":
            l = clamp(int(a.get("l", 0)), -safety.max_speed, safety.max_speed)
            r = clamp(int(a.get("r", 0)), -safety.max_speed, safety.max_speed)
            ms = clamp(int(a.get("ms", 400)), 50, safety.max_ms)
            cleaned.append({"action": "drive", "l": l, "r": r, "ms": ms, "note": a.get("note", "")})
        elif kind == "head":
            pos = clamp(int(a.get("pos", 90)), 0, 180)
            speed = clamp(int(a.get("speed", 0)), 0, 9)
            cleaned.append({"action": "head", "pos": pos, "speed": speed, "note": a.get("note", "")})
        elif kind == "stop":
            cleaned.append({"action": "stop", "note": a.get("note", "")})
        elif kind == "done":
            cleaned.append({"action": "done", "result": a.get("result", "done")})
        if len(cleaned) >= MAX_ACTIONS_PER_STEP:
            break
    return cleaned


def execute(actions: List[Dict[str, Any]], safety: Safety, history: List[str]) -> Tuple[bool, Optional[str]]:
    for a in actions:
        kind = a.get("action")
        note = a.get("note", "")

        if kind == "done":
            res = str(a.get("result", "done"))
            history.append(f"EXEC: done -> {res}")
            do_stop()
            return True, res

        if kind == "stop":
            history.append(f"EXEC: stop {note}".strip())
            do_stop()
            time.sleep(STEP_DELAY)
            continue

        if kind == "head":
            pos = a.get("pos", 90)
            speed = a.get("speed", 0)
            history.append(f"EXEC: head pos={pos} speed={speed} {note}".strip())
            do_head(pos, speed)
            time.sleep(STEP_DELAY)
            continue

        if kind == "drive":
            l = a.get("l", 0)
            r = a.get("r", 0)
            ms = a.get("ms", 400)
            history.append(f"EXEC: drive l={l} r={r} ms={ms} {note}".strip())
            do_drive(l, r, ms, safety)
            time.sleep(STEP_DELAY)
            continue

        history.append(f"EXEC: unknown -> stop {a}")
        do_stop()
        time.sleep(STEP_DELAY)

    return False, None


# ---------------- REFLEX: steer to center target ----------------
def steer_to_center(center_x: float) -> Tuple[int, int, int]:
    """
    center_x: 0..1 (0 left, 1 right), target 0.5
    returns (l, r, ms) for a short turn
    """
    err = center_x - 0.5
    if abs(err) < 0.06:
        return (0, 0, 0)

    # turn direction: if target is to the right (err>0), turn right: l>0 r<0
    direction = 1 if err > 0 else -1
    mag = min(1.0, abs(err) / 0.5)

    sp = int(120 + 60 * mag)  # 120..180
    ms = int(260 + 240 * mag) # 260..500

    if direction > 0:
        return (sp, -sp, ms)
    else:
        return (-sp, sp, ms)


def reflex_for_task(task: str, telemetry: Dict[str, Any], step: int) -> Optional[List[Dict[str, Any]]]:
    """
    If model stalls or we want stronger autonomy, do this reflex batch.
    - If opening strong: center and approach
    - Else patrol: forward + alternate turn
    - If searching people and person exists: center person and stop (or approach a bit)
    """
    t = task.lower()

    persons = telemetry.get("persons", [])
    opening = telemetry.get("opening", {"score": 0.0})
    o_score = float(opening.get("score", 0.0))

    # Person reflex: if the task is to find people
    if ("etsi" in t and "ihmi" in t) or ("find" in t and "person" in t):
        if persons:
            p0 = persons[0]
            cx = float(p0["center_x"])
            turn = steer_to_center(cx)
            actions = [{"action": "head", "pos": 90, "speed": 0, "note": "center head"}]
            if turn[2] > 0:
                actions.append({"action": "drive", "l": turn[0], "r": turn[1], "ms": turn[2], "note": "center person"})
            actions.append({"action": "stop"})
            actions.append({"action": "done", "result": f"Person found (conf={p0['conf']}, cx={p0['center_x']}, cy={p0['center_y']})"})
            return actions

    # Go-to-room reflex: opening centering + approach
    if ("toiseen huoneeseen" in t) or ("another room" in t) or ("mene" in t and "huone" in t):
        if o_score >= OPENING_MIN_SCORE and "center_x" in opening:
            cx = float(opening["center_x"])
            turn = steer_to_center(cx)
            actions = [{"action": "head", "pos": 90, "speed": 0, "note": "scan opening"}]
            if turn[2] > 0:
                actions.append({"action": "drive", "l": turn[0], "r": turn[1], "ms": turn[2], "note": "center opening"})
                actions.append({"action": "stop"})
            # approach
            if o_score >= OPENING_APPROACH_SCORE:
                actions.append({"action": "drive", "l": APPROACH_L, "r": APPROACH_R, "ms": APPROACH_MS, "note": "approach opening"})
                actions.append({"action": "stop"})
            return actions

        # Patrol if no opening
        actions: List[Dict[str, Any]] = []
        # gentle scan
        if step % 4 == 1:
            actions.append({"action": "head", "pos": 70, "speed": 0, "note": "scan up"})
        elif step % 4 == 2:
            actions.append({"action": "head", "pos": 110, "speed": 0, "note": "scan down"})
        else:
            actions.append({"action": "head", "pos": 90, "speed": 0, "note": "center"})

        actions.append({"action": "drive", "l": FWD_L, "r": FWD_R, "ms": FWD_MS, "note": "patrol forward (search doorway)"})
        actions.append({"action": "stop"})
        if step % 2 == 0:
            actions.append({"action": "drive", "l": TURN_SPEED, "r": -TURN_SPEED, "ms": TURN_MS, "note": "turn right (search)"})
        else:
            actions.append({"action": "drive", "l": -TURN_SPEED, "r": TURN_SPEED, "ms": TURN_MS, "note": "turn left (search)"})
        actions.append({"action": "stop"})
        return actions

    return None


def summarize_history(history: List[str]) -> str:
    if len(history) <= 6:
        return " | ".join(history)
    tail = history[-6:]
    return " ... | " + " | ".join(tail)


# ---------------- MAIN LOOP ----------------
def run_task(task: str, detector: PersonDetector):
    safety = Safety()
    history: List[str] = []
    state = AgentState()

    print(f"\nTASK START: {task}\n")
    try:
        do_stop()
        do_head(90, 0)
    except Exception:
        pass

    prev_fp: Optional[str] = None
    for step in range(1, MAX_STEPS + 1):
        try:
            state.mode = Mode.OBSERVE
            jpeg = get_mjpeg_frame(STREAM_URL)
            fp = quick_fp(jpeg)
            delta = fp_delta(prev_fp, fp)
            prev_fp = fp

            img = decode_jpeg(jpeg)

            # Vision detections
            persons = detector.detect(img)
            opening = detect_opening(img)
            sensors_raw = fetch_sensors()
            sensors_summary = summarize_sensors(sensors_raw)

            telemetry = {
                "step": step,
                "image_delta": round(delta, 3),
                "persons": persons[:3],  # max 3
                "opening": opening if float(opening.get("score", 0.0)) > 0 else {"score": 0.0},
                "stall_count": state.stall_count,
                "sensors": sensors_summary,
            }

            img_b64 = jpeg_to_b64(jpeg)

            # LLM plan
            state.mode = Mode.PLAN
            prompt = build_prompt(task, telemetry, history, state.memory_summary, state)
            model_out = ollama_plan(prompt, img_b64)
            actions = normalize_actions(model_out)
            actions = validate_actions(actions, safety)

            # Debug
            print(f"\n[step {step}] TELEMETRY:")
            print(json.dumps(telemetry, ensure_ascii=False))
            print(f"[step {step}] MODEL JSON:")
            print(json.dumps(model_out, ensure_ascii=False))

            # stall logic: head-only repeats
            if actions and all(a.get("action") == "head" for a in actions):
                state.stall_count += 1
            elif actions:
                state.stall_count = 0

            # Reflex override if model stalls or returns nothing
            if (not actions) or state.stall_count >= 2 or delta < 0.02:
                state.mode = Mode.RECOVER
                fb = reflex_for_task(task, telemetry, step)
                if fb:
                    print(f"[step {step}] REFLEX OVERRIDE:")
                    print(json.dumps({"actions": fb}, ensure_ascii=False))
                    done, result = execute(fb, safety, history)
                    state.stall_count = 0
                    if done:
                        print("\nTASK DONE:", result)
                        return
                    continue

            # Execute model actions
            state.mode = Mode.ACT
            done, result = execute(actions, safety, history)
            print(f"[step {step}] EXECUTED {len(actions)} action(s)")
            state.memory_summary = summarize_history(history)
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
            state.error_count += 1
            history.append(f"ERR: {repr(e)}")
            time.sleep(0.3)
            if state.error_count >= 3:
                print("[recover] Too many errors, pausing briefly.")
                time.sleep(1.0)
                state.error_count = 0

    print("\nTASK END: reached MAX_STEPS, stopping.")
    try:
        do_stop()
    except Exception:
        pass


def main():
    print("Ramblebot Vision Hybrid Agent (person YOLO + opening heuristic + multimodal LLM)")
    print(f"- Robot:  {GATEWAY_BASE}")
    print(f"- Ollama: {OLLAMA_BASE}")
    print(f"- LLM:    {MODEL}")
    print("Ctrl+C to stop.\n")

    detector = PersonDetector()

    while True:
        task = input("> ").strip()
        if not task:
            continue
        run_task(task, detector)


if __name__ == "__main__":
    main()
