"""
Ramblebot Smart Pet Agent

A unified agent that combines:
- MiDaS depth estimation for obstacle awareness
- Semantic memory for room/landmark storage
- ARCore pose tracking for localization
- YOLO person detection
- LLM planning for task execution
- Autonomous "curious pet" behaviors

The robot operates in two modes:
1. Task Mode: Execute user-given tasks ("go to kitchen and check for people")
2. Autonomous Mode: Act like a curious pet, explore, and react to its environment

Usage:
    python ramblebot_smartpet.py

Commands:
    > <task>        - Execute a task
    > /auto         - Switch to autonomous mode
    > /task         - Switch to task mode
    > /status       - Show pet status
    > /memory       - Show memory summary
    > /rooms        - List known rooms
    > /explore      - Start exploration
    > /q or exit    - Quit
"""

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

# YOLO person detector
from ultralytics import YOLO

# Our modules
from config import Config
from depth_estimator import estimate_depth, depth_to_telemetry
from semantic_memory import SemanticMemory, Pose, LandmarkType, get_memory
from pet_behaviors import PetBehaviors, ARCoreClient, Mood
from robot_speech import RobotSpeaker, RobotPersonality, CommentaryType
from slam import get_slam, SLAMMapper


# ========================
# Configuration (from config.py)
# ========================

OLLAMA_BASE = Config.OLLAMA_BASE
MODEL = Config.MODEL
GATEWAY_BASE = Config.GATEWAY_BASE

TIMEOUT = Config.TIMEOUT
SESSION = requests.Session()

STREAM_URL = Config.stream_url()
FRAME_READ_TIMEOUT = 8.0
SENSORS_URL = Config.sensors_url()

# Safety limits
MAX_SPEED = Config.MAX_SPEED
MAX_MS = Config.MAX_MS
MAX_STEPS = Config.MAX_STEPS
STEP_DELAY = 0.15
MAX_ACTIONS_PER_STEP = 6

# Movement defaults
FWD_L, FWD_R, FWD_MS = 145, 145, 520
TURN_SPEED, TURN_MS = 150, 420
APPROACH_L, APPROACH_R, APPROACH_MS = 120, 120, 450

# Vision tuning
PERSON_CONF_MIN = Config.PERSON_CONF_MIN

# Opening heuristic tuning
OPENING_MIN_SCORE = Config.OPENING_MIN_SCORE
OPENING_APPROACH_SCORE = Config.OPENING_APPROACH_SCORE

# Enable/disable depth estimation (can be slow without GPU)
USE_MIDAS_DEPTH = Config.USE_MIDAS_DEPTH

# Enable/disable Finnish speech commentary
USE_SPEECH = Config.USE_SPEECH
SPEECH_CHATTINESS = Config.SPEECH_CHATTINESS


# ========================
# System Prompt for LLM
# ========================

SYSTEM_PROMPT = """You are a smart pet robot planner. You control a telepresence robot that acts like a curious, helpful pet.

You will receive:
- task: The current goal (or "autonomous" if in pet mode)
- telemetry: Sensor data including:
  - persons[]: Detected people with position
  - opening: Doorway detection
  - depth: MiDaS depth analysis (obstacles, clearance)
  - pose: ARCore position/rotation
  - sensors: Phone accelerometer, light, etc.
- memory_context: What the robot knows about its environment
- pet_state: Mood, energy, curiosity levels
- history: Recent actions

Respond ONLY with JSON.

Allowed actions:
A) {"action":"drive","l":-255..255,"r":-255..255,"ms":50..4000,"note":"..."}
B) {"action":"head","pos":0..180,"speed":0..9,"note":"..."}
C) {"action":"stop","note":"..."}
D) {"action":"done","result":"..."}
E) {"action":"say","text":"...","lang":"fi"}  // For future TTS
F) {"action":"remember","type":"room|object|person","name":"...","note":"..."}
G) {"actions":[...], "note":"..."}  // Batch

Drive convention:
- Positive = forward, negative = backward
- RIGHT spin: l>0, r<0
- LEFT spin: l<0, r>0

DEPTH USAGE (important!):
- depth.clearance.path_ahead: "clear", "obstacle_near", or "blocked"
- depth.clearance.clearest_direction: "left", "center", or "right"
- depth.zones: average depth per zone (higher = closer obstacle)
- If path_ahead is "blocked" or "obstacle_near", turn toward clearest_direction

MEMORY USAGE:
- memory_context.known_rooms: List of rooms the robot knows
- memory_context.current_room: Where the robot thinks it is
- memory_context.nearby_landmarks: What's near the robot
- Use "remember" action to save new locations

PERSONALITY:
- Be curious and friendly
- Express interest in people (approach them gently)
- When bored, explore new areas
- Avoid repeating the same action if nothing changes
- Keep movements short (200-800ms) for safety

When in autonomous/pet mode:
- Explore if curious
- React to movement/changes
- Approach people if social
- Remember new places discovered
"""


# ========================
# Utilities
# ========================

class Mode(Enum):
    OBSERVE = "observe"
    PLAN = "plan"
    ACT = "act"
    RECOVER = "recover"


class OperatingMode(Enum):
    TASK = "task"
    AUTONOMOUS = "autonomous"


@dataclass
class Safety:
    max_speed: int = MAX_SPEED
    max_ms: int = MAX_MS


@dataclass
class AgentState:
    mode: Mode = Mode.OBSERVE
    operating_mode: OperatingMode = OperatingMode.AUTONOMOUS
    current_task: str = ""
    stall_count: int = 0
    error_count: int = 0
    last_action_fp: Optional[str] = None
    steps_taken: int = 0


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


# ========================
# MJPEG & Sensors
# ========================

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

    summary["light"] = extract_values(find_by_type("light"))
    summary["proximity"] = extract_values(find_by_type("proximity"))

    return summary


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return img


# ========================
# Person Detector
# ========================

class PersonDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        h, w = img_bgr.shape[:2]
        res = self.model.predict(img_bgr, verbose=False, conf=PERSON_CONF_MIN)
        out: List[Dict[str, Any]] = []

        if not res or res[0].boxes is None:
            return out

        boxes = res[0].boxes
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

        out.sort(key=lambda p: p["area_norm"], reverse=True)
        return out


# ========================
# Opening Detection (Doorway Heuristic)
# ========================

def detect_opening(img_bgr: np.ndarray) -> Dict[str, Any]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 160)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    vert = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        vert, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=int(h * 0.25), maxLineGap=15
    )

    if lines is None:
        return {"score": 0.0}

    candidates = []
    for (x1, y1, x2, y2) in lines[:, 0, :]:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy < int(h * 0.25) or dx > 10:
            continue
        x = int((x1 + x2) / 2)
        candidates.append((x, dy, min(y1, y2), max(y1, y2)))

    if len(candidates) < 2:
        return {"score": 0.0}

    candidates.sort(key=lambda t: t[0])
    merged = []
    for c in candidates:
        if not merged or abs(c[0] - merged[-1][0]) > 12:
            merged.append(c)
        elif c[1] > merged[-1][1]:
            merged[-1] = c

    if len(merged) < 2:
        return {"score": 0.0}

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

            mid_x1 = clamp(int(xL + width * 0.25), 0, w - 1)
            mid_x2 = clamp(int(xR - width * 0.25), 0, w - 1)
            mid_y1 = clamp(int(y_top + (y_bot - y_top) * 0.15), 0, h - 1)
            mid_y2 = clamp(int(y_bot - (y_bot - y_top) * 0.15), 0, h - 1)
            if mid_x2 <= mid_x1 or mid_y2 <= mid_y1:
                continue

            roi = edges[mid_y1:mid_y2, mid_x1:mid_x2]
            edge_density = float(np.mean(roi > 0))
            openness = 1.0 - edge_density
            strength = (min(lenL, lenR) / h) * 0.45 + openness * 0.45 + (width / w) * 0.10

            if strength > best_score:
                best_score = strength
                best = (xL, y_top, xR, y_bot)

    if best is None:
        return {"score": 0.0}

    x1, y1, x2, y2 = best
    return {
        "score": round(float(max(0, min(1, best_score))), 3),
        "center_x": round((x1 + x2) / 2.0 / w, 4),
        "center_y": round((y1 + y2) / 2.0 / h, 4),
    }


# ========================
# Gateway Actions
# ========================

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
        if kind not in {"drive", "head", "stop", "done", "say", "remember"}:
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
        elif kind == "say":
            cleaned.append({"action": "say", "text": a.get("text", ""), "lang": a.get("lang", "fi")})
        elif kind == "remember":
            cleaned.append({"action": "remember", "type": a.get("type", "custom"),
                           "name": a.get("name", "unknown"), "note": a.get("note", "")})
        if len(cleaned) >= MAX_ACTIONS_PER_STEP:
            break
    return cleaned


def execute(
    actions: List[Dict[str, Any]],
    safety: Safety,
    history: List[str],
    memory: SemanticMemory,
    arcore: ARCoreClient
) -> Tuple[bool, Optional[str]]:
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
            history.append(f"EXEC: head pos={pos} {note}".strip())
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

        if kind == "say":
            text = a.get("text", "")
            history.append(f"EXEC: say '{text}'")
            print(f"[SAY] {text}")  # Future: send to TTS
            continue

        if kind == "remember":
            lm_type = a.get("type", "custom")
            name = a.get("name", "unknown")
            pose = arcore.fetch_pose()
            if pose:
                if lm_type == "room":
                    memory.enter_room(name, pose)
                else:
                    lt = LandmarkType.OBJECT if lm_type == "object" else LandmarkType.CUSTOM
                    memory.add_landmark(name, lt, pose, description=note)
                history.append(f"EXEC: remember {lm_type}={name}")
            continue

        history.append(f"EXEC: unknown -> stop {a}")
        do_stop()
        time.sleep(STEP_DELAY)

    return False, None


# ========================
# LLM Planning
# ========================

def build_prompt(
    task: str,
    telemetry: Dict[str, Any],
    history: List[str],
    memory_context: Dict[str, Any],
    pet_state: Dict[str, Any],
    state: AgentState,
) -> str:
    short_hist = history[-8:]
    return (
        SYSTEM_PROMPT
        + "\n\nTASK:\n" + task
        + "\n\nOPERATING MODE:\n" + state.operating_mode.value
        + "\n\nPET STATE:\n" + json.dumps(pet_state, ensure_ascii=False)
        + "\n\nMEMORY CONTEXT:\n" + json.dumps(memory_context, ensure_ascii=False)
        + "\n\nTELEMETRY:\n" + json.dumps(telemetry, ensure_ascii=False)
        + "\n\nHISTORY:\n" + "\n".join(short_hist)
        + "\n\nReturn next action(s) JSON now."
    )


def ollama_plan(prompt: str, image_b64: str) -> Dict[str, Any]:
    payload = {"model": MODEL, "prompt": prompt, "images": [image_b64], "stream": False}
    r = SESSION.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    content = r.json().get("response", "")
    return safe_json_loads(content)


# ========================
# Reflex Behaviors
# ========================

def steer_to_center(center_x: float) -> Tuple[int, int, int]:
    err = center_x - 0.5
    if abs(err) < 0.06:
        return (0, 0, 0)

    direction = 1 if err > 0 else -1
    mag = min(1.0, abs(err) / 0.5)
    sp = int(120 + 60 * mag)
    ms = int(260 + 240 * mag)

    if direction > 0:
        return (sp, -sp, ms)
    else:
        return (-sp, sp, ms)


def reflex_for_depth(telemetry: Dict[str, Any], step: int) -> Optional[List[Dict[str, Any]]]:
    """Use depth info to navigate around obstacles."""
    depth = telemetry.get("depth", {})
    clearance = depth.get("clearance", {})
    path_ahead = clearance.get("path_ahead", "unknown")
    clearest = clearance.get("clearest_direction", "center")

    if path_ahead == "blocked":
        # Turn toward clearest direction
        if clearest == "left":
            return [{"action": "drive", "l": -140, "r": 140, "ms": 500, "note": "blocked, turn left"}]
        elif clearest == "right":
            return [{"action": "drive", "l": 140, "r": -140, "ms": 500, "note": "blocked, turn right"}]
        else:
            turn_dir = 1 if step % 2 == 0 else -1
            return [{"action": "drive", "l": 140 * turn_dir, "r": -140 * turn_dir, "ms": 500, "note": "blocked, turn"}]

    return None


# ========================
# Main Agent Class
# ========================

class SmartPetAgent:
    def __init__(self):
        self.detector = PersonDetector()
        self.memory = get_memory("robot_memory.json")
        self.arcore = ARCoreClient(GATEWAY_BASE)
        self.pet = PetBehaviors(self.memory, self.arcore, GATEWAY_BASE)
        self.state = AgentState()
        self.safety = Safety()
        self.history: List[str] = []
        self.prev_fp: Optional[str] = None

        # Initialize SLAM mapper
        self.slam = get_slam("slam_map.json")

        # Initialize speech system
        if USE_SPEECH:
            personality = RobotPersonality(
                name="Ramblebot",
                chattiness=SPEECH_CHATTINESS,
                enthusiasm=0.7,
                use_llm=True,
                language="fi-FI"
            )
            self.speaker = RobotSpeaker(GATEWAY_BASE, OLLAMA_BASE, personality)
        else:
            self.speaker = None

        print("Smart Pet Agent initialized.")
        print(f"- Memory: {len(self.memory.landmarks)} landmarks, {len(self.memory.episodes)} episodes")
        print(f"- SLAM: {self.slam.get_stats()['explored_percent']}% explored")
        print(f"- Mode: {self.state.operating_mode.value}")
        print(f"- Speech: {'Enabled (Finnish)' if USE_SPEECH else 'Disabled'}")

    def gather_telemetry(self, img: np.ndarray, jpeg: bytes) -> Dict[str, Any]:
        """Gather all sensor data into telemetry dict."""
        fp = quick_fp(jpeg)
        delta = fp_delta(self.prev_fp, fp)
        self.prev_fp = fp

        # Person detection
        persons = self.detector.detect(img)

        # Doorway detection
        opening = detect_opening(img)

        # Depth estimation
        depth_data = {"available": False}
        if USE_MIDAS_DEPTH:
            try:
                depth_result = estimate_depth(img)
                depth_data = depth_to_telemetry(depth_result)
            except Exception as e:
                print(f"[Depth] Error: {e}")

        # ARCore pose
        pose = self.arcore.fetch_pose()
        pose_data = {
            "available": pose is not None,
            "tracking": self.arcore.tracking_state,
        }
        if pose:
            pose_data["x"] = round(pose.x, 3)
            pose_data["y"] = round(pose.y, 3)
            pose_data["z"] = round(pose.z, 3)

        # Update SLAM with ARCore pose and depth
        slam_data = {"available": False}
        if self.arcore.tracking_state == "TRACKING":
            arcore_raw = {
                "trackingState": "TRACKING",
                "position": [pose.x, pose.y, pose.z] if pose else [0, 0, 0],
                "rotation": self.arcore.last_rotation or [0, 0, 0, 1],
            }
            self.slam.update_pose(arcore_raw)

            # Update occupancy grid with depth data
            if depth_data.get("available"):
                self.slam.update_map(depth_data)

            # Get SLAM stats for telemetry
            stats = self.slam.get_stats()
            slam_data = {
                "available": True,
                "explored_percent": stats["explored_percent"],
                "free_cells": stats["free_cells"],
                "occupied_cells": stats["occupied_cells"],
            }

        # Phone sensors
        sensors_raw = fetch_sensors()
        sensors = summarize_sensors(sensors_raw)

        return {
            "step": self.state.steps_taken,
            "image_delta": round(delta, 3),
            "persons": persons[:3],
            "opening": opening if opening.get("score", 0) > 0.1 else {"score": 0},
            "depth": depth_data,
            "pose": pose_data,
            "slam": slam_data,
            "sensors": sensors,
            "stall_count": self.state.stall_count,
        }

    def run_step(self, task: str) -> Tuple[bool, Optional[str]]:
        """Run one step of the agent loop."""
        self.state.steps_taken += 1
        self.state.mode = Mode.OBSERVE

        # Get frame
        jpeg = get_mjpeg_frame(STREAM_URL)
        img = decode_jpeg(jpeg)

        # Gather telemetry
        telemetry = self.gather_telemetry(img, jpeg)

        # Update pet state
        self.pet.update_state(telemetry)

        # Get memory context
        pose = self.arcore.last_pose
        memory_context = self.memory.get_context_for_llm(pose)
        pet_state = self.pet.get_state_summary()

        # Record person sightings in memory and comment
        if telemetry["persons"] and pose:
            self.memory.saw_person(pose, {"confidence": telemetry["persons"][0]["conf"]})
            # React to seeing a person
            if self.speaker:
                area = telemetry["persons"][0].get("area_norm", 0)
                distance = "close" if area > 0.1 else "far"
                self.speaker.react_to_person(distance)

        # Record doorways
        if telemetry["opening"].get("score", 0) > OPENING_MIN_SCORE and pose:
            self.memory.found_doorway(pose)

        img_b64 = jpeg_to_b64(jpeg)

        # Plan with LLM
        self.state.mode = Mode.PLAN

        # In autonomous mode, sometimes use pet behaviors directly
        if self.state.operating_mode == OperatingMode.AUTONOMOUS:
            pet_action = self.pet.get_behavior(telemetry)
            if pet_action and pet_action.get("behavior_score", 0) > 0.5:
                print(f"\n[step {self.state.steps_taken}] PET BEHAVIOR: {pet_action.get('note', '')}")
                print(f"  Mood: {pet_state['mood']}, Energy: {pet_state['energy']:.2f}")

                # Comment on behavior
                if self.speaker:
                    behavior_name = pet_action.get("behavior_name", "")
                    mood = pet_action.get("mood", "curious")
                    self.speaker.maybe_comment(
                        action=behavior_name,
                        observation=pet_action.get("note", ""),
                        mood=mood,
                        commentary_type=CommentaryType.ACTION
                    )

                actions = normalize_actions(pet_action)
                actions = validate_actions(actions, self.safety)
                self.state.mode = Mode.ACT
                done, result = execute(actions, self.safety, self.history, self.memory, self.arcore)
                return done, result

        # LLM planning
        try:
            prompt = build_prompt(task, telemetry, self.history, memory_context, pet_state, self.state)
            model_out = ollama_plan(prompt, img_b64)
            actions = normalize_actions(model_out)
            actions = validate_actions(actions, self.safety)

            print(f"\n[step {self.state.steps_taken}] TELEMETRY:")
            print(f"  Persons: {len(telemetry['persons'])}, Opening: {telemetry['opening'].get('score', 0):.2f}")
            print(f"  Depth: {telemetry['depth'].get('clearance', {}).get('path_ahead', 'N/A')}")
            print(f"  Pet: mood={pet_state['mood']}, energy={pet_state['energy']:.2f}")
            print(f"[step {self.state.steps_taken}] MODEL: {json.dumps(model_out, ensure_ascii=False)[:200]}")

        except Exception as e:
            print(f"[step {self.state.steps_taken}] LLM error: {e}")
            actions = []

        # Stall detection
        if actions and all(a.get("action") == "head" for a in actions):
            self.state.stall_count += 1
        elif actions:
            self.state.stall_count = 0

        # Reflex override
        if (not actions) or self.state.stall_count >= 2 or telemetry["image_delta"] < 0.02:
            self.state.mode = Mode.RECOVER

            # Try depth-based navigation first
            depth_reflex = reflex_for_depth(telemetry, self.state.steps_taken)
            if depth_reflex:
                print(f"[step {self.state.steps_taken}] DEPTH REFLEX")
                done, result = execute(depth_reflex, self.safety, self.history, self.memory, self.arcore)
                self.state.stall_count = 0
                return done, result

            # Use pet behaviors as fallback
            pet_action = self.pet.get_behavior(telemetry)
            if pet_action:
                print(f"[step {self.state.steps_taken}] PET FALLBACK: {pet_action.get('note', '')}")
                actions = normalize_actions(pet_action)
                actions = validate_actions(actions, self.safety)

        # Execute
        self.state.mode = Mode.ACT
        if actions:
            done, result = execute(actions, self.safety, self.history, self.memory, self.arcore)
            print(f"[step {self.state.steps_taken}] EXECUTED {len(actions)} action(s)")
            return done, result

        return False, None

    def run_task(self, task: str):
        """Run a task until completion or max steps."""
        self.state.current_task = task
        self.state.operating_mode = OperatingMode.TASK
        self.state.steps_taken = 0
        self.state.stall_count = 0
        self.history = []

        print(f"\n{'='*50}")
        print(f"TASK: {task}")
        print(f"{'='*50}")

        # Announce the task in Finnish
        if self.speaker:
            self.speaker.announce_task(task)

        try:
            do_stop()
            do_head(90, 0)
        except Exception:
            pass

        for _ in range(MAX_STEPS):
            try:
                done, result = self.run_step(task)
                if done:
                    print(f"\n{'='*50}")
                    print(f"TASK COMPLETE: {result}")
                    print(f"{'='*50}")
                    return result
            except KeyboardInterrupt:
                print("\nInterrupted.")
                do_stop()
                return None
            except Exception as e:
                print(f"Error: {e}")
                self.state.error_count += 1
                do_stop()
                time.sleep(0.3)
                if self.state.error_count >= 3:
                    time.sleep(1.0)
                    self.state.error_count = 0

        print("\nMax steps reached.")
        do_stop()
        return None

    def run_autonomous(self):
        """Run in autonomous pet mode until interrupted."""
        self.state.operating_mode = OperatingMode.AUTONOMOUS
        self.state.current_task = "autonomous"
        self.state.steps_taken = 0
        self.history = []

        print("\n" + "="*50)
        print("AUTONOMOUS MODE")
        print("The robot will now act like a curious pet.")
        print("Press Ctrl+C to stop.")
        print("="*50)

        # Greet when starting autonomous mode
        if self.speaker:
            self.speaker.greet()

        try:
            do_stop()
            do_head(90, 0)
        except Exception:
            pass

        while True:
            try:
                done, result = self.run_step("autonomous")
                if done:
                    print(f"Autonomous action complete: {result}")
                time.sleep(0.2)  # Small delay in autonomous mode
            except KeyboardInterrupt:
                print("\nAutonomous mode stopped.")
                do_stop()
                return
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.5)

    def show_status(self):
        """Display current status."""
        pet_state = self.pet.get_state_summary()
        slam_stats = self.slam.get_stats()
        print("\n--- Smart Pet Status ---")
        print(f"Mode: {self.state.operating_mode.value}")
        print(f"Mood: {pet_state['mood']}")
        print(f"Energy: {pet_state['energy']:.2f}")
        print(f"Curiosity: {pet_state['curiosity']:.2f}")
        print(f"Social need: {pet_state['social_need']:.2f}")
        print(f"Cells explored: {pet_state['cells_explored']}")
        print(f"ARCore tracking: {pet_state['tracking_state']}")
        print(f"SLAM explored: {slam_stats['explored_percent']}%")
        print(f"Known rooms: {len(self.memory.find_landmarks_by_type(LandmarkType.ROOM))}")
        print(f"Total landmarks: {len(self.memory.landmarks)}")
        print(f"Episodes recorded: {len(self.memory.episodes)}")

    def show_rooms(self):
        """List known rooms."""
        rooms = self.memory.find_landmarks_by_type(LandmarkType.ROOM)
        print("\n--- Known Rooms ---")
        if not rooms:
            print("No rooms known yet. Explore to discover rooms!")
        for r in rooms:
            print(f"  {r.name}: ({r.pose.x:.2f}, {r.pose.y:.2f}, {r.pose.z:.2f}) - visited {r.times_visited}x")

    def show_slam(self):
        """Display SLAM statistics."""
        stats = self.slam.get_stats()
        print("\n--- SLAM Statistics ---")
        print(f"Grid size: {stats['grid_size']} ({stats['resolution']} per cell)")
        print(f"Explored: {stats['explored_percent']}%")
        print(f"Free cells: {stats['free_cells']}")
        print(f"Occupied cells: {stats['occupied_cells']}")
        print(f"Pose history: {stats['pose_history_length']} points")
        pose = stats['current_pose']
        print(f"Current pose: X={pose['x']:.2f}, Y={pose['y']:.2f}, θ={pose['theta']}°")

    def save_slam_map(self, filename: str = "slam_map_image.png"):
        """Save SLAM map as an image."""
        try:
            img = self.slam.get_map_image(scale=3)
            import cv2
            cv2.imwrite(filename, img)
            print(f"SLAM map saved to: {filename}")
        except Exception as e:
            print(f"Failed to save map image: {e}")

        # Also save the raw map data
        self.slam.save()
        print("SLAM data saved to: slam_map.json")


# ========================
# Main Entry Point
# ========================

def main():
    print("\n" + "="*60)
    print("  RAMBLEBOT SMART PET AGENT")
    print("  - MiDaS Depth | ARCore Pose | Semantic Memory")
    print("  - YOLO Person Detection | Curious Pet Behaviors")
    print("="*60)
    print(f"\nGateway: {GATEWAY_BASE}")
    print(f"Ollama: {OLLAMA_BASE} ({MODEL})")
    print(f"MiDaS Depth: {'Enabled' if USE_MIDAS_DEPTH else 'Disabled'}")
    print("\nCommands:")
    print("  <task>    - Execute a task (e.g., 'go to kitchen')")
    print("  /auto     - Switch to autonomous mode")
    print("  /task     - Switch to task mode")
    print("  /status   - Show pet status")
    print("  /rooms    - List known rooms")
    print("  /memory   - Query memory")
    print("  /slam     - Show SLAM stats")
    print("  /savemap  - Save SLAM map image")
    print("  /explore  - Start exploration")
    print("  /q, exit  - Quit")
    print()

    agent = SmartPetAgent()

    while True:
        try:
            cmd = input("> ").strip()
            if not cmd:
                continue

            if cmd in ("/q", "/quit", "exit", "quit"):
                print("Goodbye!")
                break

            if cmd == "/auto":
                agent.run_autonomous()
                continue

            if cmd == "/task":
                agent.state.operating_mode = OperatingMode.TASK
                print("Switched to task mode. Enter a task:")
                continue

            if cmd == "/status":
                agent.show_status()
                continue

            if cmd == "/rooms":
                agent.show_rooms()
                continue

            if cmd == "/slam":
                agent.show_slam()
                continue

            if cmd == "/savemap":
                agent.save_slam_map()
                continue

            if cmd == "/explore":
                agent.run_task("Explore the area and remember what you find")
                continue

            if cmd.startswith("/memory"):
                query = cmd[7:].strip() or "what do I know"
                pose = agent.arcore.fetch_pose()
                result = agent.memory.query_memory(query, pose)
                print(f"Memory: {result['answer']}")
                continue

            # Treat as task
            agent.run_task(cmd)

        except KeyboardInterrupt:
            print("\nInterrupted. Type '/q' to quit.")
            try:
                do_stop()
            except Exception:
                pass
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
