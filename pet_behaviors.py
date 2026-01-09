"""
Curious Pet Behavior System for Ramblebot

This module implements autonomous "pet-like" behaviors:
- Curiosity: Investigate interesting things (movement, sounds, new areas)
- Patrol: Periodically check known areas
- Social: React to people, approach when appropriate
- Exploration: Discover and map new areas
- Idle behaviors: Head movements, occasional sounds

The robot acts like a curious, friendly pet when not given explicit tasks.
"""

import random
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
import requests

from semantic_memory import SemanticMemory, Pose, LandmarkType, get_memory


# ========================
# ARCore Integration
# ========================

class ARCoreClient:
    """Client to fetch ARCore pose data from Android gateway."""

    def __init__(self, gateway_base: str = "http://192.168.86.35:8765"):
        self.gateway_base = gateway_base
        self.arcore_url = f"{gateway_base}/arcore.json"
        self.session = requests.Session()
        self.last_pose: Optional[Pose] = None
        self.tracking_state: str = "UNKNOWN"

    def fetch_pose(self, timeout: float = 5.0) -> Optional[Pose]:
        """Fetch current pose from ARCore."""
        try:
            r = self.session.get(self.arcore_url, timeout=timeout)
            r.raise_for_status()
            data = r.json()

            self.tracking_state = data.get("trackingState", "UNKNOWN")

            if self.tracking_state != "TRACKING":
                return self.last_pose  # Return last known pose

            # Extract position and rotation
            pos = data.get("position", {})
            rot = data.get("rotation", {})

            pose = Pose(
                x=pos.get("x", 0),
                y=pos.get("y", 0),
                z=pos.get("z", 0),
                qx=rot.get("x", 0),
                qy=rot.get("y", 0),
                qz=rot.get("z", 0),
                qw=rot.get("w", 1),
                timestamp=time.time(),
            )

            self.last_pose = pose
            return pose

        except Exception as e:
            print(f"[ARCore] Failed to fetch pose: {e}")
            return self.last_pose

    def get_yaw(self, pose: Pose) -> float:
        """Extract yaw (rotation around vertical axis) from quaternion."""
        # Convert quaternion to yaw angle
        siny_cosp = 2 * (pose.qw * pose.qz + pose.qx * pose.qy)
        cosy_cosp = 1 - 2 * (pose.qy * pose.qy + pose.qz * pose.qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def distance_moved(self, pose1: Pose, pose2: Pose) -> float:
        """Calculate distance between two poses."""
        if pose1 is None or pose2 is None:
            return 0
        return pose1.distance_to(pose2)


# ========================
# Pet Mood System
# ========================

class Mood(Enum):
    CURIOUS = "curious"      # Wants to explore, investigate
    PLAYFUL = "playful"      # Active, responsive, bouncy movements
    CALM = "calm"            # Relaxed, slow movements
    ALERT = "alert"          # Something caught attention
    TIRED = "tired"          # Less active, slower responses
    SOCIAL = "social"        # Wants to be near people


@dataclass
class PetState:
    """Current state of the pet robot."""
    mood: Mood = Mood.CALM
    energy: float = 1.0          # 0-1, decreases over time without rest
    curiosity: float = 0.5       # 0-1, increases when nothing interesting
    social_need: float = 0.3     # 0-1, increases without people
    last_person_time: float = 0  # When we last saw a person
    last_activity_time: float = 0
    exploration_score: float = 0 # How much we've explored
    idle_since: float = 0        # How long idle


# ========================
# Behavior Triggers
# ========================

@dataclass
class BehaviorTrigger:
    """Condition that triggers a behavior."""
    name: str
    check: Callable[[Dict[str, Any], PetState], float]  # Returns 0-1 urgency
    cooldown: float = 30.0  # Seconds between triggers
    last_triggered: float = 0


class PetBehaviors:
    """
    Autonomous pet-like behavior generator.

    Given perception data and pet state, suggests appropriate actions.
    """

    def __init__(
        self,
        memory: SemanticMemory = None,
        arcore: ARCoreClient = None,
        gateway_base: str = "http://192.168.86.35:8765"
    ):
        self.memory = memory or get_memory()
        self.arcore = arcore or ARCoreClient(gateway_base)
        self.state = PetState()
        self.state.last_activity_time = time.time()
        self.state.idle_since = time.time()

        # Behavior cooldowns
        self.last_behaviors: Dict[str, float] = {}

        # Exploration targets
        self.exploration_queue: List[Tuple[float, float]] = []
        self.visited_cells: set = set()  # (grid_x, grid_y) cells we've visited
        self.grid_size = 0.5  # meters per grid cell

    def update_state(self, telemetry: Dict[str, Any], dt: float = 0.5):
        """Update pet state based on telemetry and time."""
        # Decay energy over time (restore when idle)
        if self.state.mood == Mood.TIRED:
            self.state.energy = min(1.0, self.state.energy + 0.01 * dt)
        else:
            self.state.energy = max(0.0, self.state.energy - 0.002 * dt)

        # Curiosity increases when nothing interesting
        persons = telemetry.get("persons", [])
        if not persons:
            self.state.curiosity = min(1.0, self.state.curiosity + 0.01 * dt)
            self.state.social_need = min(1.0, self.state.social_need + 0.005 * dt)
        else:
            self.state.curiosity = max(0.0, self.state.curiosity - 0.1)
            self.state.social_need = max(0.0, self.state.social_need - 0.2)
            self.state.last_person_time = time.time()

        # Update mood based on state
        self._update_mood()

        # Update exploration map
        pose = self.arcore.fetch_pose()
        if pose:
            cell = (int(pose.x / self.grid_size), int(pose.z / self.grid_size))
            self.visited_cells.add(cell)
            self.memory.last_pose = pose

    def _update_mood(self):
        """Determine current mood from state variables."""
        if self.state.energy < 0.2:
            self.state.mood = Mood.TIRED
        elif self.state.social_need > 0.7:
            self.state.mood = Mood.SOCIAL
        elif self.state.curiosity > 0.6:
            self.state.mood = Mood.CURIOUS
        elif self.state.energy > 0.7 and random.random() < 0.3:
            self.state.mood = Mood.PLAYFUL
        else:
            self.state.mood = Mood.CALM

    def get_behavior(self, telemetry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get the next autonomous behavior to execute.

        Returns:
            Action dict or None if no behavior triggered.
        """
        now = time.time()
        self.update_state(telemetry)

        # Get current pose for spatial awareness
        pose = self.arcore.last_pose

        # Score each potential behavior
        behaviors = []

        # 1. React to person (highest priority)
        person_action = self._react_to_person(telemetry, pose)
        if person_action:
            behaviors.append((0.9, "react_person", person_action))

        # 2. Investigate movement/change
        if telemetry.get("image_delta", 0) > 0.15:
            investigate_action = self._investigate_change(telemetry)
            if investigate_action:
                behaviors.append((0.7, "investigate", investigate_action))

        # 3. Curious exploration (when bored)
        if self.state.mood == Mood.CURIOUS and self._can_trigger("explore", 60):
            explore_action = self._explore_new_area(telemetry, pose)
            if explore_action:
                behaviors.append((0.5 * self.state.curiosity, "explore", explore_action))

        # 4. Check known locations
        if self.state.mood == Mood.SOCIAL and self._can_trigger("patrol", 120):
            patrol_action = self._patrol_known_areas(pose)
            if patrol_action:
                behaviors.append((0.4 * self.state.social_need, "patrol", patrol_action))

        # 5. Idle behaviors (head movements, "looking around")
        if self._can_trigger("idle", 30):
            idle_action = self._idle_behavior()
            if idle_action:
                behaviors.append((0.2, "idle", idle_action))

        # Select highest scoring behavior
        if behaviors:
            behaviors.sort(key=lambda x: -x[0])
            score, name, action = behaviors[0]

            # Only trigger if score is high enough
            if score > 0.3:
                self.last_behaviors[name] = now
                self.state.last_activity_time = now
                action["behavior_name"] = name
                action["behavior_score"] = round(score, 2)
                action["mood"] = self.state.mood.value
                return action

        return None

    def _can_trigger(self, behavior_name: str, cooldown: float) -> bool:
        """Check if a behavior can be triggered (not in cooldown)."""
        last = self.last_behaviors.get(behavior_name, 0)
        return time.time() - last >= cooldown

    def _react_to_person(self, telemetry: Dict[str, Any], pose: Optional[Pose]) -> Optional[Dict[str, Any]]:
        """React when a person is detected."""
        persons = telemetry.get("persons", [])
        if not persons:
            return None

        person = persons[0]  # Focus on closest/largest person
        cx = person.get("center_x", 0.5)
        conf = person.get("conf", 0)
        area = person.get("area_norm", 0)

        # Record in memory
        if pose:
            self.memory.saw_person(pose, {"confidence": conf, "area": area})

        # Action: turn to face person, approach slightly
        err = cx - 0.5
        actions = []

        # Look at person
        if abs(err) > 0.1:
            # Turn toward person
            turn_dir = 1 if err > 0 else -1
            turn_speed = int(120 + 60 * min(1, abs(err) * 2))
            turn_ms = int(200 + 150 * min(1, abs(err) * 2))
            actions.append({
                "action": "drive",
                "l": turn_speed * turn_dir,
                "r": -turn_speed * turn_dir,
                "ms": turn_ms,
                "note": "turn to face person"
            })

        # If person is small (far away) and we're social, approach
        if area < 0.1 and self.state.mood == Mood.SOCIAL:
            actions.append({
                "action": "drive",
                "l": 100, "r": 100, "ms": 400,
                "note": "approach person"
            })

        if actions:
            return {"actions": actions, "note": f"Reacting to person (conf={conf:.2f})"}
        return None

    def _investigate_change(self, telemetry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Investigate detected movement or change."""
        delta = telemetry.get("image_delta", 0)

        if delta < 0.15:
            return None

        # Something changed - look around
        actions = [
            {"action": "head", "pos": random.choice([70, 90, 110]), "speed": 3, "note": "look at change"}
        ]

        # If change is significant, turn toward it
        if delta > 0.25:
            turn_dir = random.choice([-1, 1])
            actions.append({
                "action": "drive",
                "l": 100 * turn_dir, "r": -100 * turn_dir, "ms": 300,
                "note": "turn to investigate"
            })

        return {"actions": actions, "note": "Investigating change"}

    def _explore_new_area(self, telemetry: Dict[str, Any], pose: Optional[Pose]) -> Optional[Dict[str, Any]]:
        """Explore areas we haven't visited."""
        depth = telemetry.get("depth", {})
        clearance = depth.get("clearance", {})

        # Check if path is clear ahead
        path_ahead = clearance.get("path_ahead", "unknown")
        clearest = clearance.get("clearest_direction", "center")

        actions = []

        if path_ahead == "clear":
            # Move forward to explore
            actions = [
                {"action": "head", "pos": 90, "speed": 0, "note": "center head"},
                {"action": "drive", "l": 130, "r": 130, "ms": 600, "note": "explore forward"},
                {"action": "stop"},
            ]
        elif path_ahead == "obstacle_near":
            # Turn toward clearest direction
            if clearest == "left":
                actions.append({
                    "action": "drive",
                    "l": -120, "r": 120, "ms": 400,
                    "note": "turn left (clearer)"
                })
            elif clearest == "right":
                actions.append({
                    "action": "drive",
                    "l": 120, "r": -120, "ms": 400,
                    "note": "turn right (clearer)"
                })
            else:
                # Turn randomly
                turn_dir = random.choice([-1, 1])
                actions.append({
                    "action": "drive",
                    "l": 120 * turn_dir, "r": -120 * turn_dir, "ms": 350,
                    "note": "turn to find path"
                })
        else:
            # Blocked - turn around
            turn_dir = random.choice([-1, 1])
            actions = [
                {"action": "drive", "l": 140 * turn_dir, "r": -140 * turn_dir, "ms": 600, "note": "blocked, turn around"},
                {"action": "stop"},
            ]

        if actions:
            return {"actions": actions, "note": f"Exploring (mood: {self.state.mood.value})"}
        return None

    def _patrol_known_areas(self, pose: Optional[Pose]) -> Optional[Dict[str, Any]]:
        """Visit known rooms to check for people."""
        rooms = self.memory.find_landmarks_by_type(LandmarkType.ROOM)
        if not rooms:
            return None

        # Find room we haven't visited in a while
        rooms_by_time = sorted(rooms, key=lambda r: r.last_seen)
        target = rooms_by_time[0]  # Oldest visited

        actions = []

        if pose:
            # Calculate direction to target
            dx = target.pose.x - pose.x
            dz = target.pose.z - pose.z
            dist = math.sqrt(dx*dx + dz*dz)

            # Get angle to target
            target_yaw = math.atan2(dx, dz)
            current_yaw = self.arcore.get_yaw(pose)
            angle_diff = target_yaw - current_yaw

            # Normalize angle to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Turn toward target
            if abs(angle_diff) > 0.3:  # >17 degrees
                turn_dir = 1 if angle_diff > 0 else -1
                turn_ms = int(min(800, abs(angle_diff) * 300))
                actions.append({
                    "action": "drive",
                    "l": 130 * turn_dir, "r": -130 * turn_dir, "ms": turn_ms,
                    "note": f"turn toward {target.name}"
                })
            else:
                # Move toward target
                move_ms = int(min(800, dist * 200))
                actions.append({
                    "action": "drive",
                    "l": 120, "r": 120, "ms": move_ms,
                    "note": f"move toward {target.name}"
                })
        else:
            # No pose - just do random patrol
            turn_dir = random.choice([-1, 1])
            actions = [
                {"action": "drive", "l": 120, "r": 120, "ms": 500, "note": "patrol forward"},
                {"action": "drive", "l": 120 * turn_dir, "r": -120 * turn_dir, "ms": 400, "note": "patrol turn"},
            ]

        if actions:
            return {"actions": actions, "note": f"Patrolling toward {target.name if rooms else 'area'}"}
        return None

    def _idle_behavior(self) -> Optional[Dict[str, Any]]:
        """Generate idle "pet-like" behavior."""
        idle_type = random.choice(["look_around", "head_tilt", "small_turn"])

        if idle_type == "look_around":
            # Look left and right
            positions = [60, 90, 120, 90]
            random.shuffle(positions[:3])
            actions = [
                {"action": "head", "pos": positions[0], "speed": 2, "note": "look around"},
            ]
            return {"actions": actions, "note": "Looking around (idle)"}

        elif idle_type == "head_tilt":
            # Curious head tilt
            pos = random.choice([70, 75, 105, 110])
            return {
                "actions": [
                    {"action": "head", "pos": pos, "speed": 1, "note": "curious tilt"},
                ],
                "note": "Curious head tilt"
            }

        elif idle_type == "small_turn":
            # Small random turn
            turn_dir = random.choice([-1, 1])
            return {
                "actions": [
                    {"action": "drive", "l": 80 * turn_dir, "r": -80 * turn_dir, "ms": 200, "note": "small turn"},
                    {"action": "head", "pos": 90, "speed": 0, "note": "center"},
                ],
                "note": "Idle movement"
            }

        return None

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of pet state for LLM context."""
        return {
            "mood": self.state.mood.value,
            "energy": round(self.state.energy, 2),
            "curiosity": round(self.state.curiosity, 2),
            "social_need": round(self.state.social_need, 2),
            "seconds_since_person": round(time.time() - self.state.last_person_time, 0) if self.state.last_person_time else None,
            "cells_explored": len(self.visited_cells),
            "tracking_state": self.arcore.tracking_state,
        }


# Test
if __name__ == "__main__":
    print("Testing PetBehaviors...")

    pet = PetBehaviors()

    # Simulate telemetry
    test_telemetry = {
        "persons": [],
        "image_delta": 0.05,
        "depth": {
            "clearance": {
                "path_ahead": "clear",
                "clearest_direction": "center"
            }
        }
    }

    # Test several cycles
    for i in range(10):
        behavior = pet.get_behavior(test_telemetry)
        if behavior:
            print(f"\n[Cycle {i}] Behavior: {behavior['note']}")
            print(f"  State: {pet.get_state_summary()}")
        else:
            print(f"[Cycle {i}] No behavior triggered")
        time.sleep(0.5)

    # Test with person detected
    print("\n--- Person detected ---")
    test_telemetry["persons"] = [{"center_x": 0.7, "conf": 0.85, "area_norm": 0.05}]
    behavior = pet.get_behavior(test_telemetry)
    if behavior:
        print(f"Behavior: {behavior['note']}")
        print(f"Actions: {behavior.get('actions', [])}")
