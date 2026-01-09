"""
Semantic Memory System for Ramblebot Smart Pet

This module provides persistent spatial and episodic memory for the robot:
- Landmark storage (rooms, objects, people locations)
- Spatial relationships (room adjacency, distances)
- Episodic memory (what happened where and when)
- Memory retrieval for navigation and task planning

The robot learns its environment over time and can recall:
- "Where is the kitchen?"
- "When did I last see a person?"
- "What's near the living room?"
"""

import json
import time
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from enum import Enum
import hashlib


class LandmarkType(Enum):
    ROOM = "room"
    OBJECT = "object"
    PERSON = "person"
    DOORWAY = "doorway"
    HAZARD = "hazard"
    CUSTOM = "custom"


@dataclass
class Pose:
    """Robot pose from ARCore (position + orientation)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    timestamp: float = 0.0

    def distance_to(self, other: "Pose") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z,
                "qx": self.qx, "qy": self.qy, "qz": self.qz, "qw": self.qw,
                "timestamp": self.timestamp}

    @staticmethod
    def from_dict(d: Dict) -> "Pose":
        return Pose(
            x=d.get("x", 0), y=d.get("y", 0), z=d.get("z", 0),
            qx=d.get("qx", 0), qy=d.get("qy", 0), qz=d.get("qz", 0), qw=d.get("qw", 1),
            timestamp=d.get("timestamp", 0)
        )


@dataclass
class Landmark:
    """A remembered location or entity."""
    id: str
    name: str
    landmark_type: str  # LandmarkType value
    pose: Pose
    confidence: float = 0.5  # 0-1, how confident we are this is correct
    last_seen: float = 0.0   # timestamp
    times_visited: int = 0
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "landmark_type": self.landmark_type,
            "pose": self.pose.to_dict(),
            "confidence": self.confidence,
            "last_seen": self.last_seen,
            "times_visited": self.times_visited,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict) -> "Landmark":
        return Landmark(
            id=d["id"],
            name=d["name"],
            landmark_type=d["landmark_type"],
            pose=Pose.from_dict(d["pose"]),
            confidence=d.get("confidence", 0.5),
            last_seen=d.get("last_seen", 0),
            times_visited=d.get("times_visited", 0),
            description=d.get("description", ""),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Episode:
    """An episodic memory - what happened at a location."""
    id: str
    timestamp: float
    pose: Pose
    event_type: str  # "saw_person", "entered_room", "found_object", etc.
    description: str
    landmark_ids: List[str] = field(default_factory=list)  # Related landmarks
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "pose": self.pose.to_dict(),
            "event_type": self.event_type,
            "description": self.description,
            "landmark_ids": self.landmark_ids,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict) -> "Episode":
        return Episode(
            id=d["id"],
            timestamp=d["timestamp"],
            pose=Pose.from_dict(d["pose"]),
            event_type=d["event_type"],
            description=d["description"],
            landmark_ids=d.get("landmark_ids", []),
            metadata=d.get("metadata", {}),
        )


class SemanticMemory:
    """
    Persistent semantic memory for the robot.

    Stores:
    - Landmarks (rooms, objects, people locations)
    - Spatial graph (adjacency, distances)
    - Episodes (events that happened)
    """

    def __init__(self, storage_path: str = "robot_memory.json"):
        self.storage_path = Path(storage_path)
        self.landmarks: Dict[str, Landmark] = {}
        self.episodes: List[Episode] = []
        self.adjacency: Dict[str, List[str]] = {}  # landmark_id -> [adjacent_ids]
        self.current_room: Optional[str] = None
        self.last_pose: Optional[Pose] = None

        # Load existing memory
        self._load()

    def _load(self):
        """Load memory from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)

                for ld in data.get("landmarks", []):
                    lm = Landmark.from_dict(ld)
                    self.landmarks[lm.id] = lm

                for ed in data.get("episodes", []):
                    self.episodes.append(Episode.from_dict(ed))

                self.adjacency = data.get("adjacency", {})
                self.current_room = data.get("current_room")

                print(f"[Memory] Loaded {len(self.landmarks)} landmarks, {len(self.episodes)} episodes")
            except Exception as e:
                print(f"[Memory] Failed to load: {e}")

    def save(self):
        """Persist memory to disk."""
        data = {
            "landmarks": [lm.to_dict() for lm in self.landmarks.values()],
            "episodes": [ep.to_dict() for ep in self.episodes[-500:]],  # Keep last 500
            "adjacency": self.adjacency,
            "current_room": self.current_room,
            "saved_at": time.time(),
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self, prefix: str, name: str) -> str:
        """Generate a unique ID for a landmark/episode."""
        hash_input = f"{prefix}_{name}_{time.time()}"
        return f"{prefix}_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"

    # ========================
    # Landmark Management
    # ========================

    def add_landmark(
        self,
        name: str,
        landmark_type: LandmarkType,
        pose: Pose,
        description: str = "",
        tags: List[str] = None,
        metadata: Dict = None,
    ) -> Landmark:
        """Add a new landmark to memory."""
        lm_id = self._generate_id("lm", name)
        lm = Landmark(
            id=lm_id,
            name=name,
            landmark_type=landmark_type.value,
            pose=pose,
            confidence=0.6,
            last_seen=time.time(),
            times_visited=1,
            description=description,
            tags=tags or [],
            metadata=metadata or {},
        )
        self.landmarks[lm_id] = lm
        self.save()
        print(f"[Memory] Added landmark: {name} ({landmark_type.value}) at ({pose.x:.2f}, {pose.y:.2f}, {pose.z:.2f})")
        return lm

    def update_landmark(self, landmark_id: str, pose: Optional[Pose] = None, **kwargs) -> Optional[Landmark]:
        """Update an existing landmark."""
        if landmark_id not in self.landmarks:
            return None

        lm = self.landmarks[landmark_id]
        lm.last_seen = time.time()
        lm.times_visited += 1

        # Update pose with weighted average (current has more weight if higher confidence)
        if pose:
            alpha = 0.3  # New observation weight
            lm.pose.x = lm.pose.x * (1 - alpha) + pose.x * alpha
            lm.pose.y = lm.pose.y * (1 - alpha) + pose.y * alpha
            lm.pose.z = lm.pose.z * (1 - alpha) + pose.z * alpha
            lm.pose.timestamp = pose.timestamp
            # Increase confidence when we see it again
            lm.confidence = min(1.0, lm.confidence + 0.05)

        for key, value in kwargs.items():
            if hasattr(lm, key):
                setattr(lm, key, value)

        self.save()
        return lm

    def find_landmark_by_name(self, name: str, fuzzy: bool = True) -> Optional[Landmark]:
        """Find a landmark by name (exact or fuzzy match)."""
        name_lower = name.lower()

        # Exact match first
        for lm in self.landmarks.values():
            if lm.name.lower() == name_lower:
                return lm

        # Fuzzy match
        if fuzzy:
            for lm in self.landmarks.values():
                if name_lower in lm.name.lower() or lm.name.lower() in name_lower:
                    return lm

        return None

    def find_landmarks_by_type(self, landmark_type: LandmarkType) -> List[Landmark]:
        """Get all landmarks of a given type."""
        return [lm for lm in self.landmarks.values() if lm.landmark_type == landmark_type.value]

    def find_nearest_landmark(
        self,
        pose: Pose,
        landmark_type: Optional[LandmarkType] = None,
        max_distance: float = float('inf')
    ) -> Optional[Tuple[Landmark, float]]:
        """Find the nearest landmark to a pose."""
        best = None
        best_dist = max_distance

        for lm in self.landmarks.values():
            if landmark_type and lm.landmark_type != landmark_type.value:
                continue
            dist = pose.distance_to(lm.pose)
            if dist < best_dist:
                best_dist = dist
                best = lm

        if best:
            return (best, best_dist)
        return None

    def get_landmarks_near(self, pose: Pose, radius: float = 2.0) -> List[Tuple[Landmark, float]]:
        """Get all landmarks within a radius of a pose."""
        results = []
        for lm in self.landmarks.values():
            dist = pose.distance_to(lm.pose)
            if dist <= radius:
                results.append((lm, dist))
        results.sort(key=lambda x: x[1])
        return results

    # ========================
    # Room / Location Tracking
    # ========================

    def enter_room(self, room_name: str, pose: Pose) -> Landmark:
        """Record entering a room."""
        lm = self.find_landmark_by_name(room_name)

        if lm and lm.landmark_type == LandmarkType.ROOM.value:
            self.update_landmark(lm.id, pose)
        else:
            lm = self.add_landmark(
                name=room_name,
                landmark_type=LandmarkType.ROOM,
                pose=pose,
                description=f"Room: {room_name}",
            )

        # Update adjacency if we came from another room
        if self.current_room and self.current_room != lm.id:
            self._add_adjacency(self.current_room, lm.id)

        self.current_room = lm.id
        self._record_episode("entered_room", f"Entered {room_name}", pose, [lm.id])

        return lm

    def _add_adjacency(self, from_id: str, to_id: str):
        """Record that two landmarks are adjacent."""
        if from_id not in self.adjacency:
            self.adjacency[from_id] = []
        if to_id not in self.adjacency[from_id]:
            self.adjacency[from_id].append(to_id)

        if to_id not in self.adjacency:
            self.adjacency[to_id] = []
        if from_id not in self.adjacency[to_id]:
            self.adjacency[to_id].append(from_id)

    def get_adjacent_rooms(self, room_id: str) -> List[Landmark]:
        """Get rooms adjacent to a given room."""
        adj_ids = self.adjacency.get(room_id, [])
        return [self.landmarks[lid] for lid in adj_ids if lid in self.landmarks]

    # ========================
    # Episode Memory
    # ========================

    def _record_episode(
        self,
        event_type: str,
        description: str,
        pose: Pose,
        landmark_ids: List[str] = None,
        metadata: Dict = None
    ) -> Episode:
        """Record an episodic memory."""
        ep = Episode(
            id=self._generate_id("ep", event_type),
            timestamp=time.time(),
            pose=pose,
            event_type=event_type,
            description=description,
            landmark_ids=landmark_ids or [],
            metadata=metadata or {},
        )
        self.episodes.append(ep)
        # Keep episodes bounded
        if len(self.episodes) > 1000:
            self.episodes = self.episodes[-500:]
        return ep

    def record_observation(
        self,
        event_type: str,
        description: str,
        pose: Pose,
        related_landmarks: List[str] = None,
        metadata: Dict = None
    ):
        """Record any observation (person seen, object found, etc.)."""
        self._record_episode(event_type, description, pose, related_landmarks, metadata)
        self.last_pose = pose
        self.save()

    def saw_person(self, pose: Pose, person_info: Dict = None):
        """Record seeing a person."""
        self._record_episode(
            "saw_person",
            f"Saw person at ({pose.x:.2f}, {pose.y:.2f})",
            pose,
            metadata=person_info or {}
        )
        self.save()

    def found_doorway(self, pose: Pose, doorway_info: Dict = None):
        """Record finding a doorway."""
        # Check if we already know about a doorway nearby
        existing = self.find_nearest_landmark(pose, LandmarkType.DOORWAY, max_distance=1.0)

        if existing:
            lm, dist = existing
            self.update_landmark(lm.id, pose)
        else:
            lm = self.add_landmark(
                name=f"doorway_{len([l for l in self.landmarks.values() if l.landmark_type == 'doorway']) + 1}",
                landmark_type=LandmarkType.DOORWAY,
                pose=pose,
                metadata=doorway_info or {}
            )

        self._record_episode("found_doorway", "Found doorway", pose, [lm.id])

    def get_recent_episodes(self, count: int = 10, event_type: str = None) -> List[Episode]:
        """Get recent episodes, optionally filtered by type."""
        eps = self.episodes
        if event_type:
            eps = [e for e in eps if e.event_type == event_type]
        return eps[-count:]

    def last_seen_person(self) -> Optional[Episode]:
        """Get the most recent person sighting."""
        person_eps = [e for e in self.episodes if e.event_type == "saw_person"]
        return person_eps[-1] if person_eps else None

    # ========================
    # Query Interface (for LLM)
    # ========================

    def query_memory(self, query: str, current_pose: Optional[Pose] = None) -> Dict[str, Any]:
        """
        Answer a natural language query about memory.
        Returns structured data the LLM can use.
        """
        query_lower = query.lower()
        result = {"query": query, "answer": "", "relevant_data": []}

        # Location queries
        if "where" in query_lower:
            # "where is the kitchen" -> find room
            for room_name in ["kitchen", "keittiö", "living", "olohuone", "bedroom",
                             "makuuhuone", "bathroom", "kylpyhuone", "hallway", "eteinen"]:
                if room_name in query_lower:
                    lm = self.find_landmark_by_name(room_name)
                    if lm:
                        result["answer"] = f"{lm.name} is at position ({lm.pose.x:.2f}, {lm.pose.y:.2f})"
                        if current_pose:
                            dist = current_pose.distance_to(lm.pose)
                            result["answer"] += f", {dist:.1f}m away"
                        result["relevant_data"].append(lm.to_dict())
                    else:
                        result["answer"] = f"I don't know where {room_name} is yet."
                    return result

        # Person queries
        if "person" in query_lower or "ihminen" in query_lower or "anyone" in query_lower:
            last = self.last_seen_person()
            if last:
                age = time.time() - last.timestamp
                if age < 60:
                    time_str = f"{int(age)} seconds ago"
                elif age < 3600:
                    time_str = f"{int(age/60)} minutes ago"
                else:
                    time_str = f"{int(age/3600)} hours ago"
                result["answer"] = f"Last saw a person {time_str} at ({last.pose.x:.2f}, {last.pose.y:.2f})"
                result["relevant_data"].append(last.to_dict())
            else:
                result["answer"] = "I haven't seen anyone yet."
            return result

        # Nearby queries
        if "near" in query_lower or "lähellä" in query_lower:
            if current_pose:
                nearby = self.get_landmarks_near(current_pose, radius=3.0)
                if nearby:
                    names = [f"{lm.name} ({dist:.1f}m)" for lm, dist in nearby]
                    result["answer"] = f"Nearby: {', '.join(names)}"
                    result["relevant_data"] = [lm.to_dict() for lm, _ in nearby]
                else:
                    result["answer"] = "Nothing known nearby."
            else:
                result["answer"] = "I don't know my current position."
            return result

        # Room list
        if "room" in query_lower or "huone" in query_lower:
            rooms = self.find_landmarks_by_type(LandmarkType.ROOM)
            if rooms:
                result["answer"] = f"Known rooms: {', '.join([r.name for r in rooms])}"
                result["relevant_data"] = [r.to_dict() for r in rooms]
            else:
                result["answer"] = "I don't know any rooms yet."
            return result

        # Default: return recent context
        result["answer"] = "I don't understand the query. Here's my recent memory."
        result["relevant_data"] = [e.to_dict() for e in self.get_recent_episodes(5)]
        return result

    def get_context_for_llm(self, current_pose: Optional[Pose] = None) -> Dict[str, Any]:
        """
        Generate a context summary for the LLM to use in planning.
        """
        context = {
            "known_rooms": [lm.name for lm in self.find_landmarks_by_type(LandmarkType.ROOM)],
            "known_doorways": len(self.find_landmarks_by_type(LandmarkType.DOORWAY)),
            "current_room": None,
            "nearby_landmarks": [],
            "recent_events": [],
            "last_person_seen": None,
        }

        # Current room
        if self.current_room and self.current_room in self.landmarks:
            context["current_room"] = self.landmarks[self.current_room].name

        # Nearby landmarks
        if current_pose:
            nearby = self.get_landmarks_near(current_pose, radius=3.0)
            context["nearby_landmarks"] = [
                {"name": lm.name, "type": lm.landmark_type, "distance": round(dist, 2)}
                for lm, dist in nearby[:5]
            ]

        # Recent events
        for ep in self.get_recent_episodes(5):
            age_sec = time.time() - ep.timestamp
            context["recent_events"].append({
                "type": ep.event_type,
                "description": ep.description,
                "seconds_ago": round(age_sec, 0),
            })

        # Last person
        last_person = self.last_seen_person()
        if last_person:
            context["last_person_seen"] = {
                "seconds_ago": round(time.time() - last_person.timestamp, 0),
                "position": {"x": last_person.pose.x, "y": last_person.pose.y},
            }

        return context


# Global instance for easy import
_memory_instance: Optional[SemanticMemory] = None


def get_memory(storage_path: str = "robot_memory.json") -> SemanticMemory:
    """Get or create the global memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = SemanticMemory(storage_path)
    return _memory_instance


# Test
if __name__ == "__main__":
    mem = SemanticMemory("test_memory.json")

    # Simulate robot exploring
    pose1 = Pose(x=0, y=0, z=0.5, timestamp=time.time())
    mem.enter_room("living room", pose1)

    pose2 = Pose(x=2.5, y=1.0, z=0.5, timestamp=time.time())
    mem.found_doorway(pose2, {"direction": "north"})

    pose3 = Pose(x=5.0, y=1.5, z=0.5, timestamp=time.time())
    mem.enter_room("kitchen", pose3)

    pose4 = Pose(x=5.5, y=2.0, z=0.5, timestamp=time.time())
    mem.saw_person(pose4, {"confidence": 0.85})

    # Query
    print("\n--- Memory Queries ---")
    print(mem.query_memory("where is the kitchen"))
    print(mem.query_memory("when did I last see a person"))
    print(mem.query_memory("what rooms do I know"))

    # Context for LLM
    print("\n--- LLM Context ---")
    ctx = mem.get_context_for_llm(pose4)
    print(json.dumps(ctx, indent=2))
