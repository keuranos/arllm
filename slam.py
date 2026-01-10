"""
Simple SLAM (Simultaneous Localization and Mapping) for Ramblebot

Combines:
- ARCore pose tracking (where am I?)
- MiDaS depth estimation (what's in front of me?)
- Occupancy grid building (map of free/occupied space)

This is a simplified 2D SLAM suitable for indoor robot navigation.
"""

import math
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


@dataclass
class Pose2D:
    """2D pose (x, y, theta)."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # radians, 0 = forward (+Z in ARCore)
    timestamp: float = 0.0


@dataclass
class OccupancyGrid:
    """
    2D occupancy grid map.

    Values:
    - -1: Unknown
    - 0: Free (robot can pass)
    - 100: Occupied (obstacle)
    """
    resolution: float = 0.1  # meters per cell
    width: int = 200         # cells
    height: int = 200        # cells
    origin_x: float = -10.0  # world X of grid origin (meters)
    origin_y: float = -10.0  # world Y of grid origin (meters)
    data: np.ndarray = field(default_factory=lambda: np.full((200, 200), -1, dtype=np.int8))

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell."""
        gx = int((wx - self.origin_x) / self.resolution)
        gy = int((wy - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid cell to world coordinates."""
        wx = self.origin_x + gx * self.resolution
        wy = self.origin_y + gy * self.resolution
        return wx, wy

    def is_valid(self, gx: int, gy: int) -> bool:
        """Check if grid cell is within bounds."""
        return 0 <= gx < self.width and 0 <= gy < self.height

    def get(self, gx: int, gy: int) -> int:
        """Get cell value (-1=unknown, 0=free, 100=occupied)."""
        if not self.is_valid(gx, gy):
            return -1
        return int(self.data[gy, gx])

    def set(self, gx: int, gy: int, value: int):
        """Set cell value."""
        if self.is_valid(gx, gy):
            self.data[gy, gx] = value

    def mark_free(self, gx: int, gy: int):
        """Mark cell as free (with decay for probabilistic update)."""
        if self.is_valid(gx, gy):
            current = self.data[gy, gx]
            if current == -1:
                self.data[gy, gx] = 0
            elif current > 0:
                self.data[gy, gx] = max(0, current - 10)

    def mark_occupied(self, gx: int, gy: int):
        """Mark cell as occupied (with increase for probabilistic update)."""
        if self.is_valid(gx, gy):
            current = self.data[gy, gx]
            if current == -1:
                self.data[gy, gx] = 50
            else:
                self.data[gy, gx] = min(100, current + 20)


class SLAMMapper:
    """
    Simple 2D SLAM mapper.

    Uses ARCore for localization and MiDaS depth for mapping.
    """

    def __init__(
        self,
        resolution: float = 0.1,
        map_size: int = 200,
        max_range: float = 4.0,
        min_range: float = 0.3,
        storage_path: str = "slam_map.json"
    ):
        """
        Initialize SLAM mapper.

        Args:
            resolution: Grid cell size in meters
            map_size: Grid dimensions (map_size x map_size cells)
            max_range: Maximum depth range to consider (meters)
            min_range: Minimum depth range (below this is noise)
            storage_path: Path to save/load map
        """
        self.storage_path = Path(storage_path)
        self.max_range = max_range
        self.min_range = min_range

        # Initialize occupancy grid
        half_size = (map_size * resolution) / 2
        self.grid = OccupancyGrid(
            resolution=resolution,
            width=map_size,
            height=map_size,
            origin_x=-half_size,
            origin_y=-half_size,
            data=np.full((map_size, map_size), -1, dtype=np.int8)
        )

        # Current pose
        self.pose = Pose2D()
        self.pose_history: List[Pose2D] = []

        # Load existing map if available
        self._load()

    def _load(self):
        """Load map from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)

                self.grid.resolution = data.get("resolution", self.grid.resolution)
                self.grid.origin_x = data.get("origin_x", self.grid.origin_x)
                self.grid.origin_y = data.get("origin_y", self.grid.origin_y)

                if "grid" in data:
                    grid_data = np.array(data["grid"], dtype=np.int8)
                    self.grid.data = grid_data
                    self.grid.height, self.grid.width = grid_data.shape

                print(f"[SLAM] Loaded map: {self.grid.width}x{self.grid.height}")
            except Exception as e:
                print(f"[SLAM] Failed to load map: {e}")

    def save(self):
        """Save map to disk."""
        data = {
            "resolution": self.grid.resolution,
            "width": self.grid.width,
            "height": self.grid.height,
            "origin_x": self.grid.origin_x,
            "origin_y": self.grid.origin_y,
            "grid": self.grid.data.tolist(),
            "saved_at": time.time(),
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f)

    def update_pose(self, arcore_data: Dict[str, Any]) -> Optional[Pose2D]:
        """
        Update robot pose from ARCore data.

        Args:
            arcore_data: ARCore JSON response with position and rotation

        Returns:
            Updated Pose2D or None if not tracking
        """
        if arcore_data.get("trackingState") != "TRACKING":
            return None

        position = arcore_data.get("position", [0, 0, 0])
        rotation = arcore_data.get("rotation", [0, 0, 0, 1])

        # Extract X and Z for 2D pose (Y is up in ARCore)
        x = position[0]
        y = position[2]  # Z in ARCore is forward

        # Extract yaw from quaternion
        qx, qy, qz, qw = rotation
        siny = 2 * (qw * qy - qz * qx)
        cosy = 1 - 2 * (qx * qx + qy * qy)
        theta = math.atan2(siny, cosy)

        self.pose = Pose2D(x=x, y=y, theta=theta, timestamp=time.time())
        self.pose_history.append(self.pose)

        # Keep history bounded
        if len(self.pose_history) > 10000:
            self.pose_history = self.pose_history[-5000:]

        return self.pose

    def update_map(self, depth_analysis: Dict[str, Any], fov_h: float = 60.0):
        """
        Update occupancy grid from MiDaS depth analysis.

        Args:
            depth_analysis: Output from depth_estimator.depth_to_telemetry()
            fov_h: Horizontal field of view in degrees
        """
        if not depth_analysis.get("available"):
            return

        zones = depth_analysis.get("zones", {})
        clearance = depth_analysis.get("clearance", {})
        obstacles = depth_analysis.get("obstacles", [])

        # Mark area in front of robot based on depth zones
        fov_rad = math.radians(fov_h)

        # Process each zone (left, center, right)
        for zone_name, depth_norm in zones.items():
            if zone_name == "full_avg":
                continue

            # Convert normalized depth (0-1, higher=closer) to meters
            # MiDaS gives relative depth, so we need to estimate
            # depth_norm ~0.3 means far (~4m), ~0.8 means close (~0.5m)
            if depth_norm < 0.1:
                distance = self.max_range
            else:
                distance = max(self.min_range, self.max_range * (1 - depth_norm))

            # Determine angle for this zone
            if zone_name == "left":
                angle_offset = fov_rad / 3
            elif zone_name == "right":
                angle_offset = -fov_rad / 3
            else:  # center
                angle_offset = 0

            ray_angle = self.pose.theta + angle_offset

            # Ray cast and update cells
            self._raycast(
                self.pose.x, self.pose.y,
                ray_angle, distance,
                mark_endpoint=(depth_norm > 0.5)  # Only mark obstacle if close
            )

        # Process specific obstacles
        for obs in obstacles:
            cx = obs.get("center_x", 0.5)
            # Map center_x (0-1) to angle within FOV
            angle_offset = (0.5 - cx) * fov_rad
            ray_angle = self.pose.theta + angle_offset

            # Obstacles are close, estimate ~1m
            distance = 1.0

            self._raycast(self.pose.x, self.pose.y, ray_angle, distance, mark_endpoint=True)

    def _raycast(
        self,
        start_x: float, start_y: float,
        angle: float, distance: float,
        mark_endpoint: bool = True
    ):
        """
        Cast a ray and update cells along the path.

        Args:
            start_x, start_y: Starting position (meters)
            angle: Direction (radians)
            distance: Distance to cast (meters)
            mark_endpoint: Whether to mark the endpoint as occupied
        """
        # Calculate endpoint
        end_x = start_x + distance * math.sin(angle)
        end_y = start_y + distance * math.cos(angle)

        # Convert to grid
        gx0, gy0 = self.grid.world_to_grid(start_x, start_y)
        gx1, gy1 = self.grid.world_to_grid(end_x, end_y)

        # Bresenham's line algorithm
        cells = self._bresenham(gx0, gy0, gx1, gy1)

        # Mark cells along ray as free (except last)
        for i, (gx, gy) in enumerate(cells[:-1]):
            self.grid.mark_free(gx, gy)

        # Mark endpoint as occupied if requested
        if mark_endpoint and cells:
            gx, gy = cells[-1]
            self.grid.mark_occupied(gx, gy)

    def _bresenham(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

            # Safety limit
            if len(cells) > 1000:
                break

        return cells

    def get_map_image(self, scale: int = 2) -> np.ndarray:
        """
        Get map as an image for visualization.

        Args:
            scale: Upscale factor

        Returns:
            BGR image (numpy array)
        """
        # Create colored image
        h, w = self.grid.data.shape
        img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                val = self.grid.data[y, x]
                if val == -1:
                    color = (50, 50, 50)  # Unknown: dark gray
                elif val == 0:
                    color = (200, 200, 200)  # Free: light gray
                elif val < 50:
                    color = (180, 180, 100)  # Probably free: yellow-gray
                else:
                    color = (0, 0, 150)  # Occupied: red

                img[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = color

        # Draw robot position
        gx, gy = self.grid.world_to_grid(self.pose.x, self.pose.y)
        if self.grid.is_valid(gx, gy):
            cx, cy = gx * scale + scale // 2, gy * scale + scale // 2

            # Robot circle
            cv2_available = False
            try:
                import cv2
                cv2.circle(img, (cx, cy), scale * 2, (0, 255, 0), -1)

                # Direction arrow
                arrow_len = scale * 4
                ax = int(cx + arrow_len * math.sin(self.pose.theta))
                ay = int(cy - arrow_len * math.cos(self.pose.theta))
                cv2.arrowedLine(img, (cx, cy), (ax, ay), (0, 255, 0), 2)
                cv2_available = True
            except ImportError:
                pass

            if not cv2_available:
                # Simple robot marker without cv2
                for dy in range(-scale, scale+1):
                    for dx in range(-scale, scale+1):
                        if 0 <= cy+dy < img.shape[0] and 0 <= cx+dx < img.shape[1]:
                            img[cy+dy, cx+dx] = (0, 255, 0)

        # Draw path
        for pose in self.pose_history[-200:]:
            gx, gy = self.grid.world_to_grid(pose.x, pose.y)
            if self.grid.is_valid(gx, gy):
                px, py = gx * scale + scale // 2, gy * scale + scale // 2
                if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                    img[py, px] = (255, 200, 100)  # Trail: light blue

        return img

    def get_stats(self) -> Dict[str, Any]:
        """Get SLAM statistics."""
        total = self.grid.width * self.grid.height
        unknown = np.sum(self.grid.data == -1)
        free = np.sum(self.grid.data == 0)
        occupied = np.sum(self.grid.data >= 50)

        return {
            "grid_size": f"{self.grid.width}x{self.grid.height}",
            "resolution": f"{self.grid.resolution}m",
            "total_cells": total,
            "unknown_cells": int(unknown),
            "free_cells": int(free),
            "occupied_cells": int(occupied),
            "explored_percent": round((1 - unknown/total) * 100, 1),
            "pose_history_length": len(self.pose_history),
            "current_pose": {
                "x": round(self.pose.x, 3),
                "y": round(self.pose.y, 3),
                "theta": round(math.degrees(self.pose.theta), 1),
            }
        }

    def query_path(self, target_x: float, target_y: float) -> Optional[List[Tuple[float, float]]]:
        """
        Find path to target using A* (simple implementation).

        Args:
            target_x, target_y: Target position in world coordinates

        Returns:
            List of waypoints or None if no path found
        """
        import heapq

        start = self.grid.world_to_grid(self.pose.x, self.pose.y)
        goal = self.grid.world_to_grid(target_x, target_y)

        if not self.grid.is_valid(*goal):
            return None

        # A* search
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    wx, wy = self.grid.grid_to_world(*current)
                    path.append((wx, wy))
                    current = came_from[current]
                path.reverse()
                return path

            # Check neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self.grid.is_valid(*neighbor):
                    continue

                # Check if passable
                cell_val = self.grid.get(*neighbor)
                if cell_val >= 50:  # Occupied
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None  # No path found


# Global instance
_slam_instance: Optional[SLAMMapper] = None


def get_slam(storage_path: str = "slam_map.json") -> SLAMMapper:
    """Get or create global SLAM instance."""
    global _slam_instance
    if _slam_instance is None:
        _slam_instance = SLAMMapper(storage_path=storage_path)
    return _slam_instance


# Test
if __name__ == "__main__":
    slam = SLAMMapper()

    # Simulate some poses and depth readings
    for i in range(50):
        angle = i * 0.1
        x = math.sin(angle) * 2
        y = math.cos(angle) * 2

        arcore_data = {
            "trackingState": "TRACKING",
            "position": [x, 0, y],
            "rotation": [0, math.sin(angle/2), 0, math.cos(angle/2)],
        }

        depth_data = {
            "available": True,
            "zones": {
                "left": 0.3 + i * 0.01,
                "center": 0.4 + i * 0.01,
                "right": 0.35 + i * 0.01,
            },
            "obstacles": [],
        }

        slam.update_pose(arcore_data)
        slam.update_map(depth_data)

    print("SLAM Stats:", slam.get_stats())
    slam.save()

    # Try to save visualization
    try:
        import cv2
        img = slam.get_map_image(scale=3)
        cv2.imwrite("slam_test.png", img)
        print("Saved slam_test.png")
    except ImportError:
        print("OpenCV not available for visualization")
