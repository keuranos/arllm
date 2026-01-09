"""
MiDaS Depth Estimation Module for Ramblebot

Uses MiDaS (Monocular Depth Estimation) to generate depth maps from single RGB images.
This enables obstacle detection and scene understanding without ToF/LiDAR sensors.

Key outputs:
- Relative depth map (normalized 0-1, closer = higher values)
- Obstacle detection zones (left, center, right)
- Floor/wall segmentation hints
- Navigation clearance estimation
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time

# MiDaS model (loaded lazily)
_midas_model = None
_midas_transform = None
_midas_device = None


def _load_midas():
    """Lazy load MiDaS model (downloads on first use)."""
    global _midas_model, _midas_transform, _midas_device

    if _midas_model is not None:
        return

    import torch

    _midas_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MiDaS] Loading model on {_midas_device}...")

    # Use MiDaS small for speed (good enough for navigation)
    # Options: "MiDaS_small", "DPT_Hybrid", "DPT_Large"
    _midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    _midas_model.to(_midas_device)
    _midas_model.eval()

    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    _midas_transform = midas_transforms.small_transform

    print("[MiDaS] Model loaded successfully.")


@dataclass
class DepthAnalysis:
    """Results from depth estimation."""
    depth_map: np.ndarray          # HxW float32, normalized 0-1 (1=close)
    zones: Dict[str, float]        # avg depth per zone (left, center, right)
    clearance: Dict[str, Any]      # navigation hints
    obstacles: list                # detected obstacle regions
    inference_time_ms: float


def estimate_depth(img_bgr: np.ndarray) -> Optional[DepthAnalysis]:
    """
    Estimate depth from a single RGB image using MiDaS.

    Args:
        img_bgr: OpenCV BGR image (HxWx3)

    Returns:
        DepthAnalysis with depth map and derived metrics
    """
    try:
        _load_midas()
    except Exception as e:
        print(f"[MiDaS] Failed to load model: {e}")
        return None

    import torch

    t0 = time.time()

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Apply MiDaS transform
    input_batch = _midas_transform(img_rgb).to(_midas_device)

    # Run inference
    with torch.no_grad():
        prediction = _midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert to numpy
    depth_raw = prediction.cpu().numpy()

    # Normalize to 0-1 range (MiDaS outputs inverse depth - closer = higher)
    depth_min = depth_raw.min()
    depth_max = depth_raw.max()
    if depth_max - depth_min > 0:
        depth_norm = (depth_raw - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth_raw)

    inference_time = (time.time() - t0) * 1000

    # Analyze zones
    zones = _analyze_zones(depth_norm)
    clearance = _compute_clearance(depth_norm, zones)
    obstacles = _detect_obstacles(depth_norm)

    return DepthAnalysis(
        depth_map=depth_norm.astype(np.float32),
        zones=zones,
        clearance=clearance,
        obstacles=obstacles,
        inference_time_ms=round(inference_time, 1)
    )


def _analyze_zones(depth: np.ndarray) -> Dict[str, float]:
    """
    Split image into left/center/right zones and compute average depth.
    Higher values = closer objects.
    """
    h, w = depth.shape

    # Bottom half of image (where obstacles matter for navigation)
    bottom = depth[h//2:, :]
    bh, bw = bottom.shape

    # Split into thirds
    left = bottom[:, :bw//3]
    center = bottom[:, bw//3:2*bw//3]
    right = bottom[:, 2*bw//3:]

    return {
        "left": round(float(np.mean(left)), 3),
        "center": round(float(np.mean(center)), 3),
        "right": round(float(np.mean(right)), 3),
        "full_avg": round(float(np.mean(bottom)), 3),
    }


def _compute_clearance(depth: np.ndarray, zones: Dict[str, float]) -> Dict[str, Any]:
    """
    Compute navigation clearance based on depth zones.

    Returns:
        Dictionary with navigation suggestions.
    """
    h, w = depth.shape

    # Focus on center-bottom region (path ahead)
    path_region = depth[int(h*0.6):, int(w*0.3):int(w*0.7)]
    path_depth = float(np.mean(path_region))

    # Threshold for "obstacle nearby" (tunable)
    CLOSE_THRESHOLD = 0.65  # Higher = closer obstacle
    VERY_CLOSE = 0.75

    # Determine clearance
    if path_depth >= VERY_CLOSE:
        ahead = "blocked"
        action = "stop_or_turn"
    elif path_depth >= CLOSE_THRESHOLD:
        ahead = "obstacle_near"
        action = "slow_or_turn"
    else:
        ahead = "clear"
        action = "proceed"

    # Find best direction
    zone_values = [(zones["left"], "left"), (zones["center"], "center"), (zones["right"], "right")]
    zone_values.sort(key=lambda x: x[0])  # Sort by depth (lower = more clear)
    best_dir = zone_values[0][1]

    # Compute relative clearance
    min_zone = min(zones["left"], zones["center"], zones["right"])
    max_zone = max(zones["left"], zones["center"], zones["right"])

    return {
        "path_ahead": ahead,
        "suggested_action": action,
        "path_depth": round(path_depth, 3),
        "clearest_direction": best_dir,
        "left_clearer_than_right": zones["left"] < zones["right"],
        "depth_variance": round(max_zone - min_zone, 3),
    }


def _detect_obstacles(depth: np.ndarray, threshold: float = 0.6) -> list:
    """
    Detect distinct obstacle regions in the depth map.

    Returns:
        List of obstacle dictionaries with position and size info.
    """
    h, w = depth.shape

    # Focus on bottom 60% of image
    roi = depth[int(h*0.4):, :]
    roi_h, roi_w = roi.shape

    # Threshold to binary (1 = obstacle/close)
    close_mask = (roi > threshold).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(close_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacles = []
    min_area = roi_h * roi_w * 0.02  # Minimum 2% of ROI

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        # Convert to normalized coordinates
        cx = (x + cw/2) / roi_w
        cy = (y + ch/2 + int(h*0.4)) / h  # Adjust for ROI offset

        # Classify position
        if cx < 0.33:
            position = "left"
        elif cx > 0.67:
            position = "right"
        else:
            position = "center"

        obstacles.append({
            "position": position,
            "center_x": round(cx, 3),
            "center_y": round(cy, 3),
            "area_ratio": round(area / (roi_h * roi_w), 4),
            "width_ratio": round(cw / roi_w, 3),
        })

    # Sort by area (largest first)
    obstacles.sort(key=lambda o: o["area_ratio"], reverse=True)
    return obstacles[:5]  # Max 5 obstacles


def depth_to_telemetry(analysis: Optional[DepthAnalysis]) -> Dict[str, Any]:
    """
    Convert depth analysis to telemetry format for LLM consumption.
    """
    if analysis is None:
        return {"available": False}

    return {
        "available": True,
        "zones": analysis.zones,
        "clearance": analysis.clearance,
        "obstacles": analysis.obstacles,
        "inference_ms": analysis.inference_time_ms,
    }


def visualize_depth(depth_map: np.ndarray, original_img: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create visualization of depth map for debugging.

    Args:
        depth_map: Normalized depth (0-1, 1=close)
        original_img: Optional original image to overlay

    Returns:
        Colorized depth visualization (BGR)
    """
    # Convert to 0-255 and apply colormap
    depth_uint8 = (depth_map * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

    if original_img is not None:
        # Resize if needed
        if depth_colored.shape[:2] != original_img.shape[:2]:
            depth_colored = cv2.resize(depth_colored, (original_img.shape[1], original_img.shape[0]))
        # Blend
        blended = cv2.addWeighted(original_img, 0.4, depth_colored, 0.6, 0)
        return blended

    return depth_colored


# Test function
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python depth_estimator.py <image_path>")
        print("Testing with webcam...")

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = estimate_depth(frame)
            if result:
                vis = visualize_depth(result.depth_map, frame)
                cv2.putText(vis, f"Path: {result.clearance['path_ahead']}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis, f"Best dir: {result.clearance['clearest_direction']}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("MiDaS Depth", vis)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"Failed to load {sys.argv[1]}")
            sys.exit(1)

        result = estimate_depth(img)
        if result:
            print("Zones:", result.zones)
            print("Clearance:", result.clearance)
            print("Obstacles:", result.obstacles)
            print(f"Inference: {result.inference_time_ms}ms")

            vis = visualize_depth(result.depth_map, img)
            cv2.imwrite("depth_output.png", vis)
            print("Saved depth_output.png")
