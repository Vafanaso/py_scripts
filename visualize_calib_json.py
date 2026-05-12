"""Visualize a Mosaic camera-rig calibration (YAML or JSON) as coordinate frames in 3D.

Input format is detected from the file suffix:
- `.json` -> Mosaic JSON camchain. R and t together encode the camera *pose*
  (cam -> lidar), so the 4x4 built from them is used directly with no inversion.
- `.yaml` -> Kalibr-style chain. cam0 stores `T_c0_lidar`, every other camera
  stores `T_cn_cnm1` (relative to the previous camera in YAML insertion order).
  Per-camera pose is reconstructed by accumulating along the chain exactly the
  way trajectory/utils/camera.py::get_cameras_parameters does:
      Tinv_0 = inv(T_c0_lidar)         (or identity if T_c0_lidar is absent)
      Tinv_n = Tinv_{n-1} @ inv(T_cn_cnm1)
  Insertion order is load-bearing: Mosaic YAMLs list cam4 last because it sits
  at the panorama stitching seam, so the chain walk must follow file order.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml


CAM_RE = re.compile(r"^cam\d+$")


def get_Ts_from_json(data: dict) -> dict[str, np.ndarray]:
    """Recover per-camera lidar-frame poses from a Mosaic JSON camchain.

    In this format R and t encode cam_n's pose in cam0's local frame. If cam0
    carries `T_c0_lidar`, we compose with its inverse to lift each pose into
    the lidar frame (matches what the YAML path computes). If T_c0_lidar is
    absent, cam0 IS the reference and poses are returned in cam0's frame.
    """
    poses: dict[str, np.ndarray] = {}
    cams = data.get("cams", {})
    cam0 = cams.get("cam0", {})
    T_lidar_cam0 = (
        np.linalg.inv(np.asarray(cam0["T_c0_lidar"], dtype=float))
        if "T_c0_lidar" in cam0 else np.eye(4)
    )
    for cam_name, cam_data in cams.items():
        if not CAM_RE.match(cam_name):
            continue
        R = np.asarray(cam_data["R"], dtype=float)
        t = np.asarray(cam_data["t"], dtype=float).reshape(3)
        M_cam0 = np.eye(4)
        M_cam0[:3, :3] = R
        M_cam0[:3, 3] = t
        poses[cam_name] = T_lidar_cam0 @ M_cam0
    return poses


def get_Ts_from_yaml(data: dict) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    Tinv = np.eye(4)
    for cam_name, entry in data.items():
        if not CAM_RE.match(cam_name):
            continue
        if cam_name == "cam0":
            T_chain = np.asarray(entry.get("T_c0_lidar", np.eye(4)), dtype=float)
        else:
            if "T_cn_cnm1" not in entry:
                raise ValueError(f"{cam_name} is missing required 'T_cn_cnm1'.")
            T_chain = np.asarray(entry["T_cn_cnm1"], dtype=float)
        if T_chain.shape != (4, 4):
            raise ValueError(f"{cam_name} chain matrix must be 4x4 (got {T_chain.shape}).")
        Tinv = Tinv @ np.linalg.inv(T_chain)
        poses[cam_name] = Tinv.copy()
    return poses


def load_poses(path: Path) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r") as f:
            return get_Ts_from_json(json.load(f))
    if suffix in (".yaml", ".yml"):
        with open(path, "r") as f:
            return get_Ts_from_yaml(yaml.safe_load(f))
    raise ValueError(f"Unsupported calibration suffix: {suffix!r} (expected .json, .yaml, or .yml).")


def visualize_cam_poses(poses: dict[str, np.ndarray]) -> None:
    if not poses:
        print("No camera poses to display.", file=sys.stderr)
        return

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("Camera Rig Visualization", 1024, 768)
    vis.show_settings = True

    for cam_name, T in poses.items():
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        mesh.transform(T)
        vis.add_geometry(cam_name, mesh)
        vis.add_3d_label(T[:3, 3], cam_name)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry("origin", origin)

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize camera rig poses from a Mosaic YAML or JSON calibration.")
    parser.add_argument("config", type=Path, help="Path to the calibration file (.yaml or .json).")
    args = parser.parse_args()

    poses = load_poses(args.config)
    print(f"Loaded {len(poses)} camera poses from {args.config.name}: {', '.join(poses.keys())}")
    visualize_cam_poses(poses)
