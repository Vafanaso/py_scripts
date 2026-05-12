"""Visualize a Mosaic JSON camchain as coordinate frames in 3D.

In this JSON format R and t together encode the camera *pose* (cam -> lidar),
not a world->cam extrinsic. Proof: the trajectory package multiplies
`T_cam2ext = pose_lidar @ M`, where M = [[R | t], [0 | 1]] comes straight from
the JSON; that composition only yields a camera-in-world pose if M is itself
cam -> lidar. Therefore the matrix built from R and t is used directly, with
no `np.linalg.inv` step.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

SENSORS = ("cam0", "cam1", "cam2", "cam3", "cam4", "cam5")


def get_Ts_from_json(data: dict) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    cams = data.get("cams", {})
    for cam_name in SENSORS:
        if cam_name not in cams:
            print(f"Warning: {cam_name} not found in JSON.")
            continue
        cam_data = cams[cam_name]
        R = np.asarray(cam_data["R"], dtype=float)
        t = np.asarray(cam_data["t"], dtype=float).reshape(3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        poses[cam_name] = T
    return poses


def visualize_cam_poses(poses: dict[str, np.ndarray]) -> None:
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
    parser = argparse.ArgumentParser(description="Visualize camera rig poses from a Mosaic JSON camchain.")
    parser.add_argument("config", type=Path, help="Path to the calibration JSON file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        data = json.load(f)

    poses = get_Ts_from_json(data)
    visualize_cam_poses(poses)
