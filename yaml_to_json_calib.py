"""Convert a Mosaic Kalibr-style YAML calibration to the Mosaic JSON camchain.

This is the exact inverse of `trajectory/utils/camera.py::_parse_json`
(in the external `trajectory` package). That function, given a JSON camchain,
walks the cameras in order and produces an in-memory dict shaped like the YAML
calibration. This script does the reverse: it consumes the YAML and emits a JSON
that, when fed back through `_parse_json`, reproduces the original YAML chain.

Forward (parser) transformation, per camera n with T_prev starting at identity:
    M_n      = [[R_n | t_n], [0 | 1]]            # built from JSON
    T_chain  = inv(M_n) @ T_prev                  # stored as YAML key
    T_prev   = M_n                                # propagated to next cam
The YAML key is `T_c0_lidar` for cam0 (only emitted when the JSON has a
top-level `lidars` entry) and `T_cn_cnm1` otherwise.

Inverse (this script):
    M_0 = inv(T_c0_lidar)                         # or identity when absent
    M_n = M_{n-1} @ inv(T_cn_cnm1)   for n >= 1
    R_n = M_n[:3, :3];  t_n = M_n[:3, 3]

The script verifies its own output by replaying the parser on the produced
JSON and checking the recovered chain matches the input YAML.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml


CAM_RE = re.compile(r"^cam(\d+)$")
PASSTHROUGH_KEYS = ("distortion_coeffs", "distortion_model", "intrinsics", "resolution")


def _ordered_cam_keys(data: dict) -> list[str]:
    """Return the camN keys in the dict's insertion order.

    Insertion order is load-bearing: the parser at trajectory/utils/camera.py
    walks `chain_json["cams"].keys()` in insertion order, and each non-cam0
    camera's `T_cn_cnm1` is interpreted as "relative to the previous camera in
    the file". For Mosaic rigs cam4 is intentionally listed last (after cam5)
    because it sits at the panorama stitching seam, so sorting by integer
    suffix would mis-chain cam4 and cam5.
    """
    return [key for key in data.keys() if CAM_RE.match(key)]


def yaml_chain_to_json_camchain(yaml_data: dict) -> dict:
    """Return a JSON-camchain dict equivalent to the input YAML calibration."""
    cam_keys = _ordered_cam_keys(yaml_data)
    if not cam_keys:
        raise ValueError("No camN entries found in the YAML root.")
    if cam_keys[0] != "cam0":
        raise ValueError(f"Camera chain must start at cam0 (found {cam_keys[0]}).")

    has_lidar = "T_c0_lidar" in yaml_data["cam0"]

    cams_json: dict[str, dict] = {}
    M_prev = np.eye(4)

    for idx, cam in enumerate(cam_keys):
        cam_data = yaml_data[cam]
        if not isinstance(cam_data, dict):
            raise ValueError(f"{cam} entry must be a mapping.")

        if idx == 0:
            if has_lidar:
                T_c0_lidar = np.asarray(cam_data["T_c0_lidar"], dtype=float)
                if T_c0_lidar.shape != (4, 4):
                    raise ValueError(f"{cam}.T_c0_lidar must be 4x4 (got {T_c0_lidar.shape}).")
                M = np.linalg.inv(T_c0_lidar)
            else:
                # cam0 still needs R/t to satisfy the validator; identity is the
                # neutral choice and matches what _parse_json effectively sees
                # (it pops R/t but skips the matrix build when "lidars" is absent).
                M = np.eye(4)
        else:
            if "T_cn_cnm1" not in cam_data:
                raise ValueError(f"{cam} is missing required key 'T_cn_cnm1'.")
            T_cn_cnm1 = np.asarray(cam_data["T_cn_cnm1"], dtype=float)
            if T_cn_cnm1.shape != (4, 4):
                raise ValueError(f"{cam}.T_cn_cnm1 must be 4x4 (got {T_cn_cnm1.shape}).")
            M = M_prev @ np.linalg.inv(T_cn_cnm1)

        cam_json: dict = {
            "R": M[:3, :3].tolist(),
            "t": M[:3, 3].tolist(),
        }
        for key in PASSTHROUGH_KEYS:
            if key not in cam_data:
                raise ValueError(f"{cam} is missing required key '{key}'.")
            cam_json[key] = cam_data[key]
        for key, value in cam_data.items():
            if key in cam_json or key in ("T_c0_lidar", "T_cn_cnm1"):
                continue
            cam_json[key] = value

        cams_json[cam] = cam_json
        M_prev = M

    result: dict = {"cams": cams_json}
    if has_lidar:
        # The parser only checks for presence of this key. Empty dict is enough.
        result["lidars"] = {}
    if "calibration_info" in yaml_data:
        result["calibration_info"] = yaml_data["calibration_info"]
    return result


def _replay_parser(json_data: dict) -> dict:
    """Faithful reimplementation of `trajectory/utils/camera.py::_parse_json` on an in-memory dict."""
    chain: dict[str, dict] = {}
    T_prev = np.eye(4)
    for cam in json_data["cams"].keys():
        chain[cam] = {}
        cam_data = dict(json_data["cams"][cam])
        if cam != "cam0" or "lidars" in json_data:
            T = np.eye(4)
            T[:3, :3] = np.asarray(cam_data["R"])
            T[:3, 3] = np.asarray(cam_data["t"])
            T, T_prev = np.linalg.inv(T) @ T_prev, T
            yaml_key = "T_c0_lidar" if cam == "cam0" else "T_cn_cnm1"
            chain[cam][yaml_key] = T
        cam_data.pop("R", None)
        cam_data.pop("t", None)
        chain[cam] |= cam_data
    return chain


def verify_roundtrip(yaml_data: dict, json_data: dict, atol: float = 1e-9) -> None:
    """Run the parser on the produced JSON and confirm the recovered chain matches the YAML input."""
    replayed = _replay_parser(json_data)
    for cam in _ordered_cam_keys(yaml_data):
        original = yaml_data[cam]
        recovered = replayed[cam]
        for pose_key in ("T_c0_lidar", "T_cn_cnm1"):
            if pose_key in original:
                if pose_key not in recovered:
                    raise AssertionError(f"Round-trip lost {cam}.{pose_key}.")
                a = np.asarray(original[pose_key], dtype=float)
                b = np.asarray(recovered[pose_key], dtype=float)
                if not np.allclose(a, b, atol=atol):
                    diff = float(np.max(np.abs(a - b)))
                    raise AssertionError(
                        f"Round-trip pose mismatch on {cam}.{pose_key}: max abs diff = {diff:g}."
                    )
        for key in PASSTHROUGH_KEYS:
            if original.get(key) != recovered.get(key):
                raise AssertionError(f"Round-trip mismatch on {cam}.{key}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Mosaic Kalibr-style YAML calibration into the Mosaic JSON camchain.",
    )
    parser.add_argument("input_yaml", type=Path, help="Path to the input .yaml calibration.")
    parser.add_argument("output_json", type=Path, help="Path to write the .json calibration.")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip round-trip verification against the parser logic.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for round-trip pose comparison (default 1e-9).",
    )
    args = parser.parse_args(argv)

    if args.input_yaml.suffix != ".yaml":
        print(f"warning: input does not have a .yaml suffix: {args.input_yaml}", file=sys.stderr)
    if args.output_json.suffix != ".json":
        print(f"warning: output does not have a .json suffix: {args.output_json}", file=sys.stderr)

    with open(args.input_yaml, "r") as f:
        yaml_data = yaml.safe_load(f)
    if not isinstance(yaml_data, dict):
        print("error: YAML root must be a mapping.", file=sys.stderr)
        return 1

    json_data = yaml_chain_to_json_camchain(yaml_data)

    if not args.no_verify:
        verify_roundtrip(yaml_data, json_data, atol=args.atol)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
