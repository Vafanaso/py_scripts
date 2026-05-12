"""Convert a Mosaic Kalibr-style YAML calibration to the Mosaic JSON camchain.

Output format matches the production Mosaic JSON (Ilia Shipachev's converter):
- Per camera, `R` and `t` encode cam_n's pose **in cam0's local frame**
  (M_0 = identity; M_n = M_{n-1} @ inv(T_cn_cnm1)).
- cam0 keeps `T_c0_lidar` from the YAML when present — that's the signal
  the parser uses to recover lidar-frame poses. JSONs without `T_c0_lidar`
  are valid; the resulting trajectories are then in cam0's frame.
- No top-level `lidars` key (that was an artifact of an earlier draft).
- Top-level `serial`, `model`, `pixel_size_mm`, `reassembly_number` lifted
  from `calibration_info.chamber.device_info` (or legacy top-level
  `device_info`) when present.
- `horizontally_flipped: true` added to cam1 and cam2 for M51 rigs.

The math through `trajectory/utils/camera.py::_parse_json` +
`get_cameras_parameters` recovers, for each camera:
    Tinv_cam0 = inv(T_c0_lidar)   if T_c0_lidar present
              = I                  otherwise
    Tinv_cam_n = Tinv_cam_{n-1} @ inv(T_cn_cnm1_recovered)
which is identical to what walking the YAML directly produces — verified
end-to-end by `verify_roundtrip` below.

Note: this JSON will NOT pass mopro's current `check_camera_json` with
`require_lid2cam=True`, because that validator requires a top-level `lidars`
key. The validator is stricter than the parser can handle and should be
fixed in tandem (separate change to mopro itself).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml


CAM_RE = re.compile(r"^cam\d+$")
PASSTHROUGH_KEYS = ("distortion_coeffs", "distortion_model", "intrinsics", "resolution")
PIXEL_SIZE_MM = {
    "M51D3": 1.55e-3,
    "MXD3": 2.74e-3,
    "XPD3": 2.74e-3,
    "Viking": 3.45e-3,
}


def _ordered_cam_keys(data: dict) -> list[str]:
    """Return camN keys in the dict's insertion order.

    Insertion order is load-bearing: the YAML's `T_cn_cnm1` is "relative to
    whatever came right before me in the file." Mosaic rigs put cam4 last
    (panorama stitching seam), so sorting by integer would mis-chain it.
    """
    return [key for key in data.keys() if CAM_RE.match(key)]


def _extract_device_info(yaml_data: dict) -> dict:
    """Extract Mosaic device metadata from the YAML's calibration_info block, with fallbacks."""
    nested = yaml_data.get("calibration_info", {}).get("chamber", {}).get("device_info", {})
    flat = yaml_data.get("device_info", {})
    yaml_di = {**flat, **nested}  # nested takes precedence on overlap

    info: dict = {}

    sn = yaml_di.get("serial_number")
    if sn is not None:
        info["serial"] = sn

    # Model: prefer hw_version ("RZSN0215, M51D3" -> "M51D3"); else probe fw_version.
    hw = yaml_di.get("hw_version", "")
    model = None
    if hw and "," in hw:
        model = hw.split(",", 1)[1].strip()
    else:
        fw = (yaml_di.get("fw_version") or "").lower()
        for token, label in (("m51", "M51D3"), ("mx", "MXD3"), ("xp", "XPD3"), ("viking", "Viking")):
            if token in fw:
                model = label
                break
    if model:
        info["model"] = model
        if model in PIXEL_SIZE_MM:
            info["pixel_size_mm"] = PIXEL_SIZE_MM[model]

    rn = yaml_di.get("reassembly_number")
    if rn is not None:
        info["reassembly_number"] = rn

    return info


def yaml_chain_to_json_camchain(yaml_data: dict) -> dict:
    """Convert a parsed YAML calibration to the Mosaic-internal-style JSON camchain dict."""
    cam_keys = _ordered_cam_keys(yaml_data)
    if not cam_keys:
        raise ValueError("No camN entries found in YAML root.")
    if cam_keys[0] != "cam0":
        raise ValueError(f"Camera chain must start at cam0 (found {cam_keys[0]}).")

    has_lidar = "T_c0_lidar" in yaml_data["cam0"]
    device_info = _extract_device_info(yaml_data)
    is_m51 = (device_info.get("model") or "").upper().startswith("M51")
    flipped_cams = {"cam1", "cam2"} if is_m51 else set()

    cams_json: dict[str, dict] = {}
    M_prev = np.eye(4)

    for idx, cam in enumerate(cam_keys):
        cam_data = yaml_data[cam]
        if not isinstance(cam_data, dict):
            raise ValueError(f"{cam} entry must be a mapping.")

        if idx == 0:
            # cam0 is the reference frame; its pose in cam0's frame is identity.
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
        if cam == "cam0" and has_lidar:
            cam_json["T_c0_lidar"] = cam_data["T_c0_lidar"]
        if cam in flipped_cams:
            cam_json["horizontally_flipped"] = True
        for key, value in cam_data.items():
            # T_cn_cnm1 is reconstructed from R/t; never echo it back verbatim.
            # T_c0_lidar is handled above (only kept on cam0).
            if key in cam_json or key in ("T_cn_cnm1", "T_c0_lidar"):
                continue
            cam_json[key] = value

        cams_json[cam] = cam_json
        M_prev = M

    result: dict = {"cams": cams_json}
    for key in ("serial", "model", "pixel_size_mm", "reassembly_number"):
        if device_info.get(key) is not None:
            result[key] = device_info[key]
    return result


def _replay_parser(json_data: dict) -> dict:
    """Faithful reimplementation of trajectory/utils/camera.py::_parse_json on an in-memory dict."""
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


def _yaml_to_tinv(yaml_data: dict) -> dict[str, np.ndarray]:
    """Replicate get_cameras_parameters' YAML path: per-camera Tinv = cam->reference pose."""
    cam_keys = _ordered_cam_keys(yaml_data)
    result: dict[str, np.ndarray] = {}
    Tinv = np.eye(4)
    for idx, cam in enumerate(cam_keys):
        entry = yaml_data[cam]
        if idx == 0:
            T = np.asarray(entry["T_c0_lidar"], dtype=float) if "T_c0_lidar" in entry else np.eye(4)
        else:
            T = np.asarray(entry["T_cn_cnm1"], dtype=float)
        Tinv = Tinv @ np.linalg.inv(T)
        result[cam] = Tinv.copy()
    return result


def _json_to_tinv(json_data: dict) -> dict[str, np.ndarray]:
    """Replicate _parse_json + get_cameras_parameters on the produced JSON dict."""
    chain = _replay_parser(json_data)
    result: dict[str, np.ndarray] = {}
    Tinv = np.eye(4)
    for cam in chain.keys():
        if cam != "cam0":
            T = np.asarray(chain[cam]["T_cn_cnm1"], dtype=float)
        elif "T_c0_lidar" in chain["cam0"]:
            T = np.asarray(chain["cam0"]["T_c0_lidar"], dtype=float)
        else:
            T = np.eye(4)
        Tinv = Tinv @ np.linalg.inv(T)
        result[cam] = Tinv.copy()
    return result


def verify_roundtrip(yaml_data: dict, json_data: dict, atol: float = 1e-9) -> None:
    """Verify the produced JSON yields the same per-camera Tinv as the source YAML."""
    yaml_tinv = _yaml_to_tinv(yaml_data)
    json_tinv = _json_to_tinv(json_data)
    if set(yaml_tinv) != set(json_tinv):
        raise AssertionError(
            f"Camera sets differ: YAML={sorted(yaml_tinv)} vs JSON={sorted(json_tinv)}"
        )
    for cam in yaml_tinv:
        diff = float(np.max(np.abs(yaml_tinv[cam] - json_tinv[cam])))
        if diff > atol:
            raise AssertionError(f"Tinv mismatch on {cam}: max abs diff = {diff:g}")
    for cam in _ordered_cam_keys(yaml_data):
        for key in PASSTHROUGH_KEYS:
            if yaml_data[cam].get(key) != json_data["cams"][cam].get(key):
                raise AssertionError(f"Pass-through field mismatch on {cam}.{key}")


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
        help="Absolute tolerance for round-trip Tinv comparison (default 1e-9).",
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
