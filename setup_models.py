"""Explicitly download the local model artifacts needed by this plugin."""

import argparse
import os
import urllib.request
from pathlib import Path


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        print(f"Already present: {target}", flush=True)
        return
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "CharacterSwapDirector/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        length = int(response.headers.get("Content-Length", 0))
        received = 0
        next_report = 0
        while chunk := response.read(4 * 1024 * 1024):
            output.write(chunk)
            received += len(chunk)
            if received >= next_report:
                print(f"{target.name}: {received / 1024**2:.0f} MB" + (f" / {length / 1024**2:.0f} MB" if length else ""), flush=True)
                next_report += 100 * 1024 * 1024
    if length and received != length:
        raise OSError("Model download did not complete")
    os.replace(temporary, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--sam2", action="store_true")
    parser.add_argument("--fun", choices=["v1", "v2"])
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--ref2va", action="store_true", help="Download the dedicated H3 Ref2VA base (about 20 GB).")
    args = parser.parse_args()
    if not (args.comfy_root / "folder_paths.py").is_file():
        parser.error("Invalid ComfyUI root")
    if args.sam2 or args.fun or args.pose:
        parser.error("SAM2/Fun/DWPose downloads are retired. Use the H3 reference workflow and --ref2va.")
    if args.ref2va:
        filename = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        download("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/" + filename,
                 args.comfy_root / "models/diffusion_models" / filename)
    if args.sam2:
        download("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt", args.comfy_root / "models/sam2/sam2.1_hiera_tiny.pt")
    if args.fun:
        filename = "minimax_h3_fun_controlnet_union_" + ("2.0_" if args.fun == "v2" else "") + "pruned_int8_convrot.safetensors"
        download("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/model_patches/" + filename, args.comfy_root / "models/model_patches" / filename)
    if args.pose:
        directory = args.comfy_root / "custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose"
        for filename in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
            download("https://huggingface.co/yzd-v/DWPose/resolve/main/" + filename, directory / filename)
