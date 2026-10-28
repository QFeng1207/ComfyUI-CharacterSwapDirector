"""Install missing plugin packages in ComfyUI Python without resolving or replacing Torch."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parent
HOST_REQUIREMENTS = ("torch>=2.5.1", "torchvision>=0.20.1", "numpy>=1.24.4", "Pillow>=9.4.0", "tqdm>=4.66.1",
                     "PyYAML>=5.1", "typing_extensions", "setuptools>=70.2.0")


def installed(requirement):
    try:
        version = importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return Version(version) in requirement.specifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Explicitly install only missing or incompatible plugin packages.")
    parser.add_argument("--verify", action="store_true", help="Verify media dependencies and local FFmpeg; no GPU model load.")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("This plugin requires Python 3.11 or newer.")
    for text in HOST_REQUIREMENTS:
        if not installed(Requirement(text)):
            parser.error(f"The existing ComfyUI runtime must provide {text}. This installer does not modify host packages.")
    torch_versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision")}
    requirements = [Requirement(line) for line in (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("#")]
    cv2_present = importlib.util.find_spec("cv2") is not None
    if cv2_present:
        import cv2

        if Version(cv2.__version__) < Version("4.7"):
            parser.error(f"Existing cv2 {cv2.__version__} is older than 4.7. Repair the existing OpenCV provider in ComfyUI; "
                         "this installer will not add a conflicting second cv2 provider.")
    missing = []
    for requirement in requirements:
        if requirement.name == "opencv-python-headless" and cv2_present:
            continue
        if not installed(requirement):
            missing.append(str(requirement))
    if missing:
        if not args.install:
            parser.error("Missing plugin dependencies: " + ", ".join(missing) + ". Run with --install --verify.")
        environment = {**os.environ, "SAM2_BUILD_CUDA": "0"}
        command = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation"]
        if any(Requirement(text).name in ("SAM-2", "iopath", "antlr4-python3-runtime") for text in missing):
            if importlib.util.find_spec("wheel") is None:
                subprocess.run(command + ["wheel>=0.43"], env=environment, check=True)
        subprocess.run(command + missing, env=environment, check=True)
    current_torch = {name: importlib.metadata.version(name) for name in torch_versions}
    if current_torch != torch_versions:
        raise RuntimeError("ComfyUI Torch/torchvision changed unexpectedly during setup.")
    if args.verify:
        import av
        import cv2
        import imageio_ffmpeg
        if not hasattr(av.VideoFrame, "rotation"):
            raise RuntimeError("PyAV display rotation support is unavailable.")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-version"], capture_output=True, check=True)
    versions = {requirement.name: importlib.metadata.version(requirement.name) for requirement in requirements
                if requirement.name != "opencv-python-headless" or installed(requirement)}
    print(json.dumps({"python": sys.executable, "torch_unchanged": current_torch, "packages": versions,
                      "backend": "H3 Ref2VA", "installed": missing}, indent=2))


if __name__ == "__main__":
    main()
