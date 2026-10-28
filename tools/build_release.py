"""Build a portable plugin ZIP, generic workflows and SHA-256 checksums."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    selected = [path for path in ROOT.iterdir() if path.is_file() and (
        path.suffix in {".py", ".md", ".toml"} or path.name in {"LICENSE", "requirements-runtime.txt"})]
    for name in ("js", "examples"):
        selected.extend(path for path in (ROOT / name).rglob("*") if path.is_file())
    archive_path = output / f"ComfyUI-CharacterSwapDirector-v{version}.zip"
    manifest = {"name": "ComfyUI-CharacterSwapDirector", "version": version, "files": []}
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            archive.writestr(f"ComfyUI-CharacterSwapDirector/{relative}", data)
            manifest["files"].append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        archive.writestr("ComfyUI-CharacterSwapDirector/RELEASE-MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for entry in manifest["files"]:
            assert hashlib.sha256(archive.read("ComfyUI-CharacterSwapDirector/" + entry["path"])).hexdigest() == entry["sha256"]
        assert not any(Path(name).suffix.lower() in {".safetensors", ".pt", ".pth", ".mp4", ".mov", ".wav", ".log", ".pyc"} for name in archive.namelist())
    assets = [archive_path]
    for name in ("character_swap_local.json", "character_swap_turbo8_local.json"):
        target = output / name
        shutil.copyfile(ROOT / "examples" / name, target)
        assets.append(target)
    sums = output / "SHA256SUMS.txt"
    sums.write_text("".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in assets), encoding="utf-8")
    print(json.dumps({"version": version, "files": len(selected), "assets": [str(path) for path in assets] + [str(sums)]}))


if __name__ == "__main__":
    build()
