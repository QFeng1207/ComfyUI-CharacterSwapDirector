# H3 Character Swap Director

[中文](README.md) · [User guide (Chinese)](USER_GUIDE.md) · [Workflows](examples) · [Releases](https://github.com/QFeng1207/ComfyUI-CharacterSwapDirector/releases)

A ComfyUI custom node package for character replacement with MiniMax H3 Ref2VA. Manage source videos, character pictures, source-person bindings, shot timelines and corrected frames inside the director node. The source video provides the action, scene and camera reference; character pictures provide the replacement appearance.

## Features

| Feature | Description |
| --- | --- |
| Embedded media library | Upload or load multiple videos and character pictures |
| Multiple characters | Bind each replacement independently, including characters appearing in alternating shots |
| Multiple pictures per character | Choose a primary picture and supporting views of the same person |
| Source localization | Record boxes, points and brush marks on the original video clock |
| Per-shot participation | Select automatic, explicitly visible or excluded roles; describe partial-body views |
| Timeline editing | Analyze source cuts, split or merge user segments, adjust boundaries and disable ranges |
| Shot action descriptions | Specify action order, framing, camera movement and occlusion |
| Timed corrected frames | Use supplied complete target frames through native H3 AddGuide |
| Segment trials and revisions | Test one range and regenerate selected ranges over a saved full-video base |
| Resolution controls | Linked width, height and megapixels, with optional canvas expansion |
| Source sound or mute | No voice-conversion model required |
| Persistent output | Original frame timing, MOV/MP4 output, reports and playable previews |

Output depends on the source material, reference pictures, prompts and sampling configuration. Test a short segment before a full render.

## Requirements

- Python 3.11 or later and a ComfyUI installation containing the official `MiniMaxH3ReferenceToVideo`, `MiniMaxH3AddGuide` and `MiniMaxH3SigmaShift` nodes.
- The existing ComfyUI PyTorch environment.
- PyAV, ImageIO-FFmpeg and an OpenCV provider. The dependency script preserves host PyTorch and reuses an existing suitable OpenCV installation.

## Installation

From `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/QFeng1207/ComfyUI-CharacterSwapDirector.git
python ComfyUI-CharacterSwapDirector/setup_dependencies.py --install --verify
```

Use the same Python interpreter that runs ComfyUI. For the official Windows portable distribution, run from its root:

```powershell
.\python_embeded\python.exe -s ComfyUI\custom_nodes\ComfyUI-CharacterSwapDirector\setup_dependencies.py --install --verify
```

Alternatively, use **Install via Git URL** in ComfyUI Manager, then run the dependency command. Restart ComfyUI and refresh the browser.

Release ZIPs contain a `ComfyUI-CharacterSwapDirector` directory to place inside `custom_nodes`.

## Models

| Model | Example filename | Directory |
| --- | --- | --- |
| H3 Ref2VA | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| Qwen3 VL | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| Optional Turbo8 LoRA | `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | `models/loras/` |

See [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) and [MiniMax H3 Turbo](https://github.com/ModelTC/Minimax-H3-Turbo). Set the CLIP Loader type to `minimax`. Model weights are not distributed with this repository. Source audio and mute modes do not require the audio VAE.

## Workflows and quick start

- [Base workflow](examples/character_swap_local.json): Ref2VA, `res_multistep` / `simple`, 20 steps.
- [Turbo8 workflow](examples/character_swap_turbo8_local.json): matching Ref2V Turbo LoRA, `euler` / `simple`, 8 steps, reference size `match`.
- [API template](examples/character_swap_api.json).

Load a workflow and select your installed model files. Upload the source and replacement pictures; create characters, select their pictures and localize the matched source people. Analyze source cuts, check participation and add shot action descriptions. Run a segment trial before rendering the full video.

For revisions, externally edit a complete frame, upload it at the matching **original-video time**, and select a saved full-video base and rerender ranges. Unselected ranges retain the chosen base. Ordinary source-scene references and corrected timed frames are separate reference types.

All references together are limited to nine pictures per inference interval. Multiple people in the same source shot are generated together; the number of localization boxes is not the number of sampler calls.

## Output

The default output uses source dimensions and frame times. Custom width/height and MP use a 32-pixel grid. MP means `1024 × 1024` pixels. Changing aspect ratio adds a background-generation stage around the replacement center.

Choose source sound or mute. Outputs and reports are saved in `ComfyUI/output/character_swap_director/`; lossless MOV results also have persistent MP4 previews.

## Version 0.3.3

Adds source matching descriptions for single-role shots, per-shot partial-person participation and action-order editing, and reference-budget handling. Existing timing, trial and revision behavior is retained. See [CHANGELOG.md](CHANGELOG.md) and [TESTING.md](TESTING.md).

## Credits and license

Built on the [ComfyUI](https://github.com/Comfy-Org/ComfyUI) MiniMax H3 nodes. Timeline UI inspiration: [Songssx](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector). Repository presentation reference: [AIMixer](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director).

Maintained by [QFeng1207](https://github.com/QFeng1207). Code: [GPL-3.0-or-later](LICENSE). Model weights retain their respective licenses.
