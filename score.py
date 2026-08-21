"""CPU-only single-image inference for the official ICCV 2025 FPEM model."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_CHECKPOINTS = (
    Path("pretrained/fpem_srcc_0.9243.pth"),
    Path("fpem_srcc_0.9243.pth"),
    Path("experiments/public_fpem/save_dir/ckpt_e_21_0.93325.pth"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score one manually cropped face with FPEM on CPU."
    )
    parser.add_argument("image", type=Path, help="PNG/JPG/JPEG face image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="official FPEM .pth file (auto-detected when omitted)",
    )
    return parser


def resolve_checkpoint(requested: Path | None, project_dir: Path) -> Path:
    if requested is not None:
        checkpoint = requested.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        return checkpoint

    for relative in DEFAULT_CHECKPOINTS:
        candidate = (project_dir / relative).resolve()
        if candidate.is_file():
            return candidate

    searched = "\n  ".join(str(project_dir / item) for item in DEFAULT_CHECKPOINTS)
    raise FileNotFoundError(
        "official FPEM checkpoint not found; searched:\n  " + searched
        + "\nDownload pretrained/fpem_srcc_0.9243.pth as described in README.md, "
        "or pass --checkpoint PATH."
    )


def resize_and_pad(image_tensor, target_size: int):
    """Official right/down zero-padding pipeline from FaceDataset."""
    from torchvision.transforms import functional as tvf

    _, height, width = image_tensor.shape
    scale_factor = target_size / max(height, width)
    scaled_height = max(1, int(height * scale_factor))
    scaled_width = max(1, int(width * scale_factor))
    resized = tvf.resize(
        image_tensor,
        [scaled_height, scaled_width],
        antialias=False,
    )
    return tvf.pad(
        resized,
        [0, 0, target_size - scaled_width, target_size - scaled_height],
        fill=0,
    )


def load_inputs(image_path: Path):
    import torch
    from PIL import Image, UnidentifiedImageError
    from torchvision.transforms import functional as tvf

    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            tensor = tvf.pil_to_tensor(rgb).to(dtype=torch.float32).div_(255.0)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"cannot read image {image_path}: {exc}") from exc

    if tensor.shape[1] == 0 or tensor.shape[2] == 0:
        raise ValueError(f"image has an empty dimension: {image_path}")

    return tuple(resize_and_pad(tensor, size).unsqueeze(0) for size in (224, 112, 160))


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint is not a state-dict mapping")
    for key in ("model", "state_dict", "state_dict_backbone"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def normalize_state_keys(state_dict, model_keys):
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    if set(state) == model_keys:
        return state

    # Official releases use one of these ModelHelper subnet names. The
    # fpem_srcc_0.9243.pth checkpoint uses `FPEM_add`.
    for wrapper in ("FPEM_add.", "FPEM."):
        if state and all(key.startswith(wrapper) for key in state):
            unwrapped = {key[len(wrapper):]: value for key, value in state.items()}
            if set(unwrapped) == model_keys:
                return unwrapped
    return state


def load_model(checkpoint_path: Path):
    import torch
    from fpem_official.Clips import FPEM

    torch.manual_seed(0)
    model = FPEM()
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:  # PyTorch < 2.0 compatibility
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state = normalize_state_keys(extract_state_dict(checkpoint), set(model.state_dict()))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint keys do not strictly match the official FPEM architecture; "
            f"refusing a partial load. Original error:\n{exc}"
        ) from exc

    del checkpoint, state
    return model.eval()


def predict_with_model(model, inputs) -> float:
    import torch

    with torch.inference_mode():
        output = model(inputs)
        prediction = output[0] if isinstance(output, tuple) else output
        score = float(prediction.reshape(-1)[0].item())

    if not math.isfinite(score):
        raise RuntimeError(f"model returned a non-finite score: {score}")
    if not 1.0 <= score <= 5.0:
        raise RuntimeError(f"model returned an out-of-scale score: {score}")
    return score


def score_image(image_path: Path, checkpoint_path: Path) -> float:
    inputs = load_inputs(image_path)
    model = load_model(checkpoint_path)
    return predict_with_model(model, inputs)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    image_path = args.image.expanduser().resolve()

    if not image_path.is_file():
        parser.error(f"image not found: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        parser.error(
            f"unsupported image format {image_path.suffix!r}; use .png, .jpg, or .jpeg"
        )

    project_dir = Path(__file__).resolve().parent
    try:
        checkpoint_path = resolve_checkpoint(args.checkpoint, project_dir)
        score = score_image(image_path, checkpoint_path)
    except (FileNotFoundError, ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"FPEM score: {score:.4f} / 5.0000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
