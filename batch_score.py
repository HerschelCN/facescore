"""Score every face image once and generate JSON plus a directly-openable HTML report."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from score import (
    SUPPORTED_SUFFIXES,
    load_inputs,
    load_model,
    predict_with_model,
    resolve_checkpoint,
)


PROJECT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score all face images on CPU and generate JSON/HTML results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_DIR / "face",
        help="image directory (default: ./face)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "face_scores.json",
        help="JSON output (default: ./face_scores.json)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=PROJECT_DIR / "face_scores.html",
        help="HTML output (default: ./face_scores.html)",
    )
    parser.add_argument("--checkpoint", type=Path, help="official FPEM checkpoint")
    return parser


def natural_key(path: Path):
    import re

    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def relative_image_url(image_path: Path, html_path: Path) -> str:
    relative = Path(os.path.relpath(image_path, start=html_path.parent))
    return "/".join(quote(part) for part in relative.parts)


def render_html(payload: dict, html_path: Path) -> str:
    cards = []
    for item in payload["results"]:
        filename = html.escape(item["filename"])
        if item["status"] == "ok":
            image_path = Path(item["absolute_path"])
            image_url = html.escape(relative_image_url(image_path, html_path), quote=True)
            score = float(item["score"])
            cards.append(
                f'''<article class="card">
  <a class="photo" href="{image_url}" target="_blank" rel="noreferrer">
    <img src="{image_url}" alt="{filename}" loading="lazy">
  </a>
  <div class="info">
    <div class="name" title="{filename}">{filename}</div>
    <div class="score"><strong>{score:.4f}</strong><span>/ 5.0000</span></div>
  </div>
</article>'''
            )
        else:
            error = html.escape(item.get("error") or "Unknown error")
            cards.append(
                f'''<article class="card failed">
  <div class="photo placeholder">无法读取图片</div>
  <div class="info">
    <div class="name" title="{filename}">{filename}</div>
    <div class="error">{error}</div>
  </div>
</article>'''
            )

    generated_at = html.escape(payload["generated_at"])
    model = html.escape(payload["model"])
    total = payload["count"]
    succeeded = payload["succeeded"]
    duration = payload["duration_seconds"]
    card_markup = "\n".join(cards)

    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FPEM 人脸评分结果</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, "Microsoft YaHei", system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f4f6f9; color: #18202b; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 22px clamp(18px, 4vw, 56px); background: rgba(255,255,255,.94); border-bottom: 1px solid #dce2ea; backdrop-filter: blur(12px); }}
    h1 {{ margin: 0 0 8px; font-size: clamp(22px, 3vw, 34px); }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 8px 18px; color: #5d6877; font-size: 14px; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 20px; padding: 28px clamp(18px, 4vw, 56px) 56px; }}
    .card {{ overflow: hidden; background: #fff; border: 1px solid #dfe5ec; border-radius: 14px; box-shadow: 0 5px 18px rgba(26,39,58,.07); }}
    .photo {{ display: flex; height: 300px; background: #e9edf2; align-items: center; justify-content: center; }}
    .photo img {{ width: 100%; height: 100%; object-fit: contain; }}
    .placeholder {{ color: #8a3b3b; }}
    .info {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px; }}
    .name {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 14px; color: #4f5b69; }}
    .score {{ flex: none; color: #1260d6; text-align: right; }}
    .score strong {{ font-size: 23px; }}
    .score span {{ margin-left: 4px; font-size: 11px; color: #788493; }}
    .error {{ font-size: 12px; color: #a12a2a; text-align: right; }}
    footer {{ padding: 0 18px 34px; color: #7c8794; text-align: center; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>FPEM 人脸评分结果</h1>
    <div class="summary">
      <span>模型：{model}</span><span>成功：{succeeded}/{total}</span>
      <span>总耗时：{duration:.2f} 秒</span><span>生成时间：{generated_at}</span>
    </div>
  </header>
  <main>
{card_markup}
  </main>
  <footer>分数是模型对照片的 1–5 分预测，不是客观颜值真值。</footer>
</body>
</html>
'''


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    html_path = args.html.expanduser().resolve()

    if not input_dir.is_dir():
        parser.error(f"input directory not found: {input_dir}")
    images = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=natural_key,
    )
    if not images:
        parser.error(f"no PNG/JPG/JPEG images found in: {input_dir}")

    try:
        checkpoint_path = resolve_checkpoint(args.checkpoint, PROJECT_DIR)
        print(f"Loading FPEM on CPU: {checkpoint_path.name}", flush=True)
        model = load_model(checkpoint_path)
    except (FileNotFoundError, RuntimeError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    results = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index:02d}/{len(images):02d}] {image_path.name}", end=" ... ", flush=True)
        try:
            score = predict_with_model(model, load_inputs(image_path))
            result = {
                "filename": image_path.name,
                "score": round(score, 4),
                "status": "ok",
                "error": None,
                "absolute_path": str(image_path),
            }
            print(f"{score:.4f}", flush=True)
        except Exception as exc:  # Keep other valid images in the report.
            result = {
                "filename": image_path.name,
                "score": None,
                "status": "error",
                "error": str(exc),
                "absolute_path": str(image_path),
            }
            print(f"ERROR: {exc}", flush=True)
        results.append(result)

    duration = time.perf_counter() - started
    payload = {
        "model": "FPEM",
        "scale": {"minimum": 1.0, "maximum": 5.0},
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_directory": str(input_dir),
        "count": len(results),
        "succeeded": sum(item["status"] == "ok" for item in results),
        "duration_seconds": round(duration, 3),
        "results": results,
    }
    atomic_write_text(output_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    # Deliberately render from the persisted JSON, so JSON is the report's
    # source of truth while the generated HTML remains directly openable.
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    atomic_write_text(html_path, render_html(persisted, html_path))

    print(f"JSON: {output_path}")
    print(f"HTML: {html_path}")
    return 0 if payload["succeeded"] == payload["count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
