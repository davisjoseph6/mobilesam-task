import os
import csv
import pathlib
import json
import numpy as np
import torch
from typing import Iterable, List, Dict, Tuple
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import Image

from tools import (
    fast_process,
    compute_segment_stats,
    draw_segment_labels_pil,
    compute_segment_stats_in_bbox,
    get_bbox_from_mask,
)

# --------- Paths & constants ----------
RESOURCES_DIR = os.environ.get("RESOURCES_DIR", "resources")
GENERATED_DIR = os.environ.get("GENERATED_DIR", "generated")
ACCEPT_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUBSETS = ("train", "valid")

# --------- Model setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam_checkpoint = "./mobile_sam.pt"
model_type = "vit_t"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
predictor = SamPredictor(mobile_sam)  # reserved for future prompt usage


def _is_image_file(p: pathlib.Path) -> bool:
    return p.suffix.lower() in ACCEPT_EXT


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def segment_everything(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    """Original segmentation function returning only the image."""
    global mask_generator
    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)
    annotations = mask_generator.generate(nd_image)

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )
    return fig


@torch.no_grad()
def analyze_segments(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    """Return (overlay image with labels, per-segment stats)."""
    global mask_generator
    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image_resized = image.resize((new_w, new_h))

    nd_image = np.array(image_resized)
    annotations = mask_generator.generate(nd_image)

    stats = compute_segment_stats(annotations, image_resized)

    overlay = fast_process(
        annotations=annotations,
        image=image_resized,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )
    overlay = draw_segment_labels_pil(overlay, annotations, stats)
    return overlay, stats


@torch.no_grad()
def analyze_segments_in_bbox0(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    """
    Return (overlay image with labels, stats) but ONLY for pixels inside
    the bbox of segment id=0. Keeps original ids; stats are computed on mask ∩ bbox0.
    """
    global mask_generator
    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image_resized = image.resize((new_w, new_h))

    nd_image = np.array(image_resized)
    annotations = mask_generator.generate(nd_image)

    if not annotations:
        return image_resized, []

    bbox0 = annotations[0].get("bbox")
    if bbox0 is None:
        x1, y1, x2, y2 = get_bbox_from_mask(annotations[0]["segmentation"])
        bbox0 = [x1, y1, x2 - x1, y2 - y1]

    stats = compute_segment_stats_in_bbox(annotations, image_resized, bbox0)

    overlay = fast_process(
        annotations=annotations,
        image=image_resized,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )
    overlay = draw_segment_labels_pil(overlay, annotations, stats)
    return overlay, stats


def _save_stats_csv(stats: List[Dict], csv_path: str) -> None:
    _ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pixels", "color_class", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        for s in stats:
            bx, by, bw, bh = (s.get("bbox_xywh") or [None, None, None, None])
            writer.writerow([s["id"], s["pixels"], s["color_class"], bx, by, bw, bh])


def _save_stats_json(stats: List[Dict], json_path: str) -> None:
    _ensure_dir(os.path.dirname(json_path))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def batch_process_resources(
    subsets: Iterable[str] = SUBSETS,
    inside_bbox0: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Process all acceptable images under resources/<subset> and save to generated/<subset>.
    Returns a manifest: {subset: [ {input, overlay, csv, json}, ... ], ...}
    """
    manifest: Dict[str, List[Dict]] = {}
    for subset in subsets:
        if subset not in SUBSETS:
            continue
        src_dir = pathlib.Path(RESOURCES_DIR) / subset
        out_dir = pathlib.Path(GENERATED_DIR) / subset
        _ensure_dir(str(out_dir))

        items: List[Dict] = []
        if not src_dir.exists():
            manifest[subset] = items
            continue

        # iterate recursively
        for p in src_dir.rglob("*"):
            if not p.is_file() or not _is_image_file(p):
                continue

            rel = p.relative_to(src_dir)
            stem = p.stem
            # output paths mirroring relative structure
            base = out_dir / rel.parent / stem
            overlay_path = str(base) + "_overlay.png"
            csv_path = str(base) + "_segments.csv"
            json_path = str(base) + "_segments.json"

            img = Image.open(p).convert("RGB")
            if inside_bbox0:
                overlay, stats = analyze_segments_in_bbox0(img)
            else:
                overlay, stats = analyze_segments(img)

            # save outputs
            _ensure_dir(os.path.dirname(overlay_path))
            overlay.save(overlay_path)
            _save_stats_csv(stats, csv_path)
            _save_stats_json(stats, json_path)

            items.append({
                "input": str(p),
                "overlay": overlay_path,
                "csv": csv_path,
                "json": json_path,
            })
        manifest[subset] = items
    return manifest


if __name__ == "__main__":
    # CLI demo: process both train and valid with bbox0 filtering
    _ensure_dir(GENERATED_DIR)
    result = batch_process_resources(subsets=SUBSETS, inside_bbox0=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))

