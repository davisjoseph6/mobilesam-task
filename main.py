import os
import csv
import pathlib
import json
import numpy as np
import torch
from typing import Iterable, List, Dict
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import Image

from tools import (
    fast_process,
    compute_segment_stats,
    draw_segment_labels_pil,
    compute_segment_stats_in_bbox,
    compute_segment_stats_in_bboxes,
    get_bbox_from_mask,
)

# --------- Paths & constants ----------
RESOURCES_DIR = os.environ.get("RESOURCES_DIR", "resources")
GENERATED_DIR = os.environ.get("GENERATED_DIR", "generated")
ACCEPT_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUBSETS = ("train", "valid")

# Large-like-0 inclusion rule
BOX0_AREA_RATIO = 0.75  # include bboxes with area >= 75% of box-0 area

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
    """Return (overlay image with labels, per-segment stats) – counts over full masks."""
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


def _collect_large_like_box0_bboxes(annotations, ratio: float = BOX0_AREA_RATIO) -> List[List[int]]:
    """
    From SAM annotations, pick every bbox whose area >= ratio * area(box-0).
    Returns a list of [x,y,w,h] (at least the box-0 bbox).
    """
    if not annotations:
        return []
    bbox0 = annotations[0].get("bbox")
    if bbox0 is None:
        x1, y1, x2, y2 = get_bbox_from_mask(annotations[0]["segmentation"])
        bbox0 = [x1, y1, x2 - x1, y2 - y1]
    _, _, w0, h0 = bbox0
    area0 = max(1, int(w0) * int(h0))

    keep: List[List[int]] = []
    for ann in annotations:
        bb = ann.get("bbox")
        if bb is None:
            x1, y1, x2, y2 = get_bbox_from_mask(ann["segmentation"])
            bb = [x1, y1, x2 - x1, y2 - y1]
        _, _, w, h = bb
        area = max(0, int(w) * int(h))
        if area >= ratio * area0:
            keep.append(bb)
    return keep


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
    NEW behavior:
    - Identify all bboxes whose area >= 75% of box-0's bbox area.
    - Build the ROI as the UNION of these bboxes.
    - Compute stats ONLY for pixels inside that ROI.
    - Draw labels only for segments that have >=1 pixel inside the ROI.
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

    # gather "large-like-0" bboxes (>= 75% of box0 area)
    large_bboxes = _collect_large_like_box0_bboxes(annotations, BOX0_AREA_RATIO)

    # stats only inside the union of those bboxes
    stats = compute_segment_stats_in_bboxes(annotations, image_resized, large_bboxes, min_pixels=1)

    # overlay (labels only for ids present in stats)
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
    Process images under resources/<subset> and save to generated/<subset>.
    If inside_bbox0=True, applies the new 'large-like-0' ROI rule described above.
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

        for p in src_dir.rglob("*"):
            if not p.is_file() or not _is_image_file(p):
                continue

            rel = p.relative_to(src_dir)
            stem = p.stem
            base = out_dir / rel.parent / stem
            overlay_path = str(base) + "_overlay.png"
            csv_path = str(base) + "_segments.csv"
            json_path = str(base) + "_segments.json"

            img = Image.open(p).convert("RGB")
            if inside_bbox0:
                overlay, stats = analyze_segments_in_bbox0(img)
            else:
                overlay, stats = analyze_segments(img)

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
    _ensure_dir(GENERATED_DIR)
    result = batch_process_resources(subsets=SUBSETS, inside_bbox0=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))

