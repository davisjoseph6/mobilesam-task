import os
import csv
import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import Image

from tools import (
    fast_process,
    compute_segment_stats,
    draw_segment_labels_pil,
    compute_segment_stats_in_bbox,
    get_bbox_from_mask,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam_checkpoint = "./mobile_sam.pt"
model_type = "vit_t"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
predictor = SamPredictor(mobile_sam)  # not used here, kept for future prompts


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

    # stats on resized image (matches mask resolution)
    stats = compute_segment_stats(annotations, image_resized)

    # visual overlay
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

    # bbox of segment 0
    bbox0 = annotations[0].get("bbox")
    if bbox0 is None:
        x1, y1, x2, y2 = get_bbox_from_mask(annotations[0]["segmentation"])
        bbox0 = [x1, y1, x2 - x1, y2 - y1]

    # stats only inside bbox0
    stats = compute_segment_stats_in_bbox(annotations, image_resized, bbox0)

    # overlay (labels will appear only for ids present in stats)
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


if __name__ == "__main__":
    input_path = "resources/dog.jpg"
    output_img = "generated/output.png"
    output_csv = "generated/output_stats.csv"

    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    image = Image.open(input_path).convert("RGB")

    # change to analyze_segments_in_bbox0(image) if you want bbox0 behavior in CLI mode
    result_img, stats = analyze_segments_in_bbox0(image=image)
    result_img.save(output_img)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pixels", "color_class", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        for s in stats:
            bx, by, bw, bh = (s["bbox_xywh"] or [None, None, None, None])
            writer.writerow([s["id"], s["pixels"], s["color_class"], bx, by, bw, bh])

    print(f"Saved segmented image: {output_img}")
    print(f"Saved stats CSV: {output_csv}")

