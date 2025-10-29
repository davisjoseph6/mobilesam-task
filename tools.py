import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def segment_image(image, bbox):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    x1, y1, x2, y2 = bbox
    segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
    segmented_image = Image.fromarray(segmented_image_array)
    white_bg = Image.new("RGB", image.size, (255, 255, 255))
    transparency_mask = np.zeros(
        (image_array.shape[0], image_array.shape[1]), dtype=np.uint8
    )
    transparency_mask[y1:y2, x1:x2] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
    white_bg.paste(segmented_image, mask=transparency_mask_image)
    return white_bg


def format_results(masks, scores, logits, filter=0):
    annotations = []
    n = len(scores)
    for i in range(n):
        annotation = {}
        mask = masks[i]
        tmp = np.where(mask != 0)
        if np.sum(mask) < filter:
            continue
        annotation["id"] = i
        annotation["segmentation"] = mask
        annotation["bbox"] = [
            np.min(tmp[0]),
            np.min(tmp[1]),
            np.max(tmp[1]),
            np.max(tmp[0]),
        ]
        annotation["score"] = scores[i]
        annotation["area"] = annotation["segmentation"].sum()
        annotations.append(annotation)
    return annotations


def filter_masks(annotations):
    annotations.sort(key=lambda x: x["area"], reverse=True)
    to_remove = set()
    for i in range(0, len(annotations)):
        a = annotations[i]
        for j in range(i + 1, len(annotations)):
            b = annotations[j]
            if i != j and j not in to_remove:
                if b["area"] < a["area"]:
                    if (a["segmentation"] & b["segmentation"]).sum() / b[
                        "segmentation"
                    ].sum() > 0.8:
                        to_remove.add(j)
    return [a for i, a in enumerate(annotations) if i not in to_remove], to_remove


def get_bbox_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x1, y1, w, h = cv2.boundingRect(contours[0])
    x2, y2 = x1 + w, y1 + h
    if len(contours) > 1:
        for b in contours:
            x_t, y_t, w_t, h_t = cv2.boundingRect(b)
            x1 = min(x1, x_t)
            y1 = min(y1, y_t)
            x2 = max(x2, x_t + w_t)
            y2 = max(y2, y_t + h_t)
    return [x1, y1, x2, y2]


# === COLOR + PIXEL ANALYSIS UTILITIES ===

def _name_color_from_hsv(h, s, v):
    """Rough mapping of mean HSV values to a coarse color class."""
    if v < 40:
        return "black"
    if s < 30 and v > 220:
        return "white"
    if s < 30:
        return "gray"
    if h < 10 or h >= 170:
        return "red"
    if h < 25:
        return "orange"
    if h < 35:
        return "brown"
    if h < 85:
        return "green"
    if h < 105:
        return "cyan"
    if h < 135:
        return "blue"
    if h < 155:
        return "purple"
    return "magenta"


def compute_segment_stats(annotations, image_rgb):
    """
    Compute per-segment pixel counts and a coarse color label from mean HSV.
    Counts over the entire mask area.
    """
    if hasattr(image_rgb, "convert"):
        img_np = np.array(image_rgb.convert("RGB"))
    else:
        img_np = image_rgb
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    stats = []
    for i, ann in enumerate(annotations):
        mask = ann["segmentation"].astype(bool)
        pixels = int(mask.sum())
        if pixels == 0:
            continue
        h_vals = hsv[..., 0][mask]
        s_vals = hsv[..., 1][mask]
        v_vals = hsv[..., 2][mask]
        h_mean = int(np.round(h_vals.mean()))
        s_mean = int(np.round(s_vals.mean()))
        v_mean = int(np.round(v_vals.mean()))
        color_class = _name_color_from_hsv(h_mean, s_mean, v_mean)
        stats.append({
            "id": i,
            "pixels": pixels,
            "bbox_xywh": ann.get("bbox", None),  # SAM gives [x,y,w,h]
            "color_class": color_class,
            "mean_hsv": [h_mean, s_mean, v_mean],
        })
    return stats


def compute_segment_stats_in_bbox(annotations, image_rgb, bbox_xywh, min_pixels=1):
    """
    Counts only pixels INSIDE a single bbox (x,y,w,h). Returns stats for all segments
    that intersect this bbox (id is the segment id).
    """
    if hasattr(image_rgb, "convert"):
        img_np = np.array(image_rgb.convert("RGB"))
    else:
        img_np = image_rgb

    H, W = img_np.shape[:2]
    if bbox_xywh is None:
        return []
    x, y, w, h = bbox_xywh
    x1 = max(0, int(round(x))); y1 = max(0, int(round(y)))
    x2 = min(W, int(round(x + w))); y2 = min(H, int(round(y + h)))
    if x1 >= x2 or y1 >= y2:
        return []

    roi = np.zeros((H, W), dtype=bool)
    roi[y1:y2, x1:x2] = True

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    stats = []
    for i, ann in enumerate(annotations):
        mask = ann["segmentation"].astype(bool)
        inter = mask & roi
        pixels = int(inter.sum())
        if pixels < min_pixels:
            continue

        h_vals = hsv[..., 0][inter]
        s_vals = hsv[..., 1][inter]
        v_vals = hsv[..., 2][inter]
        h_mean = int(np.round(h_vals.mean()))
        s_mean = int(np.round(s_vals.mean()))
        v_mean = int(np.round(v_vals.mean()))
        color_class = _name_color_from_hsv(h_mean, s_mean, v_mean)

        stats.append({
            "id": i,
            "pixels": pixels,
            "bbox_xywh": ann.get("bbox", None),
            "color_class": color_class,
            "mean_hsv": [h_mean, s_mean, v_mean],
        })
    return stats


def compute_segment_stats_in_bboxes(annotations, image_rgb, bboxes_xywh, min_pixels=1):
    """
    Counts pixels INSIDE the UNION of multiple bboxes (list of [x,y,w,h]).
    Returns stats for all segments that intersect the union ROI.
    """
    if hasattr(image_rgb, "convert"):
        img_np = np.array(image_rgb.convert("RGB"))
    else:
        img_np = image_rgb

    H, W = img_np.shape[:2]
    if not bboxes_xywh:
        return []

    roi = np.zeros((H, W), dtype=bool)
    for bbox in bboxes_xywh:
        if bbox is None:
            continue
        x, y, w, h = bbox
        x1 = max(0, int(round(x))); y1 = max(0, int(round(y)))
        x2 = min(W, int(round(x + w))); y2 = min(H, int(round(y + h)))
        if x1 < x2 and y1 < y2:
            roi[y1:y2, x1:x2] = True

    if not roi.any():
        return []

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    stats = []
    for i, ann in enumerate(annotations):
        mask = ann["segmentation"].astype(bool)
        inter = mask & roi
        pixels = int(inter.sum())
        if pixels < min_pixels:
            continue
        h_vals = hsv[..., 0][inter]
        s_vals = hsv[..., 1][inter]
        v_vals = hsv[..., 2][inter]
        h_mean = int(np.round(h_vals.mean()))
        s_mean = int(np.round(s_vals.mean()))
        v_mean = int(np.round(v_vals.mean()))
        color_class = _name_color_from_hsv(h_mean, s_mean, v_mean)
        stats.append({
            "id": i,
            "pixels": pixels,
            "bbox_xywh": ann.get("bbox", None),
            "color_class": color_class,
            "mean_hsv": [h_mean, s_mean, v_mean],
        })
    return stats


def draw_segment_labels_pil(image, annotations, stats, font_size=14):
    """Overlay '#<id>  <class> | <pixels> px' near each segment bbox."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    stats_by_id = {s["id"]: s for s in stats}
    for i, ann in enumerate(annotations):
        if i not in stats_by_id:
            continue
        s = stats_by_id[i]
        bbox = ann.get("bbox", None)
        if bbox is None:
            x1, y1, x2, y2 = get_bbox_from_mask(ann["segmentation"])
        else:
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

        label = f"#{s['id']}  {s['color_class']} | {s['pixels']} px"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=1)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        bg = [x1, max(0, y1 - th - 4), x1 + tw + 4, y1]
        draw.rectangle(bg, fill=(0, 0, 0))
        draw.text((bg[0] + 2, bg[1] + 2), label, fill=(255, 255, 255), font=font)
    return out


# === FAST SHOW / MASK OVERLAYS ===

def fast_process(
    annotations,
    image,
    device,
    scale,
    better_quality=False,
    mask_random_color=True,
    bbox=None,
    use_retina=True,
    withContours=True,
):
    if isinstance(annotations[0], dict):
        annotations = [annotation["segmentation"] for annotation in annotations]

    original_h = image.height
    original_w = image.width

    if better_quality:
        if isinstance(annotations[0], torch.Tensor):
            annotations = np.array(annotations.cpu())
        for i, mask in enumerate(annotations):
            mask = cv2.morphologyEx(mask.astype(np.uint8),
                                    cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            annotations[i] = cv2.morphologyEx(mask.astype(np.uint8),
                                              cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))

    if device == "cpu":
        annotations = np.array(annotations)
        inner_mask = fast_show_mask(
            annotations, plt.gca(), random_color=mask_random_color,
            bbox=bbox, retinamask=use_retina,
            target_height=original_h, target_width=original_w)
    else:
        if isinstance(annotations[0], np.ndarray):
            annotations = torch.from_numpy(np.array(annotations))
        inner_mask = fast_show_mask_gpu(
            annotations, plt.gca(), random_color=mask_random_color,
            bbox=bbox, retinamask=use_retina,
            target_height=original_h, target_width=original_w)

    if isinstance(annotations, torch.Tensor):
        annotations = annotations.cpu().numpy()

    if withContours:
        contour_all = []
        temp = np.zeros((original_h, original_w, 1))
        for mask in annotations:
            annotation = mask.astype(np.uint8)
            if not use_retina:
                annotation = cv2.resize(annotation, (original_w, original_h),
                                        interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_all.extend(contours)
        thickness = max(1, int(2 // max(1, scale)))
        cv2.drawContours(temp, contour_all, -1, (255, 255, 255), thickness)
        color = np.array([0 / 255, 0 / 255, 255 / 255, 0.9])
        contour_mask = temp / 255 * color.reshape(1, 1, -1)

    image = image.convert("RGBA")
    overlay_inner = Image.fromarray((inner_mask * 255).astype(np.uint8), "RGBA")
    image.paste(overlay_inner, (0, 0), overlay_inner)

    if withContours:
        overlay_contour = Image.fromarray((contour_mask * 255).astype(np.uint8), "RGBA")
        image.paste(overlay_contour, (0, 0), overlay_contour)

    return image


def fast_show_mask(annotation, ax, random_color=False, bbox=None,
                   retinamask=True, target_height=960, target_width=960):
    mask_sum = annotation.shape[0]
    height, weight = annotation.shape[1:3]
    areas = np.sum(annotation, axis=(1, 2))
    sorted_indices = np.argsort(areas)
    annotation = annotation[sorted_indices]
    index = (annotation != 0).argmax(axis=0)

    color = np.random.random((mask_sum, 1, 1, 3)) if random_color else np.ones(
        (mask_sum, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 255 / 255])
    transparency = np.ones((mask_sum, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(annotation, -1) * visual

    mask = np.zeros((height, weight, 4))
    h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing="ij")
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    mask[h_indices, w_indices, :] = mask_image[indices]

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="b", linewidth=1))
    if not retinamask:
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return mask


def fast_show_mask_gpu(annotation, ax, random_color=False, bbox=None,
                       retinamask=True, target_height=960, target_width=960):
    device = annotation.device
    mask_sum, height, weight = annotation.shape
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    index = (annotation != 0).to(torch.long).argmax(dim=0)

    color = torch.rand((mask_sum, 1, 1, 3), device=device) if random_color else \
        torch.ones((mask_sum, 1, 1, 3), device=device) * torch.tensor(
            [30 / 255, 144 / 255, 255 / 255], device=device)
    transparency = torch.ones((mask_sum, 1, 1, 1), device=device) * 0.6
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual

    h_indices, w_indices = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(weight, device=device), indexing="ij"
    )
    index = index.to(device)
    mask = torch.zeros((height, weight, 4), device=device)
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    mask[h_indices, w_indices, :] = mask_image[indices]

    mask_cpu = mask.cpu().numpy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="b", linewidth=1))
    if not retinamask:
        mask_cpu = cv2.resize(mask_cpu, (target_width, target_height),
                              interpolation=cv2.INTER_NEAREST)
    return mask_cpu

