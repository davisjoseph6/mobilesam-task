from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Query
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import base64
import json
import csv
import zipfile
import os
import pathlib
from typing import List

from main import (
    segment_everything,
    analyze_segments,
    analyze_segments_in_bbox0,
    batch_process_resources,
    RESOURCES_DIR,
    GENERATED_DIR,
    SUBSETS,
)

app = FastAPI()


# ----------------- Single-file endpoints (unchanged) -----------------

@app.post("/segment-image")
async def segment_image(file: UploadFile = File(...)):
    # Return raw PNG bytes (no stats labels)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        output_img = segment_everything(image)  # overlay without labels (original)
        buf = io.BytesIO()
        output_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment-image-with-stats")
async def segment_image_with_stats(
    file: UploadFile = File(...),
    inside_bbox0: bool = Query(
        True,
        description="If true, stats are computed only inside bbox of segment id=0",
    ),
):
    # Return JSON: base64 PNG + segments stats
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        if inside_bbox0:
            overlay, stats = analyze_segments_in_bbox0(image)
        else:
            overlay, stats = analyze_segments(image)

        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        b64_png = base64.b64encode(buf.getvalue()).decode("ascii")

        return {"image_png_base64": b64_png, "segments": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment-image-overlay")
async def segment_image_overlay(
    file: UploadFile = File(...),
    inside_bbox0: bool = Query(
        True,
        description="If true, labels reflect stats only inside bbox id=0",
    ),
):
    """Return the actual PNG image with labels drawn at their corresponding locations."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        overlay, _stats = analyze_segments_in_bbox0(image) if inside_bbox0 else analyze_segments(image)

        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment-image-overlay-with-stats")
async def segment_image_overlay_with_stats(
    file: UploadFile = File(...),
    inside_bbox0: bool = Query(
        True,
        description="If true, stats only inside bbox id=0",
    ),
):
    """
    Return a ZIP containing:
      - overlay.png       (image with labels)
      - segments.json     (stats as JSON)
      - segments.csv      (stats as CSV)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        overlay, stats = analyze_segments_in_bbox0(image) if inside_bbox0 else analyze_segments(image)

        # Prepare overlay.png
        png_buf = io.BytesIO()
        overlay.save(png_buf, format="PNG")
        png_bytes = png_buf.getvalue()

        # Prepare segments.json
        json_bytes = json.dumps(stats, ensure_ascii=False, indent=2).encode("utf-8")

        # Prepare segments.csv
        csv_sio = io.StringIO()
        writer = csv.writer(csv_sio)
        writer.writerow(["id", "pixels", "color_class", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        for s in stats:
            bx, by, bw, bh = (s.get("bbox_xywh") or [None, None, None, None])
            writer.writerow([s["id"], s["pixels"], s["color_class"], bx, by, bw, bh])
        csv_bytes = csv_sio.getvalue().encode("utf-8")

        # Build ZIP in-memory
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("overlay.png", png_bytes)
            z.writestr("segments.json", json_bytes)
            z.writestr("segments.csv", csv_bytes)
        zip_buf.seek(0)

        headers = {"Content-Disposition": 'attachment; filename="segmented_bundle.zip"'}
        return Response(content=zip_buf.getvalue(), media_type="application/zip", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- New: batch ingest & download -----------------

@app.post("/batch/process-resources")
async def batch_process(
    subset: str = Query(
        "all",
        description="Which subset to process: 'train', 'valid', or 'all'."
    ),
    inside_bbox0: bool = Query(
        True,
        description="If true, stats are computed only inside bbox of segment id=0."
    ),
):
    """
    Scan resources/train and/or resources/valid for acceptable images,
    run segmentation on each, and write results into generated/<subset>/.
    Returns a manifest of processed files and output paths.
    """
    subset = subset.lower()
    if subset == "all":
        subsets = SUBSETS
    elif subset in SUBSETS:
        subsets = (subset,)
    else:
        raise HTTPException(status_code=400, detail="subset must be 'train', 'valid', or 'all'")

    try:
        manifest = batch_process_resources(subsets=subsets, inside_bbox0=inside_bbox0)
        return {"resources_dir": RESOURCES_DIR, "generated_dir": GENERATED_DIR, "manifest": manifest}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _zip_dir_to_stream(root_dir: str, subfolders: List[str]) -> io.BytesIO:
    """
    Create an in-memory zip of generated/<subfolder>... for each in subfolders.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for sub in subfolders:
            base_path = pathlib.Path(root_dir) / sub
            if not base_path.exists():
                # include a placeholder note so caller knows it was missing
                z.writestr(f"{sub}/__EMPTY__.txt", "No files found.")
                continue
            for p in base_path.rglob("*"):
                if p.is_file():
                    arcname = str(p.relative_to(root_dir))
                    z.write(p, arcname=arcname)
    buf.seek(0)
    return buf


@app.get("/batch/download-subset")
async def batch_download_subset(
    subset: str = Query(..., description="Which subset to download: 'train' or 'valid'.")
):
    """
    Stream a ZIP of generated/<subset>.
    """
    subset = subset.lower()
    if subset not in SUBSETS:
        raise HTTPException(status_code=400, detail="subset must be 'train' or 'valid'")
    try:
        buf = _zip_dir_to_stream(GENERATED_DIR, [subset])
        filename = f"generated_{subset}.zip"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=buf.getvalue(), media_type="application/zip", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch/download-all")
async def batch_download_all():
    """
    Stream a ZIP containing both generated/train and generated/valid (if present).
    """
    try:
        buf = _zip_dir_to_stream(GENERATED_DIR, list(SUBSETS))
        headers = {"Content-Disposition": 'attachment; filename="generated_all.zip"'}
        return Response(content=buf.getvalue(), media_type="application/zip", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Root -----------------

@app.get("/")
def read_root():
    return {
        "Hello": "Welcome to MobileSAM segmentation service",
        "single_file_endpoints": [
            "POST /segment-image",
            "POST /segment-image-with-stats?inside_bbox0=true|false",
            "POST /segment-image-overlay?inside_bbox0=true|false",
            "POST /segment-image-overlay-with-stats?inside_bbox0=true|false",
        ],
        "batch_endpoints": [
            "POST /batch/process-resources?subset=train|valid|all&inside_bbox0=true|false",
            "GET  /batch/download-subset?subset=train|valid",
            "GET  /batch/download-all",
        ],
        "folders": {
            "resources_dir": RESOURCES_DIR,
            "generated_dir": GENERATED_DIR,
            "subsets": list(SUBSETS),
        },
        "acceptable_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"],
    }

