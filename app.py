from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from PIL import Image
import io
import base64
import json
import csv
import zipfile

from main import segment_everything, analyze_segments

app = FastAPI()


@app.post("/segment-image")
async def segment_image(file: UploadFile = File(...)):
    # Return raw PNG bytes (no stats labels)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        output_img = segment_everything(image)  # overlay without labels (your original)
        buf = io.BytesIO()
        output_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment-image-with-stats")
async def segment_image_with_stats(file: UploadFile = File(...)):
    # Return JSON: base64 PNG + segments stats
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        overlay, stats = analyze_segments(image)

        # overlay as base64 PNG
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        b64_png = base64.b64encode(buf.getvalue()).decode("ascii")

        return {"image_png_base64": b64_png, "segments": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment-image-overlay")
async def segment_image_overlay(file: UploadFile = File(...)):
    """
    NEW: Return the actual PNG image with labels drawn at their corresponding locations.
    Useful when you only want the visual output (no JSON).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        overlay, _stats = analyze_segments(image)

        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment-image-overlay-with-stats")
async def segment_image_overlay_with_stats(file: UploadFile = File(...)):
    """
    NEW: Return a ZIP containing:
      - overlay.png       (image with labels)
      - segments.json     (stats as JSON)
      - segments.csv      (stats as CSV)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        overlay, stats = analyze_segments(image)

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

        headers = {
            "Content-Disposition": 'attachment; filename="segmented_bundle.zip"'
        }
        return Response(content=zip_buf.getvalue(), media_type="application/zip", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"Hello": "Welcome to MobileSAM segmentation service"}

