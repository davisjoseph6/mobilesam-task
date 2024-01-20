from fastapi import FastAPI, File, UploadFile, HTTPException, status, Response
from PIL import Image
import io
import logging
from main import segment_everything

app = FastAPI()

@app.post("/segment-image")
async def segment_image(file: UploadFile = File(...)):
    # Ensure the file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File is not an image.")

    try:
        # Read the contents of the uploaded file
        contents = await file.read()

        # Convert the contents to a byte stream
        image_stream = io.BytesIO(contents)

        # Open the byte stream as an image object
        image = Image.open(image_stream).convert("RGB")

        # Process the image with the MobileSam model
        segmented_image = segment_everything(image)

        # Convert the segmented image to a byte array response
        img_byte_arr = io.BytesIO()
        segmented_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")

    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"Hello: Welcome to MobileSam segmentation service"}
