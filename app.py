from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from main import segment_everything

app = FastAPI()

@app.post("/segment-image")
async def segment_image(file: UploadFile = File(...)):
    # Read the contents of the uploaded file
    contents = await file.read()

    # Convert the contents to a byte stream
    image_stream = io.BytesIO(contents)

    # Open the byte stream as an image object
    image = Image.open(image_stream).convert("RGB")

    # Now, 'image' is a PIL Image object and can be processed

@app.get("/")
def read_root():
    return {"Hello": "World"}
