from fastapi import FastAPI, File, UploadFile
from main import segment_everything

app = FastAPI()

@app.post("/segment-image")
async def segment_image(file: UploadFile = File(...))):
    # Further processing will be done here
    pass

@app.get("/")
def read_root():
    return {"Hello": "World"}
