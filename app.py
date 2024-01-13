from fastapi import FastAPI
from main import segment_everything

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
