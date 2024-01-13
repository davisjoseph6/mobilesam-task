from fastapi iport FastAPI


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}
