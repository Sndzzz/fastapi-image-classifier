from fastapi import FastAPI, UploadFile, File
import uvicorn
from app.model import predict

app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict(image_bytes)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7001)
