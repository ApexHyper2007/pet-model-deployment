from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from predict import load_model, predict_image_bytes


@asynccontextmanager
async def lifespan(app):
    """Load the model once when the API starts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    app.state.model = load_model(device)
    yield


app = FastAPI(
    title="Pet Breed Classifier API",
    description="Predicts the breed of a cat or dog from an uploaded image.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def health_check():
    """Return a simple service status response."""
    return {"status": "ok", "service": "pet-breed-classifier"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the breed from an uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Upload a valid image file.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        predictions = predict_image_bytes(
            app.state.model,
            image_bytes,
            app.state.device,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail="Model inference failed.",
        ) from error

    return {
        "filename": file.filename,
        "predicted_breed": predictions[0][0],
        "confidence": predictions[0][1],
        "top_predictions": [
            {"breed": breed, "confidence": confidence}
            for breed, confidence in predictions
        ],
    }
