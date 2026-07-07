import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import uvicorn

from predict import predict_digit_detailed, show_preprocessed

app = FastAPI(title="Neural Digit Scanner API")

# Enable CORS for local testing if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    image_base64: str  # Base64 data URL e.g. "data:image/png;base64,..."

@app.post("/api/predict")
async def predict(req: PredictRequest):
    try:
        # Extract base64 content
        header, encoded = req.image_base64.split(",", 1) if "," in req.image_base64 else ("", req.image_base64)
        image_data = base64.b64decode(encoded)
        img_pil = Image.open(BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    try:
        # Get predictions
        digit, confidence, probabilities = predict_digit_detailed(img_pil)
        
        # Get preprocessed 28x28 image
        pre_img_np = show_preprocessed(img_pil)
        pre_img = Image.fromarray(pre_img_np)
        
        # Save to base64
        buffered = BytesIO()
        pre_img.save(buffered, format="PNG")
        pre_img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        pre_img_url = f"data:image/png;base64,{pre_img_b64}"
        
        return {
            "digit": digit,
            "confidence": confidence,
            "probabilities": probabilities,
            "preprocessed_image": pre_img_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

import os

# Mount the static files directory for the frontend if it exists (for unified hosting/local run)
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
else:
    @app.get("/")
    async def root_status():
        return {
            "status": "online",
            "message": "Neural Digit Scanner API is active. Frontend is hosted on Vercel."
        }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
