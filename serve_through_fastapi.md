To serve your [Doraemon vs. Dorami model](https://github.com/GhulamMuhammadNabeel/doraemon_vs-dorami-classification-using-cnn/blob/main/doraemon-dorami-classification-using-cnn-scratch.ipynb) with FastAPI, create an API that loads `model.keras` once during startup, preprocesses uploaded images to match the MobileNetV2 training pipeline (224×224 RGB normalized between -1 and 1), and maps the single sigmoid output to the correct class (`0: Doraemon`, `1: Dorami`).

---

### 1. Project Setup and Dependencies

Create a clean directory for your project:

```text
dorami-doraemon-api/
├── model.keras
├── main.py
└── requirements.txt

```

In `requirements.txt`, include:

```text
fastapi>=0.110.0
uvicorn[standard]>=0.28.0
python-multipart>=0.0.9
tensorflow>=2.15.0
pillow>=10.0.0
numpy>=1.24.0

```

Install them in your virtual environment:

```bash
pip install -r requirements.txt

```

---

### 2. FastAPI Application (`main.py`)

Using FastAPI's `lifespan` context manager ensures the model is loaded into memory only once when the server boots, preventing overhead and latency spikes during request handling.

```python
import io
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input

CLASSES = ["Doraemon", "Dorami"]
TARGET_SIZE = (224, 224)

# Dictionary to hold the model instance across the app lifecycle
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    try:
        ml_models["classifier"] = tf.keras.models.load_model("model.keras")
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(
    title="Doraemon vs Dorami Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)

def prepare_image(image_bytes: bytes) -> np.ndarray:
    """Decodes, resizes, and applies MobileNetV2 preprocessing."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Ensure image is in 3-channel RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize to match input shape (224, 224)
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img, dtype=np.float32)
        
        # Add batch dimension: (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Scales pixel values to [-1, 1] as used during training
        return preprocess_input(img_batch)
    except Exception as exc:
        raise ValueError("Invalid image file format.") from exc

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": "classifier" in ml_models}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate MIME type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file is not an image (received: {file.content_type}).",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file submitted.")

    try:
        processed_img = prepare_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = ml_models["classifier"]
    
    # Model inference: forward pass with training=False is faster than predict()
    prediction = model(processed_img, training=False).numpy()[0][0]
    dorami_prob = float(prediction)
    doraemon_prob = float(1.0 - dorami_prob)

    # Sigmoid output: > 0.5 indicates Dorami (label 1), <= 0.5 indicates Doraemon (label 0)
    if dorami_prob > 0.5:
        predicted_class = CLASSES[1]
        confidence = dorami_prob
    else:
        predicted_class = CLASSES[0]
        confidence = doraemon_prob

    return JSONResponse(
        content={
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Doraemon": round(doraemon_prob, 4),
                "Dorami": round(dorami_prob, 4),
            },
        }
    )

```

---

### 3. Run and Test the API

Start the Uvicorn ASGI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```

Interactive OpenAPI documentation is automatically available at `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`, where you can test the file upload directly in the browser.

#### Test with cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/test_image.jpg"

```

#### Test with Python (`requests`)

```python
import requests

url = "http://127.0.0.1:8000/predict"
file_path = "test_dorami.png"

with open(file_path, "rb") as f:
    response = requests.post(url, files={"file": (file_path, f, "image/png")})

print(response.json())

```

**Expected JSON response format:**

```json
{
  "predicted_class": "Dorami",
  "confidence": 0.9842,
  "probabilities": {
    "Doraemon": 0.0158,
    "Dorami": 0.9842
  }
}

```
