import json
import os

import mlflow
import mlflow.artifacts
import mlflow.pytorch
from fastapi import FastAPI

from fastapi import UploadFile, File
import io
from PIL import Image
import torch
from torchvision import transforms

app = FastAPI()

model = None
class_names = None

@app.on_event("startup")
def load_model():
    global model, class_names
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_uri = os.getenv("MODEL_URI", "models:/vehicle-type-classifier@champion")
    model = mlflow.pytorch.load_model(model_uri)
    print(f"Model loaded from {model_uri}")

    from mlflow.tracking import MlflowClient
    mv = MlflowClient().get_model_version_by_alias("vehicle-type-classifier", "champion")
    class_names_path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="model_meta/class_names.json"
    )
    with open(class_names_path) as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} class names")

IMG_SIZE = 224  # match training; adjust if your model expects a different size
_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = _tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].tolist()
    indexed = sorted(enumerate(probs), key=lambda p: p[1], reverse=True)
    top = [{"class": class_names[i], "confidence": float(p)} for i, p in indexed[:3]]
    all_probs = {class_names[i]: float(p) for i, p in enumerate(probs)}
    return {"top": top, "all": all_probs}