from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import torch

from model_training import train_model, model
from adversarial_defenses import adversarial_training
from adversarial_attacks import fgsm_attack
from ai_threat_intelligence import detect_future_threats
from adversarial_detection import classify_input

app = FastAPI()


class ImageInput(BaseModel):
    image: list  # Expect a (28x28) nested list or flattened 784 list


@app.post("/train")
def train():
    train_model()
    return {"message": "Model trained successfully."}


@app.post("/adversarial_training")
def defensive_train():
    adversarial_training()
    return {"message": "Adversarial training done."}


@app.post("/generate_adversarial")
def generate_adversarial(input_data: ImageInput):
    img_array = np.array(input_data.image, dtype=np.float32)
    if img_array.ndim == 1:
        img_array = img_array.reshape(1, 1, 28, 28)
    elif img_array.ndim == 2:
        img_array = img_array.reshape(1, 1, 28, 28)
    adv = fgsm_attack(model, img_array, eps=0.1)
    return {"adversarial_image": adv[0].flatten().tolist()}


@app.post("/detect_adversarial")
def detect_adversarial(input_data: ImageInput):
    img_tensor = torch.tensor(input_data.image, dtype=torch.float32).view(1, 28, 28)
    result = classify_input(img_tensor)
    return {"classification": result}


@app.get("/threat_assessment")
def threat_assessment():
    attack_history = np.random.rand(100, 2)
    risk = detect_future_threats(attack_history)
    return {"risk_level": risk}


# Example additional endpoint for image uploads (Frontend integration)
@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    # You can add PIL or OpenCV processing here to convert to a tensor/array
    # For demo, just respond success
    return {"filename": file.filename, "status": "uploaded"}

# Add more endpoints as needed for federated audit, distillation, evaluation, etc.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)