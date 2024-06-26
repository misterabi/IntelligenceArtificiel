# Imports nécessaires
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import uvicorn
from PIL import Image
import numpy as np
from pydantic import BaseModel
from typing import Any

# Déterminer le périphérique GPU si disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Définition de la classe ConvNet
class ConvNet(nn.Module):
    def __init__(self, input_size, n_kernels, output_size):
        super(ConvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=n_kernels, out_channels=n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=n_kernels*4*4, out_features=50),
            nn.Linear(in_features=50, out_features=output_size)
        )
    
    def forward(self, x):
        return self.net(x)

# Fonction pour charger le modèle pré-entraîné
def load_model(model_class, model_path):
    input_size = 28*28
    n_kernels = 6
    output_size = 10
    model = model_class(input_size, n_kernels, output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model_path = "./mnist-0.0.1.pt"
model = load_model(ConvNet, model_path)

app = FastAPI()
class ImageInput(BaseModel):
    image: bytes

# Fonction pour charger votre modèle et faire des prédictions


# Fonction pour prédire à partir de l'image reçue
def predict_image(image: list, model):
    image_np = np.array(image)
    # Prétraitement de l'image si nécessaire
    # Exemple : redimensionnement, normalisation, etc.

    image_np = image_np.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(np.expand_dims(image_np, axis=0))
    perm=torch.arange(0, 784)
    with torch.no_grad():
        image_tensor = image_tensor.view(-1, 28*28)
        image_tensor = image_tensor[:, perm]
        image_tensor = image_tensor.view(-1, 1, 28, 28)
        model_output = model(image_tensor)

    # Exemple : retourner la classe prédite (à adapter selon votre modèle)
    predicted_class = torch.argmax(model_output, dim=1).item()
    return predicted_class


class InferenceImage(BaseModel):
    image: Any

# Route FastAPI pour la prédiction
@app.post("/api/v1/predict")
def predict(inference: InferenceImage):
    try:
        print(f"COUCOUCSSSSSS : {type(inference.image)}")
        image_bytes = inference.image
        prediction = predict_image(image_bytes, model)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
