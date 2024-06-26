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

# Définir les transformations pour les données MNIST
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Charger les données d'entraînement et de test avec DataLoader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data/raw", download=True, train=True, transform=tf),
    batch_size=64, shuffle=True
)

test_load = torch.utils.data.DataLoader(
    datasets.MNIST("../data/raw", download=True, train=False, transform=tf),
    batch_size=64, shuffle=True
)

# Fonction pour afficher les 5 premières images du DataLoader
def show_images(data_loader):
    batch = next(iter(data_loader))
    x = batch[0][:5]
    y = batch[1][:5]

    # Afficher les images
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        ax = axes[i]
        image = x[i].numpy().squeeze()
        label = y[i].item()
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.show()

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

# Fonction d'entraînement
def train(model, train_loader, device, perm=torch.arange(0, 784).long(), n_epoch=1):
    model.train()
    optimizer = AdamW(model.parameters())
    
    for epoch in range(n_epoch):
        print(f"Epoch {epoch+1}/{n_epoch}")
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            data = data[:, perm]
            data = data.view(-1, 1, 28, 28)
            optimizer.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Iter: {batch_idx}, Loss: {loss.item()}')

# Fonction de test
def test(model, test_loader, device, perm=torch.arange(0, 784).long()):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            data = data[:, perm]
            data = data.view(-1, 1, 28, 28)
            logits = model(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = torch.argmax(logits, dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy * 100:.2f}%)')

# Définition du modèle MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.net(x)

def main():
    # Définir les paramètres des modèles
    n_kernels = 6
    input_size = 28*28
    output_size = 10
    hidden_size = 8

    # Créer une instance de ConvNet et de MLP
    convnet = ConvNet(input_size, n_kernels, output_size)
    mlp = MLP(input_size, hidden_size, output_size)

    # Déplacer les modèles sur le périphérique approprié
    convnet.to(device)
    mlp.to(device)

    # Afficher le nombre de paramètres pour chaque modèle
    print(f"Parameters ConvNet: {sum(p.numel() for p in convnet.parameters())/1e3}K")
    print(f"Parameters MLP: {sum(p.numel() for p in mlp.parameters())/1e3}K")

    # Entraîner et tester le modèle ConvNet
    print("\nTraining ConvNet...")
    train(convnet, train_loader, device)
    test(convnet, test_load, device)

    # Entraîner et tester le modèle MLP
    print("\nTraining MLP...")
    train(mlp, train_loader, device)
    test(mlp, test_load, device)

    # Sauvegarder le modèle ConvNet
    torch.save(convnet.state_dict(), "./model/mnist-0.0.1.pt")
    print("\nConvNet model saved as ./model/mnist-0.0.1.pt")

# Définir la transformation d'image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

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

model_path = "./model/mnist-0.0.1.pt"
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
