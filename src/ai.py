import os, time
import torch
import kagglehub
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


DATA_DIR = './data/dogs_vs_cats'
PERSISTENT_DATA = './persistent_data'
RESULTS_DIR = "./results"

ANIMALS = [ "Cat", "Dog" ]
EPOCHS = 5
BATCH = 64
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SimpleCNN(nn.Module):
    def __init__(self, total_class_num:int=len(ANIMALS)):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Calcola output shape dinamicamente
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)
            conv_out = self.conv_layers(dummy_input)
            conv_out_features = conv_out.view(1, -1).shape[1]  # Flattened size
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_features, 512),
            nn.ReLU(),
            nn.Linear(512, total_class_num)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def is_directory_empty(path: str, valid_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.csv')) -> bool:
    if not os.path.exists(path):
        return True
    for _,_, files in os.walk(path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                return False
    return True


def download_data(link="tongpython/cat-and-dog", path=DATA_DIR):
    load_dotenv()
    try:
        user = kagglehub.whoami()  # Verifica le credenziali
        print(f"Autenticated as: {user}")
    except Exception as e:
        print(f"Autentication Error: {e}")
        raise SystemExit(1) from e

    if (path is not None):
        os.makedirs(path, exist_ok=True)

    try:
        cache_path = kagglehub.dataset_download(
            link,
            force_download=True
        )
        print(f"Dataset scaricato nella cache: {cache_path}")
        for file in os.listdir(cache_path):
            src = os.path.join(cache_path, file)
            dst = os.path.join(path, file)

            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f" Dataset finale in: {path}")
        # Pulizia Cache
        user, dataset_name = link.split('/')
        cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / user / dataset_name
        shutil.rmtree(cache_root, ignore_errors=True)
        print(f"Pulizia cache: {cache_path}")
    except Exception as e:
        print(f"Errore durante l'operazione: {str(e)}")
        if os.path.exists(path):
            shutil.rmtree(path)
        raise SystemExit(1) from e


def eda(data_path=DATA_DIR, path_output=RESULTS_DIR, num_sample_img:int=1):
    train_data = ImageFolder(root=data_path+"/train", transform=TRANSFORM)
    train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
    
    test_data = ImageFolder(root=data_path+"/test", transform=TRANSFORM)
    _ = DataLoader(test_data, batch_size=BATCH, shuffle=True)

    print("EDA Reports:")
    print("- Train data size: ", len(train_data))
    print("- Test data size: ", len(test_data))
    print("- Class distribution: ", end="")
    labels = np.array(train_data.targets)
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(unique, counts))
    for class_idx, count in sorted(class_dist.items()):
        print(f"[{ANIMALS[int(class_idx)]}: {count}]", end=" ")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Plot class distribution
    plt.figure(figsize=(10,4))
    sns.barplot(x=[ANIMALS[k] for k in class_dist.keys()], y=list(class_dist.values()))
    plt.title("Class distribution")
    plt.xlabel("Animal")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path_output+"/class_distribution.png")
    plt.close()
    # Sample Images
    examples = enumerate(train_loader)
    _, (examples_data, examples_targets) = next(examples)
    plt.figure(figsize=(10,4))
    for i in range(min(num_sample_img, len(examples_data))):
        plt.subplot(2,5,i+1)
        plt.imshow(examples_data[i].permute(1, 2, 0))
        plt.title(f"Label: {ANIMALS[examples_targets[i].item()]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path_output+"/samples_images.png")
    plt.close()

# Funzione di addestramento
def train_save(model=None, model_path=PERSISTENT_DATA, data_path=DATA_DIR, epochs=EPOCHS, path_output=RESULTS_DIR):
    if model is None:
        model = SimpleCNN()
        model_path = os.path.join(model_path, "dogs_vs_cats_model.pth")
        if os.path.exists(model_path):
            print("[INFO] Modello non trovato. Addestramento in corso...")
            model.load_state_dict(torch.load(model_path))
    train_data = ImageFolder(root=data_path+"/train", transform=TRANSFORM)
    train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}")
    print(f"Training time: {time.time()-start:.2f} seconds")
    print(f"Total Loss: {running_loss/len(train_loader):.4f}")

    os.makedirs(path_output, exist_ok=True)
    torch.save(model.state_dict(), path_output+'/dogs_vs_cats_model.pth')

def evaluate(model=None, model_path=PERSISTENT_DATA, data_path=DATA_DIR, path_output=RESULTS_DIR):
    if model is None:
        model = SimpleCNN()
        model_path = os.path.join(model_path, "dogs_vs_cats_model.pth")
        if os.path.exists(model_path):
            print("[INFO] Modello non trovato. Addestramento in corso...")
            model.load_state_dict(torch.load(model_path))
    test_data = ImageFolder(root=data_path+"/test", transform=TRANSFORM)
    test_loader = DataLoader(test_data, batch_size=BATCH, shuffle=False)
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    accuracy = correct/total

    print(f"Test sull'accuratezza: {accuracy:.4f}")
    # Classification Report
    os.makedirs(path_output, exist_ok=True)
    print("Classification Report: ", classification_report(all_labels, all_preds))
    # Plot Confusion Matrix
    name = path_output+"/confusion_matrix.png"
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=ANIMALS, yticklabels=ANIMALS)
    plt.xlabel("Animale Ottenuto")
    plt.ylabel("Animale Previsto")
    plt.title("Confusion Matrix")
    plt.savefig(name)
    plt.close()
    print(f"Salvata la matrice di confusione in {name}")

def inference(img, model=None):
    if model is None:
        model = SimpleCNN()
        model_path = os.path.join(PERSISTENT_DATA, "dogs_vs_cats_model.pth")
        if not os.path.exists(model_path):
            print("[INFO] Modello non trovato. Addestramento in corso...")
            train_save(model)
        model.load_state_dict(torch.load(model_path))
    image = img.convert("RGB")
    image = TRANSFORM(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return 'Cat' if predicted.item() == 0 else 'Dog'

if __name__=="__main__":
    if is_directory_empty(DATA_DIR):
        print("[INFO] Downloading dataset...")
        download_data()
    eda()
    m = SimpleCNN()
    train_save(m)
    evaluate()