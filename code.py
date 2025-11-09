Author: Kondru Naga Sandeep
Team: JNTU KAKINADA
Date: 09/11/2025

import os
import random
import numpy as np
from zipfile import ZipFile
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder

import pennylane as qml
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#  Configuration 
DATA_DIR_CUSTOM = "./data/custom_dataset"   # path where your unzipped dataset should be
USE_CUSTOM = True                            # set False to skip custom detection and use EMNIST
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 12              # increase for better training
FEATURE_DIM = 12         # frontend CNN output size (features)
N_QUBITS = 6             # qubits in VQC (keep small: 4-8 recommended)
VQC_LAYERS = 2
SEED = 42
SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)

#  reproducibility 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# helpers
def add_noise_tensor(x, sigma=0.1):
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)

#  Dataset loading 
def load_custom_dataset(root_dir, batch_size):
    """
    Expects folder structure root_dir/class_x/*.png ...
    Returns train_loader, val_loader, test_loader, class_names
    """
    if not os.path.isdir(root_dir):
        return None

    print("Custom dataset found at", root_dir)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ])

    full_dataset = ImageFolder(root=root_dir, transform=transform)
    class_names = full_dataset.classes
    print("Detected classes:", class_names)
    # split 70/15/15
    n = len(full_dataset)
    if n < 50:
        print("Warning: very small dataset (%d samples). Consider collecting more samples." % n)
    train_n = int(0.7 * n)
    val_n = int(0.15 * n)
    test_n = n - train_n - val_n
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_n, val_n, test_n], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, test_loader, class_names

def load_emnist_balanced(batch_size):
    """
    Fallback: loads EMNIST Balanced split (digits + letters)
    Note: EMNIST images are 28x28 but rotated; torchvision handles rotation flag via split arg.
    """
    print("Loading EMNIST Balanced (fallback). This may download ~30-50MB.")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, k=1, dims=[1,2])),  # EMNIST images may need rotation; keep to display correctly
    ])
    emnist_train = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
    emnist_test = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)

    # split train into train/val
    train_size = int(0.92 * len(emnist_train))
    val_size = len(emnist_train) - train_size
    train_ds, val_ds = random_split(emnist_train, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(emnist_test, batch_size=batch_size, shuffle=False, pin_memory=True)

    # build class names for EMNIST balanced (mapping provided by dataset)
    # torchvision EMNIST has attribute 'classes'? no; we'll construct approximate labels because EMNIST balanced has labels 0..46
    # For practical purposes provide label names as numbers; user will interpret mapping if needed.
    class_names = [str(i) for i in range(47)]  # EMNIST balanced has 47 classes (may exceed 36)
    return train_loader, val_loader, test_loader, class_names

# Try custom dataset
dataset_info = None
if USE_CUSTOM:
    dataset_info = load_custom_dataset(DATA_DIR_CUSTOM, BATCH_SIZE)

if dataset_info is None:
    train_loader, val_loader, test_loader, class_names = load_emnist_balanced(BATCH_SIZE)
    print("Using EMNIST Balanced, classes count:", len(class_names))
else:
    train_loader, val_loader, test_loader, class_names = dataset_info
    print("Custom dataset classes count:", len(class_names))

NUM_CLASSES = len(class_names)
print("NUM_CLASSES =", NUM_CLASSES)

#  Model components 
class TinyCNN(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, feature_dim)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        feats = torch.sigmoid(self.fc2(x))  # normalized [0,1] for angle encoding
        return feats

#  Quantum (PennyLane) 
n_qubits = N_QUBITS
n_layers = VQC_LAYERS
dev = qml.device("default.qubit", wires=n_qubits)

def angle_encoding(theta):
    for i in range(n_qubits):
        qml.RY(theta[i], wires=i)

def variational_layer(params):
    # params shape: (n_qubits, 2)
    for i in range(n_qubits):
        qml.RZ(params[i,0], wires=i)
        qml.RY(params[i,1], wires=i)
    # ring entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    qml.CNOT(wires=[n_qubits - 1, 0])

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights):
    # inputs: length n_qubits (angles)
    angle_encoding(inputs)
    for l in range(n_layers):
        variational_layer(weights[l])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class VQCModule(nn.Module):
    def __init__(self, n_qubits=n_qubits, n_layers=n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Initialize weights (n_layers, n_qubits, 2)
        init = 0.01 * torch.randn(n_layers, n_qubits, 2, requires_grad=True)
        self.weights = nn.Parameter(init)
    def forward(self, x):
        # x: (batch, n_qubits) in radians
        outs = []
        for i in range(x.shape[0]):
            out = qnode(x[i], self.weights)
            outs.append(torch.stack(out))
        return torch.stack(outs)  # (batch, n_qubits)

#  Hybrid model 
class HybridModel(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, n_qubits=n_qubits, num_classes=NUM_CLASSES):
        super().__init__()
        self.frontend = TinyCNN(feature_dim=feature_dim)
        self.vqc = VQCModule(n_qubits=n_qubits, n_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        feats = self.frontend(x)               # (batch, feature_dim)
        q_in = feats[:, :n_qubits]             # (batch, n_qubits)
        angles = q_in * (np.pi / 8.0)          # π/8 normalization
        q_out = self.vqc(angles)               # (batch, n_qubits)
        logits = self.classifier(q_out)
        return logits

model = HybridModel(feature_dim=FEATURE_DIM, n_qubits=n_qubits, num_classes=NUM_CLASSES).to(DEVICE)
print("Model parameter count:", sum(p.numel() for p in model.parameters()))

#  Training utilities 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())
    if total == 0:
        return 0.0, np.array([]), np.array([])
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    acc = correct / total
    return acc, all_preds, all_targets

#  Training loop 
train_losses = []
val_accuracies = []

start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    n_samples = 0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    epoch_loss = running_loss / max(1, n_samples)
    train_losses.append(epoch_loss)
    val_acc, _, _ = evaluate(model, val_loader)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch}/{EPOCHS} — Loss: {epoch_loss:.4f} — Val Acc: {val_acc:.4f}")

total_time = time.time() - start_time
print(f"Training completed in {total_time:.1f}s")

# Save model
save_path = os.path.join(SAVE_DIR, "hybrid_ocr_model.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "config": {
        "feature_dim": FEATURE_DIM,
        "n_qubits": n_qubits,
        "n_layers": n_layers
    }
}, save_path)
print("Saved model to", save_path)

#  Test evaluation 
test_acc, preds, targets = evaluate(model, test_loader)
print("Test accuracy (clean): {:.4f}".format(test_acc))

if targets.size > 0:
    # If class names available, show classification report
    print("\nClassification report (test):")
    # If class count large and names generic, we still show numeric labels
    try:
        print(classification_report(targets, preds, zero_division=0))
    except Exception:
        print("Could not produce classification report (possibly many classes).")

    # Confusion matrix plot (top-left corner if many classes)
    try:
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix (test)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
    except Exception as e:
        print("Skipping confusion matrix plot:", e)

#  Robustness (noisy test) 
for sigma in [0.05, 0.1, 0.2]:
    # create noisy evaluation of test data on the fly
    model.eval()
    total=0; correct=0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb_noisy = add_noise_tensor(xb, sigma=sigma).to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb_noisy)
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
    acc = correct/total if total>0 else 0.0
    print(f"Noisy Test Accuracy (sigma={sigma}): {acc:.4f}")

print("Done.")
