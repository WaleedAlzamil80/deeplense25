import os
from PIL import Image
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.Dataset import NPYClassificationDataset

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle

from tqdm import tqdm
from specific_test_06.utils.helpful import image_to_patches, show_sample_images, random_masking, visualize_patches
from utils.helpful import print_trainable_parameters
from models.classifier import ClassifierViT

train_transforms = transforms.Compose([
    # transforms.CenterCrop(100),
    transforms.Resize(150, Image.LANCZOS),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transforms = transforms.Compose([
        transforms.Resize(150, Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset_root = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_A/Dataset/"

axion_files = sorted(glob(os.path.join(dataset_root, "axion", "*.npy")))
no_sub_files = sorted(glob(os.path.join(dataset_root, "no_sub", "*.npy")))
cdm_files = sorted(glob(os.path.join(dataset_root, "cdm", "*.npy")))

all_files = no_sub_files + axion_files + cdm_files
labels = [0] * len(no_sub_files) + [1] * len(axion_files) + [2] * len(cdm_files)

# First split: 90% train, 10% val (stratified)
train_files, val_files, train_labels, val_labels = train_test_split(
    all_files, labels, test_size=0.1, stratify=labels, random_state=42
)

# Train MAE only on no_sub_train_files
batch_size=64
train_dataset = NPYClassificationDataset(train_files, train_labels, train_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, shuffle=True)

# Validation set for MAE (later used in classification also)
val_dataset = NPYClassificationDataset(val_files, val_labels, val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10, shuffle=False)

patch_size = 10
input_dim = patch_size**2
num_patches = int(150/patch_size)**2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassifierViT(input_dim=input_dim, num_patches=num_patches).to(device)
print_trainable_parameters(model)


# Optimizer & Loss
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-6) #  , weight_decay=2e-4 , weight_decay=1e-4
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5)
criterion = nn.CrossEntropyLoss()

# Track metrics
train_losses, val_losses = [], []
train_accuracies, train_aucs = [], []
val_accuracies, val_aucs = [], []

# Training Loop
num_epochs = 50
best_val_loss = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    all_probs = []
    all_labels = []

    for images, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        images = images.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        images = image_to_patches(images, 10)
        outputs = model(images)
        print(outputs.shape)
        loss = criterion(outputs, batch_labels)

        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)
        probs = torch.softmax(outputs, dim=1) # [:, 1]  # Probability for class 1
        all_probs.extend(probs.cpu().detach().numpy())
        all_labels.extend(batch_labels.cpu().numpy())


    train_acc = correct / total
    train_loss = running_loss / len(train_loader)

    try:
        train_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        train_auc = float('nan')

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    train_aucs.append(train_auc)

    # Validation Step
    model.eval()
    val_correct, val_total = 0, 0
    val_loss = 0.0
    all_probs_test = []
    all_labels_test = []

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == batch_labels).sum().item()
            val_total += batch_labels.size(0)

            # Collect probabilities and labels for ROC/AUC
            probs = torch.softmax(outputs, dim=1) # [:, 1]  # Probability for class 1
            all_probs_test.extend(probs.cpu().detach().numpy())
            all_labels_test.extend(batch_labels.cpu().numpy())
    
    val_acc = val_correct / val_total
    val_loss = val_loss / len(val_loader)
    scheduler.step(val_loss)

    try:
        val_auc = roc_auc_score(all_labels_test, all_probs_test, multi_class='ovr')
    except ValueError:
        val_auc = float('nan') 

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_aucs.append(val_auc)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},  Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    # Save Best Model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "best_fine_tuned_vit_model_clas.pth")
        print("Model Saved (Best Validation AUC)")

sns.set(style="whitegrid", font_scale=1.2)
epochs = range(1, num_epochs + 1)
save_dir = "/home/waleed/Documents/deeplense25/specific_test_06/assets"

# Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, val_losses, label='Val Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.savefig(os.path.join(save_dir, 'Losses.png'))
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
plt.plot(epochs, val_accuracies, label='Val Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.savefig(os.path.join(save_dir, 'Accuracies.png'))
plt.show()

# Plot AUC
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_aucs, label='Train AUC', color='blue')
plt.plot(epochs, val_aucs, label='Val AUC', color='red')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('AUC over Epochs')
plt.legend()
plt.savefig(os.path.join(save_dir, 'AUC.png'))
plt.show()

state_dict = torch.load("/home/waleed/Documents/deeplense25/best_fine_tuned_vit_model_clas.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)

all_probs_test = []
all_labels_test = []
val_correct, val_total = 0, 0

with torch.no_grad():
    for batch_data, batch_labels in val_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

        outputs = model(batch_data)

        _, predicted = torch.max(outputs, 1)
        val_correct += (predicted == batch_labels).sum().item()
        val_total += batch_labels.size(0)

        probs = torch.softmax(outputs, dim=1)
        all_probs_test.extend(probs.cpu().detach().numpy())
        all_labels_test.extend(batch_labels.cpu().numpy())

val_acc = val_correct / val_total
print(f"Accuracy: {(val_acc*100):.2f}%")

# Step 1: Binarize the labels
all_labels_test = np.array(all_labels_test)
all_probs_test = np.array(all_probs_test)

n_classes = len(np.unique(all_labels_test))
all_labels_test_bin = label_binarize(all_labels_test, classes=np.arange(n_classes))

# Step 2: Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_test_bin[:, i], all_probs_test[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Step 3: Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(all_labels_test_bin.ravel(), all_probs_test.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Step 4: Plot ROC curves
label_map = {0:"no", 1:"sphere", 2: "vort"}
plt.figure(figsize=(8, 6))
colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"ROC curve of class {label_map[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot(fpr["micro"], tpr["micro"], color="deeppink", linestyle=":", linewidth=4,
         label=f"Micro-average ROC curve (AUC = {roc_auc['micro']:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend(loc="lower right")
plt.savefig(os.path.join(save_dir, 'ROC_curve.png'))
plt.show()

# Step 5: Compute AUC using roc_auc_score with multi_class='ovr'
val_auc = roc_auc_score(all_labels_test, all_probs_test, multi_class='ovr')
print(f"AUC (One-vs-Rest, macro-average): {val_auc:.2f}")

# Optional: Compute AUC with different averaging methods
val_auc_micro = roc_auc_score(all_labels_test, all_probs_test, multi_class='ovr', average='micro')
val_auc_weighted = roc_auc_score(all_labels_test, all_probs_test, multi_class='ovr', average='weighted')
print(f"AUC (One-vs-Rest, micro-average): {val_auc_micro:.2f}")
print(f"AUC (One-vs-Rest, weighted-average): {val_auc_weighted:.2f}")
