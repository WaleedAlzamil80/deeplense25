import sys
import os
sys.path.append('/home/waleed/Documents/deeplense25/specific_test_06')
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.Dataset import NPYClassificationDataset
from tqdm import tqdm
from utils.helpful import image_to_patches, show_sample_images, random_masking, visualize_patches
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
save_dir = "/home/waleed/Documents/deeplense25"

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
# Validation set for MAE (later used in classification also)
val_dataset = NPYClassificationDataset(val_files, val_labels, val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

patch_size = 10
input_dim = patch_size**2
num_patches = int(150/patch_size)**2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patch_size = 10
input_dim = patch_size**2
num_patches = int(150/patch_size)**2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassifierViT(base="tiny", embed_dim = 192, input_dim=input_dim, num_patches=num_patches)
model = nn.DataParallel(model.to(device))

print_trainable_parameters(model)

base_model = "/home/waleed/Documents/deeplense25/specific_test_06/models/checkpoints/classifier.pth"
state_dict = torch.load(base_model, map_location=device, weights_only=True)
model.load_state_dict(state_dict)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

all_probs_test = []
all_labels_test = []
all_predicted_test = []

val_correct, val_total = 0, 0

with torch.no_grad():
    for batch_data, batch_labels in val_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        batch_data = image_to_patches(batch_data, patch_size)

        outputs = model(batch_data)

        _, predicted = torch.max(outputs, 1)
        val_correct += (predicted == batch_labels).sum().item()
        val_total += batch_labels.size(0)

        probs = torch.softmax(outputs, dim=1)
        all_predicted_test.extend(predicted.cpu().detach().numpy())
        all_probs_test.extend(probs.cpu().detach().numpy())
        all_labels_test.extend(batch_labels.cpu().numpy())

val_acc = val_correct / val_total
print(f"Accuracy: {(val_acc*100):.2f}%")

# Classification report
print("Classification Report:")
print(classification_report(all_labels_test, all_predicted_test, target_names=['no_sub', 'axion', 'cdm']))

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels_test, all_predicted_test)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(all_labels_test), yticklabels=np.unique(all_labels_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
plt.show()

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
