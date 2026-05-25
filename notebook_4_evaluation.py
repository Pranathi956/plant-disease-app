# ============================================================
# NOTEBOOK 4 — EVALUATION
# Accuracy, F1-score, Confusion Matrix (paper metrics)
# ============================================================

# ── CELL 1: Imports ──────────────────────────────────────────
import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets, models
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {DEVICE}")

# ── CELL 2: Load config & class mapping ──────────────────────
with open('/kaggle/working/config.json', 'r') as f:
    config = json.load(f)

with open('/kaggle/working/idx_to_class.json', 'r') as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}

NUM_CLASSES = config['num_classes']
data_root   = config['data_root']
IMG_SIZE    = config['img_size']
BATCH_SIZE  = config['batch_size']

print(f"✅ Config loaded — {NUM_CLASSES} classes")

# ── CELL 3: Rebuild test loader ──────────────────────────────
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean, self.std = mean, std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(root=data_root, transform=val_transform)
total        = len(full_dataset)
train_size   = config['train_size']
val_size     = config['val_size']
test_size    = config['test_size']

g = torch.Generator().manual_seed(42)
_, _, test_idx = random_split(range(total), [train_size, val_size, test_size], generator=g)
test_dataset   = Subset(full_dataset, test_idx.indices)
test_loader    = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

print(f"✅ Test set: {len(test_dataset):,} images | {len(test_loader)} batches")

# ── CELL 4: Load trained model ───────────────────────────────
def build_efficientnet_b3(num_classes):
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model

model = build_efficientnet_b3(NUM_CLASSES)
model.load_state_dict(torch.load('/kaggle/working/best_model.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully")

# ── CELL 5: Get all predictions ──────────────────────────────
all_preds, all_labels, all_probs = [], [], []

print("🔄 Running inference on test set...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probs   = torch.softmax(outputs, dim=1)
        preds   = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

print(f"✅ Inference done — {len(all_preds):,} predictions")

# ── CELL 6: Overall Metrics ──────────────────────────────────
overall_acc = accuracy_score(all_labels, all_preds)
macro_f1    = f1_score(all_labels, all_preds, average='macro')
weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

print("\n" + "="*50)
print("📊 OVERALL METRICS")
print("="*50)
print(f"  Accuracy         : {overall_acc*100:.2f}%")
print(f"  Macro F1-Score   : {macro_f1*100:.2f}%")
print(f"  Weighted F1-Score: {weighted_f1*100:.2f}%")
print("="*50)

# ── CELL 7: Per-Class F1 Scores ──────────────────────────────
class_f1 = f1_score(all_labels, all_preds, average=None)
class_names_short = [idx_to_class[i].replace('___', ' | ').replace('_', ' ')
                     for i in range(NUM_CLASSES)]

# Sort by F1 for visualization
sorted_idx = np.argsort(class_f1)

plt.figure(figsize=(20, 10))
colors = ['#2ecc71' if f >= 0.8 else '#f39c12' if f >= 0.6 else '#e74c3c'
          for f in class_f1[sorted_idx]]
bars = plt.barh(range(NUM_CLASSES),
                class_f1[sorted_idx] * 100,
                color=colors, edgecolor='white')
plt.yticks(range(NUM_CLASSES),
           [class_names_short[i] for i in sorted_idx], fontsize=7)
plt.xlabel('F1-Score (%)', fontsize=12)
plt.title('Per-Class F1-Score — EfficientNet-B3', fontsize=16, fontweight='bold')
plt.axvline(x=80, color='black', linestyle='--', alpha=0.5, label='80% threshold')
plt.legend()
plt.tight_layout()
plt.savefig('/kaggle/working/per_class_f1.png', dpi=150, bbox_inches='tight')
plt.show()

# Print top 5 and bottom 5
print("\n🏆 Top 5 Classes (highest F1):")
for i in sorted_idx[-5:][::-1]:
    print(f"  {class_names_short[i]:45s}: {class_f1[i]*100:.1f}%")

print("\n⚠️ Bottom 5 Classes (lowest F1):")
for i in sorted_idx[:5]:
    print(f"  {class_names_short[i]:45s}: {class_f1[i]*100:.1f}%")

# ── CELL 8: Confusion Matrix ─────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

# Full confusion matrix
fig, ax = plt.subplots(figsize=(24, 22))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
            xticklabels=class_names_short,
            yticklabels=class_names_short,
            linewidths=0.5, ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True',      fontsize=12)
ax.set_title('Confusion Matrix — EfficientNet-B3 on PlantVillage',
             fontsize=16, fontweight='bold')
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0,  fontsize=6)
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix saved")

# ── CELL 9: Classification Report ───────────────────────────
report = classification_report(all_labels, all_preds,
                                target_names=class_names_short,
                                digits=4)
print("\n📋 CLASSIFICATION REPORT:")
print(report)

with open('/kaggle/working/classification_report.txt', 'w') as f:
    f.write(f"Overall Accuracy : {overall_acc*100:.2f}%\n")
    f.write(f"Macro F1         : {macro_f1*100:.2f}%\n")
    f.write(f"Weighted F1      : {weighted_f1*100:.2f}%\n\n")
    f.write(report)

print("✅ Classification report saved")

# ── CELL 10: Training History Plots ──────────────────────────
with open('/kaggle/working/training_history.json', 'r') as f:
    history = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train', color='steelblue', linewidth=2)
axes[0].plot(history['val_loss'],   label='Val',   color='tomato',    linewidth=2)
axes[0].set_title('Loss Curve', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot([x*100 for x in history['train_acc']], label='Train', color='steelblue', linewidth=2)
axes[1].plot([x*100 for x in history['val_acc']],   label='Val',   color='tomato',    linewidth=2)
axes[1].set_title('Accuracy Curve', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle(f'EfficientNet-B3 Training — Final Acc: {overall_acc*100:.2f}%',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/final_training_curves.png', dpi=150)
plt.show()

# ── CELL 11: Save metrics ────────────────────────────────────
metrics = {
    'overall_accuracy': overall_acc,
    'macro_f1'        : macro_f1,
    'weighted_f1'     : weighted_f1,
    'per_class_f1'    : class_f1.tolist()
}
with open('/kaggle/working/metrics.json', 'w') as f:
    json.dump(metrics, f)

print("\n✅ All evaluation complete!")
print(f"   Final Test Accuracy  : {overall_acc*100:.2f}%")
print(f"   Macro F1             : {macro_f1*100:.2f}%")
print("\n✅ Proceed to Notebook 5 — Predictions")
