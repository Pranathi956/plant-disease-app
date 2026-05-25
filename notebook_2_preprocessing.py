# ============================================================
# NOTEBOOK 2 — PREPROCESSING & DATA LOADERS
# Paper methods: 224x224, ImageNet norm, Augmentation + Gaussian noise
# ============================================================

# ── CELL 1: Imports ──────────────────────────────────────────
import os, json, random, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND — check Kaggle GPU setting!'}")

# ── CELL 2: Load class info from Notebook 1 ─────────────────
with open('/kaggle/working/class_info.json', 'r') as f:
    info = json.load(f)

data_root   = info['data_root']
classes     = info['classes']
NUM_CLASSES = info['num_classes']
print(f"✅ Loaded: {NUM_CLASSES} classes from {data_root}")

# ── CELL 3: Gaussian Noise Transform (Paper method) ─────────
class AddGaussianNoise:
    """Adds Gaussian noise — mean=0, std=0.05 (exact from paper)"""
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std  = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std})"

# ── CELL 4: Transforms (Paper exact augmentations) ───────────
IMG_SIZE = 224

# Training transforms — all paper augmentations
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),   # paper: random resized crop
    transforms.RandomHorizontalFlip(),                            # paper: horizontal flip
    transforms.RandomVerticalFlip(),                              # paper: vertical flip
    transforms.RandomRotation(degrees=20),                        # paper: rotation ±20°
    transforms.ColorJitter(brightness=0.2, contrast=0.2,         # paper: color jitter
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],             # paper: ImageNet stats
                         std=[0.229, 0.224, 0.225]),
    AddGaussianNoise(mean=0.0, std=0.05),                        # paper: gaussian noise
])

# Validation/Test transforms — no augmentation, just resize+normalize
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("✅ Transforms defined")
print(f"\nTrain transforms:\n{train_transform}")

# ── CELL 5: Load Full Dataset & Split ────────────────────────
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

full_dataset = ImageFolder(root=data_root, transform=train_transform)

# 80% train, 10% val, 10% test
total = len(full_dataset)
train_size = int(0.80 * total)
val_size   = int(0.10 * total)
test_size  = total - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Apply correct transforms to val/test (no augmentation)
val_dataset.dataset  = ImageFolder(root=data_root, transform=val_transform)
test_dataset.dataset = ImageFolder(root=data_root, transform=val_transform)

print(f"✅ Total images : {total:,}")
print(f"✅ Train        : {train_size:,}")
print(f"✅ Validation   : {val_size:,}")
print(f"✅ Test         : {test_size:,}")

# ── CELL 6: Class Weights for Imbalance (Paper method) ───────
class_counts = info['class_counts']
counts_list  = [class_counts[c] for c in classes]
class_weights = 1.0 / torch.tensor(counts_list, dtype=torch.float)
class_weights = class_weights / class_weights.sum()  # normalize

print("✅ Class weights computed (inverse frequency — paper method)")
print(f"   Max weight: {class_weights.max():.4f} | Min weight: {class_weights.min():.4f}")

# ── CELL 7: DataLoaders ──────────────────────────────────────
BATCH_SIZE = 64   # paper: batch size 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"✅ Train batches : {len(train_loader)}")
print(f"✅ Val batches   : {len(val_loader)}")
print(f"✅ Test batches  : {len(test_loader)}")

# ── CELL 8: Visualize Augmented Images ───────────────────────
# Show original vs augmented
def show_augmented(data_root, classes, n=6):
    fig, axes = plt.subplots(2, n, figsize=(18, 6))
    sample_cls = random.choice(classes)
    cls_path   = os.path.join(data_root, sample_cls)
    img_files  = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]

    original_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for i in range(n):
        img_path = os.path.join(cls_path, random.choice(img_files))
        img = Image.open(img_path).convert('RGB')

        orig = original_transform(img).permute(1,2,0).numpy()
        aug  = train_transform(img)
        # Denormalize for display
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        aug_np = aug.permute(1,2,0).numpy()
        aug_np = np.clip(aug_np * std + mean, 0, 1)

        axes[0][i].imshow(orig)
        axes[0][i].set_title("Original", fontsize=8)
        axes[0][i].axis('off')

        axes[1][i].imshow(aug_np)
        axes[1][i].set_title("Augmented", fontsize=8)
        axes[1][i].axis('off')

    plt.suptitle(f"Original vs Augmented — {sample_cls.replace('___',' | ')}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/kaggle/working/augmentation_preview.png', dpi=150)
    plt.show()

show_augmented(data_root, classes)
print("✅ Augmentation preview saved")

# ── CELL 9: Save preprocessing config ────────────────────────
import pickle

config = {
    'data_root'      : data_root,
    'classes'        : classes,
    'num_classes'    : NUM_CLASSES,
    'img_size'       : IMG_SIZE,
    'batch_size'     : BATCH_SIZE,
    'class_weights'  : class_weights.tolist(),
    'train_size'     : train_size,
    'val_size'       : val_size,
    'test_size'      : test_size,
}

with open('/kaggle/working/config.json', 'w') as f:
    json.dump(config, f)

# Save index->class mapping
idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
with open('/kaggle/working/idx_to_class.json', 'w') as f:
    json.dump({str(k): v for k, v in idx_to_class.items()}, f)

print("✅ config.json saved")
print("✅ idx_to_class.json saved")
print("\n✅ Preprocessing complete! Proceed to Notebook 3 — Training")
