# ============================================================
# NOTEBOOK 3 — MODEL TRAINING
# EfficientNet-B3 (paper best model) + Transfer Learning
# Adam optimizer + Early Stopping + Class Weights
# ============================================================

# ── CELL 1: Imports ──────────────────────────────────────────
import os, json, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# ── CELL 2: Load config ──────────────────────────────────────
with open('/kaggle/working/config.json', 'r') as f:
    config = json.load(f)

data_root    = config['data_root']
classes      = config['classes']
NUM_CLASSES  = config['num_classes']
IMG_SIZE     = config['img_size']
BATCH_SIZE   = config['batch_size']
class_weights = torch.tensor(config['class_weights']).to(DEVICE)

print(f"✅ Config loaded — {NUM_CLASSES} classes")

# ── CELL 3: Rebuild transforms & loaders ─────────────────────
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std  = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    AddGaussianNoise(0.0, 0.05),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_train = datasets.ImageFolder(root=data_root, transform=train_transform)
full_val   = datasets.ImageFolder(root=data_root, transform=val_transform)

total      = len(full_train)
train_size = config['train_size']
val_size   = config['val_size']
test_size  = config['test_size']

g = torch.Generator().manual_seed(42)
train_idx, val_idx, test_idx = random_split(range(total), [train_size, val_size, test_size], generator=g)

from torch.utils.data import Subset
train_dataset = Subset(full_train, train_idx.indices)
val_dataset   = Subset(full_val,   val_idx.indices)
test_dataset  = Subset(full_val,   test_idx.indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"✅ Loaders ready — Train:{len(train_loader)} Val:{len(val_loader)} Test:{len(test_loader)} batches")

# ── CELL 4: Build EfficientNet-B3 (paper best model) ─────────
def build_efficientnet_b3(num_classes):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    # Replace final classifier — paper method
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model

model = build_efficientnet_b3(NUM_CLASSES).to(DEVICE)
print(f"✅ EfficientNet-B3 built")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Trainable : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── CELL 5: Loss, Optimizer (Paper: Adam + class weights) ────
# Paper: inverse frequency class weights on loss
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Paper: Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# LR scheduler — reduce on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=0.5, patience=3,
                                                   verbose=True)
print("✅ Loss: CrossEntropyLoss with class weights")
print("✅ Optimizer: Adam (lr=1e-4)")
print("✅ Scheduler: ReduceLROnPlateau")

# ── CELL 6: Training Loop with Early Stopping ────────────────
def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, num_epochs=30, patience=5):
    
    best_val_acc  = 0.0
    best_weights  = copy.deepcopy(model.state_dict())
    patience_ctr  = 0
    
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60)
    
    for epoch in range(num_epochs):
        start = time.time()
        
        # ── TRAIN PHASE ──────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss    += loss.item() * inputs.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += inputs.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}", end='\r')
        
        train_loss /= train_total
        train_acc   = train_correct / train_total
        
        # ── VALIDATION PHASE ─────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs  = model(inputs)
                loss     = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds     = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += inputs.size(0)
        
        val_loss /= val_total
        val_acc   = val_correct / val_total
        elapsed   = time.time() - start
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch [{epoch+1:02d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            torch.save(best_weights, '/kaggle/working/best_model.pth')
            print(f"  ✅ Best model saved! Val Acc: {best_val_acc:.4f}")
        else:
            patience_ctr += 1
            print(f"  ⏳ No improvement ({patience_ctr}/{patience})")
            if patience_ctr >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Restore best weights
    model.load_state_dict(best_weights)
    print(f"\n✅ Training complete! Best Val Accuracy: {best_val_acc:.4f}")
    return model, history

# ── CELL 7: RUN TRAINING ─────────────────────────────────────
model, history = train_model(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    num_epochs=30, patience=5
)

# ── CELL 8: Plot Training Curves ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
axes[0].plot(history['val_loss'],   label='Val Loss',   color='orange')
axes[0].set_title('Loss Curve', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot([x*100 for x in history['train_acc']], label='Train Acc', color='blue')
axes[1].plot([x*100 for x in history['val_acc']],   label='Val Acc',   color='orange')
axes[1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('EfficientNet-B3 — Training History', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png', dpi=150)
plt.show()
print("✅ Training curves saved")

# ── CELL 9: Save everything needed for next notebooks ────────
# Save history
with open('/kaggle/working/training_history.json', 'w') as f:
    json.dump(history, f)

print("\n📦 Files saved:")
print("   /kaggle/working/best_model.pth       ← trained model weights")
print("   /kaggle/working/training_history.json ← loss/acc curves")
print("\n✅ Training complete! Proceed to Notebook 4 — Evaluation")
