# ============================================================
# NOTEBOOK 1 — EDA (Exploratory Data Analysis)
# Plant Disease Detection | PlantVillage Dataset
# ============================================================

# ── CELL 1: Imports ──────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("✅ Imports done")

# ── CELL 2: Dataset Path ─────────────────────────────────────
# Check what's available
import os

base_path = "/kaggle/input"
for item in os.listdir(base_path):
    print(item)

# ── CELL 3: Set Dataset Path ─────────────────────────────────
# After seeing output above, set the correct path
# Usually it will be something like 'plantvillage-dataset' or 'plant-village'

dataset_path = None
for item in os.listdir("/kaggle/input"):
    full = os.path.join("/kaggle/input", item)
    if os.path.isdir(full):
        # go one level deeper to find class folders
        subdirs = os.listdir(full)
        print(f"📁 {item}: {len(subdirs)} items → {subdirs[:3]}")
        dataset_path = full

print(f"\n✅ Using: {dataset_path}")

# ── CELL 4: Find correct folder with class subfolders ────────
# Navigate to the folder that has disease class folders directly
def find_class_folder(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # If this folder has 30+ subfolders, it's likely our class folder
        if len(dirnames) >= 10:
            # Check if subfolders contain images
            sample = os.path.join(dirpath, dirnames[0])
            images = [f for f in os.listdir(sample) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if len(images) > 0:
                return dirpath
    return root

data_root = find_class_folder(dataset_path)
classes = sorted(os.listdir(data_root))
classes = [c for c in classes if os.path.isdir(os.path.join(data_root, c))]

print(f"✅ Data root: {data_root}")
print(f"✅ Total classes found: {len(classes)}")
print(f"\nFirst 5 classes: {classes[:5]}")
print(f"Last 5 classes: {classes[-5:]}")

# ── CELL 5: Count images per class ───────────────────────────
class_counts = {}
total_images = 0

for cls in classes:
    cls_path = os.path.join(data_root, cls)
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    class_counts[cls] = len(imgs)
    total_images += len(imgs)

print(f"✅ Total images: {total_images}")
print(f"✅ Total classes: {len(classes)}")
print(f"\nMin images in a class: {min(class_counts.values())} ({min(class_counts, key=class_counts.get)})")
print(f"Max images in a class: {max(class_counts.values())} ({max(class_counts, key=class_counts.get)})")

# ── CELL 6: Class Distribution Bar Chart ─────────────────────
plt.figure(figsize=(20, 8))
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
names = [x[0].replace('___', '\n').replace('_', ' ')[:30] for x in sorted_classes]
counts = [x[1] for x in sorted_classes]

bars = plt.bar(range(len(names)), counts, color='steelblue', edgecolor='white')
plt.xticks(range(len(names)), names, rotation=90, fontsize=7)
plt.ylabel("Number of Images", fontsize=12)
plt.title("PlantVillage — Class Distribution", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart saved")

# ── CELL 7: Group by Plant Type ──────────────────────────────
plant_totals = {}
for cls, count in class_counts.items():
    plant = cls.split('___')[0] if '___' in cls else cls.split('_')[0]
    plant_totals[plant] = plant_totals.get(plant, 0) + count

print("📊 Images per Plant Type:")
for plant, count in sorted(plant_totals.items(), key=lambda x: x[1], reverse=True):
    print(f"  {plant:20s}: {count:,}")

# ── CELL 8: Pie Chart — Plant Distribution ───────────────────
plt.figure(figsize=(12, 8))
plants = list(plant_totals.keys())
values = list(plant_totals.values())
plt.pie(values, labels=plants, autopct='%1.1f%%', startangle=140)
plt.title("Distribution by Plant Type", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/plant_distribution.png', dpi=150)
plt.show()

# ── CELL 9: Sample Images from Random Classes ────────────────
import random
from PIL import Image

sample_classes = random.sample(classes, min(12, len(classes)))
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, cls in enumerate(sample_classes):
    cls_path = os.path.join(data_root, cls)
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    img_path = os.path.join(cls_path, random.choice(imgs))
    img = Image.open(img_path).resize((224, 224))
    axes[i].imshow(img)
    label = cls.replace('___', '\n').replace('_', ' ')
    axes[i].set_title(label[:35], fontsize=8, fontweight='bold')
    axes[i].axis('off')

plt.suptitle("Sample Images from PlantVillage Dataset", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/sample_images.png', dpi=150)
plt.show()
print("✅ Sample images saved")

# ── CELL 10: Check Image Sizes ───────────────────────────────
print("🔍 Checking sample image sizes...")
sizes = []
for cls in random.sample(classes, 5):
    cls_path = os.path.join(data_root, cls)
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for img_name in random.sample(imgs, min(3, len(imgs))):
        img = Image.open(os.path.join(cls_path, img_name))
        sizes.append(img.size)

print(f"Sample sizes: {set(sizes)}")
print("\n✅ EDA Complete! Proceed to Notebook 2")

# Save class info for next notebooks
import json
with open('/kaggle/working/class_info.json', 'w') as f:
    json.dump({
        'data_root': data_root,
        'classes': classes,
        'num_classes': len(classes),
        'class_counts': class_counts,
        'total_images': total_images
    }, f)
print("✅ class_info.json saved for next notebooks")
