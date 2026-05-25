# ============================================================
# NOTEBOOK 5 — SINGLE IMAGE PREDICTION
# Upload any leaf image → get disease + confidence
# ============================================================

# ── CELL 1: Imports ──────────────────────────────────────────
import os, json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
IMG_SIZE    = config['img_size']

print(f"✅ {NUM_CLASSES} classes loaded")

# ── CELL 3: Load model ────────────────────────────────────────
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
print("✅ Model loaded")

# ── CELL 4: Transform ─────────────────────────────────────────
predict_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── CELL 5: Disease info (treatment suggestions) ─────────────
# Short treatment info for viva/demo
disease_info = {
    'healthy': {
        'status': '✅ Healthy',
        'treatment': 'No treatment needed. Continue regular care.',
        'color': '#2ecc71'
    },
    'Apple_scab': {
        'status': '⚠️ Apple Scab',
        'treatment': 'Apply fungicides (captan/myclobutanil). Remove infected leaves.',
        'color': '#e74c3c'
    },
    'Apple_Black_rot': {
        'status': '⚠️ Apple Black Rot',
        'treatment': 'Prune infected branches. Apply copper-based fungicide.',
        'color': '#e74c3c'
    },
    'Cedar_apple_rust': {
        'status': '⚠️ Cedar Apple Rust',
        'treatment': 'Use myclobutanil fungicide. Remove nearby juniper trees.',
        'color': '#e74c3c'
    },
    'Tomato_Early_blight': {
        'status': '⚠️ Tomato Early Blight',
        'treatment': 'Apply chlorothalonil fungicide. Avoid overhead watering.',
        'color': '#e74c3c'
    },
    'Tomato_Late_blight': {
        'status': '🔴 Tomato Late Blight',
        'treatment': 'Apply metalaxyl fungicide. Remove infected plants immediately.',
        'color': '#c0392b'
    },
    'Tomato_Leaf_Mold': {
        'status': '⚠️ Tomato Leaf Mold',
        'treatment': 'Improve air circulation. Apply copper fungicide.',
        'color': '#e74c3c'
    },
    'Corn_Common_rust': {
        'status': '⚠️ Corn Common Rust',
        'treatment': 'Apply triazole fungicide. Use resistant varieties.',
        'color': '#e74c3c'
    },
    'Potato_Early_blight': {
        'status': '⚠️ Potato Early Blight',
        'treatment': 'Apply mancozeb or chlorothalonil. Ensure proper spacing.',
        'color': '#e74c3c'
    },
    'Potato_Late_blight': {
        'status': '🔴 Potato Late Blight',
        'treatment': 'Urgent: Apply metalaxyl. Destroy infected plants.',
        'color': '#c0392b'
    },
    'default': {
        'status': '⚠️ Disease Detected',
        'treatment': 'Consult local agricultural expert for treatment.',
        'color': '#e67e22'
    }
}

def get_disease_info(class_name):
    """Get treatment info based on class name"""
    if 'healthy' in class_name.lower():
        return disease_info['healthy']
    for key in disease_info:
        if key.lower().replace('_',' ') in class_name.lower().replace('_',' '):
            return disease_info[key]
    return disease_info['default']

# ── CELL 6: Prediction Function ──────────────────────────────
def predict_disease(image_path, top_k=5):
    """
    Predict plant disease from image
    Returns top-k predictions with confidence
    """
    # Load & preprocess
    img = Image.open(image_path).convert('RGB')
    tensor = predict_transform(img).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
    
    # Top-k predictions
    top_probs, top_indices = probs.topk(top_k, dim=1)
    top_probs   = top_probs[0].cpu().numpy()
    top_indices = top_indices[0].cpu().numpy()
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = idx_to_class[idx]
        # Format: "Tomato___Early_blight" -> "Tomato | Early Blight"
        parts = class_name.split('___')
        if len(parts) == 2:
            plant   = parts[0].replace('_', ' ')
            disease = parts[1].replace('_', ' ')
            display = f"{plant} — {disease}"
        else:
            display = class_name.replace('_', ' ')
        
        results.append({
            'class_name' : class_name,
            'display'    : display,
            'confidence' : float(prob) * 100,
            'info'       : get_disease_info(class_name)
        })
    
    return img, results

# ── CELL 7: Visualize Prediction ─────────────────────────────
def visualize_prediction(image_path):
    img, results = predict_disease(image_path, top_k=5)
    top = results[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Image
    axes[0].imshow(img)
    axes[0].set_title(f"Input Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Colored border based on disease/healthy
    border_color = top['info']['color']
    for spine in axes[0].spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)
    
    # Right: Bar chart of top-5 predictions
    labels      = [r['display'][:40] for r in results]
    confidences = [r['confidence'] for r in results]
    colors      = ['#2ecc71' if 'healthy' in r['class_name'].lower()
                   else '#e74c3c' for r in results]
    
    bars = axes[1].barh(labels[::-1], confidences[::-1],
                         color=colors[::-1], edgecolor='white', height=0.6)
    axes[1].set_xlabel('Confidence (%)', fontsize=12)
    axes[1].set_title('Top 5 Predictions', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 105)
    
    for bar, conf in zip(bars, confidences[::-1]):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f'{conf:.1f}%', va='center', fontsize=10, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.suptitle(
        f"🌿 Prediction: {top['display']}\n"
        f"Confidence: {top['confidence']:.1f}% | {top['info']['status']}",
        fontsize=13, fontweight='bold', color=top['info']['color']
    )
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/prediction_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print treatment
    print("\n" + "="*55)
    print(f"🌱 Plant/Disease : {top['display']}")
    print(f"📊 Confidence    : {top['confidence']:.1f}%")
    print(f"🏥 Status        : {top['info']['status']}")
    print(f"💊 Treatment     : {top['info']['treatment']}")
    print("="*55)
    
    return results

# ── CELL 8: TEST ON SAMPLE IMAGES FROM DATASET ───────────────
# Pick random test images from dataset to verify
import random

data_root   = config['data_root']
all_classes = list(idx_to_class.values())

print("🔍 Testing on 6 random images from dataset...\n")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

sample_classes = random.sample(all_classes, 6)

for i, cls in enumerate(sample_classes):
    cls_path = os.path.join(data_root, cls)
    imgs     = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    img_file = os.path.join(cls_path, random.choice(imgs))
    
    img, results = predict_disease(img_file, top_k=1)
    pred = results[0]
    
    true_label = cls.replace('___', ' | ').replace('_', ' ')
    pred_label = pred['display']
    correct    = cls == pred['class_name']
    
    axes[i].imshow(img)
    color = '#2ecc71' if correct else '#e74c3c'
    mark  = '✅' if correct else '❌'
    axes[i].set_title(
        f"{mark} True: {true_label[:30]}\n"
        f"Pred: {pred_label[:30]} ({pred['confidence']:.1f}%)",
        fontsize=8, color=color, fontweight='bold'
    )
    axes[i].axis('off')

plt.suptitle("Sample Predictions from Test Set", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/sample_predictions.png', dpi=150)
plt.show()

# ── CELL 9: HOW TO USE ON YOUR OWN IMAGE ─────────────────────
print("""
╔══════════════════════════════════════════════════════╗
║         HOW TO PREDICT ON YOUR OWN IMAGE             ║
╠══════════════════════════════════════════════════════╣
║  1. Upload image to Kaggle notebook                   ║
║  2. Run:                                              ║
║     results = visualize_prediction('/path/to/img.jpg')║
╚══════════════════════════════════════════════════════╝
""")

# Uncomment below to predict on your own image:
# results = visualize_prediction('/kaggle/input/your-image.jpg')

print("✅ Notebook 5 complete!")
print("✅ Proceed to Streamlit App (app.py)")
