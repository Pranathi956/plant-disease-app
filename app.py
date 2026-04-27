import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import numpy as np

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); }
    .prediction-box {
        background: white; border-radius: 15px;
        padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 10px 0;
    }
    .healthy  { border-left: 6px solid #2ecc71; }
    .diseased { border-left: 6px solid #e74c3c; }
    .invalid  { border-left: 6px solid #95a5a6; }
</style>
""", unsafe_allow_html=True)

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 70.0  # Below this → not a leaf

# ── Supported plants — shown to user ─────────────────────────
SUPPORTED_PLANTS = [
    "🍅 Tomato (9 diseases)",
    "🥔 Potato (2 diseases)",
    "🍎 Apple (3 diseases)",
    "🌽 Corn (3 diseases)",
    "🍇 Grape (3 diseases)",
    "🫑 Pepper (1 disease)",
    "🍓 Strawberry (1 disease)",
    "🍑 Peach (1 disease)",
    "🍒 Cherry (1 disease)",
    "🎃 Squash (1 disease)",
    "🫐 Blueberry (healthy only)",
    "🌱 Raspberry (healthy only)",
    "🍊 Orange (1 disease)",
    "🫐 Soybean (healthy only)",
]

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('config.json', 'r') as f:
        config = json.load(f)
    with open('idx_to_class.json', 'r') as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}
    NUM_CLASSES = config['num_classes']
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    return model, config, idx_to_class

predict_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Treatment info ─────────────────────────────────────────────
def get_treatment(class_name):
    treatments = {
        'healthy'             : ('✅ Healthy',              'No treatment needed. Regular care sufficient.',        '#2ecc71'),
        'Early_blight'        : ('⚠️ Early Blight',         'Apply chlorothalonil. Avoid overhead watering.',       '#e67e22'),
        'Late_blight'         : ('🔴 Late Blight',          'Urgent! Apply metalaxyl. Remove infected plants.',     '#c0392b'),
        'Leaf_Mold'           : ('⚠️ Leaf Mold',            'Improve ventilation. Apply copper fungicide.',         '#e67e22'),
        'Septoria_leaf_spot'  : ('⚠️ Septoria Leaf Spot',   'Apply mancozeb. Remove lower infected leaves.',        '#e67e22'),
        'Spider_mites'        : ('⚠️ Spider Mites',         'Apply miticide or neem oil. Increase humidity.',       '#e67e22'),
        'Target_Spot'         : ('⚠️ Target Spot',          'Apply azoxystrobin. Improve air circulation.',         '#e67e22'),
        'mosaic_virus'        : ('🔴 Mosaic Virus',         'No cure. Remove plants. Control aphids.',              '#c0392b'),
        'Yellow_Leaf_Curl'    : ('🔴 Yellow Leaf Curl',     'Control whiteflies. Use resistant varieties.',         '#c0392b'),
        'Bacterial_spot'      : ('⚠️ Bacterial Spot',       'Apply copper bactericide. Avoid wetting foliage.',     '#e67e22'),
        'scab'                : ('⚠️ Scab',                 'Apply captan fungicide. Prune infected areas.',        '#e67e22'),
        'Black_rot'           : ('⚠️ Black Rot',            'Prune branches. Apply copper fungicide.',              '#e67e22'),
        'rust'                : ('⚠️ Rust',                 'Apply triazole fungicide. Remove infected leaves.',    '#e67e22'),
        'Common_rust'         : ('⚠️ Common Rust',          'Apply triazole fungicide. Use resistant hybrids.',     '#e67e22'),
        'Northern_Leaf_Blight': ('⚠️ N. Leaf Blight',       'Apply fungicide early. Use resistant varieties.',      '#e67e22'),
        'Gray_leaf_spot'      : ('⚠️ Gray Leaf Spot',       'Apply strobilurin fungicide. Crop rotation.',          '#e67e22'),
        'Powdery_mildew'      : ('⚠️ Powdery Mildew',       'Apply sulfur or neem oil. Improve air circulation.',   '#e67e22'),
        'Leaf_blight'         : ('⚠️ Leaf Blight',          'Apply copper fungicide. Remove infected leaves.',      '#e67e22'),
        'Haunglongbing'       : ('🔴 Citrus Greening',      'No cure. Remove infected trees. Control psyllids.',    '#c0392b'),
        'Esca'                : ('🔴 Esca',                 'Prune infected vines. Apply fungicide to cuts.',       '#c0392b'),
    }
    if 'healthy' in class_name.lower():
        return treatments['healthy']
    for key, val in treatments.items():
        if key.lower().replace('_', ' ') in class_name.lower().replace('_', ' '):
            return val
    return ('⚠️ Disease Detected', 'Consult a local agricultural expert.', '#e67e22')

# ── Predict ────────────────────────────────────────────────────
def predict(image, model, idx_to_class, top_k=5):
    tensor = predict_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
    top_probs, top_indices = probs.topk(top_k, dim=1)
    results = []
    for prob, idx in zip(top_probs[0].numpy(), top_indices[0].numpy()):
        cls   = idx_to_class[int(idx)]
        parts = cls.split('___')
        display = f"{parts[0].replace('_',' ')} — {parts[1].replace('_',' ')}" \
                  if len(parts) == 2 else cls.replace('_', ' ')
        status, treatment, color = get_treatment(cls)
        results.append({
            'class': cls, 'display': display,
            'confidence': float(prob) * 100,
            'status': status, 'treatment': treatment, 'color': color
        })
    return results

# ── UI ─────────────────────────────────────────────────────────
st.markdown("# 🌿 Plant Disease Detection System")
st.markdown("**EfficientNet-B3 + Transfer Learning | PlantVillage Dataset (54,305 images)**")
st.markdown("---")

try:
    model, config, idx_to_class = load_model()
    st.success(f"✅ Model loaded — {config['num_classes']} disease classes")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(f"""
    **Model:** EfficientNet-B3  
    **Dataset:** PlantVillage (54,305 images)  
    **Classes:** {config['num_classes']} plant diseases  

    ---
    **Paper Methods (Krishna et al. 2025):**
    - ✅ EfficientNet-B3
    - ✅ Gaussian noise augmentation
    - ✅ ImageNet pretrained weights
    - ✅ Adam + Early stopping
    - ✅ Class-weighted loss

    **Our Improvements:**
    - ✅ 20x larger dataset
    - ✅ OOD-based leaf validation
    - ✅ Treatment recommendations
    - ✅ Real-time deployment
    """)

    st.markdown("---")
    st.markdown("**⚠️ Model Limitations:**")
    st.markdown("""
    - Upload **single leaf, close-up** only
    - Multiple leaves in frame → may fail
    - Only supports 14 plant types below
    - Visually similar leaves may confuse model
    """)

    st.markdown("---")
    st.markdown("**✅ Supported Plants:**")
    for p in SUPPORTED_PLANTS:
        st.markdown(f"- {p}")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📤 Upload Plant Leaf Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear, close-up photo of a single plant leaf"
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Show tips below image
        st.info("💡 For best results: single leaf, close-up, clear background")

        predict_btn = st.button("🔍 Detect Disease", type="primary", use_container_width=True)

with col2:
    if uploaded_file and predict_btn:

        with st.spinner("🔬 Analyzing image..."):
            results = predict(image, model, idx_to_class)

        top = results[0]

        # ── Leaf validation via confidence ────────────────────
        if top['confidence'] < CONFIDENCE_THRESHOLD:

            # ── NOT A LEAF — show rejection, hide top-5 ──────
            st.markdown(f"""
            <div class="prediction-box invalid">
                <h2 style="color:#7f8c8d">❌ Not a Recognizable Plant Leaf</h2>
                <br>
                <p>This image could not be matched to any plant leaf
                in the dataset with sufficient confidence.</p>
                <br>
                <p><b>Model confidence:</b> {top['confidence']:.1f}%</p>
                <p><b>Minimum required:</b> {CONFIDENCE_THRESHOLD:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.warning("⚠️ Please upload a clear, close-up photo of a single plant leaf.")

            st.markdown("**This model supports only these 14 plants:**")
            col_a, col_b = st.columns(2)
            half = len(SUPPORTED_PLANTS) // 2
            with col_a:
                for p in SUPPORTED_PLANTS[:half]:
                    st.markdown(f"- {p}")
            with col_b:
                for p in SUPPORTED_PLANTS[half:]:
                    st.markdown(f"- {p}")

        else:
            # ── VALID LEAF — show disease result + top-5 ─────
            box_class = 'healthy' if 'healthy' in top['class'].lower() else 'diseased'

            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h2 style="color:{top['color']}">{top['status']}</h2>
                <h3>🌱 {top['display']}</h3>
                <p><b>📊 Confidence:</b> {top['confidence']:.1f}%</p>
                <p><b>💊 Treatment:</b> {top['treatment']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Top-5 only shown for valid leaf
            st.markdown("### 📊 Top 5 Predictions")
            for r in results:
                c1, c2, c3 = st.columns([3, 3, 1])
                with c1:
                    st.markdown(f"**{r['display'][:38]}**")
                with c2:
                    st.progress(r['confidence'] / 100)
                with c3:
                    st.markdown(f"**{r['confidence']:.1f}%**")

st.markdown("---")
st.markdown(
    "<center><small>EfficientNet-B3 | PlantVillage 54K images | "
    "OOD-based Leaf Validation | Krishna et al. 2025 methods</small></center>",
    unsafe_allow_html=True
)