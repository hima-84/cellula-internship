import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import numpy as np
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DentalScan AI",
    page_icon="🦷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0a0c0f;
    --surface:   #111418;
    --border:    #1e2530;
    --accent:    #00e5ff;
    --accent2:   #7b61ff;
    --green:     #00ff94;
    --red:       #ff4d6d;
    --text:      #e8edf5;
    --muted:     #5a6478;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'Syne', sans-serif;
}

[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 2rem !important; max-width: 760px !important; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.8rem 1rem 1.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.22em;
    color: var(--accent);
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.25);
    border-radius: 2rem;
    padding: 0.3rem 0.9rem;
    margin-bottom: 1.1rem;
    text-transform: uppercase;
}
.hero-title {
    font-size: 3.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.05;
    background: linear-gradient(135deg, #e8edf5 30%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.6rem;
}
.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 400;
    letter-spacing: 0.01em;
    margin-bottom: 0;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.8rem 0;
}

/* ── Upload zone ── */
.upload-label {
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
    display: block;
}

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
    transition: border-color 0.25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── Image preview ── */
.preview-wrap {
    border: 1.5px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    position: relative;
    background: var(--surface);
}
.preview-corner {
    position: absolute;
    width: 14px; height: 14px;
    border-color: var(--accent);
    border-style: solid;
    opacity: 0.7;
}
.preview-corner.tl { top: 8px; left: 8px;  border-width: 2px 0 0 2px; }
.preview-corner.tr { top: 8px; right: 8px; border-width: 2px 2px 0 0; }
.preview-corner.bl { bottom: 8px; left: 8px;  border-width: 0 0 2px 2px; }
.preview-corner.br { bottom: 8px; right: 8px; border-width: 0 2px 2px 0; }

/* ── Scan button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 0 24px rgba(0,229,255,0.2) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Result card ── */
.result-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.result-class {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--accent);
    margin-bottom: 0.15rem;
}
.result-fullname {
    font-size: 0.95rem;
    color: var(--muted);
    margin-bottom: 1.2rem;
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.45rem;
}
.conf-cls {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--text);
    min-width: 42px;
}
.conf-bar-bg {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 999px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(0.34,1.56,0.64,1);
}
.conf-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    min-width: 38px;
    text-align: right;
}

/* ── Info chips ── */
.info-grid {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.info-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    font-size: 0.78rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
}
.info-chip span { color: var(--text); }

/* ── Disclaimer ── */
.disclaimer {
    text-align: center;
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 2rem;
    padding: 0 1rem;
    font-family: 'DM Mono', monospace;
    line-height: 1.6;
}

/* ── Spinner override ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Progress / success ── */
.stSuccess { background: rgba(0,255,148,0.07) !important; border-color: var(--green) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASS_NAMES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

CLASS_INFO = {
    'CaS':  ('Canker Sore',          'Aphthous ulcer — small, painful lesion inside the mouth.'),
    'CoS':  ('Cold Sore',            'Herpes labialis — viral blister near the lips.'),
    'Gum':  ('Gum Disease',          'Gingivitis or periodontitis — inflammation of gum tissue.'),
    'MC':   ('Mouth Cancer',         'Malignant neoplasm — requires immediate clinical evaluation.'),
    'OC':   ('Oral Cancer',          'Squamous cell carcinoma of oral cavity — seek specialist care.'),
    'OLP':  ('Oral Lichen Planus',   'Chronic inflammatory condition affecting oral mucosa.'),
    'OT':   ('Other / Normal',       'Healthy tissue or condition not matching the above classes.'),
}

BAR_COLORS = [
    '#00e5ff','#7b61ff','#00ff94','#ffb800','#ff4d6d','#e040fb','#40c4ff'
]

IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def _build_model():
    """Build EfficientNet-B3 with 7-class head."""
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(CLASS_NAMES)),
    )
    return model


def _load_weights(model, pth_bytes: bytes):
    """Load weights from raw bytes — always strict=False so partial saves work."""
    import io
    buf = io.BytesIO(pth_bytes)
    state = torch.load(buf, map_location='cpu', weights_only=False)

    # Unwrap common checkpoint wrappers
    if isinstance(state, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in state:
                state = state[key]
                break

    # Always use strict=False — backbone is already loaded from ImageNet,
    # this only overrides whatever keys exist in the saved file (head or full).
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith('_')]

    has_features   = any(k.startswith('features.')   for k in state.keys())
    has_classifier = any(k.startswith('classifier.') for k in state.keys())
    kind = "full" if has_features else "head" if has_classifier else "unknown"
    return kind


@st.cache_resource(show_spinner=False)
def load_model_base():
    """Cached base model — ImageNet weights only."""
    m = _build_model()
    m.eval()
    return m


@st.cache_resource(show_spinner=False)
def load_model_with_weights(file_hash: str, pth_bytes: bytes):
    """Cached model keyed by file hash — reloads only when file changes."""
    m = _build_model()
    kind = _load_weights(m, pth_bytes)
    m.eval()
    return m, kind


def predict(model, image: Image.Image):
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">Transfer Learning · EfficientNet-B3</div>
  <div class="hero-title">DentalScan AI</div>
  <div class="hero-sub">Upload an oral image — get instant pathology classification across 7 classes</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — model weights
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Weights")
    st.caption("Upload your trained `.pth` file. Without it, ImageNet weights are used (demo mode).")
    weight_file = st.file_uploader("Load .pth weights", type=["pth", "pt"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Classes**")
    for cls, (full, _) in CLASS_INFO.items():
        st.markdown(f"`{cls}` — {full}")

# ─────────────────────────────────────────────
# LOAD MODEL — store in session_state to persist across reruns
# ─────────────────────────────────────────────
import hashlib

# Initialize session state
if "model_hash" not in st.session_state:
    st.session_state.model_hash = None
if "model" not in st.session_state:
    st.session_state.model = None

# Compute current hash
current_hash = None
pth_bytes = None
if weight_file is not None:
    pth_bytes = weight_file.getvalue()
    current_hash = hashlib.md5(pth_bytes).hexdigest()

# Reload model if: (1) hash changed, or (2) model not loaded yet
need_reload = (st.session_state.model_hash != current_hash) or (st.session_state.model is None)

if need_reload:
    if weight_file is not None:
        # Load custom weights
        with st.spinner("Loading custom weights …"):
            try:
                new_model, weight_kind = load_model_with_weights(current_hash, pth_bytes)
                label = "Classifier head" if weight_kind == "head" else "Full model" if weight_kind == "full" else "Custom"
                st.sidebar.success(f"✓ {label} weights loaded successfully")
                st.session_state.model = new_model
                st.session_state.model_hash = current_hash
            except Exception as e:
                st.sidebar.error(f"⚠ Failed to load custom weights: {e}")
                st.sidebar.info("Using ImageNet backbone instead.")
                st.session_state.model = load_model_base()
                st.session_state.model_hash = None
    else:
        # Load base ImageNet model
        with st.spinner("Loading EfficientNet-B3 (ImageNet) …"):
            st.session_state.model = load_model_base()
            st.session_state.model_hash = None

model = st.session_state.model

# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────
st.markdown('<span class="upload-label">📁 Input Image</span>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop or browse a dental / oral image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    # Preview with corner brackets
    col_img, col_meta = st.columns([3, 2])
    with col_img:
        st.markdown('<div class="preview-wrap">', unsafe_allow_html=True)
        st.image(image, width="stretch")
        st.markdown("""
        <div class="preview-corner tl"></div>
        <div class="preview-corner tr"></div>
        <div class="preview-corner bl"></div>
        <div class="preview-corner br"></div>
        </div>""", unsafe_allow_html=True)

    with col_meta:
        w, h = image.size
        st.markdown(f"""
        <div style="padding-top:0.5rem;">
          <div class="info-chip" style="margin-bottom:0.5rem">📐 <span>{w} × {h} px</span></div>
          <div class="info-chip" style="margin-bottom:0.5rem">🖼 <span>{uploaded.type}</span></div>
          <div class="info-chip" style="margin-bottom:0.5rem">📦 <span>{uploaded.size/1024:.1f} KB</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    run = st.button("🔬  Analyze Image", width="stretch")

    if run:
        with st.spinner("Scanning …"):
            time.sleep(0.4)
            pred_idx, probs = predict(model, image)

        pred_cls   = CLASS_NAMES[pred_idx]
        full_name, description = CLASS_INFO[pred_cls]
        confidence = float(probs[pred_idx]) * 100

        # ── Edge case detection ──────────────────────────────────────────────
        # Entropy: uniform distribution over 7 classes = 1.946 bits (max)
        # High entropy = model is totally confused = likely unrelated image
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        max_entropy = np.log(len(CLASS_NAMES))          # ~1.946
        entropy_ratio = entropy / max_entropy            # 0 = certain, 1 = uniform

        LOW_CONF_THRESHOLD  = 35.0   # top class below this → uncertain
        HIGH_ENTROPY_THRESHOLD = 0.82  # spread too uniform → unrelated image

        is_low_confidence = confidence < LOW_CONF_THRESHOLD
        is_high_entropy   = entropy_ratio > HIGH_ENTROPY_THRESHOLD
        is_unrelated      = is_low_confidence and is_high_entropy

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        

  

        # ── Normal result card ───────────────────────────────────────────────
        if not is_unrelated:
            bars_html = ""
            sorted_pairs = sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1])
            for i, (cls, prob) in enumerate(sorted_pairs):
                pct    = prob * 100
                color  = BAR_COLORS[CLASS_NAMES.index(cls)]
                is_top = cls == pred_cls

                cls_style = "color:#00e5ff;font-weight:700" if is_top else ""
                bar_style = f"width:{pct:.1f}%;background:{color};box-shadow:0 0 8px {color}88" if is_top else f"width:{pct:.1f}%;background:{color}"
                pct_style = "color:#00e5ff" if is_top else ""

                bars_html += (
                    f'<div class="conf-row">'
                    f'<div class="conf-cls" style="{cls_style}">{cls}</div>'
                    f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="{bar_style}"></div></div>'
                    f'<div class="conf-pct" style="{pct_style}">{pct:.1f}%</div>'
                    f'</div>'
                )

            # Confidence badge color
            if confidence >= 70:
                conf_color = "#00ff94"
            elif confidence >= 45:
                conf_color = "#ffb800"
            else:
                conf_color = "#ff4d6d"

            st.markdown(f"""
            <div class="result-card">
              <div class="result-label">Predicted Class</div>
              <div class="result-class">{pred_cls}</div>
              <div class="result-fullname">
                {full_name} &nbsp;·&nbsp;
                <span style="color:{conf_color};font-weight:600">{confidence:.1f}% confidence</span>
              </div>
              <div style="height:1px;background:var(--border);margin:1rem 0"></div>
              <div class="result-label" style="margin-bottom:0.7rem">Confidence Distribution</div>
              {bars_html}
              <div style="height:1px;background:var(--border);margin:1rem 0"></div>
              <div style="font-size:0.83rem;color:var(--muted);line-height:1.6">
                <strong style="color:var(--text)">📋 Description:</strong> {description}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Urgency flag for cancer classes
            if pred_cls in ('MC', 'OC') and confidence > 50:
                st.warning("⚠️ High-confidence malignancy signal detected. Please consult a specialist immediately.", icon="🚨")

else:
    # Placeholder state
    st.markdown("""
    <div style="
        text-align:center; padding:3rem 1rem;
        border:1.5px dashed #1e2530; border-radius:14px;
        background:#111418; margin-top:0.5rem;
        color:#5a6478; font-family:'DM Mono',monospace; font-size:0.82rem; line-height:2;
    ">
        🦷 No image loaded yet<br>
        <span style="font-size:0.7rem;opacity:0.6">Supports JPG · PNG · WebP · BMP</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  ⚕️ For research & educational use only.<br>
  This tool does not constitute medical advice. Always consult a qualified dental professional.
</div>
""", unsafe_allow_html=True)