import sys
print("starting...", flush=True)
sys.stdout.flush()


import os
import io
import base64
import numpy as np
import rasterio
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use('Agg')  # no display needed on server
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify
from PIL import Image

app = Flask(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# same normalization from training — must be identical or predictions are wrong
def normalize_single(img, qa_fixed=True):
    # img shape: (12, 128, 128)
    out = np.zeros_like(img, dtype=np.float32)
    for b in range(img.shape[0]):
        band = img[b].copy()
        band = np.clip(band, 0, None)  # clip negatives

        if b == 7 and qa_fixed:  # QA band fixed range
            lo, hi = 64.0, 160.0
        else:
            lo = np.percentile(band, 2)
            hi = np.percentile(band, 98)

        out[b] = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def make_rgb(img, r=3, g=2, b=1):
    rgb = np.stack([img[r], img[g], img[b]], axis=-1).astype(np.float32)
    for c in range(3):
        lo = np.percentile(rgb[:, :, c], 2)
        hi = np.percentile(rgb[:, :, c], 98)
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb


def load_model(weights_path):
    model = smp.Unet(
        encoder_name    = 'efficientnet-b4',
        encoder_weights = None,   # no imagenet download — we load our weights
        in_channels     = 12,
        classes         = 1,
        activation      = None
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


# load model once at startup
WEIGHTS_PATH = 'D:\\IbrahimNagy\\Flask_app\\best_transfer.pth'
WEIGHTS_PATH = 'best_transfer.pth'
model = None
if os.path.exists(WEIGHTS_PATH):
    try:
        print("about to load model...", flush=True)  #  this
        model = load_model(WEIGHTS_PATH)
        print("model done", flush=True)              #  this
    except Exception as e:
        print(f'ERROR: {e}', flush=True)


@app.route('/')
def index():
    model_ready = model is not None
    return render_template('index.html', model_ready=model_ready)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Place best_transfer.pth next to app.py'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith(('.tif', '.tiff')):
        return jsonify({'error': 'Please upload a .tif file'}), 400

    # read tif from memory
    file_bytes = file.read()
    with rasterio.open(io.BytesIO(file_bytes)) as src:
        img_raw = src.read().astype(np.float32)  # (bands, H, W)

    # validate shape
    if img_raw.shape[0] != 12:
        return jsonify({'error': f'Expected 12 bands, got {img_raw.shape[0]}'}), 400

    # normalize exactly as in training
    img_norm = normalize_single(img_raw, qa_fixed=True)

    # run model
    tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob_map  = torch.sigmoid(model(tensor)).cpu().numpy()[0, 0]  # (H, W)
    pred_mask = (prob_map > 0.5).astype(np.float32)

    water_pct = float(pred_mask.mean() * 100)

    # build result figure — RGB | Prediction | Probability
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(make_rgb(img_norm))
    axes[0].set_title('RGB Composite', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(pred_mask, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title(f'Water Prediction  ({water_pct:.1f}% water)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(prob_map, cmap='RdYlGn', vmin=0, vmax=1)
    axes[2].set_title('Probability Map', fontsize=12)
    axes[2].axis('off')

    plt.suptitle(file.filename, fontsize=11, color='gray')
    plt.tight_layout()

    image_b64 = fig_to_base64(fig)

    return jsonify({
        'image'     : image_b64,
        'water_pct' : round(water_pct, 2),
        'shape'     : list(img_raw.shape)
    })


if __name__ == '__main__':
    print("launching flask", flush=True)
    app.run(debug=True, port=5000, use_reloader=False)