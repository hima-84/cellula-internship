# Water Body Segmentation — Satellite Imagery

Detect and segment water bodies and flood areas from multispectral satellite imagery using deep learning.

---

## Task Overview

Binary segmentation task — classify each pixel as **water (1)** or **not water (0)** using 12-band Harmonized Sentinel-2/Landsat patches.

---

## Data

| Property | Value |
|---|---|
| Source | Harmonized Sentinel-2 / Landsat |
| Image format | GeoTIFF (.tif) |
| Label format | PNG binary mask |
| Bands | 12 |
| Patch size | 128 × 128 pixels |
| Ground sampling distance | 30m |
| Total matched pairs | 306 |
| Water pixel ratio | ~33% |

**Bands:**
1. Coastal Aerosol
2. Blue
3. Green
4. Red
5. NIR
6. SWIR1
7. SWIR2
8. QA Band
9. Merit DEM
10. Copernicus DEM
11. ESA Land Cover
12. Water Occurrence Probability

---

## Preprocessing

- Loaded multi-band TIFFs with `rasterio` — standard image libraries (PIL, OpenCV) only support 3 channels
- Matched images and labels by filename stem, not position — string sorting caused mismatches (100 before 2)
- Clipped negative values caused by atmospheric correction in bands 2 and 3
- QA band (band 8) normalized by known Landsat flag range [64, 160] and verified against dataset distribution
- All other bands normalized using 2nd–98th percentile clipping computed across the full dataset
- Labels verified as binary [0, 1] — no rescaling needed

---

## Model

**Phase 1 — U-Net from scratch**
- Custom U-Net with 4 encoder/decoder levels
- 12 input channels, 1 output channel
- Result: **IoU ~0.70**

**Phase 2 — Transfer Learning**
- EfficientNet-B4 encoder pretrained on ImageNet
- U-Net decoder via `segmentation_models_pytorch`
- First conv layer replaced to accept 12 input channels (`in_channels=12`)
- All weights trainable — no freezing
- Result: **IoU ~0.80+**

---

## Training

| Setting | Value |
|---|---|
| Loss | BCE (weighted) + Dice |
| pos_weight | 1.97 (calculated from class ratio) |
| Optimizer | AdamW lr=1e-4 |
| Scheduler | OneCycleLR max_lr=1e-3 |
| Batch size | 8 |
| Epochs | 50–100 |
| Split | 80/20 by sample |
| Augmentation | Random horizontal/vertical flips, 90° rotations — applied consistently across all frames |

---

## Results

| Metric | Score |
|---|---|
| IoU | 0.80+ |
| F1 | 0.87+ |
| Precision | ~0.82 |
| Recall | ~0.84 |

---

## Deployment

Flask web application for local inference.

**Structure:**
```
Flask_app/
├── App.py
├── best_transfer.pth
├── requirements.txt
└── templates/
    └── index.html
```

**Run:**
```bash
python -m venv venv
source venv/Scripts/activate      # Windows
pip install -r requirements.txt
python App.py
```

Then open `http://localhost:5000`, upload a `.tif` file, and get back RGB composite, predicted water mask, and probability map.

---

## Key Lessons

- Data pairing bugs are silent and catastrophic — always verify by name not position
- Normalization must be identical between training and inference
- With small datasets, transfer learning consistently outperforms training from scratch
- BCE + Dice loss handles class imbalance better than either loss alone
