# YOLO26 — Rural Cadastral Mapping

> YOLO-based infrastructure segmentation pipeline for rural satellite imagery.  
> Training history, inference architecture, and model weights.

---

## Why this repo exists

The training notebooks live on Kaggle under a private visibility setting. Because the dataset carries strict non-disclosure requirements, keeping the notebooks public would risk leaking validation and test imagery through cached output cells. This repo is the clean counterpart — it holds everything needed to **run** inference without exposing the data.

---

## Repository Structure
```

│
├── training/
│   ├── run1/                  # Baseline — 30 epochs
│   │   ├── training-1.ipynb   # Training notebook for Run 1
│   │   └── output/            # Loss curves, MaskF1 graphs, inference outputs
│   │
│   ├── run2/                  # First fine-tune — 10 epochs
│   │   ├── training-2.ipynb   # Training notebook for Run 2
│   │   └── output/            # Loss curves, MaskF1 graphs, inference outputs
│   │
│   └── run3/                  # Final convergence — 10 epochs
│       ├── training-3.ipynb   # Training notebook for Run 3
│       └── output/            # Loss curves, MaskF1 graphs, inference outputs
│
├── weights/                   # Final model weights
│   ├── best.pt                # Current best model — promoted from Run 3
│   └── last.pt                # Last checkpoint from Run 3
│
├── inference-gpu.py           # Vectorized inference + topological snapping
└── requirements.txt
```

> **Note:** Correct the folder/file names above if they differ from your actual structure — this is the inferred layout based on your description.

---

## Training — 50 epochs across 3 runs

A staggered training strategy was used to prevent the model from memorizing the micro-gaps baked into the hand-annotated ground truth shapefiles.

### Run 1 · Baseline (30 epochs)
Full-dataset training to establish feature extraction for rural building footprints and continuous linear infrastructure.

### Run 2 · First fine-tune (10 epochs)
Hyperparameters adjusted to stabilize loss curves and sharpen the model's ability to separate faint, dusty road pixels from background noise.

### Run 3 · Final convergence (10 epochs)
Locked in `best.pt`. The MaskF1 curves from this run determined the **dual-threshold** strategy used at inference.

---

## Inference pipeline

Post-processing is split by geometry type to avoid a one-size-fits-all dilation ruining building corners.

### Roads — linear infrastructure
```
buffer-and-dissolve  →  Douglas-Peucker regularization
```
Uses GeoPandas geometry operations to bridge road gaps, then simplifies the vector output.

### Buildings — polygonal assets
```
tile-overlap merge only  (no dilation)
```
Dilation is skipped entirely. Merges happen only at tile boundaries, keeping 90° corners intact.

### Thresholds

| Class     | Confidence threshold | Strategy                          |
|-----------|----------------------|-----------------------------------|
| Buildings | `0.38`               | Strict — minimize false positives |
| Roads     | `0.15`               | High recall — catch faint tracks  |

---

## Setup
```bash
pip install -r requirements.txt
python inference-gpu.py
```


Edit the three path variables at the top of `inference-gpu.py` if your input/output locations differ:
```python
INPUT_GEOTIFF = r"input/your_image.tif"
OUTPUT_GPKG   = r"output/your_output.gpkg"
YOLO_WEIGHTS  = r"best.pt"
```

---

## Output

A regularized **GeoPackage (`.gpkg`)** — CAD-standard geometry, ready for QGIS or direct upload to government cadastral databases.
