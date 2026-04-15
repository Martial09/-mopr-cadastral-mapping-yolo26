
# 🛰️ Dual-Branch AI Pipeline for Rural Cadastral Mapping

> **Note for Evaluators:** The Google Form ZIP contains only our baseline **YOLO Seg 26m** architecture, as the portal does not allow post-submission edits. **This GitHub repository is our complete, final submission.** It includes the full dual-model pipeline (YOLO Seg 26m for buildings + LinkNet/DeepLabV3+ for roads), automated inference scripts, and the full suite of high-resolution vector PDFs in `Live-Demo-Results/`.

---

## 📐 Solution Overview & System Architecture

Rural Indian topographies present complex challenges — densely packed overlapping roofs and roads heavily occluded by tree canopies make a single-model approach insufficient. We engineered a **Dual-Branch Pipeline** that treats buildings and roads as fundamentally different topological problems.

---

### Branch A — Instance Segmentation for Buildings (YOLO Seg 26m)

Buildings are discrete objects. We use **YOLO Seg 26m** for instance segmentation (not semantic segmentation), which:

- Draws distinct bounding boxes and masks for **individual structures**, preventing adjacent houses from blending into a single polygon.
- Simultaneously extracts building footprints and classifies specific **roof material types** (`BuiltUp_Roof 1–4`) in a single pass.

---

### Branch B — Semantic Ensembling for Road Networks

Roads are continuous topological networks, not discrete objects. To handle canopy occlusion and class imbalance, we built a two-model ensemble:

| Model | Architecture | Role |
|---|---|---|
| **LinkNet** | ResNet34 | High-recall binary segmenter — aggressively identifies road pixels (threshold: `0.2`) to capture faint village lanes and dirt paths |
| **DeepLabV3+** | ResNet50 | Multiclass semantic segmenter — classifies road pixels from LinkNet into specific categories (State Highway, District Road, Village Road, Lane, Footpath) |

**Direct Masking Logic:** LinkNet acts as the high-recall base layer; DeepLabV3+ votes on specific road classification — ensuring unbroken topological continuity.

---

### Post-Processing & GIS Engine

Raw AI pixel outputs are noisy. Our Python inference scripts contain a custom GIS engine that cleans outputs before export:

- **Morphological Scissors (`cv2.morphologyEx`):** An aggressive shrink-and-restore (erosion/dilation) algorithm physically severs touching building roofs into individual property records. For roads, morphological closing bridges gaps caused by overhanging trees.
- **Geometric Filtering:** Automatically removes "spiderweb" artifacts by filtering polygons with extreme aspect ratios or areas below `MIN_AREA_PX`.
- **Metric Conversion & Attribute Generation:** Raw pixel coordinates are dynamically projected into a real-world metric CRS (`EPSG:32643`). Using `geopandas` and `shapely`, the pipeline calculates exact metric properties (`SHAPE_Area`, `SHAPE_Leng`) and compiles them into a production-ready `.gpkg` Attribute Table.

---

## 💡 Uniqueness & Innovation

- A **binary mask acts as a structural firewall** — multiclass labels are only assigned where roads actually exist, eliminating random mislabels.
- **Topological post-processing** enforces consistent classification across every road segment.
- **Morphological patching** closes pixel-level gaps, producing seamless connected road vectors.
- **YOLO Seg 26m instance segmentation** cleanly resolves densely packed structures that confuse semantic methods.

---

## 🛠️ Technology Stack

| Category | Tools / Libraries |
|---|---|
| Core Framework | Python, PyTorch, Segmentation Models PyTorch (SMP) |
| Road Segmentation | LinkNet (ResNet34), DeepLabV3+ (ResNet50) |
| Building Segmentation | YOLO Seg 26m |
| Geospatial I/O | Rasterio (sliding window inference) |
| Post-Processing | OpenCV (morphology), SciPy (connected components) |
| Vector Output | Shapely (geometry), GeoPandas (GeoPackage `.gpkg`) |

---

## 📁 Repository Structure
├── Test-Set-Results/ # Full inference results on CG & PB benchmark files │ ├── gpkg-files/ # Raw GeoPackage vector files │ └── *.pdf # High-resolution geospatial vector PDFs │ ├── models/ # Final trained model weights │ ├── yolo_seg_26m.pt │ ├── linknet_resnet34.pt │ └── deeplab_resnet50.pt │ ├── inference/ # Inference scripts │ ├── inference_buildings.py │ └── inference_roads.py │ ├── training/ # Training scripts & validation data └── requirements.txt

---

## 🚀 Running the Pipeline

**Prerequisites:** Python 3.9+

```bash
pip install -r requirements.txt
```

> **For CUDA GPU support** (highly recommended), install PyTorch manually after the above. Example for CUDA 12.1:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

1. Place your target `.tif` imagery in your designated test folder.
2. Update `INPUT_GEOTIFF` and `OUTPUT_GPKG` paths in the configuration section of the inference script.
3. Run:

```bash
python inference/inference_buildings.py
python inference/inference_roads.py
```

The scripts automatically handle sliding-window inference, morphological cleaning, coordinate deduplication, and vectorization to `.gpkg`.

---

## 📊 Test Results

The full pipeline has been executed on the official **CG (Chhattisgarh)** and **PB (Punjab)** benchmark files. Results are in `Live-Demo-Results/`.

Generated `.gpkg` attribute tables include:
- `SHAPE_Area` — area in square meters
- `SHAPE_Leng` — perimeter/length in meters
- Sub-classifications mapped precisely to the CG and PB datasets

---

## 🔍 Note to Evaluators

Start by opening the PDFs in `Test-Set-Results/` and **zoom in infinitely** to verify:
- Edge-matching algorithms
- Morphological gap-bridging for roads under tree canopy
- Clean separation of densely packed rural structures

---

## 📈 Expected Impact

- **Faster, cheaper mapping** — eliminates slow and expensive manual tracing
- **Better rural planning** — accurate cadastral maps improve resource allocation for local governments
- **Production-ready GIS data** — consistent, high-quality `.gpkg` files ready for state and federal agency use

