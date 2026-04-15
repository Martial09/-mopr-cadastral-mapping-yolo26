# 🚨 IMPORTANT NOTE FOR EVALUATORS 🚨
**Regarding the Google Form Portal Submission:**

Because the official Google Form does not allow us to edit or update documents after the initial submission, the ZIP file attached there only contains our baseline **YOLO Seg 26m** architecture. 

**THIS GitHub repository contains our complete, final, and fully integrated submission.** Please evaluate this repository as our final product. It includes our fully upgraded **dual-model pipeline** (YOLO Seg 26m for buildings + LinkNet/DeepLabV3+ for roads), the automated inference scripts, and the full suite of high-resolution vector PDFs in the `Live-Demo-Results/` folder.

---

## 1. Proposed Solution & System Architecture

Because rural Indian topographies present complex challenges—such as densely packed overlapping roofs and roads heavily occluded by tree canopies—a standard single-model approach is insufficient. We engineered a **Dual-Branch Pipeline** that treats buildings and roads as fundamentally different topological problems.

### Branch A: Instance Segmentation for Buildings
Buildings are discrete objects. We utilized **YOLO Seg 26m** to perform instance segmentation rather than semantic segmentation.
* **Why YOLO Seg 26m:** It excels at drawing distinct bounding boxes and masks for individual structures, preventing adjacent houses from blending into a single massive polygon.
* **Multi-Class Roof Detection:** The model simultaneously extracts the building footprint and classifies the specific roof material (BuiltUp_Roof 1 through 4) in a single pass.

### Branch B: Semantic Ensembling for Road Networks
Roads are continuous topological networks, not discrete objects. To solve canopy occlusion and class imbalance, we built an ensemble pipeline:
* **Model 1 (LinkNet - ResNet34):** A highly sensitive binary segmenter trained exclusively to ask: *"Is this pixel a road?"* It operates with a low threshold (`0.2`) to aggressively capture faint village lanes and dirt paths that heavier models miss.
* **Model 2 (DeepLabV3+ - ResNet50):** A multiclass semantic segmenter that takes the road pixels identified by LinkNet and classifies them into specific categories (State Highway, District Road, Village Road, Lane, Footpath).
* **Direct Masking Logic:** LinkNet acts as the high-recall base layer, and DeepLabV3+ votes on the specific road classification, ensuring unbroken topological continuity.

### The Post-Processing & GIS Engine (The "Secret Weapon")
Raw AI pixel outputs are messy. Our automated Python scripts (`inference_buildings.py` and `inference_roads.py`) contain a custom Geographic Information System (GIS) engine to clean the data before export:
* **Morphological Scissors:** We apply advanced mathematical morphology (`cv2.morphologyEx`). For buildings, an aggressive shrink-and-restore algorithm (erosion/dilation) guarantees that touching roofs are physically severed into individual property records. For roads, morphological closing bridges the gaps caused by overhanging trees.
* **Geometric Filtering:** We automatically delete "spiderweb" artifacts by filtering out polygons with extreme aspect ratios or areas below a strict pixel threshold (`MIN_AREA_PX`).
* **Metric Conversion & Attribute Generation:** The scripts dynamically project the raw pixel coordinates into a real-world metric CRS (`EPSG:32643`). Using `geopandas` and `shapely`, the pipeline calculates exact metric properties (`SHAPE_Area` and `SHAPE_Leng`) and compiles them into a finalized, production-ready `.gpkg` Attribute Table.

## 2. Uniqueness and Innovation  
Most systems blur geometry extraction and object type together. We split them. Our binary mask acts as a structural firewall—multiclass labels only go where roads actually exist, cutting out random mislabels. Topological post-processing cleans up pixel-level errors and forces every road segment to have a clear, consistent classification. Morphological tricks patch up any leftover gaps, making joined-up road vectors. And by using YOLO Seg 26m for instance segmentation, we solve the issue of packed-together structures that stump semantic methods.

## 3. Technology Stack 
The core pipeline is built in Python using PyTorch and Segmentation Models PyTorch (SMP). Road features come from LinkNet (ResNet34) and DeepLabV3+ (ResNet50). We handle instance segmentation (for discrete structures) with YOLO Seg 26m. Rasterio drives geospatial data management and sliding window inference. For post-processing, OpenCV bridges gaps, while SciPy handles connected components. We produce output vectors using Shapely for geometry checks and GeoPandas for packaging everything as GeoPackages (`.gpkg`).

## 4. Expected Impact  
By automating rural cadastral mapping, we can skip manual tracing and deliver accurate maps much faster and for less money. Better cadastral maps improve rural planning and resource allocation, helping local governments manage property and infrastructure efficiently. Consistent, reliable GIS data means state and federal agencies get high-quality, ready-to-use maps for all kinds of rural projects.

## 5. Test Results & Folder Structure
We have executed our full automated inference pipeline directly on the provided **CG** and **PB** benchmark test files. 

* 📁 **`Live-Demo-Results/`**: Contains the complete, post-processed inference results executed directly on the official CG (Chhattisgarh) and PB (Punjab) live demo files. 
  * 📂 **`gpkg-files/`**: The raw GeoPackage vector files generated by our models.
  * 📄 **High-Resolution PDFs**: Geospatial Vector PDFs (e.g., `pargaon.pdf`, `chana.pdf`, `bagai.pdf`, `basant.pdf`, `gudheli.pdf`). **Please zoom in infinitely on these PDFs** to visually verify our edge-matching algorithms, morphological gap-bridging, and clean separation of dense rural structures.
* 📁 **`models/`**: Contains the final trained weights used for inference (YOLO Seg 26m `.pt`, LinkNet `.pt`, and DeepLab `.pt`).
* 📁 **`inference/`**:Contains the Python scripts used for AI inference, executing the sliding-window predictions and converting raw GeoTIFFs into cleaned GIS vector files.
* 📁 **`training/`**: Contains the isolated training scripts and validation data used to train the dual-model architecture.

## 6. Execution Instructions
To run the automated inference pipeline locally:
1. Ensure you have Python 3.9+ installed.
2. Install the required dependencies: 
   ```bash
   pip install -r requirements.txt

NOTE: For CUDA GPU support (highly recommended for inference speed), you may need to install PyTorch manually using the appropriate wheel for your CUDA version after running the requirements file. Example for CUDA 12.1:
Bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

Place your target .tif imagery in your designated test folder.
Update the INPUT_GEOTIFF and OUTPUT_GPKG paths in the configuration section of the inference scripts.
Run the scripts. The code will automatically handle the sliding-window raster inference, morphological cleaning, coordinate deduplication, and final vectorization to .gpkg.
7. Note to Evaluators
We highly recommend beginning your review by opening the PDFs in the Live-Demo-Results/ folder. The generated attribute tables within our raw .gpkg files include calculated metric geometries (SHAPE_Area in square meters, SHAPE_Leng in meters), and specific sub-classifications mapped precisely to the CG and PB datasets.
