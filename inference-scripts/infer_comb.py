import os
import time
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import torch
import segmentation_models_pytorch as smp
import cv2
from tqdm import tqdm
from shapely.geometry import shape
from shapely.validation import make_valid
import geopandas as gpd
from scipy import ndimage

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_GEOTIFF        = r"E:\gis\test\bagai.tif"
OUTPUT_GPKG          = r"E:\gis\submission\bagai_roads.gpkg"
BINARY_WEIGHTS       = r"E:\gis\submission\model\linknet.pt"
MULTICLASS_WEIGHTS   = r"E:\gis\submission\model\deeplab.pt"

TILE_SIZE  = 768
STRIDE     = 384
BATCH_SIZE = 4

FALLBACK_CRS     = "EPSG:32643"
METRIC_CRS       = "EPSG:32643"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

BINARY_THRESHOLD = 0.2

MORPH_CLOSE_PX = 50
MORPH_OPEN_PX  = 1
MIN_AREA_PX    = 50


MODEL_ROAD_CLASSES = [1, 2, 3, 4, 5] 
EXPORT_ROAD_IDS    = [1, 3, 4, 5, 6]

# Maps the final YOLO IDs to the correct text names for the Attribute Table
TYPE_MAPPING = {
    1: "Major Road / State Highway", 
    3: "District Road",
    4: "Village Road", 
    5: "Lane",
    6: "Path / Footpath"
}

def get_time():
    return time.strftime('%H:%M:%S')

# ==============================================================================
# MODEL LOADERS
# ==============================================================================
def load_binary_model(path):
    print(f"[{get_time()}] Loading binary LinkNet...")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd   = {k.replace("module.", ""): v
            for k, v in ckpt.get("model_state_dict", ckpt).items()}
    model = smp.Linknet(encoder_name="resnet34", encoder_weights=None,
                        in_channels=3, classes=2)
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()

def load_multiclass_model(path):
    print(f"[{get_time()}] Loading multiclass DeepLab...")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd   = {k.replace("module.", ""): v
            for k, v in ckpt.get("model_state_dict", ckpt).items()}
    num_classes = sd["segmentation_head.0.weight"].shape[0]
    
    # Grab the model's valid classes and trim the export list to match if needed
    valid_model_classes = [c for c in MODEL_ROAD_CLASSES if c < num_classes]
    valid_export_ids    = EXPORT_ROAD_IDS[:len(valid_model_classes)]
    
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None,
                              in_channels=3, classes=num_classes)
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval(), valid_model_classes, valid_export_ids

def preprocess(tile_rgb):
    img  = tile_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

# ==============================================================================
# COMBINED INFERENCE (DIRECT MASKING LOGIC)
# ==============================================================================
def run_robust_inference(geotiff_path, binary_model, multiclass_model, valid_model_classes, valid_export_ids):
    print(f"\n[{get_time()}] Opening: {geotiff_path}")

    with rasterio.open(geotiff_path) as src:
        img_crs       = str(src.crs) if src.crs else FALLBACK_CRS
        img_w, img_h  = src.width, src.height
        geo_transform = src.transform
        print(f"[{get_time()}] Image: {img_w} x {img_h} | CRS: {img_crs}")

        max_x = max(0, img_w - TILE_SIZE)
        max_y = max(0, img_h - TILE_SIZE)
        xs = sorted(set(list(range(0, max_x, STRIDE)) + [max_x]))
        ys = sorted(set(list(range(0, max_y, STRIDE)) + [max_y]))
        tile_coords = [(x, y) for y in ys for x in xs]
        total_tiles = len(tile_coords)

        road_type_map = np.zeros((img_h, img_w), dtype=np.uint8)

        def flush_batch(batch):
            tensors = torch.stack([b['tensor'] for b in batch]).to(DEVICE)

            with torch.no_grad():
                ctx = (torch.cuda.amp.autocast() if DEVICE == 'cuda' else torch.no_grad.__class__())
                with ctx:
                    bin_logits   = binary_model(tensors)
                    multi_logits = multiclass_model(tensors)

            bin_probs = torch.softmax(bin_logits, dim=1)[:, 1]

            multi_probs = torch.softmax(multi_logits, dim=1)
            road_probs = multi_probs[:, valid_model_classes, :, :]
            _, best_road_idx = torch.max(road_probs, dim=1)
            
            # Instantly map the internal 0-4 index to the final 1,3,4,5,6 ID
            class_lut = torch.tensor(valid_export_ids, dtype=torch.uint8, device=DEVICE)
            best_road_class = class_lut[best_road_idx]

            bin_np = bin_probs.float().cpu().numpy()
            cls_np = best_road_class.cpu().numpy()

            for j, item in enumerate(batch):
                x, y = item['x'], item['y']
                h = min(TILE_SIZE, img_h - y)
                w = min(TILE_SIZE, img_w - x)

                b_prob = bin_np[j, :h, :w]
                r_cls  = cls_np[j, :h, :w]
                
                b_prob[item['nodata'][:h, :w]] = 0.0

                is_road = b_prob > BINARY_THRESHOLD
                final_tile = np.where(is_road, r_cls, 0).astype(np.uint8)

                current_patch = road_type_map[y:y+h, x:x+w]
                road_type_map[y:y+h, x:x+w] = np.maximum(current_patch, final_tile)

        buf = []
        print(f"\n[{get_time()}] Running direct masked inference...")
        with tqdm(total=total_tiles, desc="Tiles") as pbar:
            for x, y in tile_coords:
                w = min(TILE_SIZE, img_w - x)
                h = min(TILE_SIZE, img_h - y)

                try:
                    tile_data = src.read([1, 2, 3], window=Window(x, y, w, h))
                except Exception as e:
                    pbar.update(1)
                    continue

                tile_rgb = np.moveaxis(tile_data, 0, -1).astype(np.uint8)

                if tile_rgb.shape[:2] != (TILE_SIZE, TILE_SIZE):
                    pad = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                    pad[:h, :w] = tile_rgb
                    tile_rgb = pad

                nodata = np.all(tile_rgb == 0, axis=-1)
                if np.all(nodata):
                    pbar.update(1)
                    continue

                buf.append({'x': x, 'y': y, 'tensor': preprocess(tile_rgb), 'nodata': nodata})

                while len(buf) >= BATCH_SIZE:
                    flush_batch(buf[:BATCH_SIZE])
                    buf = buf[BATCH_SIZE:]

                pbar.update(1)

            if buf:
                flush_batch(buf)

    print(f"[{get_time()}] Road pixels found: {(road_type_map > 0).sum():,}")
    return road_type_map, geo_transform, img_crs

# ==============================================================================
# MASK CLEANUP & MAJORITY VOTING
# ==============================================================================
def clean_mask(road_type_map):
    print(f"\n[{get_time()}] Cleaning mask (morphology + majority voting)...")
    binary = (road_type_map > 0).astype(np.uint8)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_PX, MORPH_CLOSE_PX))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_PX,  MORPH_OPEN_PX))
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close);  del binary
    opened  = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  k_open);   del closed

    labels, n = ndimage.label(opened);  del opened

    if n == 0:
        return np.zeros_like(labels, dtype=np.uint8)

    sizes = np.zeros(n + 1, dtype=np.int64)
    CHUNK = 2048
    for i in range(0, labels.shape[0], CHUNK):
        sizes += np.bincount(labels[i:i+CHUNK].ravel(), minlength=n+1)

    keep       = np.where(sizes >= MIN_AREA_PX)[0]
    keep       = keep[keep > 0]
    keep_mask  = np.zeros(n + 1, dtype=bool)
    keep_mask[keep] = True

    n_road_classes = max(EXPORT_ROAD_IDS) + 1
    flat_labels = labels.ravel()
    flat_types  = road_type_map.ravel()

    valid_px     = keep_mask[flat_labels] & (flat_types > 0)
    valid_lbl    = flat_labels[valid_px]
    valid_typ    = flat_types [valid_px].astype(np.int64)

    linear_idx   = valid_lbl * n_road_classes + valid_typ
    counts       = np.bincount(linear_idx, minlength=(n + 1) * n_road_classes).reshape(n + 1, n_road_classes)

    majority_types       = np.argmax(counts[:, 1:], axis=1) + 1
    majority_types       = majority_types.astype(np.uint8)
    majority_types[0]    = 0

    clean = np.zeros_like(labels, dtype=np.uint8)
    for i in range(0, labels.shape[0], CHUNK):
        chunk = labels[i:i+CHUNK]
        valid = keep_mask[chunk]
        clean[i:i+CHUNK] = np.where(valid, majority_types[chunk], 0)

    return clean

# ==============================================================================
# VECTORISE & EXPORT (EXPANDED ATTRIBUTE TABLE)
# ==============================================================================
def vectorize_and_export(clean_map, geo_transform, img_crs, output_path):
    print(f"\n[{get_time()}] Vectorizing & building Attribute Table...")
    raw_rows = []
    
    for geom_dict, val in tqdm(shapes(clean_map, mask=(clean_map > 0), transform=geo_transform), desc="Polygons"):
        geom = shape(geom_dict)
        
        if not geom.is_valid:
            geom = make_valid(geom)
            
        if not geom.is_empty:
            raw_rows.append({
                'geometry':   geom, 
                'Class_ID':   int(val)
            })

    if not raw_rows:
        print(f"[{get_time()}] No polygons found — check threshold and model output.")
        return

    print(f"[{get_time()}] Calculating metric area and length...")
    
    # Create initial GeoDataFrame
    gdf = gpd.GeoDataFrame(raw_rows, crs=img_crs)
    
    # Convert to metric CRS to calculate accurate meters and square meters
    gdf_metric = gdf.to_crs(METRIC_CRS)
    
    # Build the expanded attribute table using fast vectorized operations
    gdf['Type']       = "Road Network"
    gdf['Road_Type']  = gdf['Class_ID'].map(TYPE_MAPPING).fillna("Unknown_Road")
    gdf['SHAPE_Leng'] = gdf_metric.geometry.length.round(2)
    gdf['SHAPE_Area'] = gdf_metric.geometry.area.round(2)
    gdf['Remarks']    = 'AI Extracted'

    # Reorder columns for a clean, professional GIS layout
    gdf = gdf[['Type', 'Road_Type', 'Class_ID', 'SHAPE_Leng', 'SHAPE_Area', 'Remarks', 'geometry']]

    print(f"[{get_time()}] Exporting {len(gdf):,} polygons into EXACTLY ONE layer...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    gdf.to_file(output_path, driver="GPKG", layer="rural_roads")
    
    print(f"[{get_time()}] ✅ Saved → {output_path}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    t0 = time.time()

    binary_model                                       = load_binary_model(BINARY_WEIGHTS)
    multiclass_model, valid_model_cls, valid_exp_ids   = load_multiclass_model(MULTICLASS_WEIGHTS)

    road_type_map, geo_transform, crs = run_robust_inference(
        INPUT_GEOTIFF, binary_model, multiclass_model, valid_model_cls, valid_exp_ids
    )

    del binary_model, multiclass_model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    clean_map = clean_mask(road_type_map)
    del road_type_map

    vectorize_and_export(clean_map, geo_transform, crs, OUTPUT_GPKG)
    print(f"\n✅ Done in {time.time()-t0:.1f}s")