# ==============================================================================
# inference-gpu.py — Professional AI Cadastral Mapping Pipeline (Ultra-Fast)
# ==============================================================================

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import torch
from ultralytics import YOLO
from shapely.geometry import shape
import geopandas as gpd
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION (Edit paths here)
# ==============================================================================
INPUT_GEOTIFF = r"E:\gis\test\pargaon.tif"
OUTPUT_GPKG   = r"E:\gis\inference\pargaon.gpkg"

# 🔥 MODEL PATH IS SET RIGHT HERE 🔥
YOLO_WEIGHTS  = r"E:\gis\models\best_m4.pt" 

TILE_SIZE  = 768
STRIDE     = 393
BATCH_SIZE = 16 

# DUAL-THRESHOLD CONFIGURATION 
BLDG_CONF_THRESH = 0.38    # Strict: High precision for rooftops
LINE_CONF_THRESH = 0.15    # Forgiving: Rescues faint roads from the noise floor

IOU_THRESH  = 0.5
MIN_AREA_M2 = 150.0        
MAX_AREA_M2 = 1000000000.0   

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FALLBACK_CRS = "EPSG:32643" 

CLASS_CATEGORY = {
    0: "Built_Up_Area", 1: "Built_Up_Area", 2: "Built_Up_Area", 3: "Built_Up_Area",
    4: "Road", 5: "Road", 6: "Road", 7: "Road", 8: "Road",
    9: "Utility", 10: "Water", 11: "Bridge", 12: "Railway",
}

GPKG_LAYERS = {
    "Built_Up_Area": "built_up_area", "Road": "road", "Utility": "utility",
    "Water": "water", "Bridge": "bridge", "Railway": "railway",
}

# ==============================================================================
# 2. INFERENCE (STABLE SINGLE-SCALE FOR YOLO26)
# ==============================================================================
def run_inference(geotiff_path, weights_path):
    print(f"Loading YOLO26-seg from {weights_path}...")
    model = YOLO(weights_path)
    model.to(DEVICE)
    if DEVICE == "cuda": model.model.half()
    
    raw_detections = []
    with rasterio.open(geotiff_path) as src:
        img_crs = str(src.crs) if src.crs is not None else FALLBACK_CRS
        img_w, img_h = src.width, src.height
        print(f"Processing {img_w}x{img_h} on {DEVICE}...")

        tile_coords = [(x, y) for y in range(0, img_h - TILE_SIZE, STRIDE) 
                              for x in range(0, img_w - TILE_SIZE, STRIDE)]
        total = len(tile_coords)

        for batch_start in range(0, total, BATCH_SIZE):
            batch_coords = tile_coords[batch_start:batch_start + BATCH_SIZE]
            if batch_start % (BATCH_SIZE * 20) == 0:
                print(f"  Tile {batch_start}/{total}...")

            batch_tiles, batch_transforms = [], []
            for x, y in batch_coords:
                window = Window(x, y, TILE_SIZE, TILE_SIZE)
                try:
                    tile_data = src.read([1, 2, 3], window=window)
                    batch_tiles.append(np.moveaxis(tile_data, 0, -1).astype(np.uint8))
                    batch_transforms.append(src.window_transform(window))
                except: continue

            if not batch_tiles: continue

            with torch.amp.autocast(DEVICE, enabled=(DEVICE == "cuda")):
                results = model(batch_tiles, conf=0.10, iou=IOU_THRESH, imgsz=TILE_SIZE, verbose=False)

            for result, tile_transform in zip(results, batch_transforms):
                if result.masks is None: continue
                
                masks_probs = result.masks.data.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                for mask_prob, cls_id, conf in zip(masks_probs, class_ids, confidences):
                    category = CLASS_CATEGORY.get(int(cls_id), "Unknown")

                    # Apply dual thresholding
                    if category in ["Road", "Railway"]:
                        if conf < LINE_CONF_THRESH: continue  
                        mask_np = (mask_prob > LINE_CONF_THRESH).astype(np.uint8) 
                    else:
                        if conf < BLDG_CONF_THRESH: continue  
                        mask_np = (mask_prob > BLDG_CONF_THRESH).astype(np.uint8) 

                    raw_detections.append({
                        'mask_np': mask_np, 'tile_transform': tile_transform,
                        'class_id': int(cls_id), 'confidence': float(conf), 'img_crs': img_crs,
                    })
    return raw_detections, img_crs

# ==============================================================================
# 3. TOPOLOGICAL VECTOR SNAPPING 
# ==============================================================================
def vectorize_and_merge(raw_detections):
    print("\nVectorizing and Snapping Geometries (Ultra-Fast Mode)...")
    
    poly_list = []
    for det in raw_detections:
        mask_uint8 = det['mask_np'].astype(np.uint8)
        for geom_dict, val in shapes(mask_uint8, transform=det['tile_transform']):
            if val > 0:
                poly = shape(geom_dict)
                if poly.is_valid and not poly.is_empty:
                    poly_list.append({
                        'geometry': poly, 'class_id': det['class_id'],
                        'confidence': det['confidence'], 'img_crs': det['img_crs']
                    })

    if not poly_list: return []

    gdf = gpd.GeoDataFrame(poly_list, crs=poly_list[0]['img_crs'])
    merged_detections = []

    for cls_id, group in gdf.groupby('class_id'):
        category = CLASS_CATEGORY.get(int(cls_id), "Unknown")
        max_conf = group['confidence'].max()
        img_crs = group['img_crs'].iloc[0]

        print(f"  -> Processing {len(group)} raw polygons for {category}...")

        if category in ["Road", "Railway"]:
            # 🔥 ROADS ONLY: Topological Snapping (Buffer & Dissolve)
            buffered = group.geometry.buffer(2.5, resolution=2)
            merged_macro = buffered.unary_union
            restored_macro = merged_macro.buffer(-2.5, resolution=2)
            
            if restored_macro.geom_type == 'MultiPolygon':
                geometries = list(restored_macro.geoms)
            elif restored_macro.geom_type == 'Polygon':
                geometries = [restored_macro]
            else:
                geometries = [g for g in getattr(restored_macro, 'geoms', []) if g.geom_type in ['Polygon', 'MultiPolygon']]
                
            for geom in geometries:
                # Regularize road vectors for CAD output
                clean_geom = geom.simplify(2.0, preserve_topology=True)
                if not clean_geom.is_empty:
                    merged_detections.append({
                        'geometry': clean_geom, 'class_id': cls_id, 
                        'confidence': max_conf, 'img_crs': img_crs
                    })

        else:
            # 🔥 BUILDINGS ONLY: Zero Snapping, Zero Dilation. Raw AI Output.
            for geom in group.geometry.tolist():
                if not geom.is_empty:
                    merged_detections.append({
                        'geometry': geom, 'class_id': cls_id, 
                        'confidence': max_conf, 'img_crs': img_crs
                    })

    print(f"After merge: {len(merged_detections)} CAD-ready instances")
    return merged_detections

# ==============================================================================
# 4. EXPORT
# ==============================================================================
def export_gpkg(detections, output_path, img_crs, metric_crs="EPSG:32643"):
    print(f"\nExporting to {output_path}...")
    by_category = defaultdict(list)
    for det in detections:
        by_category[CLASS_CATEGORY.get(det['class_id'], "Unknown")].append(det)

    for category, dets in by_category.items():
        layer_name = GPKG_LAYERS.get(category, category.lower().replace(" ", "_"))
        rows = []
        for det in dets:
            try:
                gdf_metric = gpd.GeoDataFrame([{'geometry': det['geometry']}], crs=det.get('img_crs', img_crs)).to_crs(metric_crs)
                geom_metric = gdf_metric.geometry.iloc[0]
                area_m2 = round(geom_metric.area, 2)
                
                # Apply minimum area noise filter
                if area_m2 < MIN_AREA_M2: continue

                rows.append({
                    'Type': int(det['class_id']), 'SHAPE_Leng': round(geom_metric.length, 2),
                    'SHAPE_Area': area_m2, 'Remarks': "AI Extracted",
                    'geometry': gdf_metric.to_crs(img_crs).geometry.iloc[0],
                })
            except: continue

        if rows:
            gdf = gpd.GeoDataFrame(rows, crs=img_crs)
            mode = 'w' if not os.path.exists(output_path) else 'a'
            gdf.to_file(output_path, driver="GPKG", layer=layer_name, mode=mode)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    raw_dets, img_crs = run_inference(INPUT_GEOTIFF, YOLO_WEIGHTS)
    if raw_dets:
        merged_dets = vectorize_and_merge(raw_dets)
        export_gpkg(merged_dets, OUTPUT_GPKG, img_crs)
        print("\n✅ Pipeline Complete! Open the .gpkg in QGIS.")
    else:
        print("No detections found.")