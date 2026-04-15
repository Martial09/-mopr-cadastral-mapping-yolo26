# ==============================================================================
# inference-gpu.py — Buildings + Utility + Water + Bridge (No Roads)
# ==============================================================================

import os
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import torch
from ultralytics import YOLO
from shapely.geometry import shape, Polygon as ShapelyPolygon
import geopandas as gpd
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
INPUT_GEOTIFF = r"E:\gis\test\bagai.tif"
OUTPUT_GPKG   = r"E:\gis\submission\bagai_buildings.gpkg"
YOLO_WEIGHTS  = r"E:\gis\submission\model\yolo_26_best.pt"

TILE_SIZE  = 768
STRIDE     = 384
BATCH_SIZE = 16

BLDG_CONF_THRESH  = 0.4
OTHER_CONF_THRESH = 0.30
MASK_PROB_THRESH  = 0.45
IOU_THRESH        = 0.4

MIN_AREA_M2       = 15.0
MAX_AREA_M2       = 50000.0
MIN_AREA_OTHER_M2 = 8.0
MAX_ASPECT_RATIO  = 4.0
SHRINK_M          = 1.0

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
FALLBACK_CRS = "EPSG:32643"
METRIC_CRS   = "EPSG:32643"

# ==============================================================================
# 2. CLASS MAP — mirrors training label code exactly
# ==============================================================================
# class_id → (category, roof_type, class_name)
CLASS_INFO = {
    0:  ("Built_Up_Area", 1, "BuiltUp_Roof1"),
    1:  ("Built_Up_Area", 2, "BuiltUp_Roof2"),
    2:  ("Built_Up_Area", 3, "BuiltUp_Roof3"),
    3:  ("Built_Up_Area", 4, "BuiltUp_Roof4"),
    # 4-8 = Roads  → HARD SKIP
    9:  ("Utility",       None, "Utility"),
    10: ("Water",         None, "Water"),
    11: ("Bridge",        None, "Bridge"),
    # 12 = Railway → HARD SKIP
}

SKIP_CLASS_IDS  = {4, 5, 6, 7, 8, 12}
BLDG_CLASS_IDS  = {0, 1, 2, 3}
OTHER_CLASS_IDS = {9, 10, 11}

GPKG_LAYERS = {
    "Built_Up_Area": "buildings",
    "Utility"      : "utility",
    "Water"        : "water",
    "Bridge"       : "bridge",
}

# ==============================================================================
# 3. HELPERS
# ==============================================================================
def get_aspect_ratio(geom):
    try:
        rect   = geom.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
        l1 = math.hypot(coords[0][0]-coords[1][0], coords[0][1]-coords[1][1])
        l2 = math.hypot(coords[1][0]-coords[2][0], coords[1][1]-coords[2][1])
        if l1 == 0 or l2 == 0:
            return 100.0
        return max(l1, l2) / min(l1, l2)
    except:
        return 1.0


def centroid_dedup(gdf_metric, snap_m):
    gdf_metric = gdf_metric.copy().reset_index(drop=True)
    centroids  = list(gdf_metric.geometry.centroid)
    areas      = list(gdf_metric.geometry.area)
    suppressed = set()

    for i in range(len(gdf_metric)):
        if i in suppressed:
            continue
        for j in range(i + 1, len(gdf_metric)):
            if j in suppressed:
                continue
            if centroids[i].distance(centroids[j]) < snap_m:
                suppressed.add(j if areas[i] >= areas[j] else i)

    return gdf_metric.drop(index=list(suppressed)).reset_index(drop=True)


# ==============================================================================
# 4. INFERENCE
# ==============================================================================
def run_inference(geotiff_path, weights_path):
    print(f"[1/3] Loading model from {weights_path} ...")
    model = YOLO(weights_path)
    model.to(DEVICE)
    if DEVICE == "cuda":
        model.model.half()

    raw_detections = []

    with rasterio.open(geotiff_path) as src:
        img_crs  = str(src.crs) if src.crs else FALLBACK_CRS
        img_w, img_h = src.width, src.height
        print(f"    Image: {img_w}x{img_h} px  |  Device: {DEVICE}  |  CRS: {img_crs}")

        tile_coords = []
        for y in range(0, img_h, STRIDE):
            for x in range(0, img_w, STRIDE):
                x_end   = min(x + TILE_SIZE, img_w)
                y_end   = min(y + TILE_SIZE, img_h)
                x_start = max(0, x_end - TILE_SIZE)
                y_start = max(0, y_end - TILE_SIZE)
                tile_coords.append((x_start, y_start))

        tile_coords = list(dict.fromkeys(tile_coords))
        total = len(tile_coords)
        print(f"    Total tiles: {total}")

        for batch_start in range(0, total, BATCH_SIZE):
            batch_coords = tile_coords[batch_start : batch_start + BATCH_SIZE]

            if batch_start % (BATCH_SIZE * 10) == 0:
                pct = round(100 * batch_start / total, 1)
                print(f"    Tile {batch_start}/{total}  ({pct}%)")

            batch_tiles, batch_transforms = [], []
            for x, y in batch_coords:
                window = Window(x, y, TILE_SIZE, TILE_SIZE)
                try:
                    tile_data = src.read([1, 2, 3], window=window)
                    if tile_data.shape[1] == 0 or tile_data.shape[2] == 0:
                        continue
                    batch_tiles.append(np.moveaxis(tile_data, 0, -1).astype(np.uint8))
                    batch_transforms.append(src.window_transform(window))
                except Exception as e:
                    print(f"    [WARN] Tile read failed at ({x},{y}): {e}")
                    continue

            if not batch_tiles:
                continue

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                results = model(
                    batch_tiles,
                    conf=min(BLDG_CONF_THRESH, OTHER_CONF_THRESH),
                    iou=IOU_THRESH,
                    imgsz=TILE_SIZE,
                    verbose=False,
                )

            for result, tile_transform in zip(results, batch_transforms):
                if result.masks is None:
                    continue

                masks_probs = result.masks.data.cpu().numpy()
                class_ids   = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                for mask_prob, cls_id, conf in zip(masks_probs, class_ids, confidences):
                    cls_id = int(cls_id)

                    if cls_id in SKIP_CLASS_IDS:
                        continue
                    if cls_id in BLDG_CLASS_IDS  and conf < BLDG_CONF_THRESH:
                        continue
                    if cls_id in OTHER_CLASS_IDS  and conf < OTHER_CONF_THRESH:
                        continue
                    if cls_id not in CLASS_INFO:
                        continue

                    mask_np = (mask_prob > MASK_PROB_THRESH).astype(np.uint8)
                    if mask_np.sum() < 20:
                        continue

                    raw_detections.append({
                        "mask_np"       : mask_np,
                        "tile_transform": tile_transform,
                        "class_id"      : cls_id,
                        "confidence"    : float(conf),
                        "img_crs"       : img_crs,
                    })

    print(f"    Raw detections: {len(raw_detections)}")
    return raw_detections, img_crs


# ==============================================================================
# 5. VECTORIZE + CLEAN
# ==============================================================================
def process_buildings(gdf_metric):
    cleaned = []

    for _, row in gdf_metric.iterrows():
        geom   = row["geometry"].buffer(0)
        cls_id = row["class_id"]
        conf   = row["confidence"]
        info   = CLASS_INFO[cls_id]

        geom = geom.buffer(-SHRINK_M)
        if geom.is_empty or not geom.is_valid:
            continue

        geom = geom.simplify(0.3, preserve_topology=True)
        if geom.is_empty:
            continue

        geom = geom.buffer(SHRINK_M * 0.7)
        if geom.is_empty or not geom.is_valid:
            continue

        if geom.geom_type == "Polygon" and list(geom.interiors):
            keep = [r for r in geom.interiors if ShapelyPolygon(r.coords).area > 1.5]
            geom = ShapelyPolygon(geom.exterior, keep)

        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]

        for part in parts:
            if part.is_empty or not part.is_valid:
                continue
            if part.area < MIN_AREA_M2 or part.area > MAX_AREA_M2:
                continue
            if get_aspect_ratio(part) > MAX_ASPECT_RATIO:
                continue
            cleaned.append({
                "geometry"  : part,
                "class_id"  : cls_id,
                "Roof_type" : info[1],
                "Class_Name": info[2],
                "confidence": round(conf, 3),
                "category"  : "Built_Up_Area",
            })

    return cleaned


def process_other(gdf_metric, category):
    cleaned = []

    for _, row in gdf_metric.iterrows():
        geom   = row["geometry"].buffer(0)
        cls_id = row["class_id"]
        conf   = row["confidence"]
        info   = CLASS_INFO[cls_id]

        geom = geom.simplify(0.3, preserve_topology=True)
        geom = geom.buffer(0.05).buffer(-0.05)

        if geom.is_empty or not geom.is_valid:
            continue

        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            if part.is_empty:
                continue
            if part.area < MIN_AREA_OTHER_M2 or part.area > MAX_AREA_M2:
                continue
            cleaned.append({
                "geometry"  : part,
                "class_id"  : cls_id,
                "Roof_type" : None,
                "Class_Name": info[2],
                "confidence": round(conf, 3),
                "category"  : category,
            })

    return cleaned


def vectorize_and_clean(raw_detections, img_crs):
    print("\n[2/3] Vectorizing and cleaning polygons ...")

    buckets = {"Built_Up_Area": [], "Utility": [], "Water": [], "Bridge": []}

    for det in raw_detections:
        info = CLASS_INFO.get(det["class_id"])
        if not info:
            continue
        category = info[0]
        if category not in buckets:
            continue

        for geom_dict, val in shapes(
            det["mask_np"].astype(np.uint8),
            transform=det["tile_transform"]
        ):
            if val == 0:
                continue
            poly = shape(geom_dict).buffer(0)
            if poly.is_valid and not poly.is_empty:
                buckets[category].append({
                    "geometry"  : poly,
                    "class_id"  : det["class_id"],
                    "confidence": det["confidence"],
                })

    all_results = {}

    for category, polys in buckets.items():
        if not polys:
            continue

        print(f"\n    [{category}] Raw: {len(polys)}")

        gdf_m = gpd.GeoDataFrame(polys, crs=img_crs).to_crs(METRIC_CRS)
        gdf_m["area"] = gdf_m.geometry.area

        min_a = MIN_AREA_M2 if category == "Built_Up_Area" else MIN_AREA_OTHER_M2
        gdf_m = gdf_m[(gdf_m["area"] >= min_a) & (gdf_m["area"] <= MAX_AREA_M2)].copy()
        print(f"    [{category}] After area filter: {len(gdf_m)}")

        if gdf_m.empty:
            continue

        snap  = 1.5 if category == "Built_Up_Area" else 0.8
        gdf_m = centroid_dedup(gdf_m, snap_m=snap)
        print(f"    [{category}] After dedup: {len(gdf_m)}")

        if category == "Built_Up_Area":
            cleaned = process_buildings(gdf_m)
        else:
            cleaned = process_other(gdf_m, category)

        print(f"    [{category}] Final: {len(cleaned)}")
        all_results[category] = cleaned

    return all_results


# ==============================================================================
# 6. EXPORT
# ==============================================================================
def export_gpkg(all_results, output_path, img_crs):
    print(f"\n[3/3] Exporting to {output_path} ...")

    if not all_results:
        print("    Nothing to export.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    for category, dets in all_results.items():
        if not dets:
            continue

        layer_name = GPKG_LAYERS[category]
        min_area   = MIN_AREA_M2 if category == "Built_Up_Area" else MIN_AREA_OTHER_M2
        rows       = []

        geom_list = [d["geometry"] for d in dets]
        gdf_m     = gpd.GeoDataFrame({"geometry": geom_list}, crs=METRIC_CRS)
        gdf_orig  = gdf_m.to_crs(img_crs)

        for i, det in enumerate(dets):
            try:
                geom_m  = gdf_m.geometry.iloc[i]
                area_m2 = round(geom_m.area, 2)
                length  = round(geom_m.length, 2)

                if area_m2 < min_area or area_m2 > MAX_AREA_M2:
                    continue

                row = {
                    "Type"      : det["class_id"],    # ← YOLO class ID
                    "Class_Name": det["Class_Name"],
                    "confidence": det["confidence"],
                    "SHAPE_Leng": length,
                    "SHAPE_Area": area_m2,
                    "Remarks"   : "AI Extracted",
                    "geometry"  : gdf_orig.geometry.iloc[i],
                }

                if category == "Built_Up_Area":
                    row["Roof_type"] = det["Roof_type"]

                rows.append(row)

            except Exception as e:
                print(f"    [WARN] {e}")
                continue

        if rows:
            out = gpd.GeoDataFrame(rows, crs=img_crs)
            out.to_file(output_path, driver="GPKG", layer=layer_name, mode="a")
            print(f"    ✅ {category}: {len(out)} features → '{layer_name}'")
            if category == "Built_Up_Area":
                for rt in [1, 2, 3, 4]:
                    count = sum(1 for r in rows if r.get("Roof_type") == rt)
                    if count:
                        print(f"       Roof_type {rt}: {count} buildings")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  AI Extraction — Buildings + Utility + Water + Bridge")
    print("=" * 60)

    raw_dets, img_crs = run_inference(INPUT_GEOTIFF, YOLO_WEIGHTS)

    if raw_dets:
        results = vectorize_and_clean(raw_dets, img_crs)
        export_gpkg(results, OUTPUT_GPKG, img_crs)
        print("\n✅ Done! Open the .gpkg in QGIS.")
    else:
        print("\n⚠️  No detections found.")