
import os
import time
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ===============================
# BASIC SETUP
# ===============================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MASTER_ADDR"]     = "127.0.0.1"
os.environ["MASTER_PORT"]     = "29501"   # different port from multiclass training

torch.backends.cudnn.benchmark = True

# ===============================
# CONFIGURATION
# ===============================
FAST_DEV_RUN   = False

# Same dataset as multiclass — masks get converted to binary on the fly
DATA_DIR       = "/kaggle/input/datasets/vishweshvishwakarma/geo-road-final/split/"
TRAIN_IMG_DIR  = os.path.join(DATA_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train/masks")
VAL_IMG_DIR    = os.path.join(DATA_DIR, "val/images")
VAL_MASK_DIR   = os.path.join(DATA_DIR, "val/masks")
OUTPUT_DIR     = "/kaggle/working"

# Binary: 0 = background, 1 = road (any road type)
NUM_CLASSES    = 2
INFERENCE_SIZE = 768
BATCH_SIZE     = 32
VAL_BATCH_SIZE = 4
EPOCHS         = 1 if FAST_DEV_RUN else 20
SAVE_EVERY     = 2
BASE_LR        = 1e-4
POLY_POWER     = 0.9

# How many times more costly is missing a road vs false positive
BCE_POS_WEIGHT = 8.0

# ===============================
# UTILS
# ===============================
def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"

def save_debug_grid(imgs_tensor, masks_tensor, preds_tensor, filename):
    imgs  = imgs_tensor.cpu().numpy()
    masks = masks_tensor.cpu().numpy()
    preds = preds_tensor.cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    rows = []
    for i in range(min(3, len(imgs))):
        img = (np.transpose(imgs[i], (1, 2, 0)) * std + mean).clip(0, 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        gt   = (masks[i] * 255).astype(np.uint8)
        pred = (preds[i] * 255).astype(np.uint8)
        gt   = cv2.cvtColor(gt,   cv2.COLOR_GRAY2BGR)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

        img  = cv2.resize(img,  (384, 384))
        gt   = cv2.resize(gt,   (384, 384), interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(pred, (384, 384), interpolation=cv2.INTER_NEAREST)

        rows.append(np.hstack([img, gt, pred]))

    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), np.vstack(rows))


# ===============================
# DATASET
# ===============================
class BinaryRoadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.images   = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name      = self.images[idx]
        img_path  = os.path.join(self.img_dir,  name)
        mask_path = os.path.join(self.mask_dir, name)

        with Image.open(img_path) as img:
            image = np.asarray(img.convert("RGB"), dtype=np.uint8)

        with Image.open(mask_path) as msk:
            mask = np.asarray(msk.convert("L"), dtype=np.uint8)

        mask = (mask > 0).astype(np.uint8)

        if self.transform:
            data  = self.transform(image=image, mask=mask)
            image, mask = data['image'], data['mask']

        return image, mask.long()


train_transform = A.Compose([
    A.Resize(INFERENCE_SIZE, INFERENCE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(INFERENCE_SIZE, INFERENCE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ===============================
# METRICS
# ===============================
def compute_metrics(logits, targets):
    preds = torch.argmax(logits, dim=1)
    pixel_acc = (preds == targets).float().mean()

    pred_road = preds == 1
    tgt_road  = targets == 1
    inter     = (pred_road & tgt_road).sum().float()
    union     = (pred_road | tgt_road).sum().float()
    iou       = inter / (union + 1e-8)

    tp        = inter
    fp        = (pred_road & ~tgt_road).sum().float()
    fn        = (~pred_road & tgt_road).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return pixel_acc, iou, precision, recall, f1


# ===============================
# LOSS
# ===============================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]
        tgt   = (targets == 1).float()
        inter = (probs * tgt).sum()
        total = probs.sum() + tgt.sum()

        if total < self.smooth:
            return logits.sum() * 0.0

        return 1 - (2 * inter + self.smooth) / (total + self.smooth)


# ===============================
# DDP SETUP
# ===============================
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


# ===============================
# TRAIN
# ===============================
def train_ddp(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    train_dataset = BinaryRoadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform)
    val_dataset   = BinaryRoadDataset(VAL_IMG_DIR,   VAL_MASK_DIR,   val_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE // world_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True
    )

    model = smp.Linknet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    ).to(device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler('cuda')

    pos_weight     = torch.tensor([BCE_POS_WEIGHT], device=device)
    criterion_bce  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_dice = DiceLoss()

    best_iou    = -1.0
    total_iters = EPOCHS * len(train_loader)
    global_iter = 0
    total_start = time.time()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f" LinkNet Binary Road Segmentation")
        print(f" Train: {len(train_dataset)} tiles | Val: {len(val_dataset)} tiles")
        print(f" Masks converted on-the-fly: multiclass → binary")
        print(f"{'='*60}\n")

        if FAST_DEV_RUN:
            print("⚠️  FAST DEV RUN — training will truncate\n")

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_loss_total = 0.0
        batch_count      = 0

        train_sampler.set_epoch(epoch)

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            if global_iter < 2 * len(train_loader):
                lr = BASE_LR * (global_iter / max(1, 2 * len(train_loader)))
            else:
                lr = BASE_LR * (1 - (global_iter - 2 * len(train_loader)) / 
                                max(1, total_iters)) ** POLY_POWER
                                
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                logits = model(images) 
                road_logit = logits[:, 1]
                road_target = (masks == 1).float()

                loss_bce  = criterion_bce(road_logit, road_target)
                loss_dice = criterion_dice(logits, masks)
                loss      = 0.5 * loss_bce + 0.5 * loss_dice

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()
            batch_count      += 1
            global_iter      += 1

            if rank == 0 and batch_idx % max(1, len(train_loader) // 10) == 0:
                with torch.no_grad():
                    preds        = torch.argmax(logits, dim=1)
                    unique_preds = preds.unique().tolist()
                    road_pct     = (preds == 1).float().mean().item() * 100

                print(f"  Batch {batch_idx:4d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} "
                      f"(BCE:{loss_bce.item():.4f} Dice:{loss_dice.item():.4f}) | "
                      f"LR: {lr:.6f} | "
                      f"Pred: {unique_preds} | Road%: {road_pct:.1f}%")

            if FAST_DEV_RUN and batch_idx >= 10:
                break

        train_loss_avg = train_loss_total / max(1, batch_count)

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_metrics = torch.zeros(7, device=device)  
        vis_saved   = False

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device,  non_blocking=True)

                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    road_logit  = logits[:, 1]
                    road_target = (masks == 1).float()
                    loss_bce    = criterion_bce(road_logit, road_target)
                    loss_dice   = criterion_dice(logits, masks)
                    val_loss    = 0.5 * loss_bce + 0.5 * loss_dice

                acc, iou, prec, rec, f1 = compute_metrics(logits, masks)

                val_metrics[0] += val_loss.item()
                val_metrics[1] += iou.item()
                val_metrics[2] += acc.item()
                val_metrics[3] += prec.item()
                val_metrics[4] += rec.item()
                val_metrics[5] += f1.item()
                val_metrics[6] += 1.0

                if rank == 0 and not vis_saved:
                    preds = torch.argmax(logits, dim=1)
                    save_debug_grid(
                        images, masks, preds,
                        f"val_binary_epoch_{epoch+1}.jpg"
                    )
                    vis_saved = True
                    
                if rank == 0 and batch_idx == 0:
                    preds       = torch.argmax(logits, dim=1)
                    pred_road   = (preds == 1).sum().item()
                    actual_road = (masks == 1).sum().item()
                    print(f"  [Val Batch 0] Predicted road px: {pred_road:,} | "
                          f"Actual road px: {actual_road:,} | "
                          f"Ratio: {pred_road/max(1,actual_road):.2f}")

                if FAST_DEV_RUN and batch_idx >= 5:
                    break

        dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        n = max(1, val_metrics[6].item())

        val_loss = val_metrics[0].item() / n
        val_iou  = val_metrics[1].item() / n
        val_acc  = val_metrics[2].item() / n
        val_prec = val_metrics[3].item() / n
        val_rec  = val_metrics[4].item() / n
        val_f1   = val_metrics[5].item() / n

        if rank == 0:
            elapsed = format_time(time.time() - total_start)
            print(f"\n{'='*65}")
            print(f"Epoch {epoch+1}/{EPOCHS}  |  Elapsed: {elapsed}")
            print(f"Train Loss : {train_loss_avg:.4f}")
            print(f"Val Loss   : {val_loss:.4f}  |  IoU: {val_iou:.4f}  |  Acc: {val_acc:.4f}")
            print(f"Precision  : {val_prec:.4f}  |  Recall: {val_rec:.4f}  |  F1: {val_f1:.4f}")

            if val_rec < 0.5:
                print(f"  🔴 Recall {val_rec:.3f} — model missing too many roads. "
                      f"Try increasing BCE_POS_WEIGHT.")
            elif val_rec < 0.75:
                print(f"  🟡 Recall {val_rec:.3f} — acceptable but can improve.")
            else:
                print(f"  🟢 Recall {val_rec:.3f} — good, binary model finding roads.")

            print(f"{'-'*65}")

            if val_iou >= best_iou:
                best_iou = val_iou
                torch.save(
                    model.module.state_dict(),
                    f"{OUTPUT_DIR}/binary_best.pt"
                )
                print(f"  🟢 SAVED binary_best.pt  (IoU: {best_iou:.4f})")

            if (epoch + 1) % SAVE_EVERY == 0:
                torch.save(
                    model.module.state_dict(),
                    f"{OUTPUT_DIR}/binary_epoch_{epoch+1}.pt"
                )
                print(f"  💾 SAVED binary_epoch_{epoch+1}.pt")

            torch.save(
                model.module.state_dict(),
                f"{OUTPUT_DIR}/binary_last.pt"
            )
            print(f"  🔄 SAVED binary_last.pt")
            print(f"{'='*65}\n")

    cleanup()

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)
