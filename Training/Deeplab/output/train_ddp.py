
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
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

torch.backends.cudnn.benchmark = True

# ===============================
# CONFIGURATION
# ===============================
FAST_DEV_RUN   = False

DATA_DIR       = "/kaggle/input/datasets/vishweshvishwakarma/geo-road-final/split/"
TRAIN_IMG_DIR  = os.path.join(DATA_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train/masks")
VAL_IMG_DIR    = os.path.join(DATA_DIR, "val/images")
VAL_MASK_DIR   = os.path.join(DATA_DIR, "val/masks")
OUTPUT_DIR     = "/kaggle/working"

NUM_CLASSES    = 6
INFERENCE_SIZE = 768
BATCH_SIZE     = 32
VAL_BATCH_SIZE = 4
EPOCHS         = 1 if FAST_DEV_RUN else 30
SAVE_EVERY     = 2

# FIX #3: Lowered from 4e-4 — high LR accelerates collapse to trivial solution
BASE_LR        = 1e-4
POLY_POWER     = 0.9

# Hard negative warmup: skip all-background tiles for this many epochs,
# then gradually reintroduce them. This prevents the model locking into
# the background attractor before it has learned any road features.
HARD_NEG_WARMUP_EPOCHS = 5

COLORS = np.array([
    [20, 20, 20], [255, 50, 50], [50, 255, 50],
    [50, 50, 255], [255, 255, 50], [255, 50, 255]
], dtype=np.uint8)

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

        gt   = cv2.cvtColor(COLORS[masks[i]], cv2.COLOR_RGB2BGR)
        pred = cv2.cvtColor(COLORS[preds[i]], cv2.COLOR_RGB2BGR)

        img  = cv2.resize(img,  (384, 384))
        gt   = cv2.resize(gt,   (384, 384), interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(pred, (384, 384), interpolation=cv2.INTER_NEAREST)

        rows.append(np.hstack([img, gt, pred]))

    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), np.vstack(rows))

# ===============================
# DATASET
# ===============================
class StreamingDataset(Dataset):
    """
    skip_hard_negatives: if True, filters out tiles whose mask is entirely
    background (class 0). Used during warmup epochs to prevent collapse.
    """
    def __init__(self, img_dir, mask_dir, transform=None, skip_hard_negatives=False):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        all_images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

        if skip_hard_negatives:
            print(f"[Dataset] Filtering hard negatives from {len(all_images)} tiles...")
            filtered = []
            for f in all_images:
                mask_path = os.path.join(mask_dir, f)
                with Image.open(mask_path) as m:
                    arr = np.asarray(m.convert("L"))
                if arr.max() > 0:   # at least one non-background pixel
                    filtered.append(f)
            print(f"[Dataset] Kept {len(filtered)} / {len(all_images)} tiles (removed {len(all_images)-len(filtered)} hard negatives)")
            self.images = filtered
        else:
            self.images = all_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.img_dir,  self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        with Image.open(img_path) as img:
            image = np.asarray(img.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as msk:
            mask = np.asarray(msk.convert("L"), dtype=np.uint8)

        if self.transform:
            data  = self.transform(image=image, mask=mask)
            image, mask = data['image'], data['mask']

        return image, mask.long()


train_transform = A.Compose([
    A.Resize(INFERENCE_SIZE, INFERENCE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    # Colour jitter helps generalise across different lighting / sensors
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
def compute_metrics(preds, targets):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == targets).float().sum()
    total   = torch.numel(targets)
    pixel_acc = correct / total

    ious = []
    for cls in range(1, NUM_CLASSES):
        pred_c = preds == cls
        tgt_c  = targets == cls
        inter  = (pred_c & tgt_c).sum().float()
        union  = (pred_c | tgt_c).sum().float()
        if union > 0:
            ious.append(inter / union)

    miou = torch.mean(torch.stack(ious)) if ious else torch.tensor(0.0)
    return pixel_acc, miou

# ===============================
# LOSS FUNCTIONS
# ===============================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce    = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        p_t   = torch.exp(-ce)
        focal = (1 - p_t) ** self.gamma * ce
        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        total, count = 0.0, 0
        for cls in range(1, self.num_classes):
            p     = probs[:, cls]
            t     = (targets == cls).float()
            inter = (p * t).sum()
            total += 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
            count += 1

        if count > 0:
            return total / count
        else:
            # FIX #1 (critical): keep gradient alive on all-background batches.
            # Previously returned a detached tensor → silent dead gradient on
            # every hard-negative batch → model got zero corrective signal.
            return logits.sum() * 0.0

# ===============================
# DDP SETUP
# ===============================
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# ===============================
# CLASS DISTRIBUTION DIAGNOSTIC
# ===============================
def print_class_distribution(loader, num_classes, max_batches=20, rank=0):
    """
    Run once before training starts to confirm imbalance ratio.
    If background > 90 % → you need the alpha fix. (You do.)
    """
    if rank != 0:
        return
    counts = torch.zeros(num_classes, dtype=torch.long)
    for i, (_, masks) in enumerate(loader):
        for cls in range(num_classes):
            counts[cls] += (masks == cls).sum()
        if i >= max_batches:
            break
    total = counts.sum().float()
    print("\n[CLASS DISTRIBUTION (first 20 batches)]")
    for cls in range(num_classes):
        pct = 100.0 * counts[cls] / total
        bar = "█" * int(pct / 2)
        print(f"  Class {cls}: {pct:5.1f}%  {bar}")
    print()

# ===============================
# TRAIN
# ===============================
def train_ddp(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # -------------------------------------------------------
    # FIX #2: Alpha weights
    # Background crushed to 0.05 — model is no longer rewarded
    # for lazy all-background predictions.
    # Road classes spiked to 1.5 — missing a road is expensive.
    # -------------------------------------------------------
    _alpha = torch.tensor([0.05, 1.5, 1.5, 1.5, 1.5, 1.5], device=device)

    criterion_ce    = nn.CrossEntropyLoss(weight=_alpha)
    criterion_dice  = DiceLoss(num_classes=NUM_CLASSES)
    criterion_focal = FocalLoss(gamma=2.0, alpha=_alpha)

    # Build initial dataset (hard negatives skipped during warmup)
    skip_hn = True   # will be updated each epoch
    train_dataset = StreamingDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform,
                                     skip_hard_negatives=skip_hn)
    val_dataset   = StreamingDataset(VAL_IMG_DIR,   VAL_MASK_DIR,   val_transform,
                                     skip_hard_negatives=False)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE,
                            sampler=val_sampler, num_workers=2, pin_memory=True)

    # Print class distribution once so you can confirm the imbalance
    if rank == 0:
        tmp_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        print_class_distribution(tmp_loader, NUM_CLASSES, rank=rank)

    model = smp.DeepLabV3Plus(encoder_name="resnet50", classes=NUM_CLASSES).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler('cuda')

    best_miou   = -1.0
    total_iters = EPOCHS * 1000   # approximate; updated per epoch
    global_iter = 0
    total_start = time.time()

    if rank == 0 and FAST_DEV_RUN:
        print("\n" + "⚠️  " * 15)
        print("  FAST DEV RUN IS ENABLED — TRAINING WILL TRUNCATE.")
        print("⚠️  " * 15 + "\n")

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # -------------------------------------------------------
        # Hard-negative curriculum: skip all-background tiles for
        # the first HARD_NEG_WARMUP_EPOCHS, then include them.
        # This prevents the model converging to the background
        # attractor before it has ever seen a road prediction rewarded.
        # -------------------------------------------------------
        skip_hn = (epoch < HARD_NEG_WARMUP_EPOCHS)
        train_dataset = StreamingDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform,
                                         skip_hard_negatives=skip_hn)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE // world_size,
                                  sampler=train_sampler, num_workers=2,
                                  pin_memory=True, prefetch_factor=2)

        total_iters = EPOCHS * len(train_loader)
        train_loss_total = 0.0
        batch_count      = 0

        model.train()
        train_sampler.set_epoch(epoch)

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            # Poly LR schedule
            lr = BASE_LR * (1 - global_iter / max(1, total_iters)) ** POLY_POWER
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                logits = model(images)
                # FIX #4: Triple loss
                # CE   (0.5) — hard pixel supervision, blunt forcing function out of collapse
                # Dice (0.3) — encourages contiguous road-shaped predictions
                # Focal(0.2) — cleans up hard edges (thin roads, shadow boundaries)
                loss = (0.5 * criterion_ce(logits, masks)
                      + 0.3 * criterion_dice(logits, masks)
                      + 0.2 * criterion_focal(logits, masks))

            scaler.scale(loss).backward()
            # Gradient clip: prevents early-epoch explosions that can cause collapse
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()
            batch_count      += 1
            global_iter      += 1

            if rank == 0 and batch_idx % max(1, len(train_loader) // 10) == 0:
                # Diagnostic: which classes are actually being predicted?
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    unique_preds = preds.unique().tolist()
                print(f"  Batch {batch_idx:4d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | LR: {lr:.6f} | "
                      f"Pred classes: {unique_preds}")

            if FAST_DEV_RUN and batch_idx >= 10:
                break

        train_loss_avg = train_loss_total / max(1, batch_count)

        # ================= VALIDATION =================
        model.eval()
        val_loss_total = 0.0
        miou_total     = 0.0
        acc_total      = 0.0
        count          = 0
        vis_saved      = False
        all_unique_preds = set()

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    val_loss = (0.5 * criterion_ce(logits, masks)
                              + 0.3 * criterion_dice(logits, masks)
                              + 0.2 * criterion_focal(logits, masks))

                acc, miou = compute_metrics(logits, masks)
                preds = torch.argmax(logits, dim=1)
                all_unique_preds.update(preds.unique().tolist())

                val_loss_total += val_loss.item()
                miou_total     += miou.item()
                acc_total      += acc.item()
                count          += 1

                if rank == 0 and not vis_saved:
                    save_debug_grid(images, masks, preds,
                                    f"val_debug_epoch_{epoch+1}.jpg")
                    vis_saved = True

                if FAST_DEV_RUN and batch_idx >= 5:
                    break

        metrics_tensor = torch.tensor(
            [val_loss_total, miou_total, acc_total, count], device=device
        )
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        val_loss_avg = metrics_tensor[0].item() / max(1, metrics_tensor[3].item())
        miou_avg     = metrics_tensor[1].item() / max(1, metrics_tensor[3].item())
        acc_avg      = metrics_tensor[2].item() / max(1, metrics_tensor[3].item())

        if rank == 0:
            elapsed = format_time(time.time() - total_start)
            print("\n" + "=" * 65)
            print(f"Epoch {epoch+1}/{EPOCHS}  |  Hard-neg skipped: {skip_hn}  |  Elapsed: {elapsed}")
            print(f"Train Loss : {train_loss_avg:.4f}")
            print(f"Val Loss   : {val_loss_avg:.4f}  |  mIoU: {miou_avg:.4f}  |  Acc: {acc_avg:.4f}")

            # ⚠️  Collapse detector — if you only see {0} here, collapse is ongoing
            print(f"Val predicted classes : {sorted(all_unique_preds)}")
            if all_unique_preds == {0}:
                print("  🔴 WARNING: Model is predicting ONLY background. Collapse detected.")
            elif len(all_unique_preds) < 3:
                print(f"  🟡 WARNING: Only {len(all_unique_preds)} class(es) predicted — partial collapse.")
            else:
                print(f"  🟢 Model is predicting {len(all_unique_preds)} classes — looks healthy.")

            print("-" * 65)

            if miou_avg >= best_miou:
                best_miou = miou_avg
                torch.save(model.module.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
                print(f"  🟢 SAVED best_model.pt  (mIoU: {best_miou:.4f})")

            if (epoch + 1) % SAVE_EVERY == 0:
                torch.save(model.module.state_dict(), f"{OUTPUT_DIR}/epoch_{epoch+1}.pt")
                print(f"  💾 SAVED epoch_{epoch+1}.pt")

            torch.save(model.module.state_dict(), f"{OUTPUT_DIR}/last_model.pt")
            print(f"  🔄 SAVED last_model.pt")
            print("=" * 65 + "\n")

    cleanup()

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)
