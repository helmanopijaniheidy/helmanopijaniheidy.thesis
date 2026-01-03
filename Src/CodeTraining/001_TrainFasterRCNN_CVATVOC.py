#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import math
import time
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

@dataclass
class CFG:
    epochs: int = 60
    batch_size: int = 4                # 4060 Ti 16GB: mulai 4 (aman). Naikkan ke 6/8 kalau muat.
    lr: float = 0.005
    weight_decay: float = 5e-4
    momentum: float = 0.9
    patience: int = 20

    # loader performance (i7-11700 + RAM 24GB)
    num_workers: int = 6               # coba 6-8 (jangan 16 biar tidak thrash RAM)
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # mixed precision
    amp: bool = True

    # optional resize (biarkan None untuk pakai ukuran asli)
    resize_to: int | None = None       # contoh: 640 jika kamu ingin resize persegi

    # logging
    print_skip_examples: int = 10      # tampilkan contoh alasan skip (max N)

CFG_ = CFG()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "Data" / "Datasets" / "RBC_WBC_VOC_DATASET"
RUNS_ROOT = PROJECT_ROOT / "Data" / "DataModels" / "runs_frcnn"


# =========================================================
# RUN DIR
# =========================================================
def get_run_dir(base: Path, name: str) -> Path:
    run = base / name
    (run / "weights").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(parents=True, exist_ok=True)
    return run


# =========================================================
# UTIL: safe parse VOC
# =========================================================
def safe_parse_voc_xml(xml_path: Path):
    """
    Return tuple: (filename_in_xml, boxes, labels, (w,h))
    Raise ValueError for "aneh" cases.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"XML parse error: {e}")

    filename_node = root.find("filename")
    filename = filename_node.text.strip() if filename_node is not None and filename_node.text else ""

    size = root.find("size")
    if size is None:
        # masih bisa jalan tanpa size, tapi kita keep as unknown
        img_w = img_h = None
    else:
        def _get_int(tag):
            n = size.find(tag)
            return int(float(n.text)) if (n is not None and n.text) else None
        img_w = _get_int("width")
        img_h = _get_int("height")

    boxes = []
    labels = []

    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None or not name_node.text:
            continue
        cls = name_node.text.strip()

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        def _get_float(tag):
            n = bnd.find(tag)
            if n is None or not n.text:
                return None
            return float(n.text)

        xmin = _get_float("xmin")
        ymin = _get_float("ymin")
        xmax = _get_float("xmax")
        ymax = _get_float("ymax")

        if None in (xmin, ymin, xmax, ymax):
            continue

        # basic validity
        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(cls)

    if len(boxes) == 0:
        raise ValueError("no valid boxes")

    return filename, boxes, labels, (img_w, img_h)


# =========================================================
# DATASET (CVAT VOC 1.1 style with split folders)
# =========================================================
class CVATVOCDataset(Dataset):
    """
    Expected structure:
      root/
        Annotations/Train/*.xml
        Annotations/Val/*.xml
        JPEGImages/Train/*.jpg
        JPEGImages/Val/*.jpg
        ImageSets/Main/Train.txt
        ImageSets/Main/Validation.txt
    """

    def __init__(self, root: Path, split: str, class_map: dict[str, int], resize_to: int | None = None):
        self.root = root
        self.split = split  # "Train" or "Val" etc.
        self.class_map = class_map
        self.resize_to = resize_to

        self.img_dir = root / "JPEGImages" / split
        self.ann_dir = root / "Annotations" / split

        # split file mapping
        if split.lower() == "train":
            split_file = root / "ImageSets" / "Main" / "Train.txt"
        elif split.lower() in ("val", "valid", "validation"):
            split_file = root / "ImageSets" / "Main" / "Validation.txt"
        elif split.lower() == "test":
            split_file = root / "ImageSets" / "Main" / "Test.txt"
        else:
            raise ValueError(f"Unknown split: {split}")

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        # IDs in split files can include path like "Train/BAS_0361" or just "BAS_0361"
        self.ids = [x.strip() for x in split_file.read_text().splitlines() if x.strip()]

        # stats
        self.skipped = 0
        self.skip_reasons: dict[str, int] = {}
        self._printed_examples = 0

    def __len__(self):
        return len(self.ids)

    def _record_skip(self, reason: str, example: str | None = None):
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
        if example and self._printed_examples < CFG_.print_skip_examples:
            print(f"[SKIP] {reason}: {example}")
            self._printed_examples += 1

    def __getitem__(self, idx):
        raw_id = self.ids[idx]

        # normalize id to base name
        # handle: "Train/BAS_0361" or "Test/BAS_0361" or "BAS_0361"
        base_id = Path(raw_id).name

        xml_path = self.ann_dir / f"{base_id}.xml"
        img_path = self.img_dir / f"{base_id}.jpg"

        if not xml_path.exists():
            self._record_skip("missing_xml", str(xml_path))
            return None
        if not img_path.exists():
            self._record_skip("missing_image", str(img_path))
            return None

        # parse XML safely
        try:
            _, boxes, labels_str, _ = safe_parse_voc_xml(xml_path)
        except ValueError as e:
            self._record_skip("bad_xml", f"{xml_path.name} ({e})")
            return None

        # map labels
        labels = []
        for ls in labels_str:
            if ls not in self.class_map:
                # unknown label -> skip object (not whole image)
                continue
            labels.append(self.class_map[ls])

        if len(labels) == 0:
            self._record_skip("no_known_labels", xml_path.name)
            return None

        # load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            self._record_skip("image_open_error", f"{img_path.name} ({e})")
            return None

        orig_w, orig_h = img.size

        # optional resize (square)
        if self.resize_to is not None:
            new_size = (self.resize_to, self.resize_to)
            img = img.resize(new_size)

            # scale boxes accordingly
            sx = self.resize_to / float(orig_w)
            sy = self.resize_to / float(orig_h)
            scaled_boxes = []
            for (xmin, ymin, xmax, ymax) in boxes:
                scaled_boxes.append([xmin * sx, ymin * sy, xmax * sx, ymax * sy])
            boxes = scaled_boxes

        # clamp boxes to image bounds (extra safety)
        w, h = (self.resize_to, self.resize_to) if self.resize_to else (orig_w, orig_h)
        fixed_boxes = []
        fixed_labels = []
        for b, lab in zip(boxes, labels):
            xmin, ymin, xmax, ymax = b
            xmin = max(0.0, min(float(xmin), float(w - 1)))
            ymin = max(0.0, min(float(ymin), float(h - 1)))
            xmax = max(0.0, min(float(xmax), float(w - 1)))
            ymax = max(0.0, min(float(ymax), float(h - 1)))
            if xmax <= xmin or ymax <= ymin:
                continue
            fixed_boxes.append([xmin, ymin, xmax, ymax])
            fixed_labels.append(lab)

        if len(fixed_boxes) == 0:
            self._record_skip("boxes_invalid_after_clamp", xml_path.name)
            return None

        img_tensor = F.to_tensor(img)

        target = {
            "boxes": torch.tensor(fixed_boxes, dtype=torch.float32),
            "labels": torch.tensor(fixed_labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros((len(fixed_boxes),), dtype=torch.int64),
            "area": (torch.tensor([b[2] - b[0] for b in fixed_boxes]) *
                     torch.tensor([b[3] - b[1] for b in fixed_boxes])).to(torch.float32),
        }

        return img_tensor, target


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch)) if batch else None


# =========================================================
# MODEL
# =========================================================
def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_feat, num_classes
    )
    return model


# =========================================================
# EVAL LOSS (torchvision detection stays in train mode)
# =========================================================
@torch.no_grad()
def compute_epoch_loss(model, loader, scaler_enabled: bool):
    model.train()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="Val", leave=False):
        if batch is None:
            continue
        imgs, targets = batch
        imgs = [i.to(DEVICE, non_blocking=True) for i in imgs]
        targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]

        with autocast(enabled=scaler_enabled):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

        total += float(loss.item())
        n += 1
    return total / max(n, 1)


# =========================================================
# TRAIN
# =========================================================
def train():
    # class map fixed (background=0)
    class_map = {"RBC": 1, "WBC": 2}
    num_classes = 1 + len(class_map)

    # datasets
    train_ds = CVATVOCDataset(DATASET_ROOT, "Train", class_map, resize_to=CFG_.resize_to)
    val_ds = CVATVOCDataset(DATASET_ROOT, "Val", class_map, resize_to=CFG_.resize_to)

    # loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG_.batch_size,
        shuffle=True,
        num_workers=CFG_.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=CFG_.persistent_workers if CFG_.num_workers > 0 else False,
        prefetch_factor=CFG_.prefetch_factor if CFG_.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG_.batch_size,
        shuffle=False,
        num_workers=CFG_.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=CFG_.persistent_workers if CFG_.num_workers > 0 else False,
        prefetch_factor=CFG_.prefetch_factor if CFG_.num_workers > 0 else None,
    )

    # model
    model = get_model(num_classes).to(DEVICE)

    # optimizer/scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CFG_.lr,
        momentum=CFG_.momentum,
        weight_decay=CFG_.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG_.epochs)

    # AMP scaler
    scaler = GradScaler(enabled=(CFG_.amp and DEVICE.type == "cuda"))

    # run dirs
    run_dir = get_run_dir(RUNS_ROOT, DATASET_ROOT.name)
    weights_dir = run_dir / "weights"
    logs_dir = run_dir / "logs"
    last_ckpt = weights_dir / "last.pth"
    best_ckpt = weights_dir / "best.pth"
    history_path = logs_dir / "history.jsonl"

    # resume
    start_epoch = 0
    best_val_loss = math.inf
    patience = 0

    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        patience = int(ckpt.get("patience", patience))
        print(f"🔄 Resume training from epoch {start_epoch}/{CFG_.epochs}")

    # save config
    (logs_dir / "config.json").write_text(json.dumps(CFG_.__dict__, indent=2))

    # torch performance knobs
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True  # improves conv perf on fixed sizes (OK even without resize)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # loop
    for epoch in range(start_epoch, CFG_.epochs):
        model.train()
        epoch_loss = 0.0
        step = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{CFG_.epochs}")
        for batch in pbar:
            if batch is None:
                continue

            imgs, targets = batch
            imgs = [i.to(DEVICE, non_blocking=True) for i in imgs]
            targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(CFG_.amp and DEVICE.type == "cuda")):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            epoch_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / max(step, 1)

        # val
        val_loss = compute_epoch_loss(model, val_loader, scaler_enabled=(CFG_.amp and DEVICE.type == "cuda"))

        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch+1}/{CFG_.epochs} | lr={lr_now:.6f}")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val   Loss : {val_loss:.4f}")

        # append history
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr_now,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        # save last checkpoint (resume-ready)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "patience": patience,
            },
            last_ckpt,
        )

        # best + early stop logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), best_ckpt)
            print("✅ Saved: best.pth")
        else:
            patience += 1
            print(f"⏳ Patience: {patience}/{CFG_.patience}")

        if patience >= CFG_.patience:
            print("\n⛔ Early stopping triggered")
            break

    # summary
    print("\n✅ Training selesai")
    print(f"Train skipped total : {train_ds.skipped}")
    print(f"Val skipped total   : {val_ds.skipped}")
    print(f"Output dir          : {run_dir}")

    # save skip reasons for debugging
    (logs_dir / "skip_reasons_train.json").write_text(json.dumps(train_ds.skip_reasons, indent=2))
    (logs_dir / "skip_reasons_val.json").write_text(json.dumps(val_ds.skip_reasons, indent=2))


if __name__ == "__main__":
    # quick checks
    if not DATASET_ROOT.exists():
        print(f"[ERROR] DATASET_ROOT not found: {DATASET_ROOT}")
        sys.exit(1)

    print(f"[INFO] DEVICE       : {DEVICE}")
    print(f"[INFO] DATASET_ROOT : {DATASET_ROOT}")
    print(f"[INFO] RUNS_ROOT    : {RUNS_ROOT}")
    train()
