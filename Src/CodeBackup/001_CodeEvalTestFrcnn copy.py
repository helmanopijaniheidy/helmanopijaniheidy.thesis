import torch
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Gunakan torchmetrics untuk kalkulasi mAP yang akurat
# pip install torchmetrics pycocotools
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# =========================================================
# CONFIGURATION (Menggunakan Pathlib)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent # Menuju root HELMANOPIJANIHEIDY.THESIS
DATA_DIR = BASE_DIR / "Data"

# Folder Model & Weights
MODEL_WEIGHTS = DATA_DIR / "DataModels" / "runs_frcnn" / "RBC_WBC_VOC_DATASET" / "weights" / "best.pth"

# Folder Dataset Test
DATASET_ROOT = DATA_DIR / "Datasets" / "RBC_WBC_VOC_DATASET"
IMAGE_TEST_DIR = DATASET_ROOT / "JPEGImages" / "Test"
ANNOT_TEST_DIR = DATASET_ROOT / "Annotations" / "Test"

# Folder Output
OUTPUT_DIR = DATA_DIR / "DataTesting" / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Sesuaikan dengan labelmap.txt kamu
# PENTING: Index 0 HARUS background untuk Faster R-CNN torchvision
LABEL_MAP = {
    "background": 0,
    "RBC": 1, 
    "WBC": 2
}

# =========================================================
# UTILS
# =========================================================

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    print(f"Loading weights from: {MODEL_WEIGHTS}")
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    return model.to(DEVICE).eval()

def parse_voc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in LABEL_MAP:
            continue
            
        labels.append(LABEL_MAP[name])
        bndbox = obj.find('bndbox')
        boxes.append([
            float(bndbox.find('xmin').text),
            float(bndbox.find('ymin').text),
            float(bndbox.find('xmax').text),
            float(bndbox.find('ymax').text)
        ])
    
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64)
    }

# =========================================================
# EVALUATION LOOP
# =========================================================

def run_evaluation():
    model = get_model(len(LABEL_MAP))
    metric = MeanAveragePrecision()
    
    # Ambil semua file gambar di folder Test
    test_images = list(IMAGE_TEST_DIR.glob("*.jpg")) + list(IMAGE_TEST_DIR.glob("*.png"))
    
    print(f"Found {len(test_images)} images for testing.")
    
    preds = []
    targets = []

    with torch.no_grad():
        for img_path in tqdm(test_images):
            # 1. Load Image
            img_raw = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(img_raw).unsqueeze(0).to(DEVICE)
            
            # 2. Predict
            output = model(img_tensor)[0]
            
            # 3. Load GT (Cari file XML dengan nama yang sama)
            xml_path = ANNOT_TEST_DIR / f"{img_path.stem}.xml"
            if not xml_path.exists():
                continue
                
            target = parse_voc_xml(xml_path)
            
            # 4. Format untuk Evaluator (pindah ke CPU)
            preds.append({k: v.cpu() for k, v in output.items()})
            targets.append({k: v.cpu() for k, v in target.items()})

    # Hitung mAP
    print("Calculating mAP metrics...")
    metric.update(preds, targets)
    results = metric.compute()

    # Save hasil ke JSON
    # Konversi Tensor ke format Python agar bisa di-serialize
    final_results = {}
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            final_results[k] = v.tolist() if v.numel() > 1 else v.item()

    output_path = OUTPUT_DIR / "test_evaluation_report.json"
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print("-" * 30)
    print(f"mAP @50: {final_results['map_50']:.4f}")
    print(f"mAP @75: {final_results['map_75']:.4f}")
    print(f"mAP @[50:95]: {final_results['map']:.4f}")
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    run_evaluation()