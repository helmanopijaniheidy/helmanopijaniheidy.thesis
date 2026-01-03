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
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "Data"

MODEL_WEIGHTS = DATA_DIR / "DataModels" / "runs_frcnn" / "RBC_WBC_VOC_DATASET" / "weights" / "best.pth"
IMAGE_TEST_DIR = DATA_DIR / "Datasets" / "RBC_WBC_VOC_DATASET" / "JPEGImages" / "Test"
ANNOT_TEST_DIR = DATA_DIR / "Datasets" / "RBC_WBC_VOC_DATASET" / "Annotations" / "Test"
OUTPUT_DIR = DATA_DIR / "DataTesting" / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Pastikan LABEL_MAP sesuai dengan urutan saat training
LABEL_MAP = {
    "background": 0,
    "RBC": 1, 
    "WBC": 2
}
# Map balik untuk keperluan print (id -> name)
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items() if v != 0}

# =========================================================
# UTILS
# =========================================================

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    return model.to(DEVICE).eval()

def parse_voc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name in LABEL_MAP:
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
# MAIN EVALUATION
# =========================================================

def run_evaluation():
    model = get_model(len(LABEL_MAP))
    # Aktifkan class_metrics=True untuk mendapatkan mAP per kelas
    metric = MeanAveragePrecision(class_metrics=True)
    
    test_images = list(IMAGE_TEST_DIR.glob("*.jpg")) + list(IMAGE_TEST_DIR.glob("*.png"))
    preds, targets = [], []

    with torch.no_grad():
        for img_path in tqdm(test_images, desc="Predicting"):
            img_raw = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(img_raw).unsqueeze(0).to(DEVICE)
            output = model(img_tensor)[0]
            
            xml_path = ANNOT_TEST_DIR / f"{img_path.stem}.xml"
            if xml_path.exists():
                target = parse_voc_xml(xml_path)
                preds.append({k: v.cpu() for k, v in output.items()})
                targets.append({k: v.cpu() for k, v in target.items()})

    print("\nCalculating metrics...")
    metric.update(preds, targets)
    results = metric.compute()

    # --- PERHITUNGAN TAMBAHAN (F1 & PER CLASS) ---
    mAP50 = results['map_50'].item()
    mAR100 = results['mar_100'].item()
    
    # Rumus F1-Score
    f1_score = 2 * (mAP50 * mAR100) / (mAP50 + mAR100) if (mAP50 + mAR100) > 0 else 0

    # Mengorganisir hasil untuk disimpan
    final_report = {
        "overall": {
            "mAP_50": mAP50,
            "mAP_75": results['map_75'].item(),
            "mAP_50_95": results['map'].item(),
            "mAR_100 (Recall)": mAR100,
            "F1_Score": f1_score
        },
        "per_class": {}
    }

    # Ambil mAP per kelas (mAP @50:95 per class)
    # results['map_per_class'] dan results['classes'] berisi tensor data
    for i, class_id in enumerate(results['classes'].tolist()):
        class_name = ID_TO_NAME.get(class_id, f"Class_{class_id}")
        final_report["per_class"][class_name] = {
            "mAP": results['map_per_class'][i].item(),
            "mar_100": results['mar_100_per_class'][i].item()
        }

    # Simpan ke JSON
    output_path = OUTPUT_DIR / "detailed_test_report.json"
    with open(output_path, "w") as f:
        json.dump(final_report, f, indent=4)

    # --- PRINT SUMMARY UNTUK TESIS ---
    print("\n" + "="*30)
    print("HASIL EVALUASI UNTUK TESIS")
    print("="*30)
    print(f"Overall mAP @50  : {mAP50:.4f}")
    print(f"Overall Recall   : {mAR100:.4f}")
    print(f"Overall F1-Score : {f1_score:.4f}")
    print("-" * 30)
    for cls, val in final_report["per_class"].items():
        print(f"Class {cls:10}: mAP={val['mAP']:.4f}, Recall={val['mar_100']:.4f}")
    print("="*30)
    print(f"Laporan lengkap disimpan di: {output_path}")

if __name__ == "__main__":
    run_evaluation()