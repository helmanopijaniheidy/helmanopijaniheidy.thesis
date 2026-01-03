import io
import os
import json
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torchvision
from torchvision.transforms import functional as F

import cv2


# =========================
# Konfigurasi (sesuaikan)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path checkpoint hasil training kamu
# contoh: DataModels/runs_frcnn/xxx/best_model.pth
DEFAULT_CKPT = "DataModels/runs_frcnn/RBC_WBC_VOC_DATASET/best_model.pth"

# Label map (sesuaikan dengan dataset kamu)
# Kalau kamu punya labelmap.txt, kamu bisa baca otomatis.
DEFAULT_LABELS = {
    1: "RBC",
    2: "WBC",
}
NUM_CLASSES = 1 + len(DEFAULT_LABELS)  # background + kelas


# =========================
# Utils
# =========================
def load_labelmap_from_txt(path_txt: str):
    """
    Format labelmap.txt bebas, tapi idealnya satu label per baris:
    RBC
    WBC
    """
    labels = {}
    if not os.path.exists(path_txt):
        return None
    with open(path_txt, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    for i, name in enumerate(lines, start=1):
        labels[i] = name
    return labels


def build_model(num_classes: int):
    # Faster R-CNN ResNet50 FPN (umum)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


@st.cache_resource
def load_model(ckpt_path: str, num_classes: int):
    model = build_model(num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Dua kemungkinan umum:
    # 1) ckpt langsung state_dict
    # 2) ckpt dict yang punya key 'model_state_dict'
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()
    return model


def run_inference(model, pil_img: Image.Image, score_thresh: float):
    img_tensor = F.to_tensor(pil_img).to(DEVICE)  # [C,H,W], 0..1

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs["boxes"].detach().cpu().numpy()
    scores = outputs["scores"].detach().cpu().numpy()
    labels = outputs["labels"].detach().cpu().numpy()

    keep = scores >= score_thresh
    return boxes[keep], scores[keep], labels[keep]


def draw_boxes(pil_img: Image.Image, boxes, scores, labels, label_map):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        name = label_map.get(int(lb), str(lb))
        text = f"{name} {sc:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


# =========================
# UI Streamlit
# =========================
st.set_page_config(page_title="Faster R-CNN Detector", layout="wide")
st.title("Web GUI Test - Upload Image → Deteksi Bounding Box")

with st.sidebar:
    st.header("Model & Threshold")
    ckpt_path = st.text_input("Path checkpoint (.pth)", value=DEFAULT_CKPT)

    # coba baca labelmap.txt kalau ada (project kamu ada labelmap.txt)
    labelmap_txt = "Datasets/RBC_WBC_VOC_DATASET/labelmap.txt"
    auto_labels = load_labelmap_from_txt(labelmap_txt)
    label_map = auto_labels if auto_labels else DEFAULT_LABELS

    st.caption(f"DEVICE: {DEVICE}")
    score_thresh = st.slider("Score threshold", 0.05, 0.95, 0.50, 0.05)

    show_json = st.checkbox("Tampilkan output JSON", value=False)

col1, col2 = st.columns(2)

uploaded = st.file_uploader("Upload file gambar (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Silakan upload gambar untuk mulai deteksi.")
    st.stop()

try:
    pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
except Exception as e:
    st.error(f"Gagal membaca gambar: {e}")
    st.stop()

with col1:
    st.subheader("Input")
    st.image(pil_img, use_container_width=True)

if not os.path.exists(ckpt_path):
    st.error(f"Checkpoint tidak ditemukan: {ckpt_path}")
    st.stop()

# load model (cached)
model = load_model(ckpt_path, NUM_CLASSES)

boxes, scores, labels = run_inference(model, pil_img, score_thresh)
out_img = draw_boxes(pil_img, boxes, scores, labels, label_map)

with col2:
    st.subheader("Output (Bounding Box)")
    st.image(out_img, use_container_width=True)
    st.write(f"Deteksi: **{len(boxes)}** objek")

if show_json:
    pred_list = []
    for b, s, l in zip(boxes, scores, labels):
        pred_list.append({
            "bbox_xyxy": [float(x) for x in b.tolist()],
            "score": float(s),
            "label_id": int(l),
            "label_name": label_map.get(int(l), str(int(l))),
        })
    st.json(pred_list)
