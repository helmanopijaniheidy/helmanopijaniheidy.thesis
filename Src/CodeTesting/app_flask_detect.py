import io
import os
import base64
import numpy as np
from flask import Flask, request, render_template_string
from PIL import Image

import torch
import torchvision
from torchvision.transforms import functional as F
import cv2


# =========================
# STATIC CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# best.pth setara dengan file ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(SCRIPT_DIR, "best.pth")

# Warna sesuai permintaan (RGB)
LABEL_MAP = {
    1: ("RBC", (255, 0, 0)),   # RGB (merah)
    2: ("WBC", (0, 0, 255)),   # RGB (biru)
}
NUM_CLASSES = 1 + len(LABEL_MAP)


# =========================
# Model
# =========================
def build_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


def load_model():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {CKPT_PATH}")

    model = build_model(NUM_CLASSES)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


MODEL = load_model()


def infer(pil_img: Image.Image, thr: float):
    img_tensor = F.to_tensor(pil_img).to(DEVICE)
    with torch.no_grad():
        out = MODEL([img_tensor])[0]

    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()

    keep = scores >= thr
    return boxes[keep], scores[keep], labels[keep]


def draw_boxes(pil_img: Image.Image, boxes, scores, labels):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
        lb = int(lb)
        name, rgb = LABEL_MAP.get(lb, (str(lb), (0, 255, 0)))
        color_bgr = (rgb[2], rgb[1], rgb[0])  # RGB -> BGR

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(
            img,
            f"{name} {float(sc):.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_bgr,
            2,
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def pil_to_base64_png(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def count_by_class(labels_np):
    counts = {LABEL_MAP[k][0]: 0 for k in LABEL_MAP.keys()}
    for lb in labels_np.tolist():
        name = LABEL_MAP.get(int(lb), (str(int(lb)), None))[0]
        counts[name] = counts.get(name, 0) + 1
    total = int(len(labels_np))
    return counts, total


# =========================
# Flask UI (Auto Detect on Upload)
# =========================
app = Flask(__name__)

HTML = r"""
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>RBC/WBC Detector (Flask)</title>

  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">

  <style>
    body { background: #0b1220; }
    .app-shell {
      background: radial-gradient(1200px 600px at 20% 0%, rgba(99,102,241,0.25), transparent 60%),
                  radial-gradient(900px 500px at 80% 10%, rgba(34,197,94,0.18), transparent 55%),
                  #0b1220;
      min-height: 100vh;
    }
    .glass {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      backdrop-filter: blur(10px);
    }
    .muted { color: rgba(255,255,255,0.75); }
    .tiny { font-size: 0.9rem; }
    .brand { letter-spacing: .2px; font-weight: 700; }

    .dropzone {
      border: 2px dashed rgba(255,255,255,0.25);
      border-radius: 16px;
      padding: 22px;
      cursor: pointer;
      transition: all .15s ease;
      background: rgba(255,255,255,0.03);
    }
    .dropzone:hover { border-color: rgba(99,102,241,0.65); background: rgba(99,102,241,0.08); }
    .dropzone.dragover { border-color: rgba(34,197,94,0.75); background: rgba(34,197,94,0.10); }

    .preview-img {
      max-width: 100%;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.03);
    }
    .stat-pill {
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 999px;
      padding: 10px 12px;
    }
    .kbd {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: .85rem;
      padding: 2px 8px;
      border-radius: 8px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.10);
    }
    .disabled-ui {
      pointer-events: none;
      opacity: 0.75;
    }
  </style>
</head>

<body class="text-white">
<div class="app-shell">
  <div class="container py-4 py-md-5">

    <div class="d-flex flex-column flex-md-row align-items-start align-items-md-center justify-content-between gap-3 mb-4">
      <div>
        <div class="d-flex align-items-center gap-2">
          <i class="bi bi-bounding-box-circles fs-3 text-info"></i>
          <h1 class="h3 mb-0 brand">Faster R-CNN RBC/WBC Detector</h1>
        </div>
        <div class="muted tiny mt-1">
          Device: <span class="kbd">{{ device }}</span>
          &nbsp;|&nbsp; Model: <span class="kbd">{{ model_name }}</span>
          &nbsp;|&nbsp; Local Flask Server
        </div>
      </div>

      <div class="glass p-3">
        <div class="muted tiny">Tips</div>
        <div class="tiny">
          Drag & drop gambar ke area upload, atau klik untuk memilih file.
          <div class="muted tiny mt-1">Auto-detect aktif: setelah upload, deteksi jalan otomatis.</div>
        </div>
      </div>
    </div>

    <div class="row g-4">
      <div class="col-12 col-lg-4">
        <div class="glass p-3 p-md-4 h-100">
          <h2 class="h5 mb-3">Upload & Parameter</h2>

          <form id="detectForm" method="POST" enctype="multipart/form-data">
            <input id="fileInput" type="file" name="image" accept="image/png,image/jpeg" hidden required>

            <div id="uiBlock">
              <div id="dropzone" class="dropzone mb-3" role="button" tabindex="0">
                <div class="d-flex align-items-center gap-3">
                  <div class="display-6"><i class="bi bi-cloud-arrow-up"></i></div>
                  <div>
                    <div class="fw-semibold">Upload Image</div>
                    <div class="muted tiny">Klik atau drag & drop (JPG/PNG)</div>
                  </div>
                </div>
                <div id="fileName" class="muted tiny mt-2"></div>

                <div id="clientPreviewWrap" class="mt-3" style="display:none;">
                  <div class="muted tiny mb-2">Preview (sebelum detect)</div>
                  <img id="clientPreview" class="preview-img" alt="preview"/>
                </div>
              </div>

              <div id="loadingMsg" class="muted tiny mt-2" style="display:none;">
                <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                Detecting...
              </div>

              <div class="mb-3 mt-3">
                <label class="form-label muted tiny mb-1">Score threshold</label>
                <div class="d-flex align-items-center gap-2">
                  <input id="thrRange" type="range" class="form-range" min="0.05" max="0.95" step="0.05"
                         value="{{ thr }}">
                  <input id="thrInput" class="form-control form-control-sm" style="max-width: 88px;"
                         type="number" min="0.05" max="0.95" step="0.05" name="thr" value="{{ thr }}">
                </div>
                <div class="muted tiny">Semakin tinggi → lebih selektif.</div>
              </div>
            </div>

            <!-- tombol submit disembunyikan (auto submit via JS) -->
            <button type="submit" class="d-none">Detect</button>
          </form>

          <hr class="border border-white border-opacity-10 my-4">

          <h3 class="h6 muted mb-3">Hasil Deteksi</h3>

          <div class="d-flex flex-column gap-2">
            <div class="stat-pill d-flex justify-content-between align-items-center">
              <div class="muted tiny"><i class="bi bi-droplet-fill text-danger"></i> RBC</div>
              <div class="fw-semibold">{{ counts.get('RBC', 0) }}</div>
            </div>
            <div class="stat-pill d-flex justify-content-between align-items-center">
              <div class="muted tiny"><i class="bi bi-person-fill text-primary"></i> WBC</div>
              <div class="fw-semibold">{{ counts.get('WBC', 0) }}</div>
            </div>
            <div class="stat-pill d-flex justify-content-between align-items-center">
              <div class="muted tiny"><i class="bi bi-123 text-info"></i> Total</div>
              <div class="fw-semibold">{{ total }}</div>
            </div>
          </div>

          {% if error %}
            <div class="alert alert-warning mt-3 mb-0">
              {{ error }}
            </div>
          {% endif %}
        </div>
      </div>

      <div class="col-12 col-lg-8">
        <div class="glass p-3 p-md-4">
          <div class="d-flex align-items-center justify-content-between flex-wrap gap-2 mb-3">
            <h2 class="h5 mb-0">Visualisasi</h2>
            {% if total > 0 %}
              <span class="badge text-bg-info">
                Detected {{ total }} objects
              </span>
            {% else %}
              <span class="badge text-bg-secondary">Belum ada hasil</span>
            {% endif %}
          </div>

          {% if input_img and output_img %}
          <div class="row g-3">
            <div class="col-12 col-md-6">
              <div class="muted tiny mb-2">Input</div>
              <img class="preview-img" src="data:image/png;base64,{{ input_img }}" alt="input">
            </div>
            <div class="col-12 col-md-6">
              <div class="muted tiny mb-2">Output (Bounding Box)</div>
              <img class="preview-img" src="data:image/png;base64,{{ output_img }}" alt="output">
            </div>
          </div>
          {% else %}
            <div class="text-center py-5">
              <div class="display-6 mb-2">🧪</div>
              <div class="muted">Upload gambar untuk melihat input & output deteksi.</div>
            </div>
          {% endif %}
        </div>

        <div class="muted tiny mt-3">
          &copy Helma Nopijani Heidy - 243210xx
        </div>
      </div>
    </div>

  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

<script>
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const fileName = document.getElementById('fileName');
  const clientPreview = document.getElementById('clientPreview');
  const clientPreviewWrap = document.getElementById('clientPreviewWrap');
  const loadingMsg = document.getElementById('loadingMsg');
  const form = document.getElementById('detectForm');
  const uiBlock = document.getElementById('uiBlock');

  const thrRange = document.getElementById('thrRange');
  const thrInput = document.getElementById('thrInput');

  let submitted = false;

  function setFile(file) {
    if (!file) return;
    fileName.textContent = `${file.name} (${Math.round(file.size/1024)} KB)`;
    const reader = new FileReader();
    reader.onload = (e) => {
      clientPreview.src = e.target.result;
      clientPreviewWrap.style.display = 'block';
    };
    reader.readAsDataURL(file);
  }

  function autoSubmit() {
    if (submitted) return;
    submitted = true;
    if (loadingMsg) loadingMsg.style.display = 'block';

    // Penting: JANGAN disable fileInput (kalau disabled, file tidak ikut terkirim)
    // Cukup lock UI supaya user tidak spam
    if (uiBlock) uiBlock.classList.add('disabled-ui');

    form.submit();
  }

  // Click to open picker
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
  });

  // Drag & drop
  ['dragenter', 'dragover'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      // Assign file list (umumnya works di Chromium)
      fileInput.files = files;
      setFile(files[0]);
      autoSubmit();
    }
  });

  // File chosen -> auto submit
  fileInput.addEventListener('change', () => {
    const f = fileInput.files?.[0];
    setFile(f);
    if (f) autoSubmit();
  });

  // Threshold sync (atur threshold dulu sebelum upload)
  function clampToStep(v) {
    const step = 0.05;
    v = Math.max(0.05, Math.min(0.95, v));
    return Math.round(v / step) * step;
  }
  thrRange.addEventListener('input', () => {
    thrInput.value = clampToStep(parseFloat(thrRange.value)).toFixed(2);
  });
  thrInput.addEventListener('input', () => {
    const v = clampToStep(parseFloat(thrInput.value || '0.5'));
    thrRange.value = v.toFixed(2);
    thrInput.value = v.toFixed(2);
  });
</script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    input_img = None
    output_img = None
    error = None

    thr = 0.50
    counts = {"RBC": 0, "WBC": 0}
    total = 0

    if request.method == "POST":
        try:
            thr = float(request.form.get("thr", "0.50"))
            thr = max(0.05, min(0.95, thr))
        except Exception:
            thr = 0.50

        file = request.files.get("image")
        if not file or file.filename.strip() == "":
            error = "File belum dipilih."
        else:
            try:
                pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
                boxes, scores, labels = infer(pil_img, thr)
                out_img = draw_boxes(pil_img, boxes, scores, labels)

                input_img = pil_to_base64_png(pil_img)
                output_img = pil_to_base64_png(out_img)

                counts, total = count_by_class(labels)
            except Exception as e:
                error = f"Gagal memproses gambar: {e}"

    return render_template_string(
        HTML,
        device=DEVICE,
        model_name=os.path.basename(CKPT_PATH),
        thr=f"{thr:.2f}",
        input_img=input_img,
        output_img=output_img,
        counts=counts,
        total=total,
        error=error,
    )


if __name__ == "__main__":
    # Akses PC lain: host="0.0.0.0"
    app.run(host="0.0.0.0", port=5000, debug=True)
