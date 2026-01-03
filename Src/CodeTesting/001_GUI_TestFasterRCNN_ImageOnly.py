#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tkinter GUI - Faster R-CNN (torchvision) Universal Image Testing
- Load .pth / .pt (state_dict OR checkpoint dict {"model": state_dict, ...})
- Auto-detect num_classes from checkpoint (cls_score.weight shape)
- Dynamic class counters (auto from labelmap.txt if found, else Class1..)
- Image-only (no camera/video/record)
- Pan/Zoom on left triggers re-inference on visible crop region (no IMGSZ resize)
"""

from __future__ import annotations

import time
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torchvision
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2  # pip install opencv-python
from torchvision.transforms import functional as F


# ---------------- UI CONFIG ----------------
SQUARE = 720
GAP = 5
WINDOW_WIDTH = SQUARE * 2 + GAP + 40

BTN_W = 18
BTN_H = 2

DEFAULT_SCORE_TH = 0.50  # confidence threshold
FPS_EMA_ALPHA = 0.2


# ---------------- Helpers ----------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def label_color(name: str):
    h = abs(hash(name))
    return (int(h % 200) + 30, int((h // 200) % 200) + 30, int((h // 40000) % 200) + 30)


def draw_boxes_rgb(
    img_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    names_map: Dict[int, str],
    score_th: float,
):
    out = img_rgb.copy()
    h, w = out.shape[:2]
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    for (x1, y1, x2, y2), lab, sc in zip(boxes_xyxy, labels, scores):
        sc = float(sc)
        if sc < score_th:
            continue

        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        x1i, y1i = clamp(x1i, 0, w - 1), clamp(y1i, 0, h - 1)
        x2i, y2i = clamp(x2i, 0, w - 1), clamp(y2i, 0, h - 1)
        if x2i <= x1i or y2i <= y1i:
            continue

        name = names_map.get(int(lab), str(lab))
        col = label_color(name)

        cv2.rectangle(out_bgr, (x1i, y1i), (x2i, y2i), col, 2)

        text = f"{name} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        by = max(0, y1i - th - 6)
        cv2.rectangle(out_bgr, (x1i, by), (x1i + tw + 6, by + th + 6), col, -1)
        cv2.putText(
            out_bgr,
            text,
            (x1i + 3, by + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def try_load_labelmap(model_path: Path, num_classes: int, project_root: Path) -> Optional[Dict[int, str]]:
    """
    Try to find and parse labelmap.txt to get real class names.
    We need (num_classes-1) names (excluding background=0).

    Search order:
      1) model_path.parent / "labelmap.txt"
      2) project_root / "Data" / "Datasets" / "RBC_WBC_VOC_DATASET" / "labelmap.txt"
      3) project_root / "Data" / "Datasets" / "RBC_WBC_VOC_DATASET" / "labelmap.txt" (same as above)
      4) any labelmap.txt up to 3 dirs above model_path (best effort)
    """
    want = num_classes - 1
    candidates: List[Path] = []

    candidates.append(model_path.parent / "labelmap.txt")
    candidates.append(project_root / "Data" / "Datasets" / "RBC_WBC_VOC_DATASET" / "labelmap.txt")

    # search up to 3 parents from model file
    p = model_path.parent
    for _ in range(3):
        candidates.append(p / "labelmap.txt")
        p = p.parent

    for cand in candidates:
        try:
            if not cand.exists():
                continue
            lines = []
            for ln in cand.read_text(encoding="utf-8", errors="ignore").splitlines():
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                # allow formats like "0: RBC" or "RBC"
                if ":" in ln:
                    parts = ln.split(":", 1)
                    name = parts[1].strip()
                else:
                    name = ln
                if name:
                    lines.append(name)

            if len(lines) == want:
                return {i + 1: lines[i] for i in range(want)}
        except Exception:
            continue

    return None


# ---------------- ViewTransform (zoom/pan) ----------------
@dataclass
class ViewState:
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0


def view_to_rect(view: ViewState, canvas_w: int, canvas_h: int, img_w: int, img_h: int):
    x1 = int(round((0 - view.offset_x) / view.scale))
    y1 = int(round((0 - view.offset_y) / view.scale))
    x2 = int(round((canvas_w - view.offset_x) / view.scale))
    y2 = int(round((canvas_h - view.offset_y) / view.scale))

    x1 = clamp(x1, 0, img_w)
    y1 = clamp(y1, 0, img_h)
    x2 = clamp(x2, 0, img_w)
    y2 = clamp(y2, 0, img_h)

    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, img_w, img_h
    return (x1, y1, x2, y2)


# ---------------- ImageCanvas (zoom/pan) ----------------
class ImageCanvas(tk.Canvas):
    def __init__(self, master, width=SQUARE, height=SQUARE, **kwargs):
        super().__init__(master, width=width, height=height,
                         highlightthickness=1, highlightbackground="#444", **kwargs)
        self.configure(bg="black")

        self.view = ViewState()
        self._img_rgb = None
        self._photo = None
        self._img_id = None

        self._dragging = False
        self._last_x = 0
        self._last_y = 0

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

        # Windows/Mac wheel
        self.bind("<MouseWheel>", self._on_wheel)
        # Linux wheel
        self.bind("<Button-4>", self._on_wheel_linux_up)
        self.bind("<Button-5>", self._on_wheel_linux_down)

    def set_image_rgb(self, img_rgb: np.ndarray, reset=False):
        self._img_rgb = img_rgb
        if reset:
            self.view = ViewState(scale=1.0, offset_x=0.0, offset_y=0.0)
        self._redraw()

    def set_view(self, scale: float, offset_x: float, offset_y: float):
        self.view.scale = float(scale)
        self.view.offset_x = float(offset_x)
        self.view.offset_y = float(offset_y)
        self._redraw()

    def reset_view(self):
        self.view = ViewState(scale=1.0, offset_x=0.0, offset_y=0.0)
        self._redraw()

    def get_visible_rect(self):
        if self._img_rgb is None:
            return None
        h, w = self._img_rgb.shape[:2]
        cw = int(self.cget("width"))
        ch = int(self.cget("height"))
        return view_to_rect(self.view, cw, ch, w, h)

    def _redraw(self):
        if self._img_rgb is None:
            self.delete("all")
            self._img_id = None
            self._photo = None
            return

        img = self._img_rgb
        h, w = img.shape[:2]
        scaled_w = max(1, int(round(w * self.view.scale)))
        scaled_h = max(1, int(round(h * self.view.scale)))

        pil = Image.fromarray(img)
        pil_scaled = pil.resize((scaled_w, scaled_h), resample=Image.BILINEAR)

        cw = int(self.cget("width"))
        ch = int(self.cget("height"))
        bg = Image.new("RGB", (cw, ch), (0, 0, 0))

        ox = int(round(self.view.offset_x))
        oy = int(round(self.view.offset_y))
        bg.paste(pil_scaled, (ox, oy))

        self._photo = ImageTk.PhotoImage(bg)
        if self._img_id is None:
            self._img_id = self.create_image(0, 0, image=self._photo, anchor="nw")
        else:
            self.itemconfig(self._img_id, image=self._photo)

    def _on_press(self, e):
        if self._img_rgb is None:
            return
        self._dragging = True
        self._last_x, self._last_y = e.x, e.y

    def _on_drag(self, e):
        if not self._dragging or self._img_rgb is None:
            return
        dx = e.x - self._last_x
        dy = e.y - self._last_y
        self.view.offset_x += dx
        self.view.offset_y += dy
        self._last_x, self._last_y = e.x, e.y
        self._redraw()
        self.event_generate("<<ViewChanged>>", when="tail")

    def _on_release(self, _e):
        self._dragging = False

    def _apply_zoom(self, factor: float, mx: float, my: float):
        if self._img_rgb is None:
            return
        old_scale = self.view.scale
        new_scale = clamp(old_scale * factor, 0.2, 6.0)
        if abs(new_scale - old_scale) < 1e-9:
            return

        old_img_x = (mx - self.view.offset_x) / old_scale
        old_img_y = (my - self.view.offset_y) / old_scale
        self.view.scale = new_scale
        self.view.offset_x = mx - old_img_x * new_scale
        self.view.offset_y = my - old_img_y * new_scale

        self._redraw()
        self.event_generate("<<ViewChanged>>", when="tail")

    def _on_wheel(self, e):
        factor = 1.15 if e.delta > 0 else 0.85
        self._apply_zoom(factor, e.x, e.y)

    def _on_wheel_linux_up(self, e):
        self._apply_zoom(1.15, e.x, e.y)

    def _on_wheel_linux_down(self, e):
        self._apply_zoom(0.85, e.x, e.y)


# ---------------- Faster R-CNN builder ----------------
def build_frcnn(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_feat, num_classes
    )
    return model


def extract_state_and_num_classes(ckpt_obj) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Returns: (state_dict, num_classes including background)
    Supports:
      - raw state_dict
      - checkpoint dict with key 'model'
      - DataParallel prefix 'module.'
    """
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        state = ckpt_obj["model"]
    elif isinstance(ckpt_obj, dict):
        state = ckpt_obj
    else:
        raise ValueError("Checkpoint format tidak dikenali (bukan dict).")

    key = "roi_heads.box_predictor.cls_score.weight"
    if key not in state:
        key2 = "module." + key
        if key2 in state:
            key = key2
        else:
            raise ValueError(f"Key '{key}' tidak ditemukan. Ini bukan weight Faster R-CNN yang sesuai?")

    num_classes = int(state[key].shape[0])
    return state, num_classes


# ---------------- Worker Thread (Image-only, viewport inference) ----------------
class InferenceWorker(threading.Thread):
    def __init__(self, model: torch.nn.Module, device: torch.device, out_q: queue.Queue,
                 names_map: Dict[int, str], score_th: float):
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self.out_q = out_q
        self.names_map = names_map
        self.score_th = float(score_th)

        self._running = threading.Event()
        self._running.set()

        self._trigger = threading.Event()
        self._img_rgb_full: Optional[np.ndarray] = None
        self._crop_rect = None

        self._prev_t = None
        self._fps_ema = 0.0

    def configure_image(self, img_rgb_full: np.ndarray):
        self._img_rgb_full = img_rgb_full

    def set_crop_rect(self, rect):
        self._crop_rect = rect

    def trigger(self):
        self._trigger.set()

    def stop(self):
        self._running.clear()
        self._trigger.set()

    def _update_fps(self):
        now = time.time()
        if self._prev_t is None:
            self._prev_t = now
            return 0.0
        dt = max(1e-6, now - self._prev_t)
        fps = 1.0 / dt
        self._fps_ema = FPS_EMA_ALPHA * fps + (1.0 - FPS_EMA_ALPHA) * self._fps_ema
        self._prev_t = now
        return self._fps_ema

    @torch.no_grad()
    def _infer(self, img_rgb: np.ndarray):
        t0 = time.time()
        pil = Image.fromarray(img_rgb)
        x = F.to_tensor(pil).to(self.device)

        self.model.eval()
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=True):
                y = self.model([x])[0]
            torch.cuda.synchronize()
        else:
            y = self.model([x])[0]

        dt = time.time() - t0

        boxes = y.get("boxes", torch.empty((0, 4))).detach().cpu().numpy()
        scores = y.get("scores", torch.empty((0,))).detach().cpu().numpy()
        labels = y.get("labels", torch.empty((0,), dtype=torch.int64)).detach().cpu().numpy()
        return boxes, labels, scores, dt

    def run(self):
        if self._img_rgb_full is None:
            self.out_q.put(("error", "Tidak ada gambar untuk diproses."))
            return

        self._prev_t = None
        self._fps_ema = 0.0
        self._trigger.set()

        while self._running.is_set():
            self._trigger.wait(timeout=0.5)
            if not self._running.is_set():
                break
            if not self._trigger.is_set():
                continue
            self._trigger.clear()

            img_full = self._img_rgb_full
            h, w = img_full.shape[:2]
            rect = self._crop_rect

            if rect is None:
                x1, y1, x2, y2 = 0, 0, w, h
            else:
                x1, y1, x2, y2 = rect
                x1 = clamp(int(x1), 0, w)
                x2 = clamp(int(x2), 0, w)
                y1 = clamp(int(y1), 0, h)
                y2 = clamp(int(y2), 0, h)
                if x2 <= x1 or y2 <= y1:
                    x1, y1, x2, y2 = 0, 0, w, h

            crop = img_full[y1:y2, x1:x2].copy()
            fps_stream = self._update_fps()

            try:
                boxes, labels, scores, dt = self._infer(crop)
            except Exception as e:
                self.out_q.put(("error", f"Inference error: {e}"))
                continue

            # counts dynamic
            counts = {name: 0 for name in self.names_map.values()}
            for lab, sc in zip(labels, scores):
                if float(sc) >= self.score_th:
                    name = self.names_map.get(int(lab), None)
                    if name is not None:
                        counts[name] = counts.get(name, 0) + 1

            ann_crop = draw_boxes_rgb(crop, boxes, labels, scores, self.names_map, self.score_th)
            ann_full = img_full.copy()
            ann_full[y1:y2, x1:x2] = ann_crop

            info = {
                "fps_stream": float(fps_stream),
                "dt_infer_ms": float(dt * 1000.0),
                "total": int(sum(counts.values())),
            }

            self.out_q.put(("frame", img_full, ann_full, counts, info))


# ---------------- Main Window ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("JENI - Universal Blood Cell Detection (Faster R-CNN)")
        self.geometry(f"{WINDOW_WIDTH}x{SQUARE+300}")
        self.configure(bg="#f8f8f8")
        self.resizable(False, False)

        self.project_root = Path(__file__).resolve().parents[2]
        self.default_dialog_dir = self.project_root

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model: Optional[torch.nn.Module] = None
        self.model_path: Optional[Path] = None
        self.num_classes: Optional[int] = None  # includes background
        self.names_map: Dict[int, str] = {}      # label_id -> name (1..)
        self.score_th = DEFAULT_SCORE_TH

        self.worker: Optional[InferenceWorker] = None
        self.out_q = queue.Queue()

        # ---------- Banner ----------
        banner = tk.Frame(self, bg="#f8f8f8")
        banner.pack(pady=(10, 6), padx=10, fill="x")

        banner_lbl = tk.Label(
            banner,
            text=(
                "Deteksi Sel Darah Menggunakan Faster R-CNN (Universal GUI)\n"
                "Dibuat oleh: Jeni | Mode: Upload Gambar (tanpa camera/video)\n"
                "Pan/Zoom kiri = infer viewport (cepat) | Kanan = hasil deteksi"
            ),
            bg="#f8f8f8",
            fg="#222",
            justify="center",
            font=("Arial", 11)
        )
        banner_lbl.pack()

        # ---------- Buttons ----------
        btn_frame = tk.Frame(self, bg="#f8f8f8")
        btn_frame.pack(pady=6)

        self.btn_load = tk.Button(btn_frame, text="Load Model", width=BTN_W, height=BTN_H, command=self.load_model)
        self.btn_img = tk.Button(btn_frame, text="Open Image", width=BTN_W, height=BTN_H, command=self.open_image, state="disabled")
        self.btn_reset = tk.Button(btn_frame, text="Reset Position", width=BTN_W, height=BTN_H, command=self.reset_views, state="disabled")
        self.btn_stop = tk.Button(btn_frame, text="Stop", width=BTN_W, height=BTN_H, command=self.stop_worker, state="disabled")

        for w in [self.btn_load, self.btn_img, self.btn_reset, self.btn_stop]:
            w.pack(side="left", padx=4)

        # ---------- Canvases ----------
        canvas_frame = tk.Frame(self, bg="#f8f8f8")
        canvas_frame.pack(pady=(8, 4), padx=10)

        self.left = ImageCanvas(canvas_frame, width=SQUARE, height=SQUARE)
        self.right = ImageCanvas(canvas_frame, width=SQUARE, height=SQUARE)
        self.left.pack(side="left")
        tk.Frame(canvas_frame, width=GAP, height=SQUARE, bg="#f8f8f8").pack(side="left")
        self.right.pack(side="left")

        self.left.bind("<<ViewChanged>>", self.on_view_changed)

        # ---------- Dynamic counters ----------
        self.counter_frame = tk.Frame(self, bg="#f8f8f8")
        self.counter_frame.pack(pady=(4, 6), padx=10, fill="x")
        self.class_labels: Dict[str, tk.Label] = {}

        # ---------- Status ----------
        self.status_var = tk.StringVar(value="")
        status = tk.Label(self, textvariable=self.status_var, bg="#f8f8f8", fg="#333", font=("Arial", 11))
        status.pack(pady=(4, 8))

        self.after(15, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._img_full_rgb: Optional[np.ndarray] = None

    def _enable_after_model(self, enable=True):
        self.btn_img.configure(state=("normal" if enable else "disabled"))

    def _rebuild_counters(self):
        for child in self.counter_frame.winfo_children():
            child.destroy()
        self.class_labels.clear()

        # make it look like YOLO: each class label is a small box
        # NOTE: if many classes, it will overflow horizontally; still works.
        for _, name in sorted(self.names_map.items(), key=lambda kv: kv[0]):
            lbl = tk.Label(
                self.counter_frame,
                text=f"{name}: 0",
                bg="#ffffff",
                fg="#222",
                relief="solid",
                bd=1,
                padx=10,
                pady=6,
                font=("Arial", 10)
            )
            lbl.pack(side="left", padx=4, pady=2)
            self.class_labels[name] = lbl

    # ---------- Model ----------
    def load_model(self):
        path = filedialog.askopenfilename(
            title="Pilih Model Faster R-CNN (.pth/.pt) - best/last",
            initialdir=str(self.default_dialog_dir),
            filetypes=[("PyTorch Weights", "*.pth *.pt")]
        )
        if not path:
            return

        try:
            ckpt = torch.load(path, map_location="cpu")
            state, num_classes = extract_state_and_num_classes(ckpt)

            model = build_frcnn(num_classes=num_classes)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()

            self.model = model
            self.model_path = Path(path)
            self.num_classes = num_classes

            # try labelmap first (if exists and matches)
            lm = try_load_labelmap(self.model_path, num_classes, self.project_root)
            if lm is not None:
                self.names_map = lm
            else:
                # fallback generic
                self.names_map = {i: f"Class{i}" for i in range(1, num_classes)}

            self._rebuild_counters()
            self._enable_after_model(True)

            self.status_var.set(
                f"Model dimuat: {self.model_path.name} | device={self.device} | "
                f"num_classes={num_classes} (bg+{num_classes-1}) | score_th={self.score_th:.2f}"
            )

        except Exception as e:
            messagebox.showerror("Gagal muat model", str(e))
            self.model = None
            self.model_path = None
            self.num_classes = None
            self.names_map = {}
            self._enable_after_model(False)

    # ---------- Image ----------
    def open_image(self):
        if self.model is None:
            messagebox.showwarning("Peringatan", "Muat model terlebih dahulu.")
            return

        path = filedialog.askopenfilename(
            title="Pilih Gambar",
            initialdir=str(self.default_dialog_dir),
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return

        try:
            pil = Image.open(path).convert("RGB")
            img_rgb = np.array(pil, dtype=np.uint8)
        except Exception as e:
            messagebox.showerror("Gagal membaca gambar", str(e))
            return

        self._img_full_rgb = img_rgb

        self.left.set_image_rgb(img_rgb, reset=True)
        self.right.set_image_rgb(img_rgb, reset=True)

        self.btn_reset.configure(state="normal")
        self.btn_stop.configure(state="normal")

        self.status_var.set("Image loaded. Pan/zoom di kiri untuk re-run detection pada area terlihat.")

        # restart worker
        self._stop_worker_internal()
        self.worker = InferenceWorker(
            model=self.model,
            device=self.device,
            out_q=self.out_q,
            names_map=self.names_map,
            score_th=self.score_th
        )
        self.worker.configure_image(img_rgb)
        rect = self.left.get_visible_rect()
        self.worker.set_crop_rect(rect)
        self.worker.start()
        self.worker.trigger()

    def on_view_changed(self, _evt=None):
        self.right.set_view(self.left.view.scale, self.left.view.offset_x, self.left.view.offset_y)
        if self.worker and self.worker.is_alive():
            rect = self.left.get_visible_rect()
            self.worker.set_crop_rect(rect)
            self.worker.trigger()

    # ---------- Reset / Stop ----------
    def reset_views(self):
        self.left.reset_view()
        self.right.reset_view()
        self.on_view_changed()

    def stop_worker(self):
        self._stop_worker_internal()
        self.status_var.set("Stopped.")
        self.btn_stop.configure(state="disabled")

    def _stop_worker_internal(self):
        if self.worker and self.worker.is_alive():
            self.worker.stop()
            self.worker.join(timeout=2.0)
        self.worker = None

    # ---------- Queue polling ----------
    def _poll_queue(self):
        try:
            while True:
                item = self.out_q.get_nowait()
                kind = item[0]

                if kind == "frame":
                    _, orig_rgb, ann_rgb, counts, info = item
                    self.left.set_image_rgb(orig_rgb, reset=False)
                    self.right.set_image_rgb(ann_rgb, reset=False)

                    # update labels
                    for name, lbl in self.class_labels.items():
                        val = int(counts.get(name, 0))
                        lbl.configure(text=f"{name}: {val}")
                        if val > 0:
                            lbl.configure(bg="#d0ffd0", fg="#000")
                        else:
                            lbl.configure(bg="#ffffff", fg="#222")

                    self.status_var.set(
                        f"Total={info['total']} | infer={info['dt_infer_ms']:.1f}ms | fps_stream={info['fps_stream']:.1f} | score_th={self.score_th:.2f}"
                    )

                elif kind == "error":
                    _, msg = item
                    messagebox.showerror("Error", msg)
                    self.status_var.set("Error: " + msg)

        except queue.Empty:
            pass

        self.after(15, self._poll_queue)

    def on_close(self):
        self._stop_worker_internal()
        self.destroy()


def main():
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
