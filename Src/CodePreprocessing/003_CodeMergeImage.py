"""
002_CodeMergeImage.py
--------------------
Memindahkan seluruh gambar dari PBC_dataset_split
(Train/Val/Test + 5 kelas WBC) ke satu folder: ImageMerge

Aturan:
- Tidak rename file
- Tidak overwrite
- Menggunakan pathlib (lintas OS)
- Safety: jika ImageMerge sudah ada & tidak kosong → tidak melakukan apa-apa
- Menampilkan jumlah file yang dipindahkan
"""

from pathlib import Path
import shutil
import sys


def main():
    # ==================================================
    # PROJECT ROOT
    # ==================================================
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # ==================================================
    # PATH CONFIG
    # ==================================================
    DATASETS_DIR = PROJECT_ROOT / "Data" / "Datasets"

    SRC_ROOT = DATASETS_DIR / "PBC_dataset_split" / "PBC_dataset_split"
    OUT_DIR = DATASETS_DIR / "ImageMerge"

    SPLITS = ["Train", "Val", "Test"]
    CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]

    # ==================================================
    # VALIDATION
    # ==================================================
    if not SRC_ROOT.exists():
        print(f"[ERROR] Folder sumber tidak ditemukan: {SRC_ROOT}")
        sys.exit(1)

    if OUT_DIR.exists():
        existing_files = [p for p in OUT_DIR.iterdir() if p.is_file()]
        if existing_files:
            print(
                "[INFO] Folder ImageMerge sudah ada dan tidak kosong.\n"
                f"       Lokasi: {OUT_DIR}\n"
                "       Tidak melakukan pemindahan (safety mode aktif)."
            )
            return
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=False)

    # ==================================================
    # MOVE FILES
    # ==================================================
    total_moved = 0
    total_skipped = 0

    count_by_split = {s: 0 for s in SPLITS}
    count_by_class = {c: 0 for c in CLASSES}

    for split in SPLITS:
        for cls in CLASSES:
            src_dir = SRC_ROOT / split / cls
            if not src_dir.exists():
                print(f"[WARN] Folder tidak ditemukan (skip): {src_dir}")
                continue

            for f in src_dir.iterdir():
                if not f.is_file():
                    continue

                dst = OUT_DIR / f.name

                # Safety: jangan overwrite
                if dst.exists():
                    print(f"[WARN] File duplikat, skip: {f.name}")
                    total_skipped += 1
                    continue

                shutil.move(str(f), str(dst))
                total_moved += 1
                count_by_split[split] += 1
                count_by_class[cls] += 1

    # ==================================================
    # REPORT
    # ==================================================
    print("\n[SUCCESS] Merge image selesai.")
    print(f"Lokasi output: {OUT_DIR}")
    print(f"Total file dipindahkan: {total_moved}")
    print(f"Total file di-skip (duplikat): {total_skipped}\n")

    print("Jumlah per split:")
    for s in SPLITS:
        print(f"  - {s}: {count_by_split[s]}")

    print("\nJumlah per class:")
    for c in CLASSES:
        print(f"  - {c}: {count_by_class[c]}")


if __name__ == "__main__":
    main()
