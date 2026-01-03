#!/usr/bin/env python3
"""
009_CodeUnzipDatasetCvatVoc.py
Unzip dataset VOC hasil anotasi CVAT
- Pakai pathlib (lintas OS)
- Output dibungkus folder sesuai nama zip (stem)
- Safety: tidak mengizinkan overwrite (batal jika folder target sudah ada)
- Cetak struktur folder hasil ekstraksi ke terminal
"""

from pathlib import Path
import zipfile
import sys


def print_tree(root: Path, prefix: str = ""):
    """Menampilkan struktur folder seperti perintah tree."""
    contents = sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    for index, path in enumerate(contents):
        connector = "└── " if index == len(contents) - 1 else "├── "
        print(prefix + connector + path.name)
        if path.is_dir():
            extension = "    " if index == len(contents) - 1 else "│   "
            print_tree(path, prefix + extension)


def main():
    # Root project (…/helmanopijaniheidy.thesis)
    project_root = Path(__file__).resolve().parents[2]

    # ZIP dataset
    zip_path = project_root / "Data" / "Datasets" / "RBC_WBC_VOC_DATASET.zip"

    # Folder output dibungkus sesuai nama zip (tanpa .zip)
    out_dir = zip_path.parent / zip_path.stem

    print(f"[INFO] Project root : {project_root}")
    print(f"[INFO] ZIP dataset  : {zip_path}")
    print(f"[INFO] Output dir   : {out_dir}")

    if not zip_path.exists():
        print(f"[ERROR] ZIP file tidak ditemukan: {zip_path}")
        sys.exit(1)

    # Safety: jangan overwrite
    if out_dir.exists():
        print("\n[WARNING] Folder output sudah ada. Ekstraksi dibatalkan (no overwrite)!")
        print(f"[INFO] Folder yang sudah ada: {out_dir}")
        print("[ACTION] Hapus/rename folder tersebut jika ingin ekstrak ulang.")
        sys.exit(0)

    # Buat folder output
    out_dir.mkdir(parents=True, exist_ok=False)

    # Ekstraksi ke folder output
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)

    print("\n[SUCCESS] Dataset berhasil diekstrak\n")

    # Cetak struktur hasil ekstrak (mulai dari folder output)
    print("[INFO] Struktur folder hasil ekstraksi:\n")
    print(out_dir.name)
    print_tree(out_dir)
    print()


if __name__ == "__main__":
    main()
