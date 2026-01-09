#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
011_CodeUnzipModelFrcnn.py

- Unzip dataset runs_frcnn.zip
- Lokasi: Data/DataModels/
- Perbaikan: Menghindari folder ganda (runs_frcnn/runs_frcnn)
- Menggunakan ekstraksi langsung ke parent karena ZIP sudah mengandung folder induk.
"""

from pathlib import Path
import zipfile
import sys

def print_tree(root: Path, prefix: str = ""):
    """Menampilkan struktur folder secara visual."""
    if not root.exists(): return
    contents = sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    for index, path in enumerate(contents):
        connector = "└── " if index == len(contents) - 1 else "├── "
        print(prefix + connector + path.name)
        if path.is_dir() and prefix.count("│") < 1: # Batasi kedalaman agar ringkas
            extension = "    " if index == len(contents) - 1 else "│   "
            print_tree(path, prefix + extension)

def main():
    # Base Path (Folder script ini berada)
    BASE_DIR = Path(__file__).resolve().parent

    # Lokasi ZIP
    DATASET_DIR = (BASE_DIR / ".." / ".." / "Data" / "DataModels").resolve()
    zip_path = DATASET_DIR / "runs_frcnn.zip"

    # Nama folder yang diharapkan ada di dalam ZIP
    expected_folder = DATASET_DIR / zip_path.stem

    print(f"[INFO] ZIP Path      : {zip_path}")
    print(f"[INFO] Target Folder : {expected_folder}")

    # 1. Cek apakah file ZIP ada
    if not zip_path.exists():
        print(f"❌ ERROR: File {zip_path.name} tidak ditemukan.")
        sys.exit(1)

    # 2. Safety: Jangan overwrite jika folder target SUDAH ADA
    if expected_folder.exists():
        print(f"\n⚠️ WARNING: Folder '{expected_folder.name}' sudah ada.")
        print("❌ Ekstraksi dibatalkan agar tidak menimpa data yang sudah ada.")
        sys.exit(0)

    # 3. Proses Ekstraksi
    try:
        print(f"\n📦 Mengekstrak {zip_path.name} langsung ke {DATASET_DIR.name}...")
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Kita ekstrak ke DATASET_DIR (bukan ke folder stem)
            # Karena ZIP hasil 'zip -r' sudah punya folder 'runs_frcnn' di dalamnya.
            zip_ref.extractall(DATASET_DIR)
        
        print(f"✅ Berhasil! Folder '{zip_path.stem}' kini tersedia.")

        # 4. Tampilkan struktur
        print("\n[INFO] Struktur folder hasil ekstraksi:")
        print_tree(expected_folder)

    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()