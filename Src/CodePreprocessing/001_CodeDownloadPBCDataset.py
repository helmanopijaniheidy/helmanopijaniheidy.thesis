#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
001_CodeDownloadDatasetTria.py

- Mengunduh dataset PBC_dataset_split ke folder Data/Datasets
- Menggunakan wget untuk proses download
- Proteksi overwrite jika folder sudah ada
"""

from pathlib import Path
import subprocess
import sys

# ==================================================
# BASE PATH (Folder tempat script ini berada)
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

# ==================================================
# DATASET PATH (Sesuai permintaan anda)
# ==================================================
# Struktur: helmanopijaniheidy.thesis/Data/Datasets/
DATASET_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "Datasets"
).resolve()

# URL Download Baru
ZIP_URL = "https://serverdorisjuarsafoldershare.dorisjuarsa.com/Jeni/PBC_dataset_split.zip"

# Nama file tujuan berdasarkan URL
zip_filename = "PBC_dataset_split.zip"
zip_path = DATASET_DIR / zip_filename

# ==================================================
# SAFETY CHECK: DATASET DIR
# ==================================================
# Cek apakah folder sudah ada atau file zip sudah ada
if zip_path.exists():
    print(f"⚠️ File dataset sudah ada di: {zip_path}")
    print("❌ Untuk menjaga konsistensi data, overwrite tidak diizinkan.")
    print("👉 Hapus file tersebut jika ingin download ulang.")
    sys.exit(1)

# Pastikan folder parent (Data/Datasets) sudah dibuat
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# DOWNLOAD ZIP (WGET)
# ==================================================
print("⬇️ Mengunduh dataset menggunakan wget...")
print(f"URL : {ZIP_URL}")
print(f"DEST: {zip_path}")

try:
    # Perintah wget
    # -c untuk resume jika terputus (opsional)
    # -O untuk output path
    subprocess.run(
        [
            "wget",
            "--no-check-certificate", # Tambahan jika server memiliki masalah SSL
            ZIP_URL,
            "-O",
            str(zip_path)
        ],
        check=True
    )
    print(f"✅ Download selesai: {zip_path}")
except subprocess.CalledProcessError:
    print("❌ Gagal mengunduh dataset. Pastikan 'wget' terinstal di Ubuntu (sudo apt install wget).")
    sys.exit(1)
except FileNotFoundError:
    print("❌ Perintah 'wget' tidak ditemukan. Jalankan: sudo apt install wget")
    sys.exit(1)