"""
001_CodeUnzipDataset.py
----------------------
Unzip PBC_dataset_split.zip dengan aman (tanpa overwrite).

Aturan:
- Jika folder hasil unzip sudah ada → tidak melakukan apa-apa
- Menggunakan pathlib agar lintas OS
"""

from pathlib import Path
import zipfile
import sys


def main():
    # Root project (…/helmanopijaniheidy.thesis)
    project_root = Path(__file__).resolve().parents[2]

    # Path dataset
    dataset_dir = project_root / "Data" / "Datasets"
    zip_path = dataset_dir / "PBC_dataset_split.zip"

    # Nama folder hasil unzip (otomatis dari nama zip)
    extract_dir = dataset_dir / "PBC_dataset_split"

    # Validasi file zip
    if not zip_path.exists():
        print(f"[ERROR] File zip tidak ditemukan: {zip_path}")
        sys.exit(1)

    # Safety: jangan overwrite
    if extract_dir.exists():
        print(
            "[INFO] Folder dataset sudah ada.\n"
            f"       Lokasi: {extract_dir}\n"
            "       Tidak melakukan unzip (safety mode aktif)."
        )
        return

    # Proses unzip
    print(f"[INFO] Unzipping dataset:\n       {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(
        "[SUCCESS] Dataset berhasil di-unzip.\n"
        f"          Lokasi: {extract_dir}"
    )


if __name__ == "__main__":
    main()
