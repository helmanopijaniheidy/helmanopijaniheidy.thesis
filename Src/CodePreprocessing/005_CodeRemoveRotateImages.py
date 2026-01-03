"""
004_CodeRemoveRotateImages.py
----------------------------
Menghapus SEMUA file rotate di folder ImageMerge.

Aturan:
- Hapus file jika nama diakhiri dengan:
  _0.jpg, _1.jpg, _2.jpg, _3.jpg, _4.jpg, _5.jpg
- Lintas OS (pathlib)
- Report jumlah file dihapus & disisakan
"""

from pathlib import Path
import sys


def main():
    # ==================================================
    # PROJECT ROOT
    # ==================================================
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # ==================================================
    # PATH CONFIG
    # ==================================================
    IMAGE_MERGE_DIR = PROJECT_ROOT / "Data" / "Datasets" / "ImageMerge"

    if not IMAGE_MERGE_DIR.exists():
        print(f"[ERROR] Folder ImageMerge tidak ditemukan: {IMAGE_MERGE_DIR}")
        sys.exit(1)

    # SEMUA suffix rotate yang dihapus
    suffixes_to_delete = (
        "_0.jpg",
        "_1.jpg",
        "_2.jpg",
        "_3.jpg",
        "_4.jpg",
        "_5.jpg",
    )

    deleted = 0
    kept = 0

    for p in IMAGE_MERGE_DIR.iterdir():
        if not p.is_file():
            continue

        name_lower = p.name.lower()

        # hanya target jpg
        if not name_lower.endswith(".jpg"):
            continue

        if name_lower.endswith(suffixes_to_delete):
            p.unlink()
            deleted += 1
        else:
            kept += 1

    print("\n[SUCCESS] Cleaning rotate images (0–5) selesai.")
    print(f"Folder: {IMAGE_MERGE_DIR}")
    print(f"File dihapus : {deleted}")
    print(f"File disisa  : {kept}")


if __name__ == "__main__":
    main()
