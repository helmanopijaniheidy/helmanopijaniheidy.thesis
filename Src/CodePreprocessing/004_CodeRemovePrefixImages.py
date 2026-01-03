"""
003_CodeRemovePrefixImages.py
----------------------------
Membersihkan prefix nama file di folder ImageMerge.

Aturan:
- Jika prefix sebelum '_' bukan alfabet → hapus prefix tsb
- Tidak overwrite file
- Menggunakan pathlib (lintas OS)
"""

from pathlib import Path
import sys
import re


def is_alphabetic_prefix(prefix: str) -> bool:
    """
    Return True jika prefix hanya berisi alfabet (A-Z, a-z)
    """
    return prefix.isalpha()


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

    renamed_count = 0
    skipped_count = 0
    untouched_count = 0

    for img_path in IMAGE_MERGE_DIR.iterdir():
        if not img_path.is_file():
            continue

        name = img_path.name

        # File tanpa underscore → tidak diproses
        if "_" not in name:
            untouched_count += 1
            continue

        prefix, rest = name.split("_", 1)

        # Jika prefix alfabet → tidak diubah
        if is_alphabetic_prefix(prefix):
            untouched_count += 1
            continue

        # Nama baru tanpa prefix non-alfabet
        new_name = rest
        new_path = IMAGE_MERGE_DIR / new_name

        # Safety: jangan overwrite
        if new_path.exists():
            print(f"[WARN] Target sudah ada, skip rename: {new_name}")
            skipped_count += 1
            continue

        img_path.rename(new_path)
        renamed_count += 1

    # ==================================================
    # REPORT
    # ==================================================
    print("\n[SUCCESS] Pembersihan prefix selesai.")
    print(f"Folder: {IMAGE_MERGE_DIR}")
    print(f"File di-rename        : {renamed_count}")
    print(f"File di-skip (conflict): {skipped_count}")
    print(f"File tidak berubah    : {untouched_count}")


if __name__ == "__main__":
    main()
