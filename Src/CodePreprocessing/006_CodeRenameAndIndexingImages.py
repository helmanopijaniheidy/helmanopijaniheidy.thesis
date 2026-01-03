"""
005_CodeRenameAndIndexingImages.py
---------------------------------
Rename file di folder ImageMerge:
- Mapping prefix kelas ke standar baru (CLASS_MAP)
- Indexing per class mulai 0001 (BAS_0001.jpg, BAS_0002.jpg, ...)

Aturan:
- Hanya memproses .jpg
- Prefix dianggap bagian sebelum underscore pertama
- Case-insensitive pada prefix
- Tidak overwrite file (safety)
"""

from pathlib import Path
import sys


CLASS_MAP = {
    "BA": "BAS",
    "EO": "EOS",
    "BNE": "NEU",
    "SNE": "NEU",
    "NEUTROPHIL": "NEU",
    "LY": "LIM",
    "MO": "MON",
}


def extract_prefix(filename: str) -> str | None:
    """Ambil prefix sebelum underscore pertama. Return None jika tidak ada '_'."""
    if "_" not in filename:
        return None
    return filename.split("_", 1)[0]


def main():
    # ==================================================
    # PROJECT ROOT
    # ==================================================
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    IMAGE_MERGE_DIR = PROJECT_ROOT / "Data" / "Datasets" / "ImageMerge"

    if not IMAGE_MERGE_DIR.exists():
        print(f"[ERROR] Folder ImageMerge tidak ditemukan: {IMAGE_MERGE_DIR}")
        sys.exit(1)

    # Ambil semua jpg
    all_files = sorted([p for p in IMAGE_MERGE_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])

    # Kelompokkan berdasarkan class baru (BAS/EOS/NEU/LIM/MON)
    grouped: dict[str, list[Path]] = {}

    skipped_no_prefix = 0
    skipped_unknown_prefix = 0

    for p in all_files:
        prefix = extract_prefix(p.name)
        if prefix is None:
            print(f"[WARN] Tidak ada '_' pada nama file, skip: {p.name}")
            skipped_no_prefix += 1
            continue

        prefix_key = prefix.upper()
        if prefix_key not in CLASS_MAP:
            print(f"[WARN] Prefix tidak dikenali ({prefix}), skip: {p.name}")
            skipped_unknown_prefix += 1
            continue

        new_cls = CLASS_MAP[prefix_key]  # BAS/EOS/NEU/LIM/MON
        grouped.setdefault(new_cls, []).append(p)

    # ==================================================
    # RENAME DENGAN INDEX PER CLASS
    # ==================================================
    total_renamed = 0
    total_skipped_conflict = 0

    # Untuk menghindari konflik rename "in-place", pakai 2 tahap:
    # 1) rename ke nama sementara unik
    # 2) rename dari sementara ke final
    temp_paths: list[tuple[Path, Path]] = []   # (old_path, temp_path)
    final_plans: list[tuple[Path, Path]] = []  # (temp_path, final_path)

    # Tahap 1: buat rencana rename sementara
    for new_cls in sorted(grouped.keys()):
        files = grouped[new_cls]
        # stabil: urutkan by nama asli
        files = sorted(files, key=lambda x: x.name)

        for idx, old_path in enumerate(files, start=1):
            temp_name = f"__TMP__{new_cls}__{idx:04d}__{old_path.name}"
            temp_path = IMAGE_MERGE_DIR / temp_name
            temp_paths.append((old_path, temp_path))

    # Eksekusi rename ke temp
    for old_path, temp_path in temp_paths:
        if temp_path.exists():
            # harusnya tidak terjadi, tapi safety
            print(f"[WARN] Temp target sudah ada, skip: {temp_path.name}")
            total_skipped_conflict += 1
            continue
        old_path.rename(temp_path)

    # Tahap 2: rencana final dari temp -> final
    # Kumpulkan ulang temp yang sesuai pola __TMP__
    temp_files = sorted([p for p in IMAGE_MERGE_DIR.iterdir() if p.is_file() and p.name.startswith("__TMP__")])

    # Buat counter per class dari temp file
    counters: dict[str, int] = {}
    for tp in temp_files:
        # Format temp: __TMP__{CLS}__{IDX}__{oldname}
        parts = tp.name.split("__")
        # parts: ["", "TMP", "{CLS}", "{IDX:04d}", "{oldname}"]
        if len(parts) < 5:
            print(f"[WARN] Format temp tidak valid, skip: {tp.name}")
            continue
        cls = parts[2]
        counters.setdefault(cls, 0)
        counters[cls] += 1

    # Rename final per class (urutkan dulu supaya idx naik)
    per_class_temp: dict[str, list[Path]] = {}
    for tp in temp_files:
        parts = tp.name.split("__")
        if len(parts) < 5:
            continue
        cls = parts[2]
        per_class_temp.setdefault(cls, []).append(tp)

    for cls in sorted(per_class_temp.keys()):
        # Urutkan berdasarkan temp name (sudah punya idx 0001,0002,...)
        for i, tp in enumerate(sorted(per_class_temp[cls], key=lambda x: x.name), start=1):
            final_name = f"{cls}_{i:04d}.jpg"
            final_path = IMAGE_MERGE_DIR / final_name

            if final_path.exists():
                print(f"[WARN] Final target sudah ada, skip: {final_name}")
                total_skipped_conflict += 1
                continue

            tp.rename(final_path)
            total_renamed += 1

    # ==================================================
    # REPORT
    # ==================================================
    print("\n[SUCCESS] Rename + indexing selesai.")
    print(f"Folder: {IMAGE_MERGE_DIR}")
    print(f"Total file diproses (jpg)        : {len(all_files)}")
    print(f"Total file berhasil di-rename    : {total_renamed}")
    print(f"Skip conflict (no overwrite)     : {total_skipped_conflict}")
    print(f"Skip tidak ada '_'               : {skipped_no_prefix}")
    print(f"Skip prefix tidak dikenali       : {skipped_unknown_prefix}")

    # Ringkas per class (hasil akhir)
    print("\nRingkasan hasil akhir per class (file prefix):")
    for cls in ["BAS", "EOS", "NEU", "LIM", "MON"]:
        count_cls = len(list(IMAGE_MERGE_DIR.glob(f"{cls}_*.jpg")))
        print(f"  - {cls}: {count_cls}")


if __name__ == "__main__":
    main()
