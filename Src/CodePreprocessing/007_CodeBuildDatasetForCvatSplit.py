"""
006_CodeBuildDatasetForCvatSplit.py
----------------------------------
Membuat dataset untuk CVAT secara deterministik (tanpa random sama sekali):

- Sumber: Data/Datasets/ImageMerge (file: BAS_0001.jpg dst)
- Target total: 2.000 gambar (balanced 400 per kelas: BAS/EOS/NEU/LIM/MON)
- Split per kelas: Train 70% (280), Val 20% (80), Test 10% (40)
- COPY file (bukan pindah)
- Safety: jika DatasetForCvat sudah ada dan tidak kosong -> tidak melakukan apa-apa
- Setelah selesai, buat zip: Data/Datasets/DatasetForCvat.zip (tanpa overwrite)

Catatan deterministik lintas OS:
- File dipilih berdasarkan urutan alfabet (sorted by filename), bukan iterdir order.
"""

from pathlib import Path
import shutil
import sys
import zipfile


def zip_folder(folder_path: Path, zip_path: Path) -> None:
    """Zip isi folder_path ke zip_path dengan urutan file deterministik."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # rglob + sort untuk urutan identik lintas OS
        files = sorted([p for p in folder_path.rglob("*") if p.is_file()], key=lambda p: p.as_posix())
        for f in files:
            arcname = f.relative_to(folder_path).as_posix()
            zf.write(f, arcname=arcname)


def main():
    # ==================================================
    # CONFIG
    # ==================================================
    TARGET_PER_CLASS = 400
    CLASSES = ["BAS", "EOS", "NEU", "LIM", "MON"]

    # Split per class (fixed counts, deterministic)
    TRAIN_N = 280
    VAL_N = 80
    TEST_N = 40  # 400 - 280 - 80

    # ==================================================
    # PATHS
    # ==================================================
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATASETS_DIR = PROJECT_ROOT / "Data" / "Datasets"
    SRC_DIR = DATASETS_DIR / "ImageMerge"

    OUT_ROOT = DATASETS_DIR / "DatasetForCvat"
    OUT_DIRS = {
        "Train": OUT_ROOT / "Train",
        "Val": OUT_ROOT / "Val",
        "Test": OUT_ROOT / "Test",
    }

    ZIP_PATH = DATASETS_DIR / "DatasetForCvat.zip"

    # ==================================================
    # VALIDATION
    # ==================================================
    if not SRC_DIR.exists():
        print(f"[ERROR] Folder sumber tidak ditemukan: {SRC_DIR}")
        sys.exit(1)

    # Safety: jika output folder sudah ada & tidak kosong, stop
    if OUT_ROOT.exists():
        non_empty = False
        for d in OUT_DIRS.values():
            if d.exists() and any(p.is_file() for p in d.iterdir()):
                non_empty = True
                break
        if non_empty:
            print(
                "[INFO] DatasetForCvat sudah ada dan tidak kosong.\n"
                f"       Lokasi: {OUT_ROOT}\n"
                "       Tidak melakukan split (safety mode aktif)."
            )
            return
    else:
        OUT_ROOT.mkdir(parents=True, exist_ok=False)

    # Safety: jangan overwrite zip
    if ZIP_PATH.exists():
        print(
            "[INFO] File zip sudah ada.\n"
            f"       Lokasi: {ZIP_PATH}\n"
            "       Tidak membuat zip baru (no-overwrite)."
        )
        # tetap lanjut membangun folder DatasetForCvat kalau folder belum ada
        # (zip hanya opsional setelah folder dibuat)
        # -> kita tidak return di sini.

    # Buat folder split
    for d in OUT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # ==================================================
    # COLLECT & SORT FILES PER CLASS (DETERMINISTIC)
    # ==================================================
    files_by_class: dict[str, list[Path]] = {c: [] for c in CLASSES}

    # Ambil semua jpg dan sort global by filename agar OS-independent
    all_jpg = sorted(
        [p for p in SRC_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"],
        key=lambda p: p.name,
    )

    for p in all_jpg:
        if "_" not in p.name:
            continue
        prefix = p.name.split("_", 1)[0].upper()
        if prefix in files_by_class:
            files_by_class[prefix].append(p)

    # Pastikan tiap class cukup
    for c in CLASSES:
        n = len(files_by_class[c])
        if n < TARGET_PER_CLASS:
            print(f"[ERROR] Data kelas {c} tidak cukup: tersedia {n}, butuh {TARGET_PER_CLASS}.")
            sys.exit(1)

        # Sort per class berdasarkan nama file (pasti deterministik)
        files_by_class[c] = sorted(files_by_class[c], key=lambda p: p.name)

    # ==================================================
    # PLAN SPLIT (FIRST 400 PER CLASS)
    # ==================================================
    plan: dict[str, list[Path]] = {"Train": [], "Val": [], "Test": []}

    for c in CLASSES:
        picked = files_by_class[c][:TARGET_PER_CLASS]  # ambil paling awal (non-random)

        train_part = picked[:TRAIN_N]
        val_part = picked[TRAIN_N:TRAIN_N + VAL_N]
        test_part = picked[TRAIN_N + VAL_N:TRAIN_N + VAL_N + TEST_N]

        plan["Train"].extend(train_part)
        plan["Val"].extend(val_part)
        plan["Test"].extend(test_part)

    # ==================================================
    # COPY FILES
    # ==================================================
    copied_total = 0
    copied_by_split = {"Train": 0, "Val": 0, "Test": 0}
    copied_by_class = {c: 0 for c in CLASSES}
    conflicts = 0

    def class_of(p: Path) -> str:
        return p.name.split("_", 1)[0].upper()

    for split_name in ["Train", "Val", "Test"]:
        out_dir = OUT_DIRS[split_name]
        # sort lagi supaya urutan copy konsisten (opsional tapi rapi)
        for src_path in sorted(plan[split_name], key=lambda p: p.name):
            dst_path = out_dir / src_path.name
            if dst_path.exists():
                print(f"[WARN] Target sudah ada, skip copy: {dst_path.name}")
                conflicts += 1
                continue

            shutil.copy2(src_path, dst_path)
            copied_total += 1
            copied_by_split[split_name] += 1
            cls = class_of(src_path)
            if cls in copied_by_class:
                copied_by_class[cls] += 1

    # ==================================================
    # ZIP OUTPUT (IF NOT EXISTS)
    # ==================================================
    zip_created = False
    if not ZIP_PATH.exists():
        zip_folder(OUT_ROOT, ZIP_PATH)
        zip_created = True

    # ==================================================
    # REPORT
    # ==================================================
    print("\n[SUCCESS] DatasetForCvat berhasil dibuat (COPY, non-random).")
    print(f"Sumber : {SRC_DIR}")
    print(f"Output : {OUT_ROOT}")
    print(f"Total dicopy: {copied_total}")
    print(f"Conflict (skip no-overwrite): {conflicts}\n")

    print("Ringkasan per split:")
    for s in ["Train", "Val", "Test"]:
        print(f"  - {s}: {copied_by_split[s]}")

    print("\nRingkasan per class (total across splits):")
    for c in CLASSES:
        print(f"  - {c}: {copied_by_class[c]}")

    print("\nTarget per class:")
    print(f"  - per class: {TARGET_PER_CLASS} (Train {TRAIN_N}, Val {VAL_N}, Test {TEST_N})")

    if zip_created:
        print(f"\n[SUCCESS] Zip dibuat: {ZIP_PATH}")
    else:
        print(f"\n[INFO] Zip tidak dibuat karena sudah ada: {ZIP_PATH}")


if __name__ == "__main__":
    main()
