from pathlib import Path
from shutil import copy2
import sys
import time

DATASET_ROOT = Path.home() / "Documents/Masters/Datasets/Dataset for Augment"
NORMAL_SET = DATASET_ROOT / "Normal Set"
OUTPUT_DIR = NORMAL_SET / "Subjects"

SOURCE_SETS = [
    NORMAL_SET / "TD_RGB_E_Set1",
    NORMAL_SET / "TD_RGB_E_Set2",
    NORMAL_SET / "TD_RGB_E_Set3",
    NORMAL_SET / "TD_RGB_E_Set4",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_sort_key(path):
    parts = []
    for part in path.name.split():
        parts.append(int(part) if part.isdigit() else part.lower())
    return parts


def get_images(folder):
    return sorted(
        [
            file for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_sort_key
    )


def progress_bar(current, total, start_time):
    width = 40
    percentage = current / total
    filled = int(width * percentage)

    bar = "█" * filled + "░" * (width - filled)

    elapsed = time.time() - start_time
    speed = current / elapsed if elapsed > 0 else 0
    remaining = (total - current) / speed if speed > 0 else 0

    sys.stdout.write(
        f"\rCopying subjects: |{bar}| "
        f"{current}/{total} "
        f"({percentage * 100:.1f}%) "
        f"ETA: {remaining:.1f}s"
    )
    sys.stdout.flush()


def main():
    subject_folders = []

    for source_set in SOURCE_SETS:
        if not source_set.exists():
            print(f"Warning: {source_set} does not exist.")
            continue

        for subject_folder in source_set.iterdir():
            if subject_folder.is_dir():
                subject_folders.append(subject_folder)

    subject_folders.sort(
        key=lambda folder: int(folder.name) if folder.name.isdigit() else folder.name
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_count = 0
    start_time = time.time()

    print(f"Dataset: {DATASET_ROOT}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Subjects found: {len(subject_folders)}")
    print()

    total = len(subject_folders)

    for current, subject_folder in enumerate(subject_folders, start=1):
        images = get_images(subject_folder)

        if len(images) < 5:
            print(f"\nWarning: {subject_folder} has only {len(images)} images. Skipping.")
            skipped_count += 1
            progress_bar(current, total, start_time)
            continue

        images_to_copy = images[:-1]

        output_subject_folder = OUTPUT_DIR / subject_folder.name
        output_subject_folder.mkdir(parents=True, exist_ok=True)

        for image in images_to_copy:
            copy2(image, output_subject_folder / image.name)
            copied_count += 1

        progress_bar(current, total, start_time)

    print("\n\nFinished!")
    print(f"Subjects processed: {len(subject_folders) - skipped_count}")
    print(f"Images copied: {copied_count}")
    print(f"Subjects skipped: {skipped_count}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
