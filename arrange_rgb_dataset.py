from pathlib import Path
import shutil
import csv
import random
import re

BASE_DIR = Path(
    "/home/cai/Documents/Masters/Datasets/Dataset for Augment"
)

LOSO_DIR = (
    BASE_DIR
    / "Training"
    / "results"
    / "resnet50_loso"
)

OUTPUT_DIR = BASE_DIR / "Training_RGB"

NORMAL_SOURCE = (
    BASE_DIR
    / "Normal Set"
    / "Subjects"
)

PALSY_SOURCES = [
    BASE_DIR / "Palsy Set" / "Image" / "Image",
    BASE_DIR / "Palsy Set" / "Image2" / "Image2",
    BASE_DIR / "Palsy Set" / "Image3" / "Image3",
    BASE_DIR / "Palsy Set" / "Image4" / "Image4",
]

MAX_IMAGES_PER_SUBJECT = 64
SEED = 42

NORMAL_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".JPG",
    ".JPEG"
}

PALSY_EXTENSIONS = {
    ".bmp",
    ".BMP"
}


def get_subject_number(subject_id):

    numbers = re.findall(
        r"\d+",
        str(subject_id)
    )

    if not numbers:
        return None

    return int(numbers[-1])


def read_loso_subjects():

    subjects = []

    fold_dirs = sorted(
        LOSO_DIR.glob("fold_*"),
        key=lambda x: int(
            x.name.split("_")[-1]
        )
    )

    if len(fold_dirs) != 64:
        raise RuntimeError(
            f"Expected 64 LOSO folds, "
            f"found {len(fold_dirs)}."
        )

    for fold_dir in fold_dirs:

        prediction_file = (
            fold_dir
            / "test_subject_prediction.csv"
        )

        if not prediction_file.exists():
            raise RuntimeError(
                f"Missing file:\n"
                f"{prediction_file}"
            )

        with open(
            prediction_file,
            "r",
            newline=""
        ) as f:

            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise RuntimeError(
                f"No data in:\n"
                f"{prediction_file}"
            )

        required = {
            "subject_id",
            "label",
            "fold"
        }

        missing = (
            required
            - set(rows[0].keys())
        )

        if missing:
            raise RuntimeError(
                f"Missing columns {missing} "
                f"in {prediction_file}"
            )

        row = rows[0]

        subjects.append({
            "subject_id":
                row["subject_id"].strip(),

            "label":
                int(row["label"]),

            "fold":
                int(row["fold"])
        })

    return subjects


def find_normal_subject(subject_number):

    subject_dir = (
        NORMAL_SOURCE
        / str(subject_number)
    )

    if subject_dir.exists():
        return subject_dir

    return None


def find_palsy_subject(subject_number):

    for source in PALSY_SOURCES:

        subject_dir = (
            source
            / str(subject_number)
        )

        if subject_dir.exists():
            return subject_dir

    return None


def collect_images(
    subject_dir,
    extensions
):

    images = []

    for path in subject_dir.iterdir():

        if not path.is_file():
            continue

        if path.suffix in extensions:
            images.append(path)

    images.sort()

    return images


def select_images(
    images,
    subject_id
):

    if len(images) <= MAX_IMAGES_PER_SUBJECT:
        return images

    subject_number = (
        get_subject_number(subject_id)
    )

    rng = random.Random(
        SEED + subject_number
    )

    selected = rng.sample(
        images,
        MAX_IMAGES_PER_SUBJECT
    )

    selected.sort()

    return selected


def main():

    print("=" * 70)
    print("RGB DATASET ARRANGEMENT")
    print("=" * 70)

    print("\nReading existing LOSO subjects...")

    subjects = read_loso_subjects()

    unique_subjects = {}

    for item in subjects:

        subject_id = item["subject_id"]

        if subject_id in unique_subjects:
            raise RuntimeError(
                f"Duplicate subject: "
                f"{subject_id}"
            )

        unique_subjects[
            subject_id
        ] = item

    if len(unique_subjects) != 64:

        raise RuntimeError(
            f"Expected 64 subjects, "
            f"found {len(unique_subjects)}."
        )

    normal_subjects = [
        x for x in unique_subjects.values()
        if x["label"] == 0
    ]

    palsy_subjects = [
        x for x in unique_subjects.values()
        if x["label"] == 1
    ]

    normal_subjects.sort(
        key=lambda x: x["subject_id"]
    )

    palsy_subjects.sort(
        key=lambda x:
        get_subject_number(
            x["subject_id"]
        )
    )

    print(
        f"\nNormal subjects: "
        f"{len(normal_subjects)}"
    )

    print(
        f"Palsy subjects:  "
        f"{len(palsy_subjects)}"
    )

    print(
        f"Total subjects:  "
        f"{len(unique_subjects)}"
    )

    if len(normal_subjects) != 32:
        raise RuntimeError(
            "Expected 32 Normal subjects."
        )

    if len(palsy_subjects) != 32:
        raise RuntimeError(
            "Expected 32 Palsy subjects."
        )

    print("\nLocating original RGB data...")

    locations = {}

    for item in unique_subjects.values():

        subject_id = item["subject_id"]
        label = item["label"]

        subject_number = (
            get_subject_number(subject_id)
        )

        if label == 0:

            source = find_normal_subject(
                subject_number
            )

        else:

            source = find_palsy_subject(
                subject_number
            )

        if source is None:

            class_name = (
                "Normal"
                if label == 0
                else "Palsy"
            )

            raise RuntimeError(
                f"\nCould not locate "
                f"{class_name} RGB subject "
                f"{subject_id}"
            )

        locations[
            subject_id
        ] = source

        print(
            f"{subject_id:12s} -> "
            f"{source}"
        )

    print(
        "\nAll 64 original RGB "
        "subjects located."
    )

    if OUTPUT_DIR.exists():

        print(
            f"\nRemoving existing:\n"
            f"{OUTPUT_DIR}"
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    records = []

    print("\nCopying RGB images...\n")

    ordered_subjects = sorted(
        unique_subjects.values(),
        key=lambda x: (
            x["label"],
            get_subject_number(
                x["subject_id"]
            )
        )
    )

    for item in ordered_subjects:

        subject_id = item["subject_id"]
        label = item["label"]
        fold = item["fold"]

        if label == 0:

            class_name = "normal"

            extensions = (
                NORMAL_EXTENSIONS
            )

        else:

            class_name = "palsy"

            extensions = (
                PALSY_EXTENSIONS
            )

        source = locations[
            subject_id
        ]

        images = collect_images(
            source,
            extensions
        )

        if not images:

            raise RuntimeError(
                f"No original RGB images "
                f"found for {subject_id}"
            )

        original_count = len(images)

        selected_images = select_images(
            images,
            subject_id
        )

        output_subject = (
            OUTPUT_DIR
            / class_name
            / subject_id
        )

        output_subject.mkdir(
            parents=True,
            exist_ok=True
        )

        for index, image_path in enumerate(
            selected_images,
            start=1
        ):

            destination = (
                output_subject
                / f"{index:04d}"
                f"{image_path.suffix.lower()}"
            )

            shutil.copy2(
                image_path,
                destination
            )

            records.append({

                "subject_id":
                    subject_id,

                "class":
                    class_name,

                "label":
                    label,

                "fold":
                    fold,

                "image_index":
                    index,

                "original_image_count":
                    original_count,

                "selected_image_count":
                    len(selected_images),

                "source_path":
                    str(image_path),

                "image_path":
                    str(destination)
            })

        print(
            f"{subject_id:12s} | "
            f"{class_name:6s} | "
            f"{original_count:5d} original | "
            f"{len(selected_images):2d} selected"
        )

    manifest_path = (
        OUTPUT_DIR
        / "dataset_manifest.csv"
    )

    with open(
        manifest_path,
        "w",
        newline=""
    ) as f:

        fieldnames = [
            "subject_id",
            "class",
            "label",
            "fold",
            "image_index",
            "original_image_count",
            "selected_image_count",
            "source_path",
            "image_path"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

    normal_images = sum(
        1
        for r in records
        if r["class"] == "normal"
    )

    palsy_images = sum(
        1
        for r in records
        if r["class"] == "palsy"
    )

    print("\n" + "=" * 70)
    print("RGB DATASET COMPLETE")
    print("=" * 70)

    print(
        f"Normal subjects: 32"
    )

    print(
        f"Palsy subjects:  32"
    )

    print(
        f"Total subjects:  64"
    )

    print()

    print(
        f"Normal images:   "
        f"{normal_images}"
    )

    print(
        f"Palsy images:    "
        f"{palsy_images}"
    )

    print(
        f"Total images:    "
        f"{len(records)}"
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        f"\nManifest:\n"
        f"{manifest_path}"
    )

    print("\n" + "=" * 70)
    print("READY FOR RGB LOSO TRAINING")
    print("=" * 70)


if __name__ == "__main__":
    main()