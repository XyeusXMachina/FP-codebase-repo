from pathlib import Path
import random
import shutil


BASE_DIR = Path(
    "/home/cai/Documents/Masters/Datasets/Dataset for Augment"
)

NORMAL_SOURCE = BASE_DIR / "dataset" / "normal"
PALSY_SOURCE = BASE_DIR / "Palsy Set" / "mesh" / "Subjects"

TRAINING_DIR = BASE_DIR / "Training"

TRAIN_DIR = TRAINING_DIR / "train"
TEST_DIR = TRAINING_DIR / "test"

NORMAL_TRAIN_DIR = TRAIN_DIR / "normal"
NORMAL_TEST_DIR = TEST_DIR / "normal"

PALSY_TRAIN_DIR = TRAIN_DIR / "palsy"
PALSY_TEST_DIR = TEST_DIR / "palsy"


SEED = 42
IMAGES_PER_SUBJECT = 64
NORMAL_SUBJECTS = 32
TRAIN_RATIO = 0.80

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


def get_subjects(source_dir):
    return sorted(
        [
            directory
            for directory in source_dir.iterdir()
            if directory.is_dir()
        ],
        key=lambda path: path.name,
    )


def get_images(subject_dir):
    return sorted(
        [
            image
            for image in subject_dir.rglob("*")
            if image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path),
    )


def select_images(images, count, rng):
    if len(images) <= count:
        return images.copy()

    return rng.sample(images, count)


def split_subjects(subjects, train_ratio, rng):
    subjects = subjects.copy()
    rng.shuffle(subjects)

    train_count = int(len(subjects) * train_ratio)

    train_subjects = subjects[:train_count]
    test_subjects = subjects[train_count:]

    return train_subjects, test_subjects


def copy_images(subject_images, destination_root, subject_name):
    destination_subject = destination_root / subject_name
    destination_subject.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(subject_images, start=1):
        destination_path = (
            destination_subject / f"image_{index:03d}.png"
        )

        shutil.copy2(image_path, destination_path)


def prepare_class(
    subjects,
    destination_train,
    destination_test,
    class_name,
    rng,
):
    train_subjects, test_subjects = split_subjects(
        subjects,
        TRAIN_RATIO,
        rng,
    )

    train_images = 0
    test_images = 0

    print()
    print(f"{class_name.upper()} SUBJECT SPLIT")
    print("-" * 50)
    print(f"Training subjects: {len(train_subjects)}")
    print(f"Testing subjects:  {len(test_subjects)}")

    for subject in train_subjects:
        images = get_images(subject)
        selected = select_images(
            images,
            IMAGES_PER_SUBJECT,
            rng,
        )

        copy_images(
            selected,
            destination_train,
            subject.name,
        )

        train_images += len(selected)

    for subject in test_subjects:
        images = get_images(subject)
        selected = select_images(
            images,
            IMAGES_PER_SUBJECT,
            rng,
        )

        copy_images(
            selected,
            destination_test,
            subject.name,
        )

        test_images += len(selected)

    print(f"Training images: {train_images}")
    print(f"Testing images:  {test_images}")

    return (
        len(train_subjects),
        len(test_subjects),
        train_images,
        test_images,
    )


def main():
    rng = random.Random(SEED)

    print("=" * 60)
    print("FACIAL PARALYSIS TRAINING DATASET PREPARATION")
    print("=" * 60)

    print()
    print(f"Base directory:")
    print(BASE_DIR)

    print()
    print("Configuration")
    print("-" * 60)
    print(f"Random seed:          {SEED}")
    print(f"Images per subject:   {IMAGES_PER_SUBJECT}")
    print(f"Normal subjects:      {NORMAL_SUBJECTS}")
    print(f"Train ratio:          {TRAIN_RATIO:.0%}")
    print()

    if not NORMAL_SOURCE.exists():
        raise FileNotFoundError(
            f"Normal dataset not found:\n{NORMAL_SOURCE}"
        )

    if not PALSY_SOURCE.exists():
        raise FileNotFoundError(
            f"Palsy dataset not found:\n{PALSY_SOURCE}"
        )

    normal_subjects = get_subjects(NORMAL_SOURCE)
    palsy_subjects = get_subjects(PALSY_SOURCE)

    print(f"Normal subjects available: {len(normal_subjects)}")
    print(f"Palsy subjects available:  {len(palsy_subjects)}")

    if len(normal_subjects) < NORMAL_SUBJECTS:
        raise ValueError(
            f"Only {len(normal_subjects)} normal subjects found. "
            f"Need at least {NORMAL_SUBJECTS}."
        )

    if len(palsy_subjects) == 0:
        raise ValueError("No palsy subjects found.")

    rng.shuffle(normal_subjects)

    selected_normal_subjects = normal_subjects[
        :NORMAL_SUBJECTS
    ]

    print()
    print(
        f"Selected {len(selected_normal_subjects)} "
        f"Normal subjects from AFLFP."
    )

    if TRAINING_DIR.exists():
        print()
        print("Removing existing Training directory...")
        shutil.rmtree(TRAINING_DIR)

    NORMAL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    NORMAL_TEST_DIR.mkdir(parents=True, exist_ok=True)

    PALSY_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    PALSY_TEST_DIR.mkdir(parents=True, exist_ok=True)

    normal_stats = prepare_class(
        selected_normal_subjects,
        NORMAL_TRAIN_DIR,
        NORMAL_TEST_DIR,
        "Normal",
        rng,
    )

    palsy_stats = prepare_class(
        palsy_subjects,
        PALSY_TRAIN_DIR,
        PALSY_TEST_DIR,
        "Palsy",
        rng,
    )

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print()
    print("NORMAL")
    print(f"  Train subjects: {normal_stats[0]}")
    print(f"  Test subjects:  {normal_stats[1]}")
    print(f"  Train images:   {normal_stats[2]}")
    print(f"  Test images:    {normal_stats[3]}")
    print(f"  Total images:   {normal_stats[2] + normal_stats[3]}")

    print()
    print("PALSY")
    print(f"  Train subjects: {palsy_stats[0]}")
    print(f"  Test subjects:  {palsy_stats[1]}")
    print(f"  Train images:   {palsy_stats[2]}")
    print(f"  Test images:    {palsy_stats[3]}")
    print(f"  Total images:   {palsy_stats[2] + palsy_stats[3]}")

    print()
    print("OUTPUT")
    print(TRAINING_DIR)

    print()
    print("Dataset structure:")
    print()
    print("Training/")
    print("├── train/")
    print("│   ├── normal/")
    print("│   └── palsy/")
    print("└── test/")
    print("    ├── normal/")
    print("    └── palsy/")


if __name__ == "__main__":
    main()