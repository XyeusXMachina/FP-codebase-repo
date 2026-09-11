import os
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


DATASET_DIR = Path("Training")
RESULTS_DIR = DATASET_DIR / "results" / "resnet50_loso"

SEED = 456

IMAGE_SIZE = 224
BATCH_SIZE = 32

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 50
PATIENCE = 7

# Resume behavior:
# - A fold with all final result files is skipped.
# - An interrupted fold resumes from training_checkpoint.pth when available.
# - If an older interrupted run has only logs and no checkpoint, that fold restarts.

VAL_RATIO = 0.20

NUM_CLASSES = 2

CLASS_NAMES = {
    "normal": 0,
    "palsy": 1
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rank_info():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def discover_subjects():

    subjects = []

    for class_name, label in CLASS_NAMES.items():

        class_dir = DATASET_DIR / "train" / class_name

        if not class_dir.exists():
            class_dir = DATASET_DIR / "test" / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Could not find directory for {class_name}"
            )

        search_dirs = [
            DATASET_DIR / "train" / class_name,
            DATASET_DIR / "test" / class_name
        ]

        for base_dir in search_dirs:

            if not base_dir.exists():
                continue

            for subject_dir in sorted(base_dir.iterdir()):

                if not subject_dir.is_dir():
                    continue

                images = []

                for ext in ["*.jpg", "*.jpeg", "*.png"]:

                    images.extend(
                        subject_dir.glob(ext)
                    )

                images = sorted(images)

                if len(images) == 0:
                    continue

                subjects.append({
                    "subject_id": subject_dir.name,
                    "class_name": class_name,
                    "label": label,
                    "path": subject_dir,
                    "images": images
                })

    unique = {}

    for subject in subjects:

        key = (
            subject["class_name"],
            subject["subject_id"]
        )

        unique[key] = subject

    subjects = list(unique.values())

    subjects = sorted(
        subjects,
        key=lambda x: (
            x["label"],
            int(x["subject_id"])
            if x["subject_id"].isdigit()
            else x["subject_id"]
        )
    )

    return subjects


class MeshDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None
    ):

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label, subject_id = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, subject_id


def make_samples(subjects):

    samples = []

    for subject in subjects:

        for image_path in subject["images"]:

            samples.append(
                (
                    image_path,
                    subject["label"],
                    subject["subject_id"]
                )
            )

    return samples


def split_validation(
    subjects,
    fold_subject,
    seed
):

    remaining = [
        s for s in subjects
        if not (
            s["class_name"] == fold_subject["class_name"]
            and s["subject_id"] == fold_subject["subject_id"]
        )
    ]

    rng = random.Random(seed)

    normal = [
        s for s in remaining
        if s["label"] == 0
    ]

    palsy = [
        s for s in remaining
        if s["label"] == 1
    ]

    rng.shuffle(normal)
    rng.shuffle(palsy)

    normal_val_count = max(
        1,
        round(len(normal) * VAL_RATIO)
    )

    palsy_val_count = max(
        1,
        round(len(palsy) * VAL_RATIO)
    )

    val_subjects = (
        normal[:normal_val_count]
        +
        palsy[:palsy_val_count]
    )

    val_ids = {
        (
            s["class_name"],
            s["subject_id"]
        )
        for s in val_subjects
    }

    train_subjects = [
        s for s in remaining
        if (
            s["class_name"],
            s["subject_id"]
        ) not in val_ids
    ]

    return train_subjects, val_subjects


def get_transforms():

    train_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.RandomHorizontalFlip(
            p=0.5
        ),
        transforms.RandomRotation(
            10
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    return train_transform, eval_transform


def build_model(device):

    weights = models.ResNet50_Weights.IMAGENET1K_V2

    model = models.resnet50(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model = model.to(device)

    return model


def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ) if len(np.unique(y_true)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def find_threshold(
    y_true,
    probabilities
):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities
    )

    j = tpr - fpr

    valid = np.isfinite(thresholds)

    index = np.argmax(
        np.where(
            valid,
            j,
            -np.inf
        )
    )

    return float(thresholds[index])


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device
):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels, _ in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=True
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        running_loss / total,
        correct / total
    )


@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0
    total = 0

    labels_all = []
    probabilities_all = []
    subjects_all = []

    for images, labels, subjects in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=True
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[:, 1]

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        total += labels.size(0)

        labels_all.extend(
            labels.cpu().numpy()
        )

        probabilities_all.extend(
            probabilities.cpu().numpy()
        )

        subjects_all.extend(
            subjects
        )

    labels_all = np.array(
        labels_all
    )

    probabilities_all = np.array(
        probabilities_all
    )

    return (
        running_loss / total,
        labels_all,
        probabilities_all,
        subjects_all
    )


def aggregate_subject_predictions(
    labels,
    probabilities,
    subjects
):

    df = pd.DataFrame({
        "subject_id": subjects,
        "label": labels,
        "probability": probabilities
    })

    result = (
        df.groupby("subject_id")
        .agg(
            label=("label", "first"),
            probability=("probability", "mean"),
            image_count=("probability", "count")
        )
        .reset_index()
    )

    return result



def fold_is_complete(fold_dir):
    required_files = [
        fold_dir / "test_subject_prediction.csv",
        fold_dir / "training_history.csv",
        fold_dir / "metrics.csv",
        fold_dir / "best_model.pth",
        fold_dir / "config.json",
    ]
    return all(path.exists() for path in required_files)


def get_fold_status(fold_number):
    fold_dir = RESULTS_DIR / f"fold_{fold_number:03d}"

    if not fold_dir.exists():
        return "not_started"

    if fold_is_complete(fold_dir):
        return "complete"

    checkpoint = fold_dir / "training_checkpoint.pth"

    if checkpoint.exists():
        return "resumable"

    partial_log = fold_dir / "training_history_partial.csv"

    if partial_log.exists():
        return "incomplete_log_only"

    if any(fold_dir.iterdir()):
        return "incomplete"

    return "empty"


def scan_fold_status(total_folds):
    status = {
        "complete": [],
        "resumable": [],
        "incomplete_log_only": [],
        "incomplete": [],
        "empty": [],
        "not_started": [],
    }

    for fold_number in range(1, total_folds + 1):
        status[get_fold_status(fold_number)].append(fold_number)

    return status


def print_fold_status(status, total_folds):
    complete = status["complete"]
    resumable = status["resumable"]
    incomplete_log_only = status["incomplete_log_only"]
    incomplete = status["incomplete"]
    empty = status["empty"]
    not_started = status["not_started"]

    remaining = (
        len(resumable)
        + len(incomplete_log_only)
        + len(incomplete)
        + len(empty)
        + len(not_started)
    )

    print("\n" + "=" * 70)
    print("LOSO FOLD STATUS")
    print("=" * 70)
    print(f"Expected folds : {total_folds}")
    print(f"Completed      : {len(complete)}")
    print(f"Resumable      : {len(resumable)}")
    print(f"Incomplete log : {len(incomplete_log_only)}")
    print(f"Incomplete     : {len(incomplete)}")
    print(f"Empty          : {len(empty)}")
    print(f"Not started    : {len(not_started)}")
    print(f"Remaining      : {remaining}")
    print("=" * 70)

    def show(label, folds):
        if folds:
            print(f"{label}:")
            print(" ".join(f"{n:03d}" for n in folds))
        else:
            print(f"{label}: none")
        print()

    show("Completed", complete)
    show("Resumable from checkpoint", resumable)
    show("Incomplete with log only", incomplete_log_only)
    show("Incomplete", incomplete)
    show("Empty", empty)
    show("Not started", not_started)

    print("=" * 70)
    print()


def load_completed_fold(fold_dir):
    prediction_file = fold_dir / "test_subject_prediction.csv"

    if prediction_file.exists():
        print(f"Resuming from saved result: {fold_dir}")
        return pd.read_csv(prediction_file)

    return None


def save_epoch_checkpoint(
    fold_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_loss,
    patience_counter,
    best_state,
    history,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "best_state": best_state,
        "history": history,
    }

    torch.save(
        checkpoint,
        fold_dir / "training_checkpoint.pth"
    )

    pd.DataFrame(history).to_csv(
        fold_dir / "training_history_partial.csv",
        index=False
    )


def load_epoch_checkpoint(
    fold_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
):
    checkpoint_file = fold_dir / "training_checkpoint.pth"

    if not checkpoint_file.exists():
        return None

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def train_fold(
    fold_number,
    fold_subject,
    all_subjects,
    device
):

    fold_dir = (
        RESULTS_DIR /
        f"fold_{fold_number:03d}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fold_status = get_fold_status(fold_number)

    if fold_status == "complete":
        print(
            f"Fold {fold_number:03d} is already complete. "
            f"Skipping training."
        )
        return load_completed_fold(fold_dir)

    if fold_status == "resumable":
        print(
            f"Fold {fold_number:03d} has a training checkpoint. "
            f"Resuming from the saved epoch."
        )
    elif fold_status == "incomplete_log_only":
        print(
            f"Fold {fold_number:03d} has a partial log but no "
            f"checkpoint. This fold will restart from Epoch 1."
        )
    elif fold_status == "empty":
        print(
            f"Fold {fold_number:03d} exists but is empty. "
            f"Starting from Epoch 1."
        )
    elif fold_status == "not_started":
        print(
            f"Fold {fold_number:03d} has not started. "
            f"Starting from Epoch 1."
        )
    else:
        print(
            f"Fold {fold_number:03d} is incomplete. "
            f"Starting from Epoch 1 unless a checkpoint is found."
        )

    train_subjects, val_subjects = split_validation(
        all_subjects,
        fold_subject,
        SEED + fold_number
    )

    train_samples = make_samples(
        train_subjects
    )

    val_samples = make_samples(
        val_subjects
    )

    test_samples = make_samples(
        [fold_subject]
    )

    train_transform, eval_transform = (
        get_transforms()
    )

    train_dataset = MeshDataset(
        train_samples,
        train_transform
    )

    val_dataset = MeshDataset(
        val_samples,
        eval_transform
    )

    test_dataset = MeshDataset(
        test_samples,
        eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = build_model(
        device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []
    start_epoch = 1

    checkpoint = load_epoch_checkpoint(
        fold_dir,
        model,
        optimizer,
        scheduler,
        scaler,
        device
    )

    if checkpoint is not None:
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        patience_counter = checkpoint["patience_counter"]
        best_state = checkpoint["best_state"]
        history = checkpoint["history"]

        print(
            f"Resuming Fold {fold_number:03d} "
            f"from Epoch {checkpoint['epoch'] + 1}."
        )
    else:
        partial_history_file = fold_dir / "training_history_partial.csv"

        if partial_history_file.exists():
            partial_history = pd.read_csv(partial_history_file)
            if len(partial_history) > 0:
                print(
                    f"Found partial training log for Fold "
                    f"{fold_number:03d}, but no model checkpoint. "
                    f"Training will restart this fold from Epoch 1 "
                    f"because model weights cannot be recovered."
                )

    start_time = time.time()

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1
    ):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device
        )

        val_loss, val_y, val_p, val_subjects = (
            evaluate_loader(
                model,
                val_loader,
                criterion,
                device
            )
        )

        val_subject = aggregate_subject_predictions(
            val_y,
            val_p,
            val_subjects
        )

        val_subject_metrics = calculate_metrics(
            val_subject["label"].values,
            val_subject["probability"].values,
            0.5
        )

        scheduler.step(
            val_loss
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_subject_metrics["accuracy"],
            "validation_sensitivity": val_subject_metrics["sensitivity"],
            "validation_specificity": val_subject_metrics["specificity"],
            "validation_f1": val_subject_metrics["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"]
        }

        history.append(row)

        print(
            f"Fold {fold_number:03d} | "
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Subject Acc: "
            f"{val_subject_metrics['accuracy'] * 100:.2f}%"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        save_epoch_checkpoint(
            fold_dir,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_loss,
            patience_counter,
            best_state,
            history
        )

        if patience_counter >= PATIENCE:

            print(
                f"Fold {fold_number:03d} "
                f"early stopping."
            )

            break

    training_time = (
        time.time() - start_time
    )

    model.load_state_dict(
        best_state
    )

    model.to(device)

    _, val_y, val_p, val_subjects = evaluate_loader(
        model,
        val_loader,
        criterion,
        device
    )

    val_subject = aggregate_subject_predictions(
        val_y,
        val_p,
        val_subjects
    )

    threshold = find_threshold(
        val_subject["label"].values,
        val_subject["probability"].values
    )

    _, test_y, test_p, test_subjects = evaluate_loader(
        model,
        test_loader,
        criterion,
        device
    )

    test_subject = aggregate_subject_predictions(
        test_y,
        test_p,
        test_subjects
    )

    test_metrics = calculate_metrics(
        test_subject["label"].values,
        test_subject["probability"].values,
        threshold
    )

    test_subject["fold"] = fold_number
    test_subject["threshold"] = threshold
    test_subject["predicted_label"] = (
        test_subject["probability"]
        >= threshold
    ).astype(int)

    test_subject.to_csv(
        fold_dir /
        "test_subject_prediction.csv",
        index=False
    )

    pd.DataFrame(
        history
    ).to_csv(
        fold_dir /
        "training_history.csv",
        index=False
    )

    pd.DataFrame(
        [test_metrics]
    ).to_csv(
        fold_dir /
        "metrics.csv",
        index=False
    )

    checkpoint_file = fold_dir / "training_checkpoint.pth"
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    partial_history_file = fold_dir / "training_history_partial.csv"
    if partial_history_file.exists():
        partial_history_file.unlink()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "fold": fold_number,
            "held_out_subject": fold_subject["subject_id"],
            "held_out_class": fold_subject["class_name"],
            "threshold": threshold
        },
        fold_dir /
        "best_model.pth"
    )

    config = {
        "fold": fold_number,
        "seed": SEED,
        "model": "ResNet50",
        "pretrained": True,
        "input_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "early_stopping_patience": PATIENCE,
        "validation_ratio": VAL_RATIO,
        "held_out_subject": fold_subject["subject_id"],
        "held_out_class": fold_subject["class_name"],
        "train_subject_count": len(train_subjects),
        "validation_subject_count": len(val_subjects),
        "test_subject_count": 1,
        "optimal_threshold": threshold,
        "training_time_seconds": training_time
    }

    with open(
        fold_dir /
        "config.json",
        "w"
    ) as f:
        json.dump(
            config,
            f,
            indent=4
        )

    return test_subject


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-folds",
        type=int,
        default=64
    )

    args = parser.parse_args()

    rank, world_size = get_rank_info()

    device = torch.device(
        f"cuda:{rank}"
        if torch.cuda.is_available()
        else "cpu"
    )

    set_seed(
        SEED + rank
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_subjects = discover_subjects()

    if len(all_subjects) != 64:

        print(
            f"WARNING: Expected 64 subjects, "
            f"found {len(all_subjects)}."
        )

    all_subjects = all_subjects[
        :args.max_folds
    ]

    total_folds = len(all_subjects)

    print(
        f"Rank {rank}/{world_size} "
        f"using {device}"
    )

    print(
        f"Total subjects: {total_folds}"
    )

    status = scan_fold_status(total_folds)

    if rank == 0:
        print_fold_status(status, total_folds)

    fold_results = []

    for fold_index in range(
        rank,
        len(all_subjects),
        world_size
    ):

        fold_number = fold_index + 1

        fold_subject = all_subjects[
            fold_index
        ]

        fold_dir = RESULTS_DIR / f"fold_{fold_number:03d}"

        print("\n" + "=" * 70)

        print(
            f"FOLD {fold_number}/{len(all_subjects)}"
        )

        print(
            f"Held-out subject: "
            f"{fold_subject['class_name']} "
            f"{fold_subject['subject_id']}"
        )

        print("=" * 70)

        if fold_is_complete(fold_dir):
            print(
                f"FOLD {fold_number:03d}: COMPLETE "
                f"-> loading saved result and skipping."
            )
            result = load_completed_fold(fold_dir)
        else:
            result = train_fold(
                fold_number,
                fold_subject,
                all_subjects,
                device
            )

        fold_results.append(
            result
        )

    print(
        f"\nRank {rank} completed "
        f"{len(fold_results)} folds."
    )

    if rank != 0:
        return

    print(
        "\nChecking for completed fold results..."
    )

    expected_folds = len(all_subjects)
    wait_start = time.time()
    wait_timeout = 86400

    while True:
        completed_folds = sum(
            get_fold_status(fold_number) == "complete"
            for fold_number in range(1, expected_folds + 1)
        )

        print(
            f"Completed fold results: "
            f"{completed_folds}/{expected_folds}"
        )

        if completed_folds >= expected_folds:
            break

        if time.time() - wait_start >= wait_timeout:
            print(
                "Timed out while waiting for all folds. "
                "Aggregating the completed folds only."
            )
            break

        time.sleep(10)

    all_results = []

    for fold_number in range(1, expected_folds + 1):

        fold_dir = RESULTS_DIR / f"fold_{fold_number:03d}"

        prediction_file = (
            fold_dir /
            "test_subject_prediction.csv"
        )

        if prediction_file.exists():

            df = pd.read_csv(
                prediction_file
            )

            all_results.append(
                df
            )

    if len(all_results) == 0:

        raise RuntimeError(
            "No fold results were found."
        )

    if len(all_results) < expected_folds:
        print(
            f"WARNING: Only {len(all_results)}/{expected_folds} "
            f"fold results are available. "
            f"Aggregating available completed folds."
        )

    predictions = pd.concat(
        all_results,
        ignore_index=True
    )

    predictions.to_csv(
        RESULTS_DIR /
        "loso_subject_predictions.csv",
        index=False
    )

    y_true = predictions[
        "label"
    ].astype(int).values

    probabilities = predictions[
        "probability"
    ].astype(float).values

    thresholds = predictions[
        "threshold"
    ].astype(float).values

    y_pred = np.array([
        int(
            p >= t
        )
        for p, t in zip(
            probabilities,
            thresholds
        )
    ])

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0
    )

    metrics = {
        "subjects": len(predictions),
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    pd.DataFrame(
        [metrics]
    ).to_csv(
        RESULTS_DIR /
        "overall_loso_metrics.csv",
        index=False
    )

    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Normal",
            "Palsy"
        ],
        yticklabels=[
            "Normal",
            "Palsy"
        ]
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        "LOSO Subject-Level Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "loso_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "LOSO Subject-Level ROC Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "loso_roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    summary = {
        "model": "ResNet50",
        "representation": "White facial mesh on black background",
        "seed": SEED,
        "folds": len(predictions),
        "subjects": len(predictions),
        "metrics": metrics
    }

    with open(
        RESULTS_DIR /
        "loso_summary.json",
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    print("\n" + "=" * 70)
    print("LOSO COMPLETE")
    print("=" * 70)

    print(
        f"Subjects:     {metrics['subjects']}"
    )

    print(
        f"Accuracy:     {metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision:    {metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Sensitivity:  {metrics['sensitivity'] * 100:.2f}%"
    )

    print(
        f"Specificity:  {metrics['specificity'] * 100:.2f}%"
    )

    print(
        f"F1:           {metrics['f1'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC:      {metrics['roc_auc']:.4f}"
    )

    print(
        f"TN: {tn} | FP: {fp} | "
        f"FN: {fn} | TP: {tp}"
    )

    print(
        f"\nResults saved to:\n"
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
