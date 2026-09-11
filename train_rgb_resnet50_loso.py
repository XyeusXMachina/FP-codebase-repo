from pathlib import Path
import os
import json
import random
import time
import csv
import copy

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(
    "/home/cai/Documents/Masters/Datasets/Dataset for Augment"
)

DATASET_DIR = BASE_DIR / "Training_RGB"

MANIFEST_PATH = (
    DATASET_DIR / "dataset_manifest.csv"
)

OUTPUT_DIR = (
    DATASET_DIR
    / "results"
    / "resnet50_loso"
)

SEED = 456

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 8

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 50

PATIENCE = 7

VAL_RATIO = 0.20

CLASS_NAMES = [
    "Normal",
    "Palsy"
]


def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


class RGBDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        if self.transform:

            image = self.transform(
                image
            )

        label = int(
            row["label"]
        )

        return image, label


def get_transforms():

    train_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
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

    return (
        train_transform,
        eval_transform
    )


def create_model():

    weights = (
        models.ResNet50_Weights.IMAGENET1K_V2
    )

    model = models.resnet50(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    return model


def split_subjects(
    dataframe,
    held_out_subject,
    seed
):

    remaining = dataframe[
        dataframe["subject_id"]
        != held_out_subject
    ].copy()

    normal_subjects = sorted(
        remaining.loc[
            remaining["label"] == 0,
            "subject_id"
        ].unique()
    )

    palsy_subjects = sorted(
        remaining.loc[
            remaining["label"] == 1,
            "subject_id"
        ].unique()
    )

    rng = random.Random(seed)

    rng.shuffle(
        normal_subjects
    )

    rng.shuffle(
        palsy_subjects
    )

    normal_val_count = max(
        1,
        int(
            len(normal_subjects)
            * VAL_RATIO
        )
    )

    palsy_val_count = max(
        1,
        int(
            len(palsy_subjects)
            * VAL_RATIO
        )
    )

    val_normal = set(
        normal_subjects[
            :normal_val_count
        ]
    )

    val_palsy = set(
        palsy_subjects[
            :palsy_val_count
        ]
    )

    val_subjects = (
        val_normal
        | val_palsy
    )

    train_subjects = (
        set(normal_subjects)
        | set(palsy_subjects)
    ) - val_subjects

    train_df = remaining[
        remaining["subject_id"].isin(
            train_subjects
        )
    ].copy()

    val_df = remaining[
        remaining["subject_id"].isin(
            val_subjects
        )
    ].copy()

    return (
        train_df,
        val_df
    )


def aggregate_subject_probability(
    dataframe,
    probabilities
):

    temp = dataframe[
        ["subject_id", "label"]
    ].copy()

    temp["probability"] = probabilities

    grouped = (
        temp.groupby("subject_id")
        .agg(
            label=("label", "first"),
            probability=(
                "probability",
                "mean"
            ),
            image_count=(
                "probability",
                "count"
            )
        )
        .reset_index()
    )

    return grouped


def find_best_threshold(
    labels,
    probabilities
):

    thresholds = np.linspace(
        0.01,
        0.99,
        199
    )

    best_threshold = 0.5

    best_score = -np.inf

    for threshold in thresholds:

        predictions = (
            probabilities
            >= threshold
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                labels,
                predictions,
                labels=[0, 1]
            ).ravel()
        )

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

        score = (
            sensitivity
            + specificity
            - 1
        )

        if score > best_score:

            best_score = score

            best_threshold = threshold

    return best_threshold


def calculate_metrics(
    labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1]
        ).ravel()
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    sensitivity = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    if len(
        np.unique(labels)
    ) == 2:

        auc = roc_auc_score(
            labels,
            probabilities
        )

        ap = average_precision_score(
            labels,
            probabilities
        )

    else:

        auc = float("nan")

        ap = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": auc,
        "average_precision": ap,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in loader:

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
            dtype=torch.float16
        ):

            outputs = model(
                images
            )

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
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    probabilities = []

    labels_all = []

    for images, labels in loader:

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
            dtype=torch.float16
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        probs = torch.softmax(
            outputs,
            dim=1
        )[:, 1]

        probabilities.extend(
            probs.cpu().numpy()
        )

        labels_all.extend(
            labels.cpu().numpy()
        )

    loss = (
        running_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        loss,
        accuracy,
        np.asarray(probabilities),
        np.asarray(labels_all)
    )


def save_training_curves(
    history,
    output_dir,
    fold
):

    history_df = pd.DataFrame(
        history
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        f"Fold {fold:03d} Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "training_curves.png",
        dpi=300
    )

    plt.close()

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history_df["epoch"],
        history_df["train_accuracy"]
        * 100,
        label="Training Accuracy"
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_accuracy"]
        * 100,
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        f"Fold {fold:03d} Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "accuracy_curves.png",
        dpi=300
    )

    plt.close()

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history_df["epoch"],
        history_df["learning_rate"]
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Learning Rate"
    )

    plt.title(
        f"Fold {fold:03d} Learning Rate"
    )

    plt.yscale(
        "log"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "learning_rate.png",
        dpi=300
    )

    plt.close()


def save_overall_plots(
    results_df,
    output_dir
):

    labels = (
        results_df["label"]
        .astype(int)
        .values
    )

    probabilities = (
        results_df["probability"]
        .astype(float)
        .values
    )

    predictions = (
        results_df["predicted_label"]
        .astype(int)
        .values
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1]
        ).ravel()
    )

    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "RGB ResNet50 LOSO Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        CLASS_NAMES
    )

    plt.yticks(
        [0, 1],
        CLASS_NAMES
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    row_sums = cm.sum(
        axis=1,
        keepdims=True
    )

    normalized_cm = (
        cm / np.maximum(
            row_sums,
            1
        )
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        normalized_cm,
        interpolation="nearest"
    )

    plt.title(
        "Normalized Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        CLASS_NAMES
    )

    plt.yticks(
        [0, 1],
        CLASS_NAMES
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                f"{normalized_cm[i, j] * 100:.1f}%",
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "normalized_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    fpr, tpr, _ = roc_curve(
        labels,
        probabilities
    )

    auc = roc_auc_score(
        labels,
        probabilities
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.4f}"
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
        "RGB ResNet50 LOSO ROC Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "roc_curve.png",
        dpi=300
    )

    plt.close()

    precision, recall, _ = (
        precision_recall_curve(
            labels,
            probabilities
        )
    )

    ap = average_precision_score(
        labels,
        probabilities
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        recall,
        precision,
        label=f"AP = {ap:.4f}"
    )

    plt.xlabel(
        "Recall / Sensitivity"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "RGB ResNet50 Precision-Recall Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "precision_recall_curve.png",
        dpi=300
    )

    plt.close()

    normal_probs = (
        probabilities[
            labels == 0
        ]
    )

    palsy_probs = (
        probabilities[
            labels == 1
        ]
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.hist(
        normal_probs,
        bins=20,
        alpha=0.6,
        label="Normal"
    )

    plt.hist(
        palsy_probs,
        bins=20,
        alpha=0.6,
        label="Palsy"
    )

    plt.axvline(
        0.5,
        linestyle="--",
        label="Threshold 0.5"
    )

    plt.xlabel(
        "Predicted Palsy Probability"
    )

    plt.ylabel(
        "Number of Subjects"
    )

    plt.title(
        "Subject-Level Prediction Probability Distribution"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "prediction_distribution.png",
        dpi=300
    )

    plt.close()

    plt.figure(
        figsize=(12, 6)
    )

    fold_numbers = (
        results_df["fold"]
        .astype(int)
        .values
    )

    fold_accuracy = (
        results_df["accuracy"]
        .astype(float)
        .values
        * 100
    )

    plt.bar(
        fold_numbers,
        fold_accuracy
    )

    plt.axhline(
        fold_accuracy.mean(),
        linestyle="--",
        label=(
            f"Mean = "
            f"{fold_accuracy.mean():.2f}%"
        )
    )

    plt.xlabel(
        "LOSO Fold"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "Accuracy Across LOSO Folds"
    )

    plt.xticks(
        fold_numbers,
        rotation=90
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "fold_accuracy.png",
        dpi=300
    )

    plt.close()

    fold_loss = (
        results_df["best_val_loss"]
        .astype(float)
        .values
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        fold_numbers,
        fold_loss,
        marker="o",
        markersize=3
    )

    plt.xlabel(
        "LOSO Fold"
    )

    plt.ylabel(
        "Best Validation Loss"
    )

    plt.title(
        "Best Validation Loss Across LOSO Folds"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "fold_loss.png",
        dpi=300
    )

    plt.close()

    sensitivity = (
        results_df["sensitivity"]
        .astype(float)
        .values
        * 100
    )

    specificity = (
        results_df["specificity"]
        .astype(float)
        .values
        * 100
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        fold_numbers,
        sensitivity,
        marker="o",
        markersize=3,
        label="Sensitivity"
    )

    plt.plot(
        fold_numbers,
        specificity,
        marker="o",
        markersize=3,
        label="Specificity"
    )

    plt.xlabel(
        "LOSO Fold"
    )

    plt.ylabel(
        "Percentage (%)"
    )

    plt.title(
        "Sensitivity and Specificity Across LOSO Folds"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "fold_sensitivity_specificity.png",
        dpi=300
    )

    plt.close()


def save_training_summary(
    results_df,
    output_dir
):

    metrics = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc"
    ]

    rows = []

    for metric in metrics:

        values = (
            results_df[metric]
            .astype(float)
            .values
        )

        rows.append({
            "metric": metric,
            "mean": np.nanmean(values),
            "std": np.nanstd(
                values,
                ddof=1
            ),
            "min": np.nanmin(values),
            "max": np.nanmax(values)
        })

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        output_dir
        / "training_summary.csv",
        index=False
    )


def main():

    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            0
        )
    )

    rank = int(
        os.environ.get(
            "RANK",
            0
        )
    )

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            1
        )
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        f"cuda:{local_rank}"
    )

    seed_everything(
        SEED + rank
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe = pd.read_csv(
        MANIFEST_PATH
    )

    required_columns = {
        "subject_id",
        "class",
        "label",
        "fold",
        "image_path"
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    subjects = (
        dataframe[
            [
                "subject_id",
                "label",
                "fold"
            ]
        ]
        .drop_duplicates()
        .sort_values("fold")
    )

    if len(subjects) != 64:

        raise RuntimeError(
            f"Expected 64 subjects, "
            f"found {len(subjects)}"
        )

    if set(
        subjects["fold"]
    ) != set(
        range(1, 65)
    ):

        raise RuntimeError(
            "Expected LOSO folds 1 through 64."
        )

    if rank == 0:

        print("=" * 70)
        print("RGB RESNET50 LOSO")
        print("=" * 70)

        print(
            f"Subjects:       {len(subjects)}"
        )

        print(
            f"Images:         {len(dataframe)}"
        )

        print(
            f"GPUs:           {world_size}"
        )

        print(
            f"Input:          Original RGB"
        )

        print(
            f"Architecture:   ResNet50"
        )

        print(
            f"Seed:           {SEED}"
        )

        print(
            f"Batch/GPU:      {BATCH_SIZE}"
        )

        print(
            f"Image size:     "
            f"{IMAGE_SIZE}x{IMAGE_SIZE}"
        )

        print("=" * 70)

    train_transform, eval_transform = (
        get_transforms()
    )

    fold_results = []

    assigned_folds = list(
        range(
            rank + 1,
            65,
            world_size
        )
    )

    print(
        f"GPU {local_rank} assigned folds: "
        f"{assigned_folds}"
    )

    for fold in assigned_folds:

        fold_start = time.time()

        fold_info = subjects[
            subjects["fold"] == fold
        ]

        if len(fold_info) != 1:

            raise RuntimeError(
                f"Fold {fold} does not contain "
                f"exactly one subject."
            )

        held_out_subject = (
            fold_info.iloc[0]["subject_id"]
        )

        held_out_label = int(
            fold_info.iloc[0]["label"]
        )

        fold_dir = (
            OUTPUT_DIR
            / f"fold_{fold:03d}"
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"\nGPU {local_rank} | "
            f"Fold {fold}/64 | "
            f"Test subject: "
            f"{held_out_subject}"
        )

        train_df, val_df = (
            split_subjects(
                dataframe,
                held_out_subject,
                SEED + fold
            )
        )

        test_df = dataframe[
            dataframe["subject_id"]
            == held_out_subject
        ].copy()

        train_dataset = RGBDataset(
            train_df,
            train_transform
        )

        val_dataset = RGBDataset(
            val_df,
            eval_transform
        )

        test_dataset = RGBDataset(
            test_df,
            eval_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(
                NUM_WORKERS > 0
            )
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(
                NUM_WORKERS > 0
            )
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(
                NUM_WORKERS > 0
            )
        )

        model = create_model().to(
            device
        )

        criterion = (
            nn.CrossEntropyLoss()
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2
            )
        )

        scaler = torch.amp.GradScaler(
            "cuda"
        )

        best_val_loss = float(
            "inf"
        )

        best_state = None

        best_epoch = 0

        patience_counter = 0

        history = []

        for epoch in range(
            1,
            MAX_EPOCHS + 1
        ):

            epoch_start = time.time()

            train_loss, train_accuracy = (
                train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    scaler,
                    device
                )
            )

            val_loss, val_accuracy, _, _ = (
                evaluate(
                    model,
                    val_loader,
                    criterion,
                    device
                )
            )

            scheduler.step(
                val_loss
            )

            current_lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            epoch_time = (
                time.time()
                - epoch_start
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy":
                    train_accuracy,
                "val_loss": val_loss,
                "val_accuracy":
                    val_accuracy,
                "learning_rate":
                    current_lr,
                "epoch_seconds":
                    epoch_time
            })

            print(
                f"GPU {local_rank} | "
                f"Fold {fold:03d} | "
                f"Epoch {epoch:02d} | "
                f"Train Loss "
                f"{train_loss:.4f} | "
                f"Train Acc "
                f"{train_accuracy * 100:.2f}% | "
                f"Val Loss "
                f"{val_loss:.4f} | "
                f"Val Acc "
                f"{val_accuracy * 100:.2f}% | "
                f"LR "
                f"{current_lr:.2e}"
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                best_epoch = epoch

                patience_counter = 0

                best_state = copy.deepcopy(
                    model.state_dict()
                )

            else:

                patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    f"GPU {local_rank} | "
                    f"Fold {fold:03d} | "
                    f"Early stopping at "
                    f"epoch {epoch}"
                )

                break

        if best_state is None:

            raise RuntimeError(
                f"No best model for fold {fold}"
            )

        model.load_state_dict(
            best_state
        )

        torch.save(
            best_state,
            fold_dir
            / "best_model.pth"
        )

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            fold_dir
            / "training_history.csv",
            index=False
        )

        save_training_curves(
            history,
            fold_dir,
            fold
        )

        _, _, val_probabilities, _ = (
            evaluate(
                model,
                val_loader,
                criterion,
                device
            )
        )

        val_subject = (
            aggregate_subject_probability(
                val_df,
                val_probabilities
            )
        )

        threshold = find_best_threshold(
            val_subject["label"].values,
            val_subject["probability"].values
        )

        _, _, test_probabilities, _ = (
            evaluate(
                model,
                test_loader,
                criterion,
                device
            )
        )

        test_subject = (
            aggregate_subject_probability(
                test_df,
                test_probabilities
            )
        )

        test_metrics = calculate_metrics(
            test_subject["label"].values,
            test_subject["probability"].values,
            threshold
        )

        test_probability = float(
            test_subject.iloc[0][
                "probability"
            ]
        )

        predicted_label = int(
            test_probability
            >= threshold
        )

        fold_time = (
            time.time()
            - fold_start
        )

        result = {

            "subject_id":
                held_out_subject,

            "label":
                held_out_label,

            "probability":
                test_probability,

            "predicted_label":
                predicted_label,

            "image_count":
                int(
                    test_subject.iloc[0][
                        "image_count"
                    ]
                ),

            "fold":
                fold,

            "threshold":
                float(threshold),

            "accuracy":
                test_metrics["accuracy"],

            "precision":
                test_metrics["precision"],

            "sensitivity":
                test_metrics["sensitivity"],

            "specificity":
                test_metrics["specificity"],

            "f1":
                test_metrics["f1"],

            "roc_auc":
                test_metrics["roc_auc"],

            "average_precision":
                test_metrics[
                    "average_precision"
                ],

            "tn":
                test_metrics["tn"],

            "fp":
                test_metrics["fp"],

            "fn":
                test_metrics["fn"],

            "tp":
                test_metrics["tp"],

            "best_val_loss":
                best_val_loss,

            "best_epoch":
                best_epoch,

            "epochs_completed":
                len(history),

            "training_seconds":
                fold_time,

            "gpu":
                local_rank
        }

        fold_results.append(
            result
        )

        prediction_df = pd.DataFrame([
            {
                "subject_id":
                    held_out_subject,

                "label":
                    held_out_label,

                "probability":
                    test_probability,

                "image_count":
                    result["image_count"],

                "fold":
                    fold,

                "threshold":
                    float(threshold),

                "predicted_label":
                    predicted_label
            }
        ])

        prediction_df.to_csv(
            fold_dir
            / "test_subject_prediction.csv",
            index=False
        )

        metrics_df = pd.DataFrame([
            test_metrics
        ])

        metrics_df.to_csv(
            fold_dir
            / "metrics.csv",
            index=False
        )

        print(
            f"GPU {local_rank} | "
            f"Fold {fold:03d} COMPLETE | "
            f"Subject {held_out_subject} | "
            f"Actual {held_out_label} | "
            f"Predicted {predicted_label} | "
            f"Probability "
            f"{test_probability:.4f} | "
            f"Threshold "
            f"{threshold:.4f}"
        )

    process_results = pd.DataFrame(
        fold_results
    )

    process_results.to_csv(
        OUTPUT_DIR
        / f"worker_{rank:02d}_results.csv",
        index=False
    )

    print(
        f"\nGPU {local_rank} completed "
        f"{len(fold_results)} folds."
    )

    if rank == 0:

        print(
            "\nWaiting for the other "
            "GPU workers to finish..."
        )

        expected_folds = set(
            range(1, 65)
        )

        while True:

            worker_files = list(
                OUTPUT_DIR.glob(
                    "worker_*_results.csv"
                )
            )

            completed_folds = set()

            for worker_file in worker_files:

                worker_df = pd.read_csv(
                    worker_file
                )

                if "fold" in worker_df:

                    completed_folds.update(
                        worker_df["fold"]
                        .astype(int)
                        .tolist()
                    )

            if completed_folds >= expected_folds:

                break

            time.sleep(5)

        all_worker_files = list(
            OUTPUT_DIR.glob(
                "worker_*_results.csv"
            )
        )

        all_results = []

        for worker_file in all_worker_files:

            worker_df = pd.read_csv(
                worker_file
            )

            all_results.append(
                worker_df
            )

        results_df = pd.concat(
            all_results,
            ignore_index=True
        )

        results_df = (
            results_df
            .sort_values("fold")
            .reset_index(drop=True)
        )

        if len(results_df) != 64:

            raise RuntimeError(
                f"Expected 64 fold results, "
                f"found {len(results_df)}."
            )

        if results_df["fold"].nunique() != 64:

            raise RuntimeError(
                "Duplicate or missing LOSO folds."
            )

        results_df.to_csv(
            OUTPUT_DIR
            / "loso_subject_predictions.csv",
            index=False
        )

        labels = (
            results_df["label"]
            .astype(int)
            .values
        )

        probabilities = (
            results_df["probability"]
            .astype(float)
            .values
        )

        predictions = (
            results_df["predicted_label"]
            .astype(int)
            .values
        )

        tn, fp, fn, tp = (
            confusion_matrix(
                labels,
                predictions,
                labels=[0, 1]
            ).ravel()
        )

        accuracy = accuracy_score(
            labels,
            predictions
        )

        precision = precision_score(
            labels,
            predictions,
            zero_division=0
        )

        sensitivity = recall_score(
            labels,
            predictions,
            zero_division=0
        )

        specificity = (
            tn / (tn + fp)
        )

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        auc = roc_auc_score(
            labels,
            probabilities
        )

        ap = average_precision_score(
            labels,
            probabilities
        )

        overall = {

            "subjects":
                len(labels),

            "accuracy":
                accuracy,

            "precision":
                precision,

            "sensitivity":
                sensitivity,

            "specificity":
                specificity,

            "f1":
                f1,

            "roc_auc":
                auc,

            "average_precision":
                ap,

            "tn":
                int(tn),

            "fp":
                int(fp),

            "fn":
                int(fn),

            "tp":
                int(tp)
        }

        pd.DataFrame([
            overall
        ]).to_csv(
            OUTPUT_DIR
            / "overall_loso_metrics.csv",
            index=False
        )

        save_training_summary(
            results_df,
            OUTPUT_DIR
        )

        save_overall_plots(
            results_df,
            OUTPUT_DIR
        )

        config = {

            "model":
                "ResNet50",

            "pretrained":
                True,

            "input":
                "Original RGB",

            "input_size":
                IMAGE_SIZE,

            "subjects":
                64,

            "loso_folds":
                64,

            "batch_size_per_gpu":
                BATCH_SIZE,

            "gpu_count":
                world_size,

            "folds_per_gpu":
                "Independent LOSO folds",

            "epochs":
                MAX_EPOCHS,

            "early_stopping_patience":
                PATIENCE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "optimizer":
                "AdamW",

            "scheduler":
                "ReduceLROnPlateau",

            "augmentation": [
                "RandomHorizontalFlip",
                "RandomRotation(10)",
                "RandomAffine"
            ],

            "normalization":
                "ImageNet",

            "seed":
                SEED,

            "classification":
                "Normal vs Palsy",

            "evaluation":
                "Subject-level LOSO",

            "subject_aggregation":
                "Mean image probability",

            "threshold":
                "Youden J from validation subjects",

            "manifest":
                str(MANIFEST_PATH)
        }

        with open(
            OUTPUT_DIR
            / "experiment_config.json",
            "w"
        ) as f:

            json.dump(
                config,
                f,
                indent=2
            )

        print("\n")
        print("=" * 70)
        print("RGB RESNET50 LOSO COMPLETE")
        print("=" * 70)

        print(
            f"Subjects:     {len(labels)}"
        )

        print(
            f"Accuracy:     "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Precision:    "
            f"{precision * 100:.2f}%"
        )

        print(
            f"Sensitivity:  "
            f"{sensitivity * 100:.2f}%"
        )

        print(
            f"Specificity:  "
            f"{specificity * 100:.2f}%"
        )

        print(
            f"F1:           "
            f"{f1 * 100:.2f}%"
        )

        print(
            f"ROC-AUC:      "
            f"{auc:.4f}"
        )

        print(
            f"Average Prec: "
            f"{ap:.4f}"
        )

        print(
            f"TN: {tn} | "
            f"FP: {fp} | "
            f"FN: {fn} | "
            f"TP: {tp}"
        )

        print("\nResults:")

        print(
            OUTPUT_DIR
        )

        print("\nVisualizations:")

        print(
            "  confusion_matrix.png"
        )

        print(
            "  normalized_confusion_matrix.png"
        )

        print(
            "  roc_curve.png"
        )

        print(
            "  precision_recall_curve.png"
        )

        print(
            "  prediction_distribution.png"
        )

        print(
            "  fold_accuracy.png"
        )

        print(
            "  fold_loss.png"
        )

        print(
            "  fold_sensitivity_specificity.png"
        )

        print(
            "  training_summary.csv"
        )

        print("=" * 70)


if __name__ == "__main__":
    main()