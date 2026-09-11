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
    roc_curve,
    precision_recall_curve,
    classification_report
)

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms, models


BASE_DIR = Path.home() / "Documents" / "Masters" / "Datasets" / "Dataset for Augment"
DATASET_DIR = BASE_DIR / "Training"
RESULTS_DIR = DATASET_DIR / "results" / "resnet50"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7
NUM_WORKERS = 4
SEED = 1000

CLASS_NAMES = ["normal", "palsy"]

CLASS_TO_INDEX = {
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


def setup_ddp():

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(
            backend="nccl"
        )

        torch.cuda.set_device(local_rank)

        return True, rank, world_size, local_rank

    return False, 0, 1, 0


def cleanup_ddp():

    if dist.is_initialized():
        dist.destroy_process_group()


def format_time(seconds):

    seconds = int(max(0, seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"

    if minutes > 0:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"


class MeshDataset(Dataset):

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label, subject = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return (
            image,
            label,
            str(image_path),
            subject
        )


def collect_samples(directory):

    samples = []

    for class_name in CLASS_NAMES:

        class_dir = directory / class_name

        if not class_dir.exists():
            continue

        for subject_dir in sorted(class_dir.iterdir()):

            if not subject_dir.is_dir():
                continue

            subject = subject_dir.name

            image_files = []

            for extension in ["*.png", "*.jpg", "*.jpeg"]:
                image_files.extend(
                    subject_dir.glob(extension)
                )

            for image_path in sorted(image_files):

                samples.append(
                    (
                        image_path,
                        CLASS_TO_INDEX[class_name],
                        subject
                    )
                )

    return samples


def split_subjects(samples, validation_ratio=0.20):

    subjects_by_class = {}

    for class_name in CLASS_NAMES:

        subjects = sorted(
            list(
                set(
                    subject
                    for _, label, subject in samples
                    if label == CLASS_TO_INDEX[class_name]
                )
            )
        )

        rng = random.Random(
            SEED + CLASS_TO_INDEX[class_name]
        )

        rng.shuffle(subjects)

        validation_count = max(
            1,
            int(len(subjects) * validation_ratio)
        )

        validation_subjects = set(
            subjects[:validation_count]
        )

        training_subjects = set(
            subjects[validation_count:]
        )

        subjects_by_class[class_name] = {
            "train": training_subjects,
            "val": validation_subjects
        }

    train_samples = []
    val_samples = []

    for sample in samples:

        image_path, label, subject = sample

        class_name = CLASS_NAMES[label]

        if subject in subjects_by_class[class_name]["val"]:
            val_samples.append(sample)

        else:
            train_samples.append(sample)

    return (
        train_samples,
        val_samples,
        subjects_by_class
    )


def create_transforms():

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

    evaluation_transform = transforms.Compose([
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
        evaluation_transform
    )


def create_model():

    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    return model


def reduce_metrics(
    loss_sum,
    correct,
    total,
    device
):

    values = torch.tensor(
        [
            loss_sum,
            correct,
            total
        ],
        dtype=torch.float64,
        device=device
    )

    if dist.is_initialized():

        dist.all_reduce(
            values,
            op=dist.ReduceOp.SUM
        )

    global_loss_sum = values[0].item()
    global_correct = values[1].item()
    global_total = values[2].item()

    return (
        global_loss_sum / global_total,
        global_correct / global_total
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    rank
):

    model.train()

    loss_sum = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}",
        disable=(rank != 0),
        leave=True
    )

    for images, labels, _, _ in progress:

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

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        batch_size = labels.size(0)

        loss_sum += (
            loss.item() * batch_size
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        current_loss = (
            loss_sum / total
        )

        current_accuracy = (
            correct / total
        )

        elapsed = (
            time.time() - start_time
        )

        batches_done = progress.n
        total_batches = len(loader)

        if batches_done > 0:

            batch_time = (
                elapsed / batches_done
            )

            remaining = (
                batch_time *
                (total_batches - batches_done)
            )

            progress.set_postfix({
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_accuracy * 100:.2f}%",
                "ETA": format_time(remaining)
            })

    return reduce_metrics(
        loss_sum,
        correct,
        total,
        device
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    rank,
    description="Evaluation"
):

    model.eval()

    loss_sum = 0.0
    total = 0

    local_labels = []
    local_predictions = []
    local_probabilities = []
    local_paths = []
    local_subjects = []

    progress = tqdm(
        loader,
        desc=description,
        disable=(rank != 0)
    )

    for images, labels, paths, subjects in progress:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        batch_size = labels.size(0)

        loss_sum += (
            loss.item() * batch_size
        )

        total += batch_size

        local_labels.extend(
            labels.cpu().numpy().tolist()
        )

        local_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        local_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
            .tolist()
        )

        local_paths.extend(paths)

        local_subjects.extend(subjects)

    local_data = {
        "labels": local_labels,
        "predictions": local_predictions,
        "probabilities": local_probabilities,
        "paths": local_paths,
        "subjects": local_subjects,
        "loss_sum": loss_sum,
        "total": total
    }

    if dist.is_initialized():

        gathered_data = [
            None
            for _ in range(
                dist.get_world_size()
            )
        ]

        dist.all_gather_object(
            gathered_data,
            local_data
        )

        if rank == 0:

            labels = []
            predictions = []
            probabilities = []
            paths = []
            subjects = []

            global_loss_sum = 0
            global_total = 0

            for data in gathered_data:

                labels.extend(
                    data["labels"]
                )

                predictions.extend(
                    data["predictions"]
                )

                probabilities.extend(
                    data["probabilities"]
                )

                paths.extend(
                    data["paths"]
                )

                subjects.extend(
                    data["subjects"]
                )

                global_loss_sum += (
                    data["loss_sum"]
                )

                global_total += (
                    data["total"]
                )

            loss = (
                global_loss_sum /
                global_total
            )

            accuracy = accuracy_score(
                labels,
                predictions
            )

            return {
                "loss": loss,
                "accuracy": accuracy,
                "labels": np.array(labels),
                "predictions": np.array(predictions),
                "probabilities": np.array(probabilities),
                "paths": paths,
                "subjects": subjects
            }

        return None

    accuracy = accuracy_score(
        local_labels,
        local_predictions
    )

    return {
        "loss": loss_sum / total,
        "accuracy": accuracy,
        "labels": np.array(local_labels),
        "predictions": np.array(local_predictions),
        "probabilities": np.array(local_probabilities),
        "paths": local_paths,
        "subjects": local_subjects
    }


def save_training_plots(
    history,
    output_dir
):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_dir /
        "training_validation_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_dir /
        "training_validation_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["learning_rate"],
        label="Learning Rate"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(
        "Learning Rate Schedule"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_dir /
        "learning_rate.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_predictions(
    results,
    output_dir,
    filename
):

    df = pd.DataFrame({
        "image": results["paths"],
        "subject": results["subjects"],
        "actual": [
            CLASS_NAMES[x]
            for x in results["labels"]
        ],
        "predicted": [
            CLASS_NAMES[x]
            for x in results["predictions"]
        ],
        "palsy_probability": results[
            "probabilities"
        ]
    })

    df.to_csv(
        output_dir / filename,
        index=False
    )

    return df


def calculate_metrics(results):

    labels = results["labels"]

    predictions = results[
        "predictions"
    ]

    probabilities = results[
        "probabilities"
    ]

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

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

    auc = roc_auc_score(
        labels,
        probabilities
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1 Score": f1,
        "ROC-AUC": auc
    }, cm


def save_evaluation_results(
    results,
    output_dir,
    prefix
):

    metrics, cm = calculate_metrics(
        results
    )

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        output_dir /
        f"{prefix}_metrics.csv",
        index=False
    )

    report = classification_report(
        results["labels"],
        results["predictions"],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df.to_csv(
        output_dir /
        f"{prefix}_classification_report.csv"
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f"{prefix.capitalize()} Confusion Matrix"
    )

    plt.savefig(
        output_dir /
        f"{prefix}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    normalized_cm = (
        cm.astype(float) /
        cm.sum(
            axis=1,
            keepdims=True
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        normalized_cm,
        annot=True,
        fmt=".2%",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f"{prefix.capitalize()} Normalized Confusion Matrix"
    )

    plt.savefig(
        output_dir /
        f"{prefix}_normalized_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    fpr, tpr, _ = roc_curve(
        results["labels"],
        results["probabilities"]
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {metrics['ROC-AUC']:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"{prefix.capitalize()} ROC Curve"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_dir /
        f"{prefix}_roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            results["labels"],
            results["probabilities"]
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        recall_curve,
        precision_curve
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"{prefix.capitalize()} Precision-Recall Curve"
    )

    plt.grid(True)

    plt.savefig(
        output_dir /
        f"{prefix}_precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return metrics


def save_dataset_summary(
    train_samples,
    val_samples,
    test_samples,
    output_dir
):

    rows = []

    for split_name, samples in [
        ("train", train_samples),
        ("validation", val_samples),
        ("test", test_samples)
    ]:

        for class_name in CLASS_NAMES:

            label = CLASS_TO_INDEX[
                class_name
            ]

            class_samples = [
                sample
                for sample in samples
                if sample[1] == label
            ]

            subjects = set(
                sample[2]
                for sample in class_samples
            )

            rows.append({
                "split": split_name,
                "class": class_name,
                "subjects": len(subjects),
                "images": len(class_samples)
            })

    pd.DataFrame(rows).to_csv(
        output_dir /
        "dataset_summary.csv",
        index=False
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE
    )

    args = parser.parse_args()

    ddp, rank, world_size, local_rank = (
        setup_ddp()
    )

    set_seed(
        SEED + rank
    )

    device = torch.device(
        f"cuda:{local_rank}"
    )

    if rank == 0:

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        print("=" * 70)
        print(
            "RESNET50 FACIAL PALSY CLASSIFICATION"
        )
        print("=" * 70)

        print(
            f"Dataset: {DATASET_DIR}"
        )

        print(
            f"Device: {device}"
        )

        print(
            f"GPU count: {world_size}"
        )

        print(
            f"Batch size per GPU: "
            f"{args.batch_size}"
        )

        print(
            f"Effective batch size: "
            f"{args.batch_size * world_size}"
        )

        print(
            f"Epochs: {args.epochs}"
        )

        print(
            f"Learning rate: "
            f"{LEARNING_RATE}"
        )

        print(
            f"Seed: {SEED}"
        )

        print("=" * 70)

    all_train_samples = collect_samples(
        DATASET_DIR / "train"
    )

    test_samples = collect_samples(
        DATASET_DIR / "test"
    )

    train_samples, val_samples, subject_split = (
        split_subjects(
            all_train_samples
        )
    )

    if rank == 0:

        print("\nDATASET")

        print(
            f"Training images: "
            f"{len(train_samples):,}"
        )

        print(
            f"Validation images: "
            f"{len(val_samples):,}"
        )

        print(
            f"Testing images: "
            f"{len(test_samples):,}"
        )

        print("\nSUBJECT SPLIT")

        for class_name in CLASS_NAMES:

            print(
                f"{class_name.upper()}: "
                f"{len(subject_split[class_name]['train'])} "
                f"train, "
                f"{len(subject_split[class_name]['val'])} "
                f"validation"
            )

        print(
            f"\nTest subjects: "
            f"{len(set(x[2] for x in test_samples))}"
        )

    train_transform, evaluation_transform = (
        create_transforms()
    )

    train_dataset = MeshDataset(
        train_samples,
        train_transform
    )

    val_dataset = MeshDataset(
        val_samples,
        evaluation_transform
    )

    test_dataset = MeshDataset(
        test_samples,
        evaluation_transform
    )

    train_sampler = None
    val_sampler = None
    test_sampler = None

    if ddp:

        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True
        )

        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False
        )

        test_sampler = DistributedSampler(
            test_dataset,
            shuffle=False
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    model = create_model()

    model = model.to(device)

    if ddp:

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank]
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

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "learning_rate": []
    }

    best_val_loss = float("inf")

    epochs_without_improvement = 0

    training_start = time.time()

    if rank == 0:
        print("\nSTARTING TRAINING\n")

    completed_epochs = 0

    for epoch in range(
        1,
        args.epochs + 1
    ):

        epoch_start = time.time()

        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            rank
        )

        val_results = evaluate(
            model,
            val_loader,
            criterion,
            device,
            rank,
            "Validation"
        )

        stop_training = False

        if rank == 0:

            val_loss = val_results["loss"]

            val_accuracy = val_results[
                "accuracy"
            ]

            scheduler.step(
                val_loss
            )

            current_lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            history[
                "train_loss"
            ].append(
                train_loss
            )

            history[
                "val_loss"
            ].append(
                val_loss
            )

            history[
                "train_accuracy"
            ].append(
                train_accuracy
            )

            history[
                "val_accuracy"
            ].append(
                val_accuracy
            )

            history[
                "learning_rate"
            ].append(
                current_lr
            )

            epoch_time = (
                time.time() -
                epoch_start
            )

            total_elapsed = (
                time.time() -
                training_start
            )

            average_epoch_time = (
                total_elapsed / epoch
            )

            remaining_epochs = (
                args.epochs - epoch
            )

            estimated_remaining = (
                average_epoch_time *
                remaining_epochs
            )

            print(
                "\n" + "=" * 70
            )

            print(
                f"Epoch {epoch}/{args.epochs}"
            )

            print(
                "=" * 70
            )

            print(
                f"Train Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Train Accuracy: "
                f"{train_accuracy * 100:.2f}%"
            )

            print(
                f"Validation Loss: "
                f"{val_loss:.4f}"
            )

            print(
                f"Validation Accuracy: "
                f"{val_accuracy * 100:.2f}%"
            )

            print(
                f"Learning Rate: "
                f"{current_lr:.2e}"
            )

            print(
                f"Epoch Time: "
                f"{format_time(epoch_time)}"
            )

            print(
                f"Estimated Remaining: "
                f"{format_time(estimated_remaining)}"
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                epochs_without_improvement = 0

                model_to_save = (
                    model.module
                    if ddp
                    else model
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict":
                            model_to_save.state_dict(),
                        "optimizer_state_dict":
                            optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_accuracy":
                            val_accuracy
                    },
                    RESULTS_DIR /
                    "best_model.pth"
                )

                print(
                    "Best model saved."
                )

            else:

                epochs_without_improvement += 1

                print(
                    f"No improvement: "
                    f"{epochs_without_improvement}/"
                    f"{PATIENCE}"
                )

                if (
                    epochs_without_improvement
                    >= PATIENCE
                ):

                    stop_training = True

                    print(
                        "\nEarly stopping triggered."
                    )

            completed_epochs = epoch

        if ddp:

            stop_tensor = torch.tensor(
                [int(stop_training)],
                device=device
            )

            dist.broadcast(
                stop_tensor,
                src=0
            )

            stop_training = bool(
                stop_tensor.item()
            )

        if stop_training:
            break

    if rank == 0:

        total_training_time = (
            time.time() -
            training_start
        )

        model_to_save = (
            model.module
            if ddp
            else model
        )

        torch.save(
            {
                "epoch": completed_epochs,
                "model_state_dict":
                    model_to_save.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict()
            },
            RESULTS_DIR /
            "final_model.pth"
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "TRAINING COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"Completed epochs: "
            f"{completed_epochs}"
        )

        print(
            f"Total training time: "
            f"{format_time(total_training_time)}"
        )

        save_training_plots(
            history,
            RESULTS_DIR
        )

        pd.DataFrame(
            history
        ).to_csv(
            RESULTS_DIR /
            "training_history.csv",
            index=False
        )

        save_dataset_summary(
            train_samples,
            val_samples,
            test_samples,
            RESULTS_DIR
        )

        config = {
            "model": "ResNet50",
            "pretrained": True,
            "input_size": IMAGE_SIZE,
            "batch_size_per_gpu":
                args.batch_size,
            "effective_batch_size":
                args.batch_size * world_size,
            "gpu_count": world_size,
            "epochs_requested":
                args.epochs,
            "epochs_completed":
                completed_epochs,
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss",
            "scheduler":
                "ReduceLROnPlateau",
            "early_stopping_patience":
                PATIENCE,
            "random_seed": SEED,
            "dataset": "YFP + AFLFP",
            "representation":
                "White facial mesh on black background",
            "classification":
                "Normal vs Palsy",
            "split":
                "Subject-level",
            "validation_ratio":
                0.20
        }

        with open(
            RESULTS_DIR /
            "experiment_config.json",
            "w"
        ) as f:

            json.dump(
                config,
                f,
                indent=4
            )

    if ddp:
        dist.barrier()

    test_results = evaluate(
        model,
        test_loader,
        criterion,
        device,
        rank,
        "Test"
    )

    if rank == 0:

        save_predictions(
            val_results,
            RESULTS_DIR,
            "validation_predictions.csv"
        )

        save_predictions(
            test_results,
            RESULTS_DIR,
            "test_predictions.csv"
        )

        validation_metrics = (
            save_evaluation_results(
                val_results,
                RESULTS_DIR,
                "validation"
            )
        )

        test_metrics = (
            save_evaluation_results(
                test_results,
                RESULTS_DIR,
                "test"
            )
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "VALIDATION RESULTS"
        )

        print(
            "=" * 70
        )

        for metric, value in validation_metrics.items():

            if metric == "ROC-AUC":

                print(
                    f"{metric}: "
                    f"{value:.4f}"
                )

            else:

                print(
                    f"{metric}: "
                    f"{value * 100:.2f}%"
                )

        print(
            "\n" + "=" * 70
        )

        print(
            "TEST RESULTS"
        )

        print(
            "=" * 70
        )

        for metric, value in test_metrics.items():

            if metric == "ROC-AUC":

                print(
                    f"{metric}: "
                    f"{value:.4f}"
                )

            else:

                print(
                    f"{metric}: "
                    f"{value * 100:.2f}%"
                )

        print(
            "\nResults saved to:"
        )

        print(
            RESULTS_DIR
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()