#!/usr/bin/env python3
"""
Mesh-Only Comparative CNN Study — Single-Command Pipeline
==========================================================
Trains ResNet50, MobileNetV2, EfficientNet-B0 on MediaPipe Face Mesh images.
3 architectures x 5 seeds = 15 runs, distributed across available GPUs.

Usage:
    python run_comparative_study.py              # Full experiment
    python run_comparative_study.py --dry-run    # Show plan, no training
    python run_comparative_study.py --smoke-test # 1 epoch per run
"""

import sys
import os
import json
import time
import random
import argparse
import subprocess
import logging
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = Path("/home/cai/Documents/Masters/Datasets/Dataset for Augment")
DATASET_DIR = BASE_DIR / "Training"
STUDY_DIR = BASE_DIR / "Comparative Study"

IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 7
VAL_RATIO = 0.20
NUM_WORKERS = 4

SEEDS = [42, 123, 456, 789, 1000]
ARCHITECTURES = ["resnet50", "mobilenetv2", "efficientnet_b0"]

ARCH_LABELS = {
    "resnet50": "ResNet50",
    "mobilenetv2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
}

CLASS_NAMES = ["normal", "palsy"]
CLASS_TO_IDX = {"normal": 0, "palsy": 1}

# Sub-directories
CONFIGS_DIR = STUDY_DIR / "configs"
CKPT_DIR = STUDY_DIR / "checkpoints"
LOG_DIR = STUDY_DIR / "logs"
RES_DIR = STUDY_DIR / "results"
FIG_DIR = STUDY_DIR / "figures"
MET_DIR = STUDY_DIR / "metrics"

# ================================================================
# TERMINAL COLORS
# ================================================================

class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

def cprint(msg, color=C.RESET):
    print(f"{color}{msg}{C.RESET}", flush=True)

# ================================================================
# UTILITIES
# ================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_gpus():
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def print_gpu_info(gpus):
    cprint("\n" + "=" * 65, C.CYAN)
    cprint("  GPU AVAILABILITY", C.CYAN)
    cprint("=" * 65, C.CYAN)
    if not gpus:
        cprint("  No GPU detected on this machine.", C.YELLOW)
        cprint("  (GPUs are required for training, but not for --dry-run)", C.DIM)
        return
    cprint(f"  Detected {len(gpus)} GPU(s):", C.GREEN)
    for i in gpus:
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        cprint(f"    GPU {i}: {name} ({mem:.1f} GB)", C.GREEN)
    cprint("=" * 65 + "\n", C.CYAN)


def ensure_dirs():
    for d in [CONFIGS_DIR, FIG_DIR, MET_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for arch in ARCHITECTURES:
        for base in [CKPT_DIR, LOG_DIR, RES_DIR]:
            (base / arch).mkdir(parents=True, exist_ok=True)


# ================================================================
# DATA LOADING
# ================================================================

class MeshDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, subject = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(path), subject


def collect_samples(directory):
    samples = []
    for cls in CLASS_NAMES:
        cls_dir = directory / cls
        if not cls_dir.exists():
            continue
        for subj_dir in sorted(cls_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            sid = subj_dir.name
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for p in sorted(subj_dir.glob(ext)):
                    samples.append((p, CLASS_TO_IDX[cls], sid))
    return samples


def split_train_val(all_train, val_ratio=VAL_RATIO, seed=456):
    subjects = sorted(set((s[2], s[1]) for s in all_train))
    normal = sorted([s for s, l in subjects if l == 0])
    palsy = sorted([s for s, l in subjects if l == 1])

    rng = random.Random(seed)
    rng.shuffle(normal)
    rng.shuffle(palsy)

    n_val = max(1, int(len(normal) * val_ratio))
    p_val = max(1, int(len(palsy) * val_ratio))

    val_set = set(normal[:n_val]) | set(palsy[:p_val])
    train = [s for s in all_train if s[2] not in val_set]
    val = [s for s in all_train if s[2] in val_set]
    return train, val


def get_transforms():
    train_t = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_t, eval_t


# ================================================================
# MODEL BUILDING
# ================================================================

def build_model(arch):
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == "mobilenetv2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.last_channel, 2)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


# ================================================================
# TRAINING & EVALUATION
# ================================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(images)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    probs, labels_all, paths_all, subjects_all = [], [], [], []
    for images, labels, paths, subjects in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(images)
            loss = criterion(out, labels)
        loss_sum += loss.item() * labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
        p = torch.softmax(out, 1)[:, 1]
        probs.extend(p.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        paths_all.extend(paths)
        subjects_all.extend(subjects)
    return (
        loss_sum / total,
        correct / total,
        np.array(probs),
        np.array(labels_all),
        paths_all,
        subjects_all,
    )


# ================================================================
# METRICS
# ================================================================

def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    two_class = len(np.unique(labels)) == 2
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if two_class else float("nan"),
        "avg_precision": float(average_precision_score(labels, probs)) if two_class else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def find_youden_threshold(labels, probs):
    fpr, tpr, thr = roc_curve(labels, probs)
    j = tpr - fpr
    valid = np.isfinite(thr)
    idx = np.argmax(np.where(valid, j, -np.inf))
    return float(thr[idx])


def subject_level_metrics(labels, probs, subjects, threshold):
    df = pd.DataFrame({"label": labels, "prob": probs, "subject": subjects})
    agg = df.groupby("subject").agg(
        label=("label", "first"), prob=("prob", "mean"), count=("prob", "count")
    ).reset_index()
    m = compute_metrics(agg["label"].values, agg["prob"].values, threshold)
    return m, agg


# ================================================================
# VISUALIZATION
# ================================================================

def save_training_curves(history, out_dir, arch, seed):
    df = pd.DataFrame(history)
    ep = df["epoch"].values

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(ep, df["train_loss"], label="Training Loss")
    plt.plot(ep, df["val_loss"], label="Validation Loss")
    best_idx = df["val_loss"].idxmin()
    plt.axvline(df.loc[best_idx, "epoch"], linestyle="--", alpha=0.5,
                label=f"Best Epoch {int(df.loc[best_idx, 'epoch'])}")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{ARCH_LABELS[arch]} Seed {seed} — Loss")
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=300); plt.close()

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(ep, np.array(df["train_acc"]) * 100, label="Training Accuracy")
    plt.plot(ep, np.array(df["val_acc"]) * 100, label="Validation Accuracy")
    plt.axvline(df.loc[best_idx, "epoch"], linestyle="--", alpha=0.5,
                label=f"Best Epoch {int(df.loc[best_idx, 'epoch'])}")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title(f"{ARCH_LABELS[arch]} Seed {seed} — Accuracy")
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curves.png", dpi=300); plt.close()

    # Learning Rate
    plt.figure(figsize=(8, 5))
    plt.plot(ep, df["lr"])
    plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.yscale("log")
    plt.title(f"{ARCH_LABELS[arch]} Seed {seed} — Learning Rate")
    plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(out_dir / "learning_rate.png", dpi=300); plt.close()

    # Dashboard
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(ep, df["train_loss"], label="Train"); axes[0].plot(ep, df["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(ep, np.array(df["train_acc"]) * 100, label="Train"); axes[1].plot(ep, np.array(df["val_acc"]) * 100, label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)"); axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.25)
    axes[2].plot(ep, df["lr"]); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR"); axes[2].set_title("Learning Rate"); axes[2].set_yscale("log"); axes[2].grid(alpha=0.25)
    fig.suptitle(f"{ARCH_LABELS[arch]} — Seed {seed} Training Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(out_dir / "training_dashboard.png", dpi=300); plt.close()


def save_eval_plots(labels, probs, threshold, out_dir, arch, seed, prefix=""):
    p = prefix + "_" if prefix else ""

    # Confusion Matrix
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    cm = np.array([[tn, fp], [fn, tp]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title(f"{p}Confusion Matrix"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1]); axes[0].set_xticklabels(["Normal", "Palsy"]); axes[0].set_yticklabels(["Normal", "Palsy"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=axes[0])
    cm_n = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im2 = axes[1].imshow(cm_n, interpolation="nearest", cmap="Blues")
    axes[1].set_title(f"{p}Normalized CM"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1]); axes[1].set_xticklabels(["Normal", "Palsy"]); axes[1].set_yticklabels(["Normal", "Palsy"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm_n[i, j]*100:.1f}%", ha="center", va="center", fontsize=14, fontweight="bold")
    plt.colorbar(im2, ax=axes[1])
    plt.tight_layout(); plt.savefig(out_dir / f"{p}confusion_matrix.png", dpi=300); plt.close()

    # ROC + PR + Pred Distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if len(np.unique(labels)) == 2:
        fpr, tpr, _ = roc_curve(labels, probs); auc = roc_auc_score(labels, probs)
        axes[0].plot(fpr, tpr, label=f"AUC = {auc:.4f}"); axes[0].plot([0, 1], [0, 1], "k--")
        axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC Curve"); axes[0].legend(); axes[0].grid(alpha=0.25)
        prec, rec, _ = precision_recall_curve(labels, probs); ap = average_precision_score(labels, probs)
        axes[1].plot(rec, prec, label=f"AP = {ap:.4f}"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("PR Curve"); axes[1].legend(); axes[1].grid(alpha=0.25)
    n_probs, p_probs = probs[labels == 0], probs[labels == 1]
    axes[2].hist(n_probs, bins=20, alpha=0.6, label="Normal"); axes[2].hist(p_probs, bins=20, alpha=0.6, label="Palsy")
    axes[2].axvline(threshold, linestyle="--", label=f"Thr={threshold:.3f}")
    axes[2].set_xlabel("Palsy Probability"); axes[2].set_ylabel("Count"); axes[2].set_title("Prediction Distribution"); axes[2].legend(); axes[2].grid(alpha=0.25)
    plt.suptitle(f"{ARCH_LABELS[arch]} Seed {seed}", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(out_dir / f"{p}roc_pr_distribution.png", dpi=300); plt.close()


# ================================================================
# SAVE EXPERIMENT RECORDS
# ================================================================

def save_config(arch, seed, gpu_id, train_n, val_n, test_n, params, out_dir):
    cfg = {
        "architecture": arch, "label": ARCH_LABELS[arch], "seed": seed, "gpu_id": gpu_id,
        "dataset": "MediaPipe Face Mesh", "rgb_used": False,
        "train_images": train_n, "val_images": val_n, "test_images": test_n,
        "input_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW", "scheduler": "ReduceLROnPlateau",
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "amp": True,
        "augmentation": ["RandomHorizontalFlip", "RandomRotation(10)", "RandomAffine"],
        "normalization": "ImageNet",
        "python_version": sys.version, "torch_version": torch.__version__,
        "timestamp": datetime.now().isoformat(), **params,
    }
    with open(out_dir / "experiment_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def save_model_complexity(arch, model, out_dir):
    info = count_params(model)
    est_mb = info["total"] * 4 / 1e6
    data = {**info, "architecture": arch, "label": ARCH_LABELS[arch], "est_size_mb": round(est_mb, 1)}
    with open(out_dir / "model_complexity.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


# ================================================================
# WORKER — Single architecture + seed training run
# ================================================================

def run_worker(arch, seed, gpu_id, smoke_test=False):
    set_seed(seed)
    device = torch.device(f"cuda:{gpu_id}")
    max_ep = 2 if smoke_test else MAX_EPOCHS

    arch_dir = RES_DIR / arch / f"seed_{seed}"
    arch_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CKPT_DIR / arch
    log_file = LOG_DIR / arch / f"seed_{seed}.log"

    logging.basicConfig(filename=str(log_file), level=logging.INFO, filemode="w")
    log = logging.getLogger(arch + str(seed))

    cprint(f"  [{ARCH_LABELS[arch]}] Seed {seed} on GPU {gpu_id}", C.CYAN)

    # --- Data ---
    all_train = collect_samples(DATASET_DIR / "train")
    test_samples = collect_samples(DATASET_DIR / "test")
    train_s, val_s = split_train_val(all_train, VAL_RATIO, seed)
    test_subs = sorted(set(s[2] for s in test_samples))

    log.info(f"Train={len(train_s)} Val={len(val_s)} Test={len(test_samples)}")
    cprint(f"    Data: {len(train_s)} train / {len(val_s)} val / {len(test_samples)} test images", C.DIM)

    train_t, eval_t = get_transforms()
    tr_loader = DataLoader(MeshDataset(train_s, train_t), batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    vl_loader = DataLoader(MeshDataset(val_s, eval_t), batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    te_loader = DataLoader(MeshDataset(test_samples, eval_t), batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

    # --- Model ---
    model = build_model(arch).to(device)
    complexity = save_model_complexity(arch, model, arch_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda")

    # --- Train ---
    best_val_loss, best_state, best_ep, patience_ctr, history = float("inf"), None, 0, 0, []
    t0 = time.time()

    for epoch in range(1, max_ep + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, optimizer, criterion, scaler, device)
        vl_loss, vl_acc, _, _, _, _ = evaluate(model, vl_loader, criterion, device)
        scheduler.step(vl_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": vl_loss, "val_acc": vl_acc, "lr": lr})

        elapsed = time.time() - t0
        eta_s = (elapsed / epoch) * (max_ep - epoch)
        marker = ""
        if vl_loss < best_val_loss:
            best_val_loss, best_ep, patience_ctr = vl_loss, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = f" {C.GREEN}*{C.RESET}"
        else:
            patience_ctr += 1

        cprint(f"    Ep {epoch:2d}/{max_ep} | "
               f"TrL {tr_loss:.4f} TrA {tr_acc*100:5.1f}% | "
               f"VlL {vl_loss:.4f} VlA {vl_acc*100:5.1f}% | "
               f"LR {lr:.1e} | {elapsed/60:.1f}m ETA {eta_s/60:.1f}m{marker}",
               C.GREEN if marker else C.RESET)

        if patience_ctr >= PATIENCE:
            cprint(f"    Early stopping at epoch {epoch}", C.YELLOW)
            break

    total_time = time.time() - t0

    if best_state is None:
        raise RuntimeError(f"No best model for {arch} seed {seed}")

    model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "arch": arch, "seed": seed,
                "best_epoch": best_ep, "val_loss": best_val_loss},
               arch_dir / "best_model.pth")
    # Also save in checkpoints/
    torch.save(best_state, CKPT_DIR / arch / f"seed_{seed}_best.pth")

    pd.DataFrame(history).to_csv(arch_dir / "training_history.csv", index=False)
    save_training_curves(history, arch_dir, arch, seed)

    # --- Test Evaluation ---
    _, _, te_probs, te_labels, te_paths, te_subs = evaluate(model, te_loader, criterion, device)
    thr = find_youden_threshold(
        *evaluate_subject_probs(model, vl_loader, criterion, device)[:2]
    )
    img_m = compute_metrics(te_labels, te_probs, thr)
    sub_m, sub_df = subject_level_metrics(te_labels, te_probs, te_subs, thr)

    # Save predictions
    pd.DataFrame({"image_path": te_paths, "subject_id": te_subs,
                  "true_label": te_labels, "probability_palsy": te_probs,
                  "predicted_label": (te_probs >= thr).astype(int),
                  "correct": ((te_probs >= thr).astype(int) == te_labels)}).to_csv(
        arch_dir / "test_predictions.csv", index=False)
    sub_df.to_csv(arch_dir / "subject_predictions.csv", index=False)

    # Save metrics
    metrics = {"arch": arch, "seed": seed, "best_epoch": best_ep,
               "best_val_loss": best_val_loss, "threshold": thr,
               "training_time_s": total_time, "params": complexity["total"],
               "image_level": img_m, "subject_level": sub_m}
    with open(arch_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([{**{"arch": arch, "seed": seed, "level": "image"}, **img_m},
                  {**{"arch": arch, "seed": seed, "level": "subject"}, **sub_m}]).to_csv(
        arch_dir / "metrics.csv", index=False)

    # Evaluation plots
    save_eval_plots(te_labels, te_probs, thr, arch_dir, arch, seed)

    cprint(f"    DONE | Best ep {best_ep} | ValLoss {best_val_loss:.4f} | "
           f"TestAcc {img_m['accuracy']*100:.1f}% | AUC {img_m['roc_auc']:.4f} | "
           f"Time {total_time/60:.1f}m", C.GREEN)
    log.info(f"Complete: {json.dumps(metrics)}")
    return metrics


def evaluate_subject_probs(model, loader, criterion, device):
    """Get subject-level probabilities from validation set for threshold."""
    _, _, probs, labels, _, subjects = evaluate(model, loader, criterion, device)
    df = pd.DataFrame({"label": labels, "prob": probs, "subject": subjects})
    agg = df.groupby("subject").agg(label=("label", "first"), prob=("prob", "mean")).reset_index()
    return agg["label"].values, agg["prob"].values


# ================================================================
# MASTER ORCHESTRATOR
# ================================================================

def run_master(dry_run=False, smoke_test=False, force=False):
    gpus = detect_gpus()
    print_gpu_info(gpus)

    # Build work queue
    queue = []
    for arch in ARCHITECTURES:
        for seed in SEEDS:
            queue.append((arch, seed))

    cprint(f"Experiment plan: {len(queue)} runs "
           f"({len(ARCHITECTURES)} architectures x {len(SEEDS)} seeds)", C.BOLD)
    cprint(f"Available GPUs: {len(gpus)}", C.BOLD)

    if dry_run:
        cprint("\n--- DRY RUN — No training will occur ---\n", C.YELLOW)
        n_gpus = max(len(gpus), 1)
        for i, (arch, seed) in enumerate(queue):
            gpu = gpus[i % len(gpus)] if gpus else (i % 4)
            out = RES_DIR / arch / f"seed_{seed}"
            cprint(f"  [{i+1:2d}/15] GPU {gpu} | {ARCH_LABELS[arch]:15s} | Seed {seed:4d} | {out}", C.CYAN)
        cprint("\nDry run complete. Remove --dry-run to start training.", C.YELLOW)
        return

    if not gpus:
        cprint("ERROR: No GPU found. This pipeline requires at least one CUDA GPU.", C.RED)
        sys.exit(1)

    ensure_dirs()

    # Check for completed runs
    pending = []
    for arch, seed in queue:
        d = RES_DIR / arch / f"seed_{seed}"
        if not force and (d / "metrics.json").exists() and (d / "best_model.pth").exists():
            cprint(f"  [SKIP] {ARCH_LABELS[arch]} seed {seed} — already complete", C.DIM)
        else:
            pending.append((arch, seed))

    if not pending:
        cprint("\nAll 15 runs already complete. Proceeding to aggregation.", C.GREEN)
    else:
        cprint(f"\n{len(pending)} runs to execute. Launching...\n", C.BOLD)

        # Assign to GPUs round-robin
        assignments = {gpu: [] for gpu in gpus}
        for i, task in enumerate(pending):
            gpu = gpus[i % len(gpus)]
            assignments[gpu].append(task)

        for gpu, tasks in assignments.items():
            names = [f"{ARCH_LABELS[a]} s{s}" for a, s in tasks]
            cprint(f"  GPU {gpu}: {', '.join(names)}", C.CYAN)

        # Launch one subprocess per GPU
        procs = []
        for gpu, tasks in assignments.items():
            task_str = "|".join(f"{a},{s}" for a, s in tasks)
            log_f = open(LOG_DIR / f"gpu_{gpu}.log", "w")
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--worker", "--gpu", str(gpu), "--tasks", task_str]
            if smoke_test:
                cmd.append("--smoke-test")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            procs.append((proc, gpu, tasks, log_f))

        # Wait for all
        for proc, gpu, tasks, log_f in procs:
            proc.wait()
            log_f.close()
            status = "COMPLETE" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            color = C.GREEN if proc.returncode == 0 else C.RED
            cprint(f"  GPU {gpu}: {status}", color)

    # --- Aggregation ---
    cprint("\n" + "=" * 65, C.MAGENTA)
    cprint("  AGGREGATION & COMPARISON", C.MAGENTA)
    cprint("=" * 65, C.MAGENTA)
    aggregate_and_compare()


# ================================================================
# AGGREGATION & COMPARISON
# ================================================================

def aggregate_and_compare():
    all_metrics = []
    for arch in ARCHITECTURES:
        for seed in SEEDS:
            mf = RES_DIR / arch / f"seed_{seed}" / "metrics.json"
            if mf.exists():
                with open(mf) as f:
                    all_metrics.append(json.load(f))

    if not all_metrics:
        cprint("  No results found for aggregation!", C.RED)
        return

    df = pd.DataFrame(all_metrics)

    # Per-seed metrics table
    rows = []
    for _, r in df.iterrows():
        il = r.get("image_level", {})
        sl = r.get("subject_level", {})
        rows.append({
            "Architecture": ARCH_LABELS.get(r["arch"], r["arch"]),
            "Seed": r["seed"],
            "Best Epoch": r.get("best_epoch", ""),
            "Threshold": r.get("threshold", ""),
            "Img Accuracy": il.get("accuracy", ""),
            "Img Sensitivity": il.get("sensitivity", ""),
            "Img Specificity": il.get("specificity", ""),
            "Img Precision": il.get("precision", ""),
            "Img F1": il.get("f1", ""),
            "Img AUC": il.get("roc_auc", ""),
            "Sub Accuracy": sl.get("accuracy", ""),
            "Sub Sensitivity": sl.get("sensitivity", ""),
            "Sub Specificity": sl.get("specificity", ""),
            "Sub F1": sl.get("f1", ""),
            "Sub AUC": sl.get("roc_auc", ""),
            "Time (s)": round(r.get("training_time_s", 0), 1),
        })
    pd.DataFrame(rows).to_csv(MET_DIR / "per_seed_metrics.csv", index=False)

    # Aggregated comparison table
    agg_rows = []
    for arch in ARCHITECTURES:
        adf = df[df["arch"] == arch]
        if len(adf) == 0:
            continue
        il_metrics = [r.get("image_level", {}) for _, r in adf.iterrows()]
        for metric in ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]:
            vals = [m.get(metric, float("nan")) for m in il_metrics]
            agg_rows.append({
                "Architecture": ARCH_LABELS[arch],
                "Metric": metric,
                "Mean": np.nanmean(vals), "SD": np.nanstd(vals, ddof=1),
                "Min": np.nanmin(vals), "Max": np.nanmax(vals),
            })
    pd.DataFrame(agg_rows).to_csv(MET_DIR / "aggregated_metrics.csv", index=False)

    # Markdown comparison
    lines = ["# Model Comparison (Image-Level, Mean +/- SD)", "",
             "| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC |",
             "|-------|----------|-------------|-------------|-----------|----|---------|"]
    for arch in ARCHITECTURES:
        adf = df[df["arch"] == arch]
        if len(adf) == 0:
            continue
        il = [r.get("image_level", {}) for _, r in adf.iterrows()]
        vals = {}
        for m in ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]:
            v = [x.get(m, float("nan")) for x in il]
            vals[m] = f"{np.nanmean(v)*100:.2f}+/-{np.nanstd(v, ddof=1)*100:.2f}"
        lines.append(f"| {ARCH_LABELS[arch]} | {vals['accuracy']} | {vals['sensitivity']} | "
                     f"{vals['specificity']} | {vals['precision']} | {vals['f1']} | {vals['roc_auc']} |")
    with open(MET_DIR / "comparison_table.md", "w") as f:
        f.write("\n".join(lines))

    # --- Comparison Figures ---
    generate_comparison_figures(df)

    # --- Final Report ---
    generate_final_report(df)

    # --- Experiment Log ---
    generate_experiment_log(df)

    cprint("\n" + "=" * 65, C.MAGENTA)
    cprint("  MESH-ONLY COMPARATIVE STUDY COMPLETE", C.MAGENTA)
    cprint("=" * 65, C.MAGENTA)
    for arch in ARCHITECTURES:
        n = len([f for f in (RES_DIR / arch).iterdir() if f.is_dir()]) if (RES_DIR / arch).exists() else 0
        cprint(f"  {ARCH_LABELS[arch]:15s}: {n}/{len(SEEDS)} seeds complete", C.GREEN)
    cprint(f"\n  Results: {STUDY_DIR}", C.BOLD)
    cprint("=" * 65 + "\n", C.MAGENTA)


def generate_comparison_figures(df):
    # 1. Bar chart: accuracy/sensitivity/specificity/precision/f1/auc per architecture
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metric_names = ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
    titles = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC-AUC"]
    for idx, (metric, title) in enumerate(zip(metric_names, titles)):
        ax = axes[idx // 3][idx % 3]
        means, sds, labels_x = [], [], []
        for arch in ARCHITECTURES:
            adf = df[df["arch"] == arch]
            vals = [r.get("image_level", {}).get(metric, float("nan")) for _, r in adf.iterrows()]
            means.append(np.nanmean(vals) * 100)
            sds.append(np.nanstd(vals, ddof=1) * 100)
            labels_x.append(ARCH_LABELS[arch])
        colors = ["#2196F3", "#4CAF50", "#FF9800"]
        bars = ax.bar(labels_x, means, yerr=sds, capsize=5, color=colors, alpha=0.85)
        ax.set_title(title, fontweight="bold"); ax.set_ylabel("%")
        ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.25)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{m:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Architecture Comparison — Image-Level Metrics (Mean +/- SD)", fontsize=15, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIG_DIR / "architecture_comparison.png", dpi=300); plt.close()

    # 2. ROC curves overlay (all seeds, per architecture)
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"resnet50": "#2196F3", "mobilenetv2": "#4CAF50", "efficientnet_b0": "#FF9800"}
    for arch in ARCHITECTURES:
        for seed in SEEDS:
            pf = RES_DIR / arch / f"seed_{seed}" / "test_predictions.csv"
            if not pf.exists():
                continue
            pdf = pd.read_csv(pf)
            if len(np.unique(pdf["true_label"])) < 2:
                continue
            fpr, tpr, _ = roc_curve(pdf["true_label"], pdf["probability_palsy"])
            auc = roc_auc_score(pdf["true_label"], pdf["probability_palsy"])
            ax.plot(fpr, tpr, color=colors[arch], alpha=0.3, linewidth=1)
        # Mean ROC
        aucs = []
        for seed in SEEDS:
            pf = RES_DIR / arch / f"seed_{seed}" / "test_predictions.csv"
            if pf.exists():
                pdf = pd.read_csv(pf)
                if len(np.unique(pdf["true_label"])) >= 2:
                    aucs.append(roc_auc_score(pdf["true_label"], pdf["probability_palsy"]))
        if aucs:
            ax.plot([], [], color=colors[arch], linewidth=2.5,
                    label=f"{ARCH_LABELS[arch]} (AUC={np.mean(aucs):.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves — All Architectures & Seeds"); ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(FIG_DIR / "roc_curves.png", dpi=300); plt.close()

    # 3. Confusion matrices side-by-side (seed 456 representative)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, arch in enumerate(ARCHITECTURES):
        pf = RES_DIR / arch / "seed_456" / "test_predictions.csv"
        if pf.exists():
            pdf = pd.read_csv(pf)
            preds = (pdf["probability_palsy"].values >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(pdf["true_label"], preds, labels=[0, 1]).ravel()
            cm = np.array([[tn, fp], [fn, tp]])
            axes[i].imshow(cm, cmap="Blues"); axes[i].set_title(ARCH_LABELS[arch], fontweight="bold")
            axes[i].set_xticks([0, 1]); axes[i].set_yticks([0, 1])
            axes[i].set_xticklabels(["Normal", "Palsy"]); axes[i].set_yticklabels(["Normal", "Palsy"])
            for r in range(2):
                for c in range(2):
                    axes[i].text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=16, fontweight="bold")
        else:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[i].transAxes)
    fig.suptitle("Confusion Matrices (Seed 456)", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIG_DIR / "confusion_matrices.png", dpi=300); plt.close()

    # 4. Training curves overlay (all seeds)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for arch in ARCHITECTURES:
        for seed in SEEDS:
            hf = RES_DIR / arch / f"seed_{seed}" / "training_history.csv"
            if hf.exists():
                hdf = pd.read_csv(hf)
                axes[0].plot(hdf["epoch"], hdf["val_loss"], alpha=0.4, label=f"{ARCH_LABELS[arch]} s{seed}")
                axes[1].plot(hdf["epoch"], np.array(hdf["val_acc"]) * 100, alpha=0.4, label=f"{ARCH_LABELS[arch]} s{seed}")
                axes[2].plot(hdf["epoch"], hdf["lr"], alpha=0.4, label=f"{ARCH_LABELS[arch]} s{seed}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val Loss"); axes[0].set_title("Validation Loss"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Accuracy (%)"); axes[1].set_title("Validation Accuracy"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.25)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate"); axes[2].set_title("Learning Rate"); axes[2].set_yscale("log"); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.25)
    fig.suptitle("Training Curves — All Architectures & Seeds", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIG_DIR / "training_curves.png", dpi=300); plt.close()


def generate_final_report(df):
    lines = [
        "# Mesh-Only Comparative CNN Study — Final Report\n",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Objective",
        "Compare ResNet50, MobileNetV2, and EfficientNet-B0 on MediaPipe Face Mesh images",
        "for facial paralysis classification (Normal vs Palsy).\n",
        "## Dataset",
        "- **Source**: MediaPipe Face Mesh (white mesh on black background)",
        "- **Subjects**: 64 (32 Normal, 32 Palsy)",
        "- **Split**: Fixed holdout — 50 train / 14 test subjects",
        "- **Images**: All PNG mesh representations, 224x224\n",
        "## Training Configuration",
        f"- Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE} | Weight decay: {WEIGHT_DECAY}",
        f"- Optimizer: AdamW | Scheduler: ReduceLROnPlateau",
        f"- Max epochs: {MAX_EPOCHS} | Early stopping patience: {PATIENCE}",
        f"- AMP: enabled | Seeds: {SEEDS}\n",
        "## Image-Level Results (Mean +/- SD across 5 seeds)\n",
        "| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC |",
        "|-------|----------|-------------|-------------|-----------|----|---------|",
    ]
    for arch in ARCHITECTURES:
        adf = df[df["arch"] == arch]
        if len(adf) == 0:
            continue
        il = [r.get("image_level", {}) for _, r in adf.iterrows()]
        parts = []
        for m in ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]:
            v = [x.get(m, float("nan")) for x in il]
            parts.append(f"{np.nanmean(v)*100:.2f}+/-{np.nanstd(v,ddof=1)*100:.2f}")
        lines.append(f"| {ARCH_LABELS[arch]} | {' | '.join(parts)} |")

    lines += ["", "## Subject-Level Results\n",
              "| Model | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC |",
              "|-------|----------|-------------|-------------|----|---------|"]
    for arch in ARCHITECTURES:
        adf = df[df["arch"] == arch]
        if len(adf) == 0:
            continue
        sl = [r.get("subject_level", {}) for _, r in adf.iterrows()]
        parts = []
        for m in ["accuracy", "sensitivity", "specificity", "f1", "roc_auc"]:
            v = [x.get(m, float("nan")) for x in sl]
            parts.append(f"{np.nanmean(v)*100:.2f}+/-{np.nanstd(v,ddof=1)*100:.2f}")
        lines.append(f"| {ARCH_LABELS[arch]} | {' | '.join(parts)} |")

    # Training time
    lines += ["", "## Training Time\n", "| Model | Mean (s) | SD (s) |",
              "|-------|----------|--------|"]
    for arch in ARCHITECTURES:
        adf = df[df["arch"] == arch]
        times = [r.get("training_time_s", 0) for _, r in adf.iterrows()]
        lines.append(f"| {ARCH_LABELS[arch]} | {np.mean(times):.1f} | {np.std(times, ddof=1):.1f} |")

    lines += ["", "## Model Complexity\n", "| Model | Parameters | Est. Size (MB) |",
              "|-------|-----------|----------------|"]
    for arch in ARCHITECTURES:
        cf = RES_DIR / arch / f"seed_{SEEDS[0]}" / "model_complexity.json"
        if cf.exists():
            with open(cf) as f:
                c = json.load(f)
            lines.append(f"| {ARCH_LABELS[arch]} | {c['total']:,} | {c.get('est_size_mb', 'N/A')} |")

    lines += ["", "## Notes",
              "- All runs used mesh images only (no RGB)",
              "- Threshold selected via Youden J on validation set",
              "- Results are from the fixed test set (14 subjects)",
              "- Do not interpret differences as statistically significant without formal testing",
              ""]
    with open(STUDY_DIR / "README_experimental_record.md", "w") as f:
        f.write("\n".join(lines))


def generate_experiment_log(df):
    rows = []
    for _, r in df.iterrows():
        il = r.get("image_level", {})
        rows.append({
            "Model": ARCH_LABELS.get(r["arch"], r["arch"]),
            "Seed": r["seed"], "Best Epoch": r.get("best_epoch", ""),
            "Threshold": r.get("threshold", ""),
            "Test Accuracy": il.get("accuracy", ""),
            "Test Sensitivity": il.get("sensitivity", ""),
            "Test Specificity": il.get("specificity", ""),
            "Test Precision": il.get("precision", ""),
            "Test F1": il.get("f1", ""),
            "Test ROC-AUC": il.get("roc_auc", ""),
            "Training Time (s)": round(r.get("training_time_s", 0), 1),
            "Status": "complete",
        })
    pd.DataFrame(rows).to_csv(MET_DIR / "experiment_log.csv", index=False)


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Mesh-Only Comparative CNN Study")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without training")
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch per run for testing")
    parser.add_argument("--force", action="store_true", help="Rerun completed experiments")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--tasks", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        tasks = [t.split(",") for t in args.tasks.split("|") if t]
        for arch, seed in tasks:
            try:
                run_worker(arch, int(seed), args.gpu, smoke_test=args.smoke_test)
            except Exception as e:
                cprint(f"  ERROR: {ARCH_LABELS.get(arch, arch)} seed {seed}: {e}", C.RED)
                traceback.print_exc()
                # Log error
                err_dir = LOG_DIR / arch
                err_dir.mkdir(parents=True, exist_ok=True)
                with open(err_dir / f"seed_{seed}_error.log", "w") as f:
                    f.write(traceback.format_exc())
    else:
        run_master(dry_run=args.dry_run, smoke_test=args.smoke_test, force=args.force)


if __name__ == "__main__":
    main()
