# Research Codebase Context and Alignment Document

## Purpose

This document is a working context file for future conversations about
the facial paralysis computer vision and deep learning research project.

It records the known project structure, dataset organization,
preprocessing pipeline, training scripts, experiment protocol, results,
known issues, and current research direction.

Use this document as the primary project-context reference when
continuing work. Do not assume that an item is currently true if a later
conversation explicitly changes it.

------------------------------------------------------------------------

# 1. Project Overview

The project is a research-grade facial paralysis
screening/classification system using computer vision and deep learning.

Current binary classification task:

-   `Normal`
-   `Palsy`

The project compares different image representations and CNN approaches,
with particular emphasis on subject-independent evaluation.

The current main CNN architecture is:

-   ResNet50
-   ImageNet pretrained
-   224 x 224 input
-   Binary classification

The two primary input representations are:

1.  MediaPipe facial mesh rendered as white lines on a pure black
    background
2.  Original RGB facial images

The main evaluation protocol is:

-   Subject-level Leave-One-Subject-Out (LOSO)
-   64 total subjects
-   64 LOSO folds
-   Every subject is held out exactly once

The current research goal is to determine how well the models generalize
to unseen subjects and to compare RGB versus facial-mesh
representations.

------------------------------------------------------------------------

# 2. HPC Environment

## Operating environment

The project is being run on a Linux HPC/server environment.

Project root:

``` text
/home/cai/Documents/Masters/Datasets/Dataset for Augment
```

Virtual environment:

``` text
/home/cai/Documents/Masters/Datasets/Dataset for Augment/.venv
```

Activation:

``` bash
cd "$HOME/Documents/Masters/Datasets/Dataset for Augment"
source .venv/bin/activate
```

Python version:

``` text
Python 3.12.3
```

## GPU hardware

The server has:

``` text
4 x Tesla V100-PCIE-32GB
32 GB VRAM per GPU
```

Previously verified NVIDIA environment:

``` text
NVIDIA Driver: 535.309.01
CUDA capability reported by nvidia-smi: 12.2
```

GPU availability and CUDA matrix multiplication were previously verified
successfully.

Recommended PyTorch installation used for the environment:

``` bash
python -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

GPU monitoring:

``` bash
watch -n 1 nvidia-smi
```

Compact GPU monitoring:

``` bash
watch -n 1 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv"
```

------------------------------------------------------------------------

# 3. Python Dependencies

Important packages used by the project include:

``` text
numpy
pandas
matplotlib
seaborn
mediapipe
opencv-python
scikit-learn
pillow
tqdm
torch
torchvision
torchaudio
statsmodels
```

Current pinned basic requirements used:

``` text
numpy==1.26.4
pandas==2.2.3
matplotlib==3.10.1
seaborn==0.13.2
```

MediaPipe compatibility issue was solved by using:

``` bash
python -m pip install mediapipe==0.10.21
```

Verification:

``` bash
python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"
```

------------------------------------------------------------------------

# 4. Original Dataset Structure

The project originally contained:

``` text
Dataset for Augment/
├── Normal Set/
│   ├── TD_RGB_E_Set1/
│   ├── TD_RGB_E_Set2/
│   ├── TD_RGB_E_Set3/
│   ├── TD_RGB_E_Set4/
│   └── Subjects/
│
└── Palsy Set/
    ├── Image/
    │   └── Image/
    ├── Image2/
    │   └── Image2/
    ├── Image3/
    │   └── Image3/
    ├── Image4/
    │   └── Image4/
    └── Subjects/
```

------------------------------------------------------------------------

# 5. Normal Dataset

The normal source is AFLFP.

The original normal set is organized into source sets:

``` text
Normal Set/
├── TD_RGB_E_Set1/
│   └── subjects 1-25
├── TD_RGB_E_Set2/
│   └── subjects 26-50
├── TD_RGB_E_Set3/
│   └── subjects 51-75
├── TD_RGB_E_Set4/
│   └── subjects 76-113
└── Subjects/
    └── subjects 1-113
```

The project ultimately uses 32 AFLFP subjects for the balanced binary
experiment.

Normal subject identifiers in the training/LOSO manifest use the form:

``` text
aflfp_001
aflfp_002
...
aflfp_113
```

The selected experiment uses 32 normal subjects.

------------------------------------------------------------------------

# 6. Palsy Dataset

The palsy source is YFP.

The source dataset is divided as follows:

``` text
Palsy Set/Image/Image/
    subjects 1-8

Palsy Set/Image2/Image2/
    subjects 9-16

Palsy Set/Image3/Image3/
    subjects 17-24

Palsy Set/Image4/Image4/
    subjects 25-32
```

Palsy subjects are identified numerically in the prepared RGB
dataset/manifest:

``` text
1
2
3
...
32
```

Important discovery:

YFP original RGB files are BMP files, not JPG/JPEG.

Example original RGB files:

``` text
Palsy Set/Image/Image/1/2371.bmp
Palsy Set/Image/Image/1/6016.bmp

Palsy Set/Image3/Image3/18/2576.bmp
Palsy Set/Image3/Image3/18/8111.bmp
Palsy Set/Image3/Image3/18/8336.bmp
Palsy Set/Image3/Image3/18/9956.bmp
```

Previously observed file counts in the Palsy Set:

``` text
71,306 BMP files
35,427 PNG files
```

The PNG files correspond to generated facial-mesh images.

------------------------------------------------------------------------

# 7. AFLFP Internal Structure

AFLFP subject directories contain facial-expression/action folders.

Example:

``` text
AFLFP/
├── 1/
│   ├── 1-brow raise/
│   │   └── key frames/
│   ├── 1-close smile/
│   │   └── key frames/
│   ├── 1-frown/
│   │   └── key frames/
│   ├── ...
│   └── 1-tight eye closure/
│       └── key frames/
├── 2/
└── ...
```

Only clean original images are used for mesh generation.

Files such as:

``` text
.jpg
.jpeg
```

are used as clean originals.

Files such as:

``` text
_lmk.png
.pts
```

are not used as the original RGB image input.

------------------------------------------------------------------------

# 8. Facial Mesh Preprocessing

The initial mesh representation uses MediaPipe Face Mesh.

Representation:

``` text
white facial mesh
on
pure black background
```

The mesh images are stored as PNG.

YFP mesh generation result:

``` text
35,427 completed
226 skipped
```

Output:

``` text
Palsy Set/mesh/Subjects
```

AFLFP mesh generation:

``` text
88 subjects processed
6,476 images processed
0 skipped
```

Output:

``` text
dataset/normal
```

The final binary classification ignores individual facial action type
and classifies at the subject/image level as:

``` text
Normal vs Palsy
```

------------------------------------------------------------------------

# 9. Balanced Dataset Protocol

The experiment was designed to balance the number of subjects.

Final subject counts:

``` text
Normal: 32 subjects
Palsy:  32 subjects
Total:  64 subjects
```

Palsy:

``` text
All 32 YFP subjects
```

Normal:

``` text
32 AFLFP subjects selected from the 113 available normal subjects
```

Maximum images per subject:

``` text
64
```

If a subject has more than 64 usable images:

``` text
randomly sample 64
```

If a subject has fewer than 64:

``` text
use all available images
```

The original source datasets are preserved and not modified.

------------------------------------------------------------------------

# 10. Original Training Dataset

An initial training dataset was created as:

``` text
Training/
├── train/
│   ├── normal/
│   └── palsy/
└── test/
    ├── normal/
    └── palsy/
```

The initial fixed holdout protocol used:

``` text
25 normal training subjects
25 palsy training subjects

7 normal test subjects
7 palsy test subjects
```

This gives:

``` text
50 training subjects
14 test subjects
64 total subjects
```

Within the training process, the training subjects were further split at
subject level for validation.

The validation split used:

``` text
20% validation
```

Approximate split:

``` text
16 normal train
16 palsy train

4 normal validation
4 palsy validation

7 normal test
7 palsy test
```

This fixed-holdout experiment was used before moving to LOSO.

------------------------------------------------------------------------

# 11. Training_RGB Dataset

The RGB ablation uses the exact same 64 subjects and LOSO fold
assignments as the mesh experiment.

Structure:

``` text
Training_RGB/
├── normal/
│   ├── aflfp_003/
│   ├── ...
│
├── palsy/
│   ├── 1/
│   ├── 2/
│   ├── ...
│
└── dataset_manifest.csv
```

The purpose is to make the RGB experiment directly comparable to the
mesh experiment.

The manifest contains information including:

``` text
subject_id
class
label
fold
image_path
```

Expected labels:

``` text
0 = Normal
1 = Palsy
```

The LOSO fold assignment is stored in:

``` text
Training_RGB/dataset_manifest.csv
```

The same 64 subjects are used for both the mesh and RGB LOSO
experiments.

------------------------------------------------------------------------

# 12. Subject-Level LOSO Protocol

The primary evaluation protocol is Leave-One-Subject-Out.

There are:

``` text
64 subjects
64 folds
```

For each fold:

``` text
1 subject = test subject

63 subjects = development data
```

The 63 remaining subjects are split into:

``` text
training subjects
validation subjects
```

The held-out test subject is never used for:

``` text
training
augmentation
validation
threshold selection
early stopping
```

The model is trained only using the remaining subjects.

The held-out subject is evaluated after the best model is selected.

This avoids image-level leakage caused by having images from the same
person in both training and testing.

------------------------------------------------------------------------

# 13. Subject-Level Prediction Aggregation

Each subject may contain multiple images.

The current LOSO protocol aggregates image predictions using:

``` text
mean image probability
```

For each subject:

``` text
subject_probability =
mean(probability of all subject images)
```

The final subject prediction is:

``` text
probability >= threshold
```

for Palsy.

Otherwise:

``` text
Normal
```

The current LOSO threshold strategy is:

``` text
Youden J from validation subjects
```

The threshold is selected using validation subjects and then applied to
the held-out test subject.

Important methodological note:

The current implementation uses a separate threshold for each LOSO fold.

For a final paper analysis, fixed 0.5 and/or a global out-of-fold
threshold may also be worth reporting as a sensitivity analysis.

------------------------------------------------------------------------

# 14. ResNet50 Architecture

Primary CNN:

``` text
ResNet50
```

Pretraining:

``` text
ImageNet
```

Torchvision weights:

``` text
ResNet50_Weights.IMAGENET1K_V2
```

Final fully connected layer:

``` text
2 outputs
```

for:

``` text
Normal
Palsy
```

Input size:

``` text
224 x 224
```

------------------------------------------------------------------------

# 15. RGB ResNet50 Preprocessing

RGB pipeline:

``` text
Original RGB
    ↓
Resize 224 x 224
    ↓
Geometric augmentation during training
    ↓
Tensor conversion
    ↓
ImageNet normalization
    ↓
ResNet50
```

ImageNet normalization:

``` text
mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]
```

Training augmentation:

``` text
RandomHorizontalFlip(p=0.5)
RandomRotation(10 degrees)
RandomAffine(
    translate=(0.05, 0.05),
    scale=(0.95, 1.05)
)
```

Heavy color augmentation, heavy blur, and aggressive cropping are
intentionally avoided.

------------------------------------------------------------------------

# 16. Training Hyperparameters

Current primary ResNet50 settings:

``` text
Batch size per GPU: 32
GPU count: 4
Learning rate: 1e-4
Weight decay: 1e-4
Optimizer: AdamW
Scheduler: ReduceLROnPlateau
Scheduler factor: 0.5
Scheduler patience: 2
Maximum epochs: 50
Early stopping patience: 7
Validation ratio: 20%
Seed: 456
```

Mixed precision:

``` text
CUDA AMP / float16
```

------------------------------------------------------------------------

# 17. Four-GPU LOSO Strategy

Important implementation decision:

LOSO folds are independent experiments.

Therefore, the preferred four-GPU strategy is:

``` text
GPU 0 -> folds 1, 5, 9, 13, ...
GPU 1 -> folds 2, 6, 10, 14, ...
GPU 2 -> folds 3, 7, 11, 15, ...
GPU 3 -> folds 4, 8, 12, 16, ...
```

Each GPU independently trains its assigned folds.

DDP should NOT be used to make the four GPUs cooperate on a single fold
while the ranks are simultaneously executing different folds.

The current RGB LOSO script was specifically updated to use independent
fold processes under torchrun.

Launch:

``` bash
torchrun --standalone --nproc_per_node=4 train_rgb_resnet50_loso.py
```

------------------------------------------------------------------------

# 18. Important Codebase Scripts

Known scripts and their purposes:

## prepare_training_dataset.py

Purpose:

Creates the balanced fixed-holdout training dataset.

Uses:

``` text
32 normal subjects
32 palsy subjects
maximum 64 images per subject
subject-level split
```

Output:

``` text
Training/
```

------------------------------------------------------------------------

## train_resnet50.py

Purpose:

Fixed-holdout MediaPipe mesh ResNet50 training.

Uses:

``` text
4 GPUs
DDP
ImageNet pretrained ResNet50
224 x 224
AdamW
ReduceLROnPlateau
AMP
early stopping
```

Launch:

``` bash
torchrun --standalone --nproc_per_node=4 train_resnet50.py --epochs 50
```

Smoke test:

``` bash
torchrun --standalone --nproc_per_node=4 train_resnet50.py --epochs 2
```

------------------------------------------------------------------------

## loso_resnet50.py

Purpose:

Primary MediaPipe mesh LOSO ResNet50 experiment.

Output:

``` text
Training/results/resnet50_loso/
```

Contains:

``` text
fold_001/
fold_002/
...
fold_064/

loso_subject_predictions.csv
overall_loso_metrics.csv
confusion_matrix.png
roc_curve.png
summary JSON
```

Launch:

``` bash
torchrun --standalone --nproc_per_node=4 loso_resnet50.py
```

------------------------------------------------------------------------

## RGB dataset arrangement script

Purpose:

Creates the RGB dataset using the exact 64 subjects and existing LOSO
fold assignments.

Important source mappings:

``` text
Normal:
Normal Set/Subjects/<numeric subject>

Palsy:
1-8   -> Palsy Set/Image/Image/
9-16  -> Palsy Set/Image2/Image2/
17-24 -> Palsy Set/Image3/Image3/
25-32 -> Palsy Set/Image4/Image4/
```

Palsy original files are:

``` text
.bmp
```

Normal source files are:

``` text
.jpg
.jpeg
```

Output:

``` text
Training_RGB/
```

------------------------------------------------------------------------

## train_rgb_resnet50_loso.py

Purpose:

RGB ResNet50 LOSO ablation.

Uses:

``` text
Original RGB
64 subjects
64 LOSO folds
ResNet50 ImageNet pretrained
224 x 224
AdamW
ReduceLROnPlateau
AMP
subject-level aggregation
validation-derived Youden threshold
4 independent GPU workers
```

It produces:

``` text
Training_RGB/results/resnet50_loso/
```

with per-fold training and evaluation files plus aggregate
visualizations.

------------------------------------------------------------------------

# 19. Expected RGB LOSO Output Structure

Current intended output:

``` text
Training_RGB/results/resnet50_loso/
│
├── fold_001/
│   ├── best_model.pth
│   ├── training_history.csv
│   ├── training_curves.png
│   ├── accuracy_curves.png
│   ├── learning_rate.png
│   ├── test_subject_prediction.csv
│   └── metrics.csv
│
├── fold_002/
├── ...
├── fold_064/
│
├── loso_subject_predictions.csv
├── overall_loso_metrics.csv
├── training_summary.csv
├── confusion_matrix.png
├── normalized_confusion_matrix.png
├── roc_curve.png
├── precision_recall_curve.png
├── prediction_distribution.png
├── fold_accuracy.png
├── fold_loss.png
├── fold_sensitivity_specificity.png
├── experiment_config.json
└── worker_XX_results.csv
```

------------------------------------------------------------------------

# 20. Fixed-Holdout Results: Seed 42

Mesh ResNet50, fixed holdout.

Test results:

``` text
Accuracy:     92.67%
Precision:    95.72%
Sensitivity:  89.96%
Specificity:  95.63%
F1:           92.75%
ROC-AUC:      0.9794
```

Subject-level:

``` text
Accuracy:     92.86%
Precision:    100.00%
Sensitivity:  85.71%
Specificity:  100.00%
F1:           92.31%
ROC-AUC:      1.0000
```

Subject confusion matrix:

``` text
TN = 7
FP = 0
FN = 1
TP = 6
```

Validation-derived threshold observed:

``` text
approximately 0.8176
```

------------------------------------------------------------------------

# 21. Fixed-Holdout Results: Seed 123

Mesh ResNet50.

Validation:

``` text
Accuracy:     89.84%
Precision:    85.79%
Sensitivity:  96.25%
Specificity:  83.00%
F1:           90.72%
ROC-AUC:      0.9878
```

Test:

``` text
Accuracy:     94.65%
Precision:    98.09%
Sensitivity:  91.52%
Specificity:  98.06%
F1:           94.69%
ROC-AUC:      0.9922
```

Subject-level:

``` text
Accuracy:     85.71%
Precision:    100.00%
Sensitivity:  71.43%
Specificity:  100.00%
F1:           83.33%
```

------------------------------------------------------------------------

# 22. Fixed-Holdout Results: Seed 456

Best fixed-holdout mesh run.

Validation:

``` text
Accuracy:     90.58%
Precision:    86.80%
Sensitivity:  96.56%
Specificity:  84.12%
F1:           91.42%
ROC-AUC:      0.9698
```

Test:

``` text
Accuracy:     96.86%
Precision:    97.09%
Sensitivity:  96.88%
Specificity:  96.84%
F1:           96.98%
ROC-AUC:      0.9959
```

Subject-level:

``` text
14/14 correct
Accuracy:     100.00%
Precision:    100.00%
Sensitivity:  100.00%
Specificity:  100.00%
F1:           100.00%
ROC-AUC:      1.0000
```

Important:

The subject-level result is based on only 14 test subjects and must not
be interpreted as robust population-level generalization.

------------------------------------------------------------------------

# 23. Fixed-Holdout Results: Seed 789

Mesh ResNet50.

Validation:

``` text
Accuracy:     80.61%
Precision:    76.25%
Sensitivity:  90.31%
Specificity:  70.39%
F1:           82.69%
ROC-AUC:      0.9366
```

Test:

``` text
Accuracy:     78.49%
Precision:    70.97%
Sensitivity:  99.33%
Specificity:  55.83%
F1:           82.79%
ROC-AUC:      0.9844
```

Subject-level:

``` text
Accuracy:     78.57%
Precision:    70.00%
Sensitivity:  100.00%
Specificity:  57.14%
F1:           82.35%
ROC-AUC:      1.0000
```

Subject confusion matrix:

``` text
TN = 4
FP = 3
FN = 0
TP = 7
```

Important observation:

All seven Palsy subjects ranked above all seven Normal subjects,
producing subject-level AUC 1.0, but the default threshold caused three
Normal subjects to be classified as Palsy.

This demonstrates that ranking quality and threshold calibration are
different.

------------------------------------------------------------------------

# 24. Fixed-Holdout Results: Seed 1000

Mesh ResNet50.

Validation:

``` text
Accuracy:     84.35%
Precision:    77.67%
Sensitivity:  97.81%
Specificity:  70.00%
F1:           86.58%
ROC-AUC:      0.9728
```

Test:

``` text
Accuracy:     96.74%
Precision:    96.88%
Sensitivity:  96.88%
Specificity:  96.60%
F1:           96.88%
ROC-AUC:      0.9945
```

Subject-level:

``` text
Accuracy:     100.00%
Precision:    100.00%
Sensitivity:  100.00%
Specificity:  100.00%
F1:           100.00%
ROC-AUC:      1.0000
```

------------------------------------------------------------------------

# 25. Five-Seed Fixed-Holdout Aggregate

Test results:

``` text
Seed 42:
Accuracy 92.67%
Sensitivity 89.96%
Specificity 95.63%
F1 92.75%
AUC 0.9794

Seed 123:
Accuracy 94.65%
Sensitivity 91.52%
Specificity 98.06%
F1 94.69%
AUC 0.9922

Seed 456:
Accuracy 96.86%
Sensitivity 96.88%
Specificity 96.84%
F1 96.98%
AUC 0.9959

Seed 789:
Accuracy 78.49%
Sensitivity 99.33%
Specificity 55.83%
F1 82.79%
AUC 0.9844

Seed 1000:
Accuracy 96.74%
Sensitivity 96.88%
Specificity 96.60%
F1 96.88%
AUC 0.9945
```

Mean +/- SD:

``` text
Accuracy:     91.88 +/- 7.40%
Sensitivity:  92.91 +/- 4.26%
Specificity:  88.59 +/- 16.41%
F1:           92.82 +/- 5.97%
ROC-AUC:      0.9893 +/- 0.0070
```

Seed 789 must NOT be removed from the primary analysis.

It is an important demonstration of seed variability.

------------------------------------------------------------------------

# 26. Primary Mesh LOSO Result

Completed 64-subject mesh ResNet50 LOSO experiment:

``` text
Subjects:     64
Accuracy:     90.62%
Precision:    96.43%
Sensitivity:  84.38%
Specificity:  96.88%
F1:           90.00%
ROC-AUC:      0.9873

TN = 31
FP = 1
FN = 5
TP = 27
```

Therefore:

``` text
58 / 64 subjects correct
31 / 32 Normal correct
27 / 32 Palsy correct
```

This is currently the primary subject-independent mesh baseline.

------------------------------------------------------------------------

# 27. Current RGB LOSO Result

The RGB ResNet50 LOSO experiment has been completed.

Configuration:

``` text
Model: ResNet50
Pretrained: ImageNet
Input: Original RGB
Input size: 224 x 224
Subjects: 64
LOSO folds: 64
Batch size/GPU: 32
GPUs: 4
Optimizer: AdamW
Learning rate: 0.0001
Weight decay: 0.0001
Scheduler: ReduceLROnPlateau
Maximum epochs: 50
Early stopping patience: 7
Seed: 456
Classification: Normal vs Palsy
Evaluation: Subject-level LOSO
Subject aggregation: Mean image probability
Threshold: Youden J from validation subjects
```

Source configuration is confirmed in the uploaded experiment
configuration. fileciteturn2file0L2-L16
fileciteturn2file0L22-L28

Reported overall RGB result:

``` text
Accuracy:     100.00%
Precision:    100.00%
Sensitivity:  100.00%
Specificity:  100.00%
F1:           100.00%
ROC-AUC:      1.0000
Average Precision: 1.0000

TN = 32
FP = 0
FN = 0
TP = 32
```

This means:

``` text
64 / 64 subjects correctly classified
32 / 32 Normal correctly classified
32 / 32 Palsy correctly classified
```

------------------------------------------------------------------------

# 28. RGB Result Concerns Requiring Audit

The perfect RGB result is unusually strong and should be audited before
being presented as the definitive scientific conclusion.

Observed issues:

## 28.1 Nearly perfect probability separation

Normal subject probabilities were observed extremely close to zero.

Palsy probabilities were observed extremely close to one.

This suggests almost perfect separation.

Possible explanations:

``` text
Genuine discriminative information
Dataset/domain bias
Acquisition differences
Background differences
Camera differences
Resolution differences
Compression differences
Framing differences
Other source-specific artifacts
```

------------------------------------------------------------------------

## 28.2 Thresholds

The RGB LOSO outputs showed thresholds around:

``` text
0.01
```

for folds.

The configuration says thresholds are selected using Youden J on
validation subjects.

The fact that thresholds are repeatedly at 0.01 should be investigated.

This does not automatically invalidate the experiment, but it should be
verified.

------------------------------------------------------------------------

## 28.3 Per-fold metrics

Do not average per-fold precision, sensitivity, or specificity across
individual LOSO folds.

Each LOSO fold has only one held-out subject.

Therefore, many per-fold classification metrics are undefined or
degenerate.

The appropriate primary metrics should be calculated from the pooled 64
held-out subject predictions.

------------------------------------------------------------------------

## 28.4 One Palsy subject has only one image

The RGB result contains a Palsy subject with:

``` text
image_count = 1
```

while many other subjects have:

``` text
image_count = 64
```

This needs to be verified against the source dataset and arrangement
process.

If genuine, it should be documented.

If caused by preprocessing or arrangement, the experiment should be
corrected and rerun.

------------------------------------------------------------------------

# 29. Current Research Interpretation

The current comparison is:

``` text
                    Mesh ResNet50     RGB ResNet50

Subjects                  64                64
LOSO folds                 64                64
Architecture            ResNet50          ResNet50
Pretraining             ImageNet         ImageNet
Resolution              224x224          224x224
Optimizer               AdamW             AdamW
Learning rate            1e-4              1e-4
Weight decay             1e-4              1e-4
Evaluation              Subject LOSO      Subject LOSO

Accuracy                 90.62%            100.00%
Sensitivity              84.38%            100.00%
Specificity              96.88%            100.00%
F1                       90.00%            100.00%
ROC-AUC                   0.9873             1.0000
```

The initial finding is that RGB substantially outperformed the mesh
representation.

However, this must be interpreted carefully because Normal and Palsy
originate from different source datasets:

``` text
Normal = AFLFP
Palsy  = YFP
```

This creates a potential cross-dataset/domain bias.

The model may be learning dataset-specific visual characteristics rather
than only facial paralysis characteristics.

------------------------------------------------------------------------

# 30. Recommended Dataset Audit

Before further expensive training, perform:

## Subject count audit

Verify:

``` text
32 Normal subjects
32 Palsy subjects
64 total subjects
64 LOSO folds
```

## Duplicate audit

Check for:

``` text
duplicate images
duplicate file paths
same image assigned to multiple subjects
```

## Dimension audit

Compare Normal versus Palsy:

``` text
image width
image height
aspect ratio
```

## File statistics audit

Compare:

``` text
file size
image format
compression characteristics
```

## Pixel statistics audit

Compare:

``` text
mean brightness
standard deviation
channel means
channel standard deviations
```

between Normal and Palsy.

## Visual audit

Create representative image grids:

``` text
Normal samples | Palsy samples
```

and inspect:

``` text
background
framing
face location
camera characteristics
lighting
image quality
```

## Probability audit

Plot subject-level probability distributions.

Expected current pattern:

``` text
Normal probabilities -> near 0
Palsy probabilities  -> near 1
```

------------------------------------------------------------------------

# 31. Future Research Plan

Recommended progression:

## Phase 1: Validate current RGB result

Perform the dataset and prediction audit.

Do not immediately rerun the expensive LOSO experiment.

## Phase 2: Mesh versus RGB ablation

Compare:

``` text
Mesh ResNet50
RGB ResNet50
```

using the exact same 64 subjects and LOSO folds.

## Phase 3: Additional CNN architectures

Potential comparisons:

``` text
ResNet50
EfficientNet
DenseNet
ConvNeXt
```

Only after the baseline pipeline is validated.

## Phase 4: Representation ablation

Potential experiments:

``` text
RGB
Mesh
RGB + Mesh
```

## Phase 5: Robustness

Potential tests:

``` text
different random seeds
different augmentation settings
different subject splits
cross-dataset testing if possible
```

## Phase 6: Statistical reporting

Final paper reporting should include:

``` text
Accuracy
Precision
Sensitivity
Specificity
F1
ROC-AUC
Average Precision
Confusion matrix
ROC curve
PR curve
Confidence intervals
```

and where appropriate:

``` text
mean +/- SD
```

for repeated experiments.

------------------------------------------------------------------------

# 32. Important Research Limitations

Current limitations include:

1.  Normal and Palsy classes originate from different datasets.
2.  Cross-dataset domain bias may affect RGB performance.
3.  The dataset contains only 64 subjects in the current balanced
    experiment.
4.  Some subjects may contain fewer than 64 usable images.
5.  Subject-level sample size is much more important than total image
    count.
6.  A perfect LOSO result requires careful audit before being treated as
    definitive.
7.  The current fold-specific threshold methodology should be documented
    clearly.
8.  The model may exploit acquisition artifacts in original RGB images.
9.  Mesh preprocessing removes some appearance information, so RGB
    versus mesh is a representation comparison rather than a pure
    architecture comparison.
10. External validation on an independent dataset would provide stronger
    evidence of clinical/generalization performance.

------------------------------------------------------------------------

# 33. Current Working Principle

When continuing this project:

-   Preserve the original datasets.
-   Do not change the established 64-subject fold assignment unless
    explicitly conducting a new experiment.
-   Keep subject-level separation.
-   Never allow images from a held-out subject into training or
    validation.
-   Treat the 64-subject pooled LOSO metrics as the primary evaluation.
-   Do not interpret per-fold single-subject precision/sensitivity
    averages as meaningful aggregate metrics.
-   Keep RGB and mesh experiments directly comparable.
-   Record every experiment configuration.
-   Save model checkpoints and prediction files.
-   Prefer reproducible seeds.
-   Use all four GPUs efficiently for independent LOSO folds.
-   Investigate unexpectedly perfect performance before accepting it.
-   Do not silently remove outlier seeds.
-   Clearly distinguish exploratory findings from final paper results.

------------------------------------------------------------------------

# 34. Useful Commands

## Activate environment

``` bash
cd "$HOME/Documents/Masters/Datasets/Dataset for Augment"
source .venv/bin/activate
```

## Check Python

``` bash
python --version
```

## Check GPUs

``` bash
nvidia-smi
```

## Live GPU monitoring

``` bash
watch -n 1 nvidia-smi
```

## Compact GPU monitoring

``` bash
watch -n 1 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv"
```

## Check four GPUs from Python

``` bash
python -c "import torch; print('GPUs:', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

## Check script syntax

``` bash
python -m py_compile train_rgb_resnet50_loso.py
```

## Run RGB LOSO

``` bash
torchrun --standalone --nproc_per_node=4 train_rgb_resnet50_loso.py
```

## Run mesh LOSO

``` bash
torchrun --standalone --nproc_per_node=4 loso_resnet50.py
```

------------------------------------------------------------------------

# 35. Current Project State

At the latest known point:

``` text
Dataset preparation       COMPLETE
Mesh generation           COMPLETE
Balanced subject setup    COMPLETE
Fixed-holdout mesh runs   COMPLETE
5-seed mesh evaluation    COMPLETE
Mesh LOSO                 COMPLETE
RGB dataset arrangement   COMPLETE
RGB ResNet50 LOSO         COMPLETE
RGB visual outputs        GENERATED
```

Current primary mesh baseline:

``` text
90.62% subject-level LOSO accuracy
0.9873 subject-level ROC-AUC
```

Current RGB LOSO result:

``` text
100.00% subject-level LOSO accuracy
1.0000 ROC-AUC
```

The next recommended task is NOT another expensive training run.

The next task is:

``` text
AUDIT THE RGB DATASET AND PERFECT LOSO RESULT
```

especially:

``` text
dataset leakage
dataset/domain bias
image counts
image dimensions
source-specific artifacts
probability distributions
threshold behavior
```

Only after this audit should additional model experiments be considered.

------------------------------------------------------------------------

# 36. Conversation Alignment Rule

When discussing this project in future conversations, use the
terminology and structure in this document.

If a future uploaded file, terminal output, script, or user statement
conflicts with this document, treat the newer direct evidence as
authoritative and update the working context accordingly.

Do not assume a script is identical to a previously discussed version.
When the user provides a new script, inspect that exact script before
recommending modifications.

Do not infer unseen folder contents. If exact current filesystem
contents are required, ask the user for the relevant command output or
inspect a provided file/output.

This document is a project-context snapshot, not a claim that every path
or result remains unchanged forever.
