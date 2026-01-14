#  Corneal Ulcer Segmentation with ROI-Guided ResUNet

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![ResNet](https://img.shields.io/badge/Encoder-ResNet34-blue?style=for-the-badge)
![Dice](https://img.shields.io/badge/Best%20Dice-0.760-success?style=for-the-badge)

**Automatic segmentation of corneal ulcers using transfer learning and anatomical ROI masking**

*SUSTech-SYSU Fluorescein Staining Dataset*

[Overview](#-overview) •
[Dataset](#-dataset) •
[Methods](#-methodology) •
[Results](#-results) •
[Usage](#-how-to-run)

</div>

---

##  Table of Contents

- [Overview](#-overview)
- [Background](#-background)
- [Dataset](#-dataset)
- [Dataset Examples](#-dataset-examples)
- [Methodology](#-methodology)
- [Pipeline](#-pipeline)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Requirements](#-requirements)
- [References](#-references)

---

##  Overview

This project implements a **U-Net style decoder with a ResNet34 encoder** for automatic segmentation of corneal ulcers from fluorescein-stained slit-lamp images. We explore how **transfer learning** and **anatomical priors** (cornea-based ROI masking) can improve segmentation accuracy compared to training from scratch.

### Experiments

| # | Experiment | Description |
|:-:|------------|-------------|
| 1️ | **Scratch (no ROI)** | Train from scratch without any masking |
| 2 | **Pretrained (no ROI)** | ImageNet pretrained encoder, no masking |
| 3️ | **Pretrained + ROI** | Pretrained encoder + mask outside cornea region |

---

##  Pipeline

<img src="assets/pipeline.PNG" alt="Pipeline overview" width="800">

---

##  Background

Corneal ulcers are a serious ophthalmic condition that can lead to **vision loss** if not diagnosed and monitored accurately. Manual annotation of ulcer regions is:

-  **Time-consuming** - requires significant clinical expertise
-  **Subjective** - varies between clinicians
-  **Inconsistent** - difficult to track progression over time

**Automatic segmentation** aims to assist clinical workflows by providing:
-  Fast, real-time analysis
-  Consistent and objective measurements
-  Accurate localization of ulcer regions

---

##  Dataset

We use the **SUSTech-SYSU** corneal ulcer fluorescein-staining dataset.

### Dataset Components

| Component | Description |
|-----------|-------------|
| `rawImages/` | Original fluorescein-stained slit-lamp photographs |
| `corneaLabels/` | Binary masks delineating the cornea boundary |
| `ulcerLabels/` | Binary masks for ulcer regions (ground truth) |
| `corneaOverlay/` | Visualization of cornea annotations |
| `ulcerOverlay/` | Visualization of ulcer annotations |

### Data Splits

| Split | Samples | Purpose |
|-------|:-------:|---------|
| Train | 247 | Model training |
| Validation | 53 | Hyperparameter tuning |
| Test | 54 | Final evaluation |

>  Dataset download link available in `data/dataset_url`

---

##  Dataset Examples

Below is a sample image from the dataset showing a fluorescein-stained corneal image with ulcer annotation overlay:

<img src="assets/sample_overlay.jpg" alt="Corneal ulcer with annotation overlay" width="500">

*Fluorescein-stained slit-lamp image with ulcer boundaries marked in green and cornea boundary in red. The green/yellow fluorescence indicates damaged corneal epithelium (ulcer regions).*

---

##  Methodology

### Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ResUNet Architecture                      │
├─────────────────────────────────────────────────────────────┤
│  Encoder: ResNet34 (pretrained on ImageNet)                 │
│  Decoder: U-Net style with skip connections                 │
│  Output:  1-channel sigmoid (binary segmentation)           │
└─────────────────────────────────────────────────────────────┘
```

### Loss Function

Combined loss for handling class imbalance:

```
Loss = 0.5 × BCEWithLogitsLoss + 0.5 × DiceLoss
```

### ROI Masking Strategy

The **ROI (Region of Interest)** approach masks everything **outside the cornea** using `corneaLabels`:

```
masked_image = original_image × cornea_mask
```

This focuses the model's attention on the relevant anatomical region, reducing false positives from background artifacts (eyelids, eyelashes, etc.).

> **Note:** When ROI masking is enabled, pixels outside the cornea boundary are set to zero (black) before being fed to the model. This helps the network focus exclusively on the corneal region.

### Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Dice** | 2\|A∩B\| / (\|A\|+\|B\|) | Overlap similarity (F1 for segmentation) |
| **IoU** | \|A∩B\| / \|A∪B\| | Intersection over Union (Jaccard Index) |

---

##  Results

### Quantitative Results

| Method | Test Dice ↑ | Test IoU ↑ | Improvement |
|--------|:-----------:|:----------:|:-----------:|
| Scratch (no ROI) | 0.656 | 0.552 | baseline |
| Pretrained (no ROI) | 0.712 | 0.603 | +8.5% Dice |
| **Pretrained + ROI** | **0.760** | **0.657** | **+15.8% Dice** |

### Performance Comparison

```
Dice Score Comparison
═══════════════════════════════════════════════════
Scratch (no ROI)      █████████████░░░░░░░░░░░  0.656
Pretrained (no ROI)   ██████████████░░░░░░░░░░  0.712
Pretrained + ROI      ███████████████░░░░░░░░░  0.760
═══════════════════════════════════════════════════
                      0.0        0.5        1.0
```

### Training Details

| Hyperparameter | Value |
|----------------|-------|
| Image Size | 256×256 |
| Batch Size | 8 |
| Epochs | 15 |
| Learning Rate | 2e-4 |
| Weight Decay | 1e-4 |
| Optimizer | AdamW |
| LR Scheduler | ReduceLROnPlateau |

---

##  Key Findings

### 1. Transfer Learning Matters
> **+8.5% Dice improvement** when using ImageNet pretrained weights vs. training from scratch

Pretrained encoders provide better feature extraction, especially beneficial for small medical imaging datasets (only 247 training samples).

### 2. Anatomical Priors Boost Performance
> **+6.7% additional Dice improvement** when adding ROI masking to Pretrained model

Masking irrelevant background regions (eyelids, eyelashes) helps the model focus on the target anatomy and reduces false positives.

### 3. Combined Approach is Best
> **+15.8% total improvement** using both pretrained weights + ROI masking

The synergy between transfer learning and domain-specific preprocessing yields the best results.

### 4. Faster Convergence with ROI
The Pretrained+ROI model reached val_dice > 0.77 by epoch 6, while the Pretrained model reached similar performance only by epoch 11.

---

##  Repository Structure

```
 Corneal-Ulcer-Segmentation/
├──  data/
│   ├──  dataset_url              # Link to download the dataset
│   ├──  dataset_index.csv        # Dataset split index
│   ├──  rawImages/               # Original images
│   ├──  corneaLabels/            # Cornea masks
│   ├──  ulcerLabels/             # Ulcer masks (ground truth)
│   ├──  corneaOverlay/           # Cornea visualizations
│   └──  ulcerOverlay/            # Ulcer visualizations
│
├──  notebooks/
│   ├──  EDA.ipynb                # Dataset validation + visuals
│   └──  training_and_evaluation.ipynb  # Training + evaluation + plots
│
├──  src/
│   └──  build_dataset_index.py   # Builds dataset index and splits
│
├──  assets/
│   ├──  pipeline.PNG             # Pipeline visualization
│   ├──  sample_image.jpg         # Example dataset image
│   └──  sample_mask.png          # Example ulcer mask
│
└──  README.md
```

---

##  How to Run

### Google Colab (Recommended)

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Navigate to project folder
%cd /content/drive/MyDrive/path/to/project

# 3. Verify dataset index exists
!ls data/dataset_index.csv

# 4. Run training notebook
# Open notebooks/training_and_evaluation.ipynb and run all cells
```

### Local Setup

```bash
# Clone repository
git clone <repository-url>
cd Corneal-Ulcer-Segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build dataset index (if needed)
python src/build_dataset_index.py

# Run Jupyter
jupyter notebook notebooks/training_and_evaluation.ipynb
```

---

##  Requirements

```txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
albumentations>=1.0.0
segmentation-models-pytorch>=0.3.0
jupyter>=1.0.0
tqdm>=4.62.0
```

---

##  References

1. **Dataset**: Deng, L., et al. "SUSTech-SYSU: A Benchmark for Clinical Corneal Ulcer Fluorescein Staining Image Segmentation"

2. **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"

3. **ResNet**: He, K., et al. (2016). "Deep Residual Learning for Image Recognition"

4. **Transfer Learning in Medical Imaging**: Tajbakhsh, N., et al. (2016). "Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?"

---

##  License

This project is for **academic and research purposes** only.

