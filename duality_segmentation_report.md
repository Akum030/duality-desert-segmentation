# Duality AI Offroad Autonomy — Semantic Segmentation Report

**Track:** Semantic Segmentation (Duality Falcon synthetic desert environments)  
**Team:** (your team name here)  
**Date:** 2026-02-26  

---

## 1. Problem & Dataset Overview

### Task
Train a robust semantic segmentation model on synthetic desert environments captured via **Duality's Falcon digital twin platform**, then evaluate on unseen desert test images. The model must correctly label every pixel with one of 10 semantic classes covering typical off-road desert terrain.

### Dataset
| Split | Images | Source |
|-------|--------|--------|
| Train | 2857   | `Offroad_Segmentation_Training_Dataset/train/` |
| Val   | 317    | `Offroad_Segmentation_Training_Dataset/val/`   |
| Test  | TBD    | `Offroad_Segmentation_testImages/`             |

### Semantic Classes (10 total)

| ID | Class | Description |
|----|-------|-------------|
| 0  | Background | All unlabelled areas |
| 1  | Trees | Tall tree canopy |
| 2  | Lush Bushes | Dense green bushes |
| 3  | Dry Grass | Dry/yellow grass |
| 4  | Dry Bushes | Sparse dry bushes |
| 5  | Ground Clutter | Mixed small objects on ground |
| 6  | Logs | Fallen logs/branches |
| 7  | Rocks | Boulders and rocks |
| 8  | Landscape | General desert ground / terrain |
| 9  | Sky | Sky region |

Segmentation masks use raw pixel values (0, 100, 200, 300, 500, 550, 700, 800, 7100, 10000) which are remapped to class IDs 0–9 at load time.

---

## 2. Model & Training Approach

### Backbone
We use **ResNet50 pretrained on ImageNet** (torchvision `IMAGENET1K_V1`) as a **frozen** feature extractor. The final average-pooling and FC layers are removed. The output is the last convolutional feature map (`[B, 2048, H', W']`).

### Segmentation Head — SegmentationHeadConvNeXt
A lightweight ConvNeXt-style decoder trained on top of the frozen features:

```
ResNet50 last conv (2048-ch)
  → bilinear resize to token grid (19×34 = 646 tokens)
  → stem: Conv2d(2048→128, k=7) + GELU
  → block1: depthwise Conv2d(k=7) + GELU + pointwise Conv2d(1×1) + GELU
  → block2: depthwise Conv2d(k=7) + GELU + pointwise Conv2d(1×1) + GELU
  → classifier: Conv2d(128→10, k=1)
  → bilinear upsample back to input resolution
```

Input images are resized to **266×476** (h×w). Features are extracted at a 19×34 grid and upsampled back to full resolution.

### Loss Function
**CrossEntropyLoss** — standard multi-class pixel-wise cross-entropy, equal weight for all classes.

### Optimizer
**SGD with Nesterov momentum** (momentum=0.9, weight_decay=1e-4)  
**Scheduler: CosineAnnealingLR** — smooth LR decay from LR to LR×0.01 over all epochs.

---

## 3. Training Strategy & Improvements

### Baseline Run (v1) — 2026-02-26
| Parameter | Value |
|-----------|-------|
| Head depth | 1 ConvNeXt block |
| Epochs | 10 |
| Batch size | 2 |
| LR | 1e-4 |
| Augmentation | Resize + normalize only |
| Optimizer | SGD (no weight decay, no Nesterov) |
| Scheduler | None |

### Improved Run (v2) — 2026-02-26
| Parameter | Value | Why |
|-----------|-------|-----|
| Head depth | **2 ConvNeXt blocks** | Deeper head captures richer patterns |
| Epochs | **15** | More convergence within ~50 min (batch_size=4) |
| Batch size | **4** | Stable gradients, fewer iterations/epoch |
| LR | **3e-4** | Higher initial LR + cosine decay |
| Augmentation | **HorizontalFlip + ColorJitter + BrightnessContrast + GaussNoise** | See below |
| Optimizer | **SGD + Nesterov + weight_decay=1e-4** | Better convergence |
| Scheduler | **CosineAnnealingLR** | Smooth LR decay |

#### Augmentation Details
Implemented with **albumentations** (already installed, v2.0.8) for exact image+mask alignment:

| Augmentation | Why for Desert |
|---|---|
| `HorizontalFlip(p=0.5)` | Desert terrain is horizontally symmetric; doubles effective dataset |
| `ColorJitter(bright/contrast/sat/hue)` | Invariance to synthetic lighting conditions across day times |
| `RandomBrightnessContrast` | Handles bloom, shadows in Falcon renderer |
| `GaussNoise` | Robustness to rendering artefacts |

---

## 4. Results & Metrics

### Summary Comparison Table

| Run | Epochs | Batch | LR | Augmentation | Best Val IoU | Best Val Dice | Best Val Acc | Best Epoch |
|-----|--------|-------|----|---|:---:|:---:|:---:|:---:|
| **v1 Baseline** | 10 | 2 | 1e-4 | None | 0.2305 | 0.3633 | 0.6608 | 9 |
| **v2 Improved** | 15 | 4 | 3e-4 | Strong | *running* | *running* | *running* | *pending* |

> Update the v2 row from `train_stats/v2/evaluation_metrics_v2.txt` once training completes.

### v1 Baseline — Key Highlights
- **Best validation IoU: 0.2305** at Epoch 9
- **Best validation Dice: 0.3633** at Epoch 9
- **Final validation pixel accuracy: 0.6608** (Epoch 10)
- Metric curves: `train_stats/all_metrics_curves.png`

### v1 Per-Epoch History

| Epoch | Train Loss | Val Loss | Train IoU | Val IoU | Val Dice | Val Acc |
|-------|-----------|----------|-----------|---------|----------|---------|
| 1  | 1.3927 | 1.1641 | 0.1867 | 0.1795 | 0.2993 | 0.6026 |
| 2  | 1.1138 | 1.0868 | 0.2101 | 0.1971 | 0.3231 | 0.6219 |
| 3  | 1.0609 | 1.0468 | 0.2223 | 0.2066 | 0.3343 | 0.6366 |
| 4  | 1.0314 | 1.0239 | 0.2240 | 0.2092 | 0.3377 | 0.6425 |
| 5  | 1.0115 | 1.0038 | 0.2372 | 0.2188 | 0.3500 | 0.6488 |
| 6  | 0.9968 | 0.9928 | 0.2414 | 0.2230 | 0.3552 | 0.6524 |
| 7  | 0.9857 | 0.9826 | 0.2468 | 0.2278 | 0.3610 | 0.6551 |
| 8  | 0.9763 | 0.9729 | 0.2449 | 0.2262 | 0.3585 | 0.6577 |
| 9  | 0.9685 | 0.9670 | 0.2491 | **0.2305** | **0.3633** | 0.6593 |
| 10 | 0.9614 | 0.9592 | 0.2480 | 0.2303 | 0.3632 | **0.6608** |

### Plot Locations

| File | Content |
|------|---------|
| `train_stats/training_curves.png` | Loss + pixel accuracy (v1) |
| `train_stats/iou_curves.png` | IoU per epoch (v1) |
| `train_stats/dice_curves.png` | Dice per epoch (v1) |
| `train_stats/all_metrics_curves.png` | 4-panel combined (v1) |
| `train_stats/v2/training_curves.png` | Loss + pixel accuracy (v2) |
| `train_stats/v2/iou_curves.png` | IoU per epoch (v2) |
| `train_stats/v2/dice_curves.png` | Dice per epoch (v2) |
| `train_stats/v2/all_metrics_curves.png` | 4-panel combined (v2) |

---

## 5. Failure Cases & Future Work

### Current Failure Modes (observed from v1 metrics)

- **Rare classes (Logs, Ground Clutter):** Very few training pixels → model avoids predicting them → drags down mean IoU. Class-weighted loss would directly address this.
- **Visually similar classes:** Dry Grass / Dry Bushes / Landscape are texturally similar in Falcon synthetic renders. The model frequently confuses them, especially at class boundaries.
- **Small/thin objects (Logs, thin Rocks):** The 19×34 token grid has a large effective receptive field but low resolution. Thin structures are merged into background.
- **Sky-landscape boundary:** Hard edge artifacts at the sky/terrain horizon sometimes get misclassified.

### Proposed Future Improvements

| Improvement | Expected Gain | Effort |
|---|---|---|
| Class-weighted CrossEntropyLoss (up-weight Logs, Rocks, Clutter) | +3–5% mIoU rare classes | Low |
| Partially unfreeze ResNet50 layer4, LR=1e-5 | +2–4% overall mIoU | Low |
| Stronger backbone (ResNet101 / ConvNeXt-Base) | +5–10% mIoU | Medium |
| Multi-scale feature fusion (FPN-style) | +3–6% mIoU | Medium |
| Test-time augmentation (horizontal flip avg) | +1–2% mIoU | Very Low |
| Replace bilinear up-sampling with learnable transpose conv | +1–2% fine detail mIoU | Low |
| Elastic distortion augmentation | +1–2% mIoU | Low |

---

## 6. How to Reproduce

```bash
# 1. Setup
cd /home3/indiamart/gbht

# 2. Train (v2 — estimated ~50 min on A100/V100)
sudo bash -c "cd /home3/indiamart/gbht && \
  /home3/indiamart/gbht/edu_env/bin/python3 train_segmentation.py \
  > duality_train_v2.log 2>&1"

# 3. Monitor
tail -f /home3/indiamart/gbht/duality_train_v2.log

# 4. Run inference on test images
sudo bash -c "cd /home3/indiamart/gbht && \
  /home3/indiamart/gbht/edu_env/bin/python3 test_segmentation.py"

# 5. Results
cat train_stats/v2/evaluation_metrics_v2.txt
cat test_metrics.txt
```

### File Map

| File | Description |
|------|-------------|
| `train_segmentation.py` | Full training script with all v2 improvements |
| `test_segmentation.py` | Inference script — loads best weights, saves colored masks |
| `segmentation_head_best.pth` | **Best-epoch model weights** (use for submission) |
| `segmentation_head.pth` | Last-epoch model weights |
| `duality_train_v2.log` | Full training log |
| `train_stats/v2/evaluation_metrics_v2.txt` | Complete v2 metrics |
| `train_stats/v2/*.png` | Training curves |
| `Offroad_Segmentation_testImages/Predicted_Masks/` | Color-coded test predictions |
| `test_metrics.txt` | Test inference summary |

---

## 7. Design Decisions

### Why ResNet50 instead of DINOv2?
The original code attempted DINOv2 but it requires internet download and has complex API (`forward_features`). ResNet50 is available offline via torchvision and provides strong 2048-dim spatial features at 1/32 resolution — well suited for dense prediction with bilinear upsampling.

### Why Keep the Backbone Frozen?
~2857 training images and ~1 hour GPU budget. Fine-tuning a full ResNet50 risks overfitting and extends training time by ~3×. The frozen backbone provides stable ImageNet features, and the task-specific segmentation head has sufficient capacity.

---

*Report auto-generated. Update the v2 results table once `train_stats/v2/evaluation_metrics_v2.txt` is available.*
