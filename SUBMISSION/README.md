# Off-Road Desert Semantic Segmentation using Synthetic Data – Team Aidhunik

## Overview

This project trains a semantic segmentation model on synthetic desert scenes from the Duality dataset, provided as part of the Hack For Green Bharat hackathon. The model learns to label each pixel in an image as one of ten terrain and object classes — things like trees, rocks, dry grass, sky, and ground. We used ResNet50 as a frozen feature extractor and trained a small multi-scale FPN segmentation head on top of it. The backbone produces features at three spatial scales, which the head fuses together to produce detailed per-pixel predictions. Our goal was to build something that generalises well from one simulated desert environment to an unseen test set, using only the provided training images.

## Repository Structure

```
train_segmentation.py       training script (ResNet50 + FPN segmentation head)
train_segmentation_v3.py    full v3 training script with multi-scale FPN and combined loss
test_segmentation.py        inference script — runs on test images and saves predicted masks
visualize.py                optional helper for visualising predictions
models/
    segmentation_head_best.pth    best trained model weights (120 MB)
train_stats/
    evaluation_metrics.txt        v1 baseline training results
    training_curves.png           loss and accuracy curves
    iou_curves.png                IoU curve over epochs
    dice_curves.png               Dice score curve
    all_metrics_curves.png        combined metrics plot
    v3/
        evaluation_metrics_v3.txt  v3 FPN training results
        *.png                      v3 training plots
requirements.txt            Python package requirements
README.md                   this file
REPORT.md                   detailed hackathon report
FORM_SNIPPETS.md            ready-to-copy answers for the submission form
SUBMISSION/                 packaged folder for final zip submission
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

Download the dataset from the official problem statement link (not included in this repo). The expected folder layout is:

```
Offroad_Segmentation_Training_Dataset/
  train/
    Color_Images/
    Segmentation/
  val/
    Color_Images/
    Segmentation/

Offroad_Segmentation_testImages/
  Color_Images/
  Segmentation/     (optional, only if ground truth is available)
```

**Important:** the test images from `Offroad_Segmentation_testImages/` were only used for final inference — they were never used during training or validation.

## Training

```bash
python3 train_segmentation_v3.py
# or in the background:
nohup python3 train_segmentation_v3.py > train.log 2>&1 &
```

Training outputs:
- Best model weights saved to `models/segmentation_head_best.pth`
- Training curves and metrics saved under `train_stats/v3/`

Expected time: about 3–4 minutes per epoch on a T4 GPU. We trained for 15 epochs.

## Testing / Inference

```bash
python3 test_segmentation.py
```

This loads the best weights from `models/segmentation_head_best.pth`, runs inference on all images in `Offroad_Segmentation_testImages/Color_Images/`, and saves:
- Colour-coded segmentation masks to `outputs_test/`
- A metrics summary to `test_metrics_v3.txt`

## Results

Our best model (v3, epoch 15) achieves:

| Metric         | Baseline (v1, epoch 9) | Best (v3, epoch 15) |
|----------------|------------------------|----------------------|
| Val IoU        | 0.2305                 | **0.3903**           |
| Val Dice       | 0.3633                 | **~0.5615**          |
| Val Accuracy   | 0.6608                 | **0.8120**           |

That is roughly a **+70% relative improvement in IoU** over our starting baseline.

## Notes

- The dataset belongs to the hackathon organisers and is not redistributed here.
- This project was built under hackathon conditions — limited GPU time (a single T4) and a tight deadline.
- The model uses ResNet50 pretrained on ImageNet. We did not use any test images for training.
