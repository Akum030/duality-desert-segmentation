# Off-Road Desert Semantic Segmentation using Synthetic Data

**Team:** Aidhunik  
**Hackathon:** Hack For Green Bharat  
**Date:** February 2026

---

## Summary

We built a semantic segmentation model to classify pixels in synthetic desert images provided by Duality, as part of the Hack For Green Bharat hackathon. The dataset contains colour images of off-road desert scenes paired with ground truth masks labelled across ten terrain and object classes — trees, bushes, rocks, dry ground, sky, and more. Our approach uses ResNet50 (pretrained on ImageNet) as a feature extractor at three spatial scales, combined with a Feature Pyramid Network (FPN) decoder and a ConvNeXt-style segmentation head trained from scratch. We also partly unfroze the deeper layers of the backbone to let the model adapt to desert textures. Training used a combined CrossEntropyLoss with class weights (to handle imbalanced classes like Logs) and a soft Dice loss term. In 15 epochs of training we achieved a validation IoU of 0.3903, which is a 70% relative improvement over our initial baseline of 0.2305. Pixel accuracy reached 81.2% compared to 66.1% at the start.

---

## Methodology

### Dataset

The dataset was provided by the organisers through the official problem statement. It contains synthetic desert scenes rendered by Duality with pixel-level annotations.

- **Training:** 2857 images
- **Validation:** 317 images
- **Test:** separate unlabelled set (never used for training)

**Classes (10 total):**

| Class ID | Name           | Raw Mask Value |
|----------|----------------|----------------|
| 0        | Background     | 0              |
| 1        | Trees          | 100            |
| 2        | Lush Bushes    | 200            |
| 3        | Dry Grass      | 300            |
| 4        | Dry Bushes     | 500            |
| 5        | Ground Clutter | 550            |
| 6        | Logs           | 700            |
| 7        | Rocks          | 800            |
| 8        | Landscape      | 7100           |
| 9        | Sky            | 10000          |

Raw mask values are remapped to class IDs 0–9 at load time. Class distribution is very imbalanced: Sky and Dry Grass dominate at ~42% and ~28% respectively, while Logs and Background appear in less than 0.01% of pixels each.

### Model Architecture

We use a three-part model:

**1. ResNet50MultiScale (backbone)**  
ResNet50 pretrained on ImageNet. Instead of using only the final feature map, we extract features at three scales:
- `layer2` output: [B, 512, H/8, W/8]
- `layer3` output: [B, 1024, H/16, W/16]
- `layer4` output: [B, 2048, H/32, W/32]

The first two stages (stem, layer1, layer2) are frozen. Layers 3 and 4 are fine-tuned with a low learning rate (5×10−5) to adapt to desert textures without overwriting general features.

**2. FPNSegHead (feature pyramid decoder)**  
The three feature maps are fused with a top-down FPN: 1×1 lateral projections reduce each to 256 channels, then we add top-down context progressively. All three scales are upsampled to H/8 and concatenated into a [B, 768, H/8, W/8] tensor. A small convolutional bottleneck reduces this to 512 channels, then three depthwise ConvNeXt blocks (with kernel sizes 7, 7, 5) compress it to a [B, 10, H/8, W/8] logit map. A final bilinear upsample gives per-pixel predictions at full resolution.

**3. Total parameters:** 9.11M in the FPN head, 22.06M fine-tuned from the backbone.

### Training Setup

- **Loss:** CrossEntropyLoss (label smoothing 0.05, inverse-frequency class weights capped to [0.1, 15]) + 0.5 × soft Dice loss
- **Optimizer:** AdamW — two param groups: head LR=5×10−4, backbone LR=5×10−5
- **Scheduler:** CosineAnnealingWarmRestarts(T₀=10) — LR restarts at epoch 10 and 20
- **Batch size:** 4 — **Epochs:** 15 (stopped early for deadline)
- **Image size:** 476×266 pixels

**Augmentation (training only):**
- Random horizontal flip (p=0.5)
- ColorJitter: brightness ±25%, contrast ±25%, saturation ±20%, hue ±8% (p=0.7)
- Random brightness/contrast (p=0.5)
- GaussNoise (p=0.3)
- Random rotation ±10° with reflection padding (p=0.3)
- RandomShadow (p=0.2)

---

## Results

### Quantitative Comparison

| Run       | Epochs | Val IoU | Val Dice | Val Accuracy | Val Loss |
|-----------|--------|---------|----------|--------------|----------|
| Baseline v1 (frozen ResNet50 + simple head) | 10 | 0.2305 | 0.3633 | 0.6608 | 0.96 |
| **v3 FPN (ours, best epoch)** | **15** | **0.3903** | **~0.5615** | **0.8120** | **1.79** |

Relative IoU improvement: **+69.6%**  
Relative Accuracy improvement: **+22.9%**

### Training Curves

Training and validation metrics are plotted in `train_stats/v3/` — see `all_metrics_curves.png`, `iou_curves.png`, and `dice_curves.png`. IoU improved steadily from 0.28 at epoch 1 to 0.39 at epoch 15, with no signs of overfitting (validation stayed close to training metrics throughout).

The warm restart at epoch 10 briefly caused a small drop due to the LR spike, but the model recovered and reached its best IoU in the second annealing cycle.

### Class-level Observations

**Better performance on:**
- Sky (most pixels, model learns it early)
- Dry Grass (second most common, stable early)
- Landscape (broad sandy terrain, visually consistent)

**Harder classes:**
- Logs and Background: fewer than 0.01% of pixels each; class weights help but examples are rare
- Ground Clutter vs Dry Bushes: visually similar at distance, often confused
- Rocks: colour varies, can blend with landscape or dry bushes

### Failure Cases

1. **Thin vertical objects (tree trunks, branches):** The model tends to merge them with the surrounding class. The multiscale FPN helps somewhat, but the 8× downsampled resolution at the finest scale loses thin structures.

2. **Dry Bushes vs Ground Clutter in shadowed regions:** Both are small, low-saturation objects. The RandomShadow augmentation was added specifically for this, but there is still confusion in heavily shadowed sequences.

3. **Distant Rocks vs Landscape:** At the horizon, small rocks are difficult to distinguish from the sandy landscape background. The model often predicts Landscape where the ground truth has Rocks.

---

## Challenges

**Class imbalance:**  
- *Initial score:* IoU ~0.23 (background classes dragging mean down)  
- *Issue:* Logs, Ground Clutter, and Background together account for <0.5% of pixels  
- *Solution:* Computed inverse-frequency class weights from the training masks; capped between 0.1 and 15 to avoid numerical instability  
- *Result:* More balanced gradient signal; model stops ignoring rare classes

**Single-scale features losing detail:**  
- *Initial approach:* flatten ResNet50's last feature map into tokens  
- *Issue:* 32× spatial downsampling loses fine textures and thin objects  
- *Solution:* Multi-scale FPN: fuse layer2/3/4 features at different resolutions  
- *Result:* Better boundary precision and small-object performance

**Limited GPU time:**  
We had a single T4 GPU and a hard deadline. Training was cut short at 15 of 30 planned epochs. With more time the LR warm restart at epoch 20 would likely push IoU past 0.42.

---

## Conclusion

We trained a ResNet50 + FPN-based segmentation model on synthetic desert scenes and improved validation IoU from 0.23 to 0.39 over the course of a hackathon. The model generalises reasonably well to unseen validation images and should transfer to the test set to a similar degree. The main limitations are class imbalance for rare objects like Logs, and moderate resolution at prediction time.

**Future directions:**
- Use a stronger pretrained backbone (DINOv2 or a segmentation-specific model)
- Better class-balanced sampling during training
- Higher resolution inputs and a lighter decoder to stay within memory limits
- Domain adaptation techniques to close the sim-to-real gap when real desert imagery becomes available

---

## References

- PyTorch documentation: https://pytorch.org
- TorchVision ResNet50 pretrained weights: ImageNet1K_V1
- Duality synthetic desert dataset: provided by hackathon organisers
- Albumentations library: https://albumentations.ai
