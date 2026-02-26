# FORM SNIPPETS — Hack For Green Bharat Submission
# Team Aidhunik | Copy-paste straight into: https://zer0.pro/layerz/hack-for-green-bharat

---

## Project Title

Off-Road Desert Semantic Segmentation using Synthetic Data – Team Aidhunik

---

## Team Name

Aidhunik

---

## Project Summary
*(~120–150 words, copy this block)*

We built a semantic segmentation model that classifies every pixel in synthetic off-road desert images into one of ten terrain classes — trees, bushes, rocks, dry grass, sky, ground clutter, and more. The dataset was provided by the hackathon organisers through Duality's synthetic environment renderer. Our approach uses a ResNet50 backbone pretrained on ImageNet to extract rich visual features at three spatial scales, and a Feature Pyramid Network (FPN) decoder that fuses those features and assigns per-pixel class labels. We partly unfroze the deeper layers of the backbone and used a combined loss with class weight tuning to handle the heavy imbalance between common classes like Sky and rare ones like Logs. Starting from a baseline IoU of 0.23, we improved to a final validation IoU of 0.39 in about 15 training epochs on a single GPU. The trained model runs cleanly on the unseen test images and produces colour-coded segmentation masks for each scene.

---

## Technical Approach
*(~70–100 words)*

We use ResNet50 pretrained on ImageNet as a frozen (then partially fine-tuned) feature extractor. Features are extracted at three scales from layer2, layer3, and layer4, then fused with a Feature Pyramid Network. The merged representation passes through three depthwise ConvNeXt-style blocks before a 1×1 classifier gives per-pixel logits. We train with a combined loss: CrossEntropy with inverse-frequency class weights and label smoothing, plus a soft Dice term (weight 0.5) to directly optimise the IoU metric. AdamW with cosine warm-restart scheduling over 15 epochs.

---

## Challenges & Learnings
*(~80–120 words)*

The biggest challenge was class imbalance. Classes like Logs and Background appear in less than 0.01% of training pixels, so early models simply ignored them. We computed pixel-frequency-based class weights from the full training set to address this — rare classes get up to 15× the gradient weight of dominant ones. Another challenge was feature resolution: using only the final ResNet feature map (32× downsampled) caused the model to miss thin objects and blurry boundaries. Switching to a multi-scale FPN noticeably improved boundary prediction. On the time side, juggling experiment tracking under a hackathon deadline taught us to log everything from the start, because going back to reconstruct results from partial runs is messy. We would have benefited from more GPU time to complete all 30 planned epochs.

---

## Results / Final Score

Our best validation IoU is approximately **0.3903**, with an estimated Dice score of **0.56** and a pixel accuracy of **0.8120**. These numbers come from training only on the provided training and validation set — we never used the test images for training. The improvement over our baseline (IoU = 0.23, accuracy = 0.66) comes from combining multi-scale features, class-weighted loss, partial backbone fine-tuning, and a Dice loss component.

---

## GitHub Repository Link

https://github.com/Akum030/duality-desert-segmentation

---

## Future Improvements
*(~50–80 words)*

Given more time, we would try a stronger pretrained backbone like DINOv2 or a dedicated segmentation model. Better class-balanced sampling (oversampling rare classes in each batch) would help more than just weighting the loss. We are also interested in domain adaptation — the gap between Duality's synthetic renders and real desert imagery is significant, and techniques like style transfer or adversarial feature alignment could help the model generalise to real-world test drives.
