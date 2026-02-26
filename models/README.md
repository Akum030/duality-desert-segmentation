# Model Weights

The trained weights file (`segmentation_head_best.pth`, ~120 MB) is not tracked in git because it exceeds GitHub's file size limit.

**Architecture:** ResNet50MultiScale + FPNSegHead (v3)  
**Best Val IoU:** 0.3903 (epoch 15 of 15 trained)  
**Best Val Accuracy:** 0.8120

To use the pretrained model, either:
1. Run `train_segmentation_v3.py` to train from scratch (saves to `models/segmentation_head_best.pth`)
2. Or contact the team for the weights file.

Expected path: `models/segmentation_head_best.pth`
