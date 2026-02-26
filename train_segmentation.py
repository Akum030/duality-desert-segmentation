"""
Segmentation Training Script - v2 (Improved)
ResNet50 backbone + ConvNeXt-style segmentation head
Improvements over v1:
  - Strong albumentations augmentation (flip, color jitter) applied consistently to image+mask
  - Higher LR (3e-4) + CosineAnnealingLR scheduler + Nesterov momentum
  - Larger batch size (4) for faster convergence
  - 15 epochs
  - Best-model checkpoint (saved per epoch based on val IoU)
  - Extra ConvNeXt block in the head (deeper head for better feature processing)
  - All v2 outputs written to train_stats/v2/
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.models as models
from PIL import Image
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Configuration  — change these flags to reproduce different experiments
# ============================================================================

USE_STRONG_AUG  = True    # albumentations augmentation on training set
NUM_EPOCHS      = 15      # v1 used 10; extra 5 epochs within time budget
BATCH_SIZE      = 4       # v1 used 2; larger batch → fewer iters per epoch
LR              = 3e-4    # v1 used 1e-4; slightly higher, cosine decay handles it
MOMENTUM        = 0.9
WEIGHT_DECAY    = 1e-4
OUTPUT_SUBDIR   = 'v2'    # all v2 artefacts go into train_stats/v2/


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.moveaxis(img, 0, -1)
    img  = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

value_map = {
    0:     0,   # background
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9    # Sky
}
n_classes = len(value_map)


def convert_mask(mask_np):
    """Convert raw mask pixel values to class IDs (operates on numpy array)."""
    new_arr = np.zeros_like(mask_np, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[mask_np == raw_value] = new_value
    return new_arr


# ============================================================================
# Dataset — albumentations version (joint image+mask transforms)
# ============================================================================

class MaskDataset(Dataset):
    """Dataset with albumentations augmentation (image+mask aligned)."""

    def __init__(self, data_dir, aug_transform=None, val_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.aug_transform = aug_transform    # albumentations pipeline
        self.val_transform = val_transform    # also albumentations (no aug)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id   = self.data_ids[idx]
        img_path  = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = np.array(Image.open(img_path).convert("RGB"))   # H x W x 3, uint8
        mask  = np.array(Image.open(mask_path))                  # H x W, raw values
        mask  = convert_mask(mask)                               # H x W, class IDs

        pipeline = self.aug_transform if self.aug_transform is not None else self.val_transform
        result = pipeline(image=image, mask=mask)
        image  = result['image']    # torch.Tensor C x H x W (float32, normalised)
        mask   = result['mask'].long()   # torch.Tensor H x W

        return image, mask


# ============================================================================
# Backbone & Token Extraction
# ============================================================================

def build_backbone(device):
    """Load ResNet50 (ImageNet pretrained) with FC+avgpool removed."""
    full_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    backbone    = nn.Sequential(*list(full_resnet.children())[:-2])  # [B, 2048, H', W']
    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def extract_tokens(backbone, imgs, tokenH, tokenW):
    """
    ResNet50 → [B, N, C] token grid.
    Resizes the last conv feature map to (tokenH, tokenW) and flattens spatial dims.
    """
    feats  = backbone(imgs)                                                                     # [B, 2048, Hf, Wf]
    feats  = F.interpolate(feats, size=(tokenH, tokenW), mode="bilinear", align_corners=False)  # [B, 2048, tokenH, tokenW]
    B, C, _, _ = feats.shape
    tokens = feats.view(B, C, -1).permute(0, 2, 1)                                             # [B, N, C]
    return tokens


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        # Extra block vs v1 — doubles head capacity with minimal parameter overhead
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
        pred_inds   = pred   == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union        = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).item())

    return float(np.nanmean(iou_per_class))


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds   = pred   == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        dice_score   = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)
        dice_per_class.append(dice_score.item())

    return float(np.mean(dice_per_class))


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return float((pred_classes == target).float().mean().item())


def evaluate_metrics(model, backbone, data_loader, device, tokenH, tokenW, num_classes=10):
    """Evaluate IoU, Dice, and pixel accuracy on a dataset."""
    iou_scores, dice_scores, pixel_accuracies = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Evaluating", leave=False, unit="batch"):
            imgs   = imgs.to(device)
            labels = labels.to(device).long()
            if labels.dim() == 4:
                labels = labels.squeeze(1)

            tokens  = extract_tokens(backbone, imgs, tokenH, tokenW)
            logits  = model(tokens)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            iou_scores.append(compute_iou(outputs, labels, num_classes=num_classes))
            dice_scores.append(compute_dice(outputs, labels, num_classes=num_classes))
            pixel_accuracies.append(compute_pixel_accuracy(outputs, labels))

    model.train()
    return float(np.nanmean(iou_scores)), float(np.mean(dice_scores)), float(np.mean(pixel_accuracies))


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    print(f"Saved training curves to '{output_dir}/training_curves.png'")

    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.title('Train IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Validation IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'))
    plt.close()
    print(f"Saved IoU curves to '{output_dir}/iou_curves.png'")

    # Plot 3: Dice curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.title('Train Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Validation Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'))
    plt.close()
    print(f"Saved Dice curves to '{output_dir}/dice_curves.png'")

    # Plot 4: Combined metrics plot
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='train')
    plt.plot(history['val_iou'], label='val')
    plt.title('IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='train')
    plt.plot(history['val_dice'], label='val')
    plt.title('Dice Score vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    print(f"Saved combined metrics curves to '{output_dir}/all_metrics_curves.png'")


def save_history_to_file(history, output_dir, filename='evaluation_metrics.txt',
                         config_info=None):
    """Save training history and config to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")

        if config_info:
            f.write("Configuration:\n")
            for k, v in config_info.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Best Results:\n")
        best_iou_ep  = int(np.argmax(history['val_iou']))  + 1
        best_dice_ep = int(np.argmax(history['val_dice'])) + 1
        best_acc_ep  = int(np.argmax(history['val_pixel_acc'])) + 1
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f}  (Epoch {best_iou_ep})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f}  (Epoch {best_dice_ep})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f}  (Epoch {best_acc_ep})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f}  (Epoch {int(np.argmin(history['val_loss'])) + 1})\n")
        f.write("=" * 60 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 104 + "\n")
        hdr = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
               'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*hdr))
        f.write("-" * 104 + "\n")
        for i in range(len(history['train_loss'])):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],      history['val_loss'][i],
                history['train_iou'][i],       history['val_iou'][i],
                history['train_dice'][i],      history['val_dice'][i],
                history['train_pixel_acc'][i], history['val_pixel_acc'][i],
            ))

    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Image size (same as v1 for fair comparison)
    w = int(((960 / 2) // 14) * 14)   # 476
    h = int(((540 / 2) // 14) * 14)   # 266
    tokenW = w // 14
    tokenH = h // 14

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats', OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Augmentation pipelines
    # ------------------------------------------------------------------
    train_aug = A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]) if USE_STRONG_AUG else A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_aug = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    data_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir  = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    trainset = MaskDataset(data_dir=data_dir, aug_transform=train_aug)
    valset   = MaskDataset(data_dir=val_dir,  val_transform=val_aug)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # ------------------------------------------------------------------
    # Backbone (ResNet50, frozen)
    # ------------------------------------------------------------------
    print("Loading ResNet50 backbone...")
    backbone = build_backbone(device)
    print("Backbone loaded successfully!")

    # Probe embedding dimension
    imgs_probe, _ = next(iter(train_loader))
    imgs_probe = imgs_probe.to(device)
    with torch.no_grad():
        toks = extract_tokens(backbone, imgs_probe, tokenH, tokenW)
    n_embedding = toks.shape[2]
    print(f"Embedding dimension: {n_embedding}")
    print(f"Token grid: {tokenH} x {tokenW}  →  {tokenH * tokenW} tokens per image")

    # ------------------------------------------------------------------
    # Segmentation head
    # ------------------------------------------------------------------
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=tokenW,
        tokenH=tokenH,
    ).to(device)

    # ------------------------------------------------------------------
    # Loss, optimiser, scheduler
    # ------------------------------------------------------------------
    loss_fct  = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

    # Config info for metrics file
    config_info = {
        'USE_STRONG_AUG': USE_STRONG_AUG,
        'NUM_EPOCHS':     NUM_EPOCHS,
        'BATCH_SIZE':     BATCH_SIZE,
        'LR':             LR,
        'MOMENTUM':       MOMENTUM,
        'WEIGHT_DECAY':   WEIGHT_DECAY,
        'image_size':     f'{h}x{w}',
        'tokenH_tokenW':  f'{tokenH}x{tokenW}',
        'n_embedding':    n_embedding,
        'n_classes':      n_classes,
        'optimizer':      'SGD+Nesterov+CosineAnnealingLR',
    }

    # Training history
    history = {k: [] for k in [
        'train_loss', 'val_loss',
        'train_iou',  'val_iou',
        'train_dice', 'val_dice',
        'train_pixel_acc', 'val_pixel_acc',
    ]}

    best_val_iou    = -1.0
    best_epoch      = -1
    best_model_path = os.path.join(script_dir, 'segmentation_head_best.pth')

    print("\nStarting training...")
    print("=" * 80)
    print(f"  Epochs: {NUM_EPOCHS},  Batch size: {BATCH_SIZE},  LR: {LR}")
    print(f"  Strong augmentation: {USE_STRONG_AUG}")
    print(f"  Output dir: {output_dir}")
    print("=" * 80)

    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # ---- training phase ----
        classifier.train()
        train_losses = []

        for imgs, labels in tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Train]",
                                  leave=False, unit="batch"):
            imgs   = imgs.to(device)
            labels = labels.to(device).long()

            with torch.no_grad():
                tokens = extract_tokens(backbone, imgs, tokenH, tokenW)

            logits  = classifier(tokens)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            loss    = loss_fct(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ---- validation loss ----
        classifier.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Val]",
                                      leave=False, unit="batch"):
                imgs   = imgs.to(device)
                labels = labels.to(device).long()
                tokens  = extract_tokens(backbone, imgs, tokenH, tokenW)
                logits  = classifier(tokens)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                val_losses.append(loss_fct(outputs, labels).item())

        # ---- compute full metrics ----
        train_iou, train_dice, train_acc = evaluate_metrics(
            classifier, backbone, train_loader, device, tokenH, tokenW, n_classes)
        val_iou, val_dice, val_acc = evaluate_metrics(
            classifier, backbone, val_loader, device, tokenH, tokenW, n_classes)

        epoch_train_loss = float(np.mean(train_losses))
        epoch_val_loss   = float(np.mean(val_losses))

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_acc)
        history['val_pixel_acc'].append(val_acc)

        # LR scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch   = epoch + 1
            torch.save(classifier.state_dict(), best_model_path)

        epoch_pbar.set_postfix(
            train_loss=f"{epoch_train_loss:.3f}",
            val_loss=f"{epoch_val_loss:.3f}",
            val_iou=f"{val_iou:.3f}",
            val_acc=f"{val_acc:.3f}",
            lr=f"{current_lr:.2e}",
        )

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    print("\nSaving training curves...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir,
                         filename='evaluation_metrics_v2.txt',
                         config_info=config_info)

    # Last epoch weights (for reference)
    last_model_path = os.path.join(script_dir, 'segmentation_head.pth')
    torch.save(classifier.state_dict(), last_model_path)
    print(f"Saved last-epoch model  → '{last_model_path}'")
    print(f"Saved best-epoch model  → '{best_model_path}'  (epoch {best_epoch}, val IoU {best_val_iou:.4f})")

    print("\nFinal evaluation results:")
    print(f"  Final Val Loss:     {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU:      {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:     {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history['val_pixel_acc'][-1]:.4f}")
    print(f"\n  BEST Val IoU:       {best_val_iou:.4f}  (Epoch {best_epoch})")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

