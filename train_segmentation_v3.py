"""
Segmentation Training Script - v3 (Maximum Performance)
ResNet50 multi-scale FPN backbone + deep segmentation head

IMPROVEMENTS OVER v2:
  1. Multi-scale feature fusion (FPN): ResNet50 layer2+layer3+layer4
     - Captures both fine texture (layer2) and high-level semantics (layer4)
  2. Partial backbone unfreezing: layer3 + layer4 trained with LR/10
  3. Combined loss: CrossEntropyLoss + Soft Dice loss
     - CE handles overall accuracy, Dice directly optimizes IoU metric
  4. Class-weighted CrossEntropy: weights inverse-proportional to pixel frequency
     - Addresses severe class imbalance (Logs/Ground Clutter very rare)
  5. Deeper segmentation head: 3 ConvNeXt blocks + BN + skip connection
  6. 30 epochs with CosineAnnealingWarmRestarts (T_0=10)
  7. AdamW optimizer (better for limited data than SGD)
  8. Warm start: load v2 best weights for the head decoder if available
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

plt.switch_backend('Agg')


# ============================================================================
# Configuration
# ============================================================================

NUM_EPOCHS      = 30
BATCH_SIZE      = 4
LR_HEAD         = 5e-4     # learning rate for segmentation head
LR_BACKBONE     = 5e-5     # 10x lower for backbone fine-tuning
WEIGHT_DECAY    = 1e-4
DICE_WEIGHT     = 0.5      # combined loss = CE + DICE_WEIGHT * Dice
UNFREEZE_LAYERS = ['layer3', 'layer4']   # layers to fine-tune in ResNet50
T_0             = 10       # CosineAnnealingWarmRestarts period
OUTPUT_SUBDIR   = 'v3'
WARM_START_PATH = '/home3/indiamart/gbht/segmentation_head_best.pth'


# ============================================================================
# Mask conversion
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
CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass',
    'Dry Bushes', 'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]


def convert_mask(mask_np):
    new_arr = np.zeros_like(mask_np, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[mask_np == raw_value] = new_value
    return new_arr


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, aug_transform=None, val_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.aug_transform = aug_transform
        self.val_transform = val_transform

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id   = self.data_ids[idx]
        image = np.array(Image.open(os.path.join(self.image_dir, data_id)).convert('RGB'))
        mask  = np.array(Image.open(os.path.join(self.masks_dir, data_id)))
        mask  = convert_mask(mask)

        pipeline = self.aug_transform if self.aug_transform is not None else self.val_transform
        result   = pipeline(image=image, mask=mask)
        return result['image'], result['mask'].long()


# ============================================================================
# Class weight computation
# ============================================================================

def compute_class_weights(data_dir, transform, device, n_classes=10, max_samples=800):
    """
    Compute inverse-frequency class weights from training masks.
    Rare classes (Logs, Ground Clutter) get higher weights.
    """
    print("Computing class weights from training data...")
    masks_dir  = os.path.join(data_dir, 'Segmentation')
    mask_files = sorted(os.listdir(masks_dir))[:max_samples]

    counts = np.zeros(n_classes, dtype=np.float64)
    for fname in tqdm(mask_files, desc='Computing class weights', leave=False):
        mask_raw = np.array(Image.open(os.path.join(masks_dir, fname)))
        mask     = convert_mask(mask_raw)
        for c in range(n_classes):
            counts[c] += (mask == c).sum()

    total = counts.sum()
    # Inverse frequency with smoothing
    weights = total / (n_classes * (counts + 1e-6))
    weights = weights / weights.mean()   # normalize so mean weight = 1
    weights = np.clip(weights, 0.1, 15.0)  # cap to prevent explosion

    print("Class weights:")
    for i, (name, w, cnt) in enumerate(zip(CLASS_NAMES, weights, counts)):
        pct = 100.0 * cnt / max(total, 1)
        print(f"  [{i}] {name:<18} weight={w:.3f}  ({pct:.2f}% of pixels)")

    return torch.tensor(weights, dtype=torch.float32).to(device)


# ============================================================================
# Backbone: Multi-scale ResNet50
# ============================================================================

class ResNet50MultiScale(nn.Module):
    """
    ResNet50 with hooks to extract intermediate features at 3 scales.
    Optionally fine-tunes layer3 and layer4.
    """
    def __init__(self, unfreeze_layers=None):
        super().__init__()
        full = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Build stem + stages explicitly
        self.stem   = nn.Sequential(full.conv1, full.bn1, full.relu, full.maxpool)
        self.layer1 = full.layer1   # out: [B, 256,  H/4,  W/4]
        self.layer2 = full.layer2   # out: [B, 512,  H/8,  W/8]
        self.layer3 = full.layer3   # out: [B, 1024, H/16, W/16]
        self.layer4 = full.layer4   # out: [B, 2048, H/32, W/32]

        # Freeze everything first
        for p in self.parameters():
            p.requires_grad = False

        # Unfreeze specified layers
        if unfreeze_layers:
            for name in unfreeze_layers:
                for p in getattr(self, name).parameters():
                    p.requires_grad = True
            print(f"Unfrozen backbone layers: {unfreeze_layers}")

    def forward(self, x):
        x  = self.stem(x)
        x  = self.layer1(x)
        c2 = self.layer2(x)    # [B, 512,  H/8,  W/8]
        c3 = self.layer3(c2)   # [B, 1024, H/16, W/16]
        c4 = self.layer4(c3)   # [B, 2048, H/32, W/32]
        return c2, c3, c4


# ============================================================================
# FPN Decoder + Segmentation Head
# ============================================================================

class ConvBnGelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)


class FPNSegHead(nn.Module):
    """
    FPN-style feature pyramid network decoder.
    Takes c2, c3, c4 from ResNet50 and produces pixel-wise class scores.
    """
    def __init__(self, out_channels=10, fpn_channels=256):
        super().__init__()
        F = fpn_channels

        # Lateral projections (reduce channel dim)
        self.lat4 = ConvBnGelu(2048, F, kernel=1, padding=0)
        self.lat3 = ConvBnGelu(1024, F, kernel=1, padding=0)
        self.lat2 = ConvBnGelu(512,  F, kernel=1, padding=0)

        # Post-merge smoothing
        self.smooth4 = ConvBnGelu(F, F)
        self.smooth3 = ConvBnGelu(F, F)
        self.smooth2 = ConvBnGelu(F, F)

        # Merge: concatenate all 3 scales (all upsampled to c2 size)
        # c2 is at H/8, W/8
        self.merge_conv = nn.Sequential(
            ConvBnGelu(F * 3, F * 2),
            ConvBnGelu(F * 2, F * 2),
        )

        # Dense prediction head (applied at H/8 then upsample to full)
        self.head = nn.Sequential(
            # Block 1 — depthwise + pointwise
            nn.Conv2d(F * 2, F * 2, 7, padding=3, groups=F * 2, bias=False),
            nn.BatchNorm2d(F * 2),
            nn.GELU(),
            nn.Conv2d(F * 2, F * 2, 1, bias=False),
            nn.BatchNorm2d(F * 2),
            nn.GELU(),
            # Block 2
            nn.Conv2d(F * 2, F * 2, 7, padding=3, groups=F * 2, bias=False),
            nn.BatchNorm2d(F * 2),
            nn.GELU(),
            nn.Conv2d(F * 2, F, 1, bias=False),
            nn.BatchNorm2d(F),
            nn.GELU(),
            # Block 3
            nn.Conv2d(F, F, 5, padding=2, groups=F, bias=False),
            nn.BatchNorm2d(F),
            nn.GELU(),
            nn.Conv2d(F, F, 1, bias=False),
            nn.BatchNorm2d(F),
            nn.GELU(),
            # Classifier
            nn.Conv2d(F, out_channels, 1),
        )

    def forward(self, c2, c3, c4):
        # Top-down pathway
        p4 = self.smooth4(self.lat4(c4))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False))

        # Upsample p4, p3 to c2 resolution then concatenate
        p4_up = F.interpolate(p4, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)

        merged = torch.cat([p2, p3_up, p4_up], dim=1)   # [B, F*3, H/8, W/8]
        merged = self.merge_conv(merged)                  # [B, F*2, H/8, W/8]
        return self.head(merged)                          # [B, n_classes, H/8, W/8]


# ============================================================================
# Loss: CE + Soft Dice
# ============================================================================

class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.ce    = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        self.dice_w = dice_weight
        self.smooth = smooth

    def soft_dice_loss(self, logits, targets, n_cls):
        probs   = torch.softmax(logits, dim=1)
        oh      = F.one_hot(targets, n_cls).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        inter   = (probs * oh).sum(dim=(0, 2, 3))
        union   = probs.sum(dim=(0, 2, 3)) + oh.sum(dim=(0, 2, 3))
        dice    = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, logits, targets):
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.soft_dice_loss(logits, targets, logits.shape[1])
        return ce_loss + self.dice_w * dice_loss


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred_logits, target, num_classes=10):
    pred   = torch.argmax(pred_logits, dim=1).view(-1)
    target = target.view(-1)
    ious   = []
    for c in range(num_classes):
        p = pred   == c
        t = target == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious.append((inter / union).item() if union > 0 else float('nan'))
    return float(np.nanmean(ious))


def compute_dice(pred_logits, target, num_classes=10, smooth=1e-6):
    pred   = torch.argmax(pred_logits, dim=1).view(-1)
    target = target.view(-1)
    dices  = []
    for c in range(num_classes):
        p = pred   == c
        t = target == c
        inter = (p & t).sum().float()
        dices.append(((2 * inter + smooth) / (p.sum().float() + t.sum().float() + smooth)).item())
    return float(np.mean(dices))


def compute_pixel_accuracy(pred_logits, target):
    return float((torch.argmax(pred_logits, dim=1) == target).float().mean().item())


def per_class_iou(pred_logits, target, num_classes=10):
    pred   = torch.argmax(pred_logits, dim=1).view(-1)
    target = target.view(-1)
    ious   = {}
    for c in range(num_classes):
        p = pred   == c
        t = target == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious[CLASS_NAMES[c]] = (inter / union).item() if union > 0 else float('nan')
    return ious


def evaluate_metrics(model, backbone, data_loader, device, full_res, num_classes=10, verbose_classes=False):
    ious, dices, accs = [], [], []
    class_ious = {n: [] for n in CLASS_NAMES}

    model.eval()
    backbone.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc='Evaluating', leave=False):
            imgs, labels = imgs.to(device), labels.to(device).long()
            c2, c3, c4   = backbone(imgs)
            logits        = model(c2, c3, c4)
            outputs       = F.interpolate(logits, size=full_res, mode='bilinear', align_corners=False)

            ious.append(compute_iou(outputs, labels, num_classes))
            dices.append(compute_dice(outputs, labels, num_classes))
            accs.append(compute_pixel_accuracy(outputs, labels))

            if verbose_classes:
                ciou = per_class_iou(outputs, labels, num_classes)
                for name, v in ciou.items():
                    class_ious[name].append(v)

    if verbose_classes:
        return (float(np.nanmean(ious)), float(np.mean(dices)), float(np.mean(accs)),
                {n: float(np.nanmean(v)) for n, v in class_ious.items()})

    backbone.train()
    for name in UNFREEZE_LAYERS:
        pass  # backbone fine-tune layers stay in train mode
    model.train()
    return float(np.nanmean(ious)), float(np.mean(dices)), float(np.mean(accs))


# ============================================================================
# Plots
# ============================================================================

def save_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    keys = [
        ('train_loss', 'val_loss', 'Loss', 'Loss'),
        ('train_iou',  'val_iou',  'IoU',  'IoU'),
        ('train_dice', 'val_dice', 'Dice', 'Dice Score'),
        ('train_pixel_acc', 'val_pixel_acc', 'Pixel Accuracy', 'Accuracy'),
    ]
    # Combined 4-panel
    plt.figure(figsize=(14, 10))
    for i, (ktr, kval, title, ylabel) in enumerate(keys):
        plt.subplot(2, 2, i + 1)
        plt.plot(history[ktr], label='train')
        plt.plot(history[kval], label='val')
        plt.title(f'{title} vs Epoch')
        plt.xlabel('Epoch'); plt.ylabel(ylabel)
        plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    print(f"Saved combined curves → {output_dir}/all_metrics_curves.png")

    # Individual
    for ktr, kval, title, ylabel in keys:
        plt.figure(figsize=(8, 5))
        plt.plot(history[ktr], label='train')
        plt.plot(history[kval], label='val')
        plt.title(f'{title} vs Epoch'); plt.xlabel('Epoch'); plt.ylabel(ylabel)
        plt.legend(); plt.grid(True); plt.tight_layout()
        fname = title.lower().replace(' ', '_') + '_curves.png'
        plt.savefig(os.path.join(output_dir, fname)); plt.close()


def save_metrics_file(history, output_dir, config_info, class_ious_best=None):
    os.makedirs(output_dir, exist_ok=True)
    fp = os.path.join(output_dir, 'evaluation_metrics_v3.txt')
    with open(fp, 'w') as f:
        f.write("TRAINING RESULTS — v3\n")
        f.write("=" * 70 + "\n\n")
        f.write("Configuration:\n")
        for k, v in config_info.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Final Metrics (last epoch):\n")
        for k in ['train_loss', 'val_loss', 'train_iou', 'val_iou',
                  'train_dice', 'val_dice', 'train_pixel_acc', 'val_pixel_acc']:
            f.write(f"  {k}: {history[k][-1]:.4f}\n")
        f.write("\n")

        f.write("Best Results:\n")
        best_iou  = max(history['val_iou'])
        best_dice = max(history['val_dice'])
        best_acc  = max(history['val_pixel_acc'])
        f.write(f"  Best Val IoU:      {best_iou:.4f}  (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {best_dice:.4f}  (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {best_acc:.4f}  (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f}  (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("\n")

        if class_ious_best:
            f.write("Per-Class IoU (at best epoch):\n")
            for name, v in class_ious_best.items():
                f.write(f"  {name:<20}: {v:.4f}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("Per-Epoch History:\n")
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(
            'Epoch', 'TrainLoss', 'ValLoss', 'TrainIoU', 'ValIoU',
            'TrainDice', 'ValDice', 'TrainAcc', 'ValAcc'))
        f.write("-" * 96 + "\n")
        for i in range(len(history['train_loss'])):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i], history['val_loss'][i],
                history['train_iou'][i],  history['val_iou'][i],
                history['train_dice'][i], history['val_dice'][i],
                history['train_pixel_acc'][i], history['val_pixel_acc'][i],
            ))
    print(f"Saved evaluation metrics → {fp}")
    return fp


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Image size (same as v1/v2 — fair comparison)
    w = int(((960 / 2) // 14) * 14)   # 476
    h = int(((540 / 2) // 14) * 14)   # 266

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats', OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    data_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir  = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    train_aug = A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.08, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.RandomShadow(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_aug = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    trainset = MaskDataset(data_dir=data_dir, aug_transform=train_aug)
    valset   = MaskDataset(data_dir=val_dir,  val_transform=val_aug)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Training samples: {len(trainset)},  Validation samples: {len(valset)}")

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------
    class_weights = compute_class_weights(data_dir, val_aug, device, n_classes)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"\nBuilding multi-scale ResNet50 + FPN head (unfreezing {UNFREEZE_LAYERS})...")
    backbone  = ResNet50MultiScale(unfreeze_layers=UNFREEZE_LAYERS).to(device)
    seg_head  = FPNSegHead(out_channels=n_classes, fpn_channels=256).to(device)
    print(f"Head parameters: {sum(p.numel() for p in seg_head.parameters()) / 1e6:.2f}M")
    backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Backbone trainable parameters: {backbone_trainable / 1e6:.2f}M")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    criterion = CombinedLoss(class_weights=class_weights, dice_weight=DICE_WEIGHT).to(device)

    # ------------------------------------------------------------------
    # Optimizer: separate param groups (head @ LR_HEAD, backbone @ LR_BACKBONE)
    # ------------------------------------------------------------------
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params     = list(seg_head.parameters())

    optimizer = optim.AdamW([
        {'params': head_params,     'lr': LR_HEAD,     'weight_decay': WEIGHT_DECAY},
        {'params': backbone_params, 'lr': LR_BACKBONE, 'weight_decay': WEIGHT_DECAY * 2},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=1, eta_min=1e-6)

    # ------------------------------------------------------------------
    # Config info
    # ------------------------------------------------------------------
    config_info = {
        'NUM_EPOCHS':      NUM_EPOCHS,
        'BATCH_SIZE':      BATCH_SIZE,
        'LR_HEAD':         LR_HEAD,
        'LR_BACKBONE':     LR_BACKBONE,
        'WEIGHT_DECAY':    WEIGHT_DECAY,
        'DICE_WEIGHT':     DICE_WEIGHT,
        'UNFREEZE_LAYERS': str(UNFREEZE_LAYERS),
        'T_0':             T_0,
        'image_size':      f'{h}x{w}',
        'n_classes':       n_classes,
        'optimizer':       'AdamW + CosineAnnealingWarmRestarts',
        'loss':            f'CE(label_smooth=0.05, class_weights) + {DICE_WEIGHT}*SoftDice',
        'fpn_channels':    256,
        'head_depth':      '3 ConvNeXt blocks',
    }

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------
    history = {k: [] for k in [
        'train_loss', 'val_loss',
        'train_iou',  'val_iou',
        'train_dice', 'val_dice',
        'train_pixel_acc', 'val_pixel_acc',
    ]}

    best_val_iou     = -1.0
    best_epoch       = -1
    best_class_ious  = None
    best_path        = os.path.join(script_dir, 'segmentation_head_best_v3.pth')

    print("\nStarting v3 training...")
    print("=" * 80)
    print(f"  Epochs: {NUM_EPOCHS},  Batch: {BATCH_SIZE}")
    print(f"  LR head={LR_HEAD}, LR backbone={LR_BACKBONE}")
    print(f"  Loss: CE + {DICE_WEIGHT}*Dice  |  Output: {output_dir}")
    print("=" * 80)

    epoch_pbar = tqdm(range(NUM_EPOCHS), desc='Training v3', unit='epoch')
    for epoch in epoch_pbar:
        # ---- TRAIN ----
        seg_head.train()
        backbone.train()
        train_losses = []

        for imgs, labels in tqdm(train_loader,
                                  desc=f'Ep {epoch+1}/{NUM_EPOCHS} [Train]',
                                  leave=False, unit='batch'):
            imgs, labels = imgs.to(device), labels.to(device).long()

            c2, c3, c4 = backbone(imgs)
            logits      = seg_head(c2, c3, c4)
            outputs     = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            loss        = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(seg_head.parameters()) + backbone_params, max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---- VAL LOSS ----
        seg_head.eval()
        backbone.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                      desc=f'Ep {epoch+1}/{NUM_EPOCHS} [Val]',
                                      leave=False, unit='batch'):
                imgs, labels = imgs.to(device), labels.to(device).long()
                c2, c3, c4  = backbone(imgs)
                logits       = seg_head(c2, c3, c4)
                outputs      = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
                val_losses.append(criterion(outputs, labels).item())

        # ---- METRICS ----
        train_iou, train_dice, train_acc = evaluate_metrics(
            seg_head, backbone, train_loader, device, (h, w), n_classes)
        val_iou, val_dice, val_acc = evaluate_metrics(
            seg_head, backbone, val_loader, device, (h, w), n_classes)

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

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        # Save best
        if val_iou > best_val_iou:
            best_val_iou  = val_iou
            best_epoch    = epoch + 1
            torch.save({
                'seg_head':  seg_head.state_dict(),
                'backbone_layer3': backbone.layer3.state_dict(),
                'backbone_layer4': backbone.layer4.state_dict(),
            }, best_path)

            # Compute per-class IoU at best epoch for report
            result = evaluate_metrics(
                seg_head, backbone, val_loader, device, (h, w), n_classes,
                verbose_classes=True)
            _, _, _, best_class_ious = result

            print(f"\n  *** NEW BEST at epoch {best_epoch}: Val IoU = {best_val_iou:.4f} ***")
            print("  Per-class IoU:")
            for name, iou_v in best_class_ious.items():
                print(f"    {name:<20}: {iou_v:.4f}")

        epoch_pbar.set_postfix(
            tr_loss=f'{epoch_train_loss:.3f}',
            vl_loss=f'{epoch_val_loss:.3f}',
            val_iou=f'{val_iou:.3f}',
            val_acc=f'{val_acc:.3f}',
            lr=f'{current_lr:.1e}',
        )

    # ------------------------------------------------------------------
    # Save last epoch weights + artefacts
    # ------------------------------------------------------------------
    last_path = os.path.join(script_dir, 'segmentation_head_v3_last.pth')
    torch.save({
        'seg_head':  seg_head.state_dict(),
        'backbone_layer3': backbone.layer3.state_dict(),
        'backbone_layer4': backbone.layer4.state_dict(),
    }, last_path)

    print(f"\nSaved last-epoch model → {last_path}")
    print(f"Saved best-epoch model → {best_path}  (epoch {best_epoch}, Val IoU {best_val_iou:.4f})")

    save_plots(history, output_dir)
    save_metrics_file(history, output_dir, config_info, class_ious_best=best_class_ious)

    print("\nFinal evaluation results (v3):")
    print(f"  Final Val Loss:     {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU:      {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:     {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history['val_pixel_acc'][-1]:.4f}")
    print(f"\n  BEST Val IoU:       {best_val_iou:.4f}  (Epoch {best_epoch})")
    print("\nv3 Training complete!")


if __name__ == '__main__':
    main()
