"""
Segmentation Inference Script — v3 (FPN + multi-scale ResNet50)
Loads segmentation_head_best_v3.pth and runs inference on test images.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

value_map = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}
n_classes = len(value_map)
CLASS_NAMES = ['Background','Trees','Lush Bushes','Dry Grass','Dry Bushes','Ground Clutter','Logs','Rocks','Landscape','Sky']
CLASS_COLORS = {0:(0,0,0),1:(34,139,34),2:(0,200,80),3:(210,180,50),4:(139,90,43),5:(128,64,0),6:(80,40,0),7:(150,150,150),8:(200,160,100),9:(135,206,235)}

def convert_mask(mask_input):
    arr = np.array(mask_input)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        out[arr == raw] = new
    return out

def class_mask_to_color(pred_mask_np):
    H, W = pred_mask_np.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        color_img[pred_mask_np == cls_id] = color
    return color_img

class ConvBnGelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())
    def forward(self, x): return self.block(x)

class ResNet50MultiScale(nn.Module):
    def __init__(self, unfreeze_layers=None):
        super().__init__()
        full = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(full.conv1, full.bn1, full.relu, full.maxpool)
        self.layer1 = full.layer1
        self.layer2 = full.layer2
        self.layer3 = full.layer3
        self.layer4 = full.layer4
        for p in self.parameters(): p.requires_grad = False
        if unfreeze_layers:
            for name in unfreeze_layers:
                for p in getattr(self, name).parameters(): p.requires_grad = True
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x)
        c2 = self.layer2(x); c3 = self.layer3(c2); c4 = self.layer4(c3)
        return c2, c3, c4

class FPNSegHead(nn.Module):
    def __init__(self, out_channels=10, fpn_channels=256):
        super().__init__()
        Fc = fpn_channels
        self.lat4 = ConvBnGelu(2048, Fc, kernel=1, padding=0)
        self.lat3 = ConvBnGelu(1024, Fc, kernel=1, padding=0)
        self.lat2 = ConvBnGelu(512,  Fc, kernel=1, padding=0)
        self.smooth4 = ConvBnGelu(Fc, Fc)
        self.smooth3 = ConvBnGelu(Fc, Fc)
        self.smooth2 = ConvBnGelu(Fc, Fc)
        self.merge_conv = nn.Sequential(ConvBnGelu(Fc*3, Fc*2), ConvBnGelu(Fc*2, Fc*2))
        self.head = nn.Sequential(
            nn.Conv2d(Fc*2, Fc*2, 7, padding=3, groups=Fc*2, bias=False), nn.BatchNorm2d(Fc*2), nn.GELU(),
            nn.Conv2d(Fc*2, Fc*2, 1, bias=False), nn.BatchNorm2d(Fc*2), nn.GELU(),
            nn.Conv2d(Fc*2, Fc*2, 7, padding=3, groups=Fc*2, bias=False), nn.BatchNorm2d(Fc*2), nn.GELU(),
            nn.Conv2d(Fc*2, Fc, 1, bias=False), nn.BatchNorm2d(Fc), nn.GELU(),
            nn.Conv2d(Fc, Fc, 5, padding=2, groups=Fc, bias=False), nn.BatchNorm2d(Fc), nn.GELU(),
            nn.Conv2d(Fc, Fc, 1, bias=False), nn.BatchNorm2d(Fc), nn.GELU(),
            nn.Conv2d(Fc, out_channels, 1),
        )
    def forward(self, c2, c3, c4):
        p4 = self.smooth4(self.lat4(c4))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False))
        p4u = F.interpolate(p4, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p3u = F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        merged = self.merge_conv(torch.cat([p2, p3u, p4u], dim=1))
        return self.head(merged)

def compute_iou(pred_logits, target, num_classes=10):
    pred = torch.argmax(pred_logits, dim=1).view(-1); target = target.view(-1)
    ious = []
    for c in range(num_classes):
        p = pred==c; t = target==c
        inter=(p&t).sum().float(); union=(p|t).sum().float()
        ious.append((inter/union).item() if union>0 else float('nan'))
    return float(np.nanmean(ious))

def compute_dice(pred_logits, target, num_classes=10, smooth=1e-6):
    pred = torch.argmax(pred_logits, dim=1).view(-1); target = target.view(-1)
    dices = []
    for c in range(num_classes):
        p=pred==c; t=target==c
        inter=(p&t).sum().float()
        dices.append(((2*inter+smooth)/(p.sum().float()+t.sum().float()+smooth)).item())
    return float(np.mean(dices))

def compute_pixel_accuracy(pred_logits, target):
    return float((torch.argmax(pred_logits, dim=1)==target).float().mean().item())

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    w = int(((960/2)//14)*14)   # 476
    h = int(((540/2)//14)*14)   # 266
    script_dir = os.path.dirname(os.path.abspath(__file__))

    models_best = os.path.join(script_dir, 'models', 'segmentation_head_best.pth')
    v3_best     = os.path.join(script_dir, 'segmentation_head_best_v3.pth')
    v3_last     = os.path.join(script_dir, 'segmentation_head_v3_last.pth')
    if os.path.exists(models_best):
        weights_path = models_best
        print(f"Loading weights from models/: {weights_path}")
    elif os.path.exists(v3_best):
        weights_path = v3_best
        print(f"Loading v3 best-epoch weights: {weights_path}")
    elif os.path.exists(v3_last):
        weights_path = v3_last
        print(f"Loading v3 last-epoch weights: {weights_path}")
    else:
        raise FileNotFoundError(
            f"No weights found. Run train_segmentation_v3.py first.\n"
            f"  Expected: {models_best}")


    print("Building ResNet50MultiScale + FPNSegHead...")
    backbone = ResNet50MultiScale(unfreeze_layers=['layer3', 'layer4']).to(device)
    seg_head = FPNSegHead(out_channels=n_classes, fpn_channels=256).to(device)

    ckpt = torch.load(weights_path, map_location=device)
    seg_head.load_state_dict(ckpt['seg_head'])
    backbone.layer3.load_state_dict(ckpt['backbone_layer3'])
    backbone.layer4.load_state_dict(ckpt['backbone_layer4'])
    backbone.eval(); seg_head.eval()

    epoch_tag = ckpt.get('epoch', '?')
    best_iou  = ckpt.get('val_iou', float('nan'))
    print(f"Loaded: epoch={epoch_tag}, best_val_iou={best_iou:.4f}")
    print(f"FPNSegHead params: {sum(p.numel() for p in seg_head.parameters())/1e6:.2f}M")

    infer_aug = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])

    test_img_dir = os.path.join(script_dir, 'Offroad_Segmentation_testImages', 'Color_Images')
    test_gt_dir  = os.path.join(script_dir, 'Offroad_Segmentation_testImages', 'Segmentation')
    pred_out_dir = os.path.join(script_dir, 'Offroad_Segmentation_testImages', 'Predicted_Masks_v3')
    os.makedirs(pred_out_dir, exist_ok=True)

    has_gt = os.path.isdir(test_gt_dir)
    print(f"Ground truth: {'yes' if has_gt else 'no'}")
    test_images = sorted(os.listdir(test_img_dir))
    print(f"Found {len(test_images)} test images.")

    per_image_rows = []
    all_ious, all_dices, all_accs = [], [], []
    class_pixel_counts = np.zeros(n_classes, dtype=np.int64)

    with torch.no_grad():
        for fname in tqdm(test_images, desc="Inference", unit="img"):
            img_np = np.array(Image.open(os.path.join(test_img_dir, fname)).convert('RGB'))
            orig_h, orig_w = img_np.shape[:2]
            image_t = infer_aug(image=img_np)['image'].unsqueeze(0).to(device)

            c2, c3, c4 = backbone(image_t)
            logits = seg_head(c2, c3, c4)
            outputs = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

            for cls in range(n_classes):
                class_pixel_counts[cls] += int((pred_mask == cls).sum())

            cv2.imwrite(os.path.join(pred_out_dir, fname), class_mask_to_color(pred_mask)[:,:,::-1])
            row = {'image': fname}

            if has_gt:
                gt_mask = convert_mask(np.array(Image.open(os.path.join(test_gt_dir, fname))))
                gt_t = torch.from_numpy(gt_mask).long()
                pred_t = torch.from_numpy(pred_mask).long()
                out1h = F.one_hot(pred_t, n_classes).permute(2,0,1).unsqueeze(0).float()
                iou  = compute_iou(out1h, gt_t.unsqueeze(0), n_classes)
                dice = compute_dice(out1h, gt_t.unsqueeze(0), n_classes)
                acc  = float((pred_t==gt_t).float().mean().item())
                all_ious.append(iou); all_dices.append(dice); all_accs.append(acc)
                row.update({'iou': f'{iou:.4f}', 'dice': f'{dice:.4f}', 'pixel_acc': f'{acc:.4f}'})
            else:
                most_common = int(np.bincount(pred_mask.reshape(-1), minlength=n_classes).argmax())
                row.update({'dominant_class': CLASS_NAMES[most_common]})
            per_image_rows.append(row)

    metrics_path = os.path.join(script_dir, 'test_metrics_v3.txt')
    total_pixels = class_pixel_counts.sum()
    with open(metrics_path, 'w') as f:
        f.write("TEST INFERENCE RESULTS — v3 (FPN)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model weights  : {weights_path}\n")
        f.write(f"Epoch: {epoch_tag},  Best Val IoU: {best_iou:.4f}\n")
        f.write(f"Images count   : {len(test_images)}\n")
        f.write(f"Output masks   : {pred_out_dir}\n\n")
        if has_gt and all_ious:
            f.write("Aggregate Metrics:\n")
            f.write(f"  Mean IoU            : {np.mean(all_ious):.4f}\n")
            f.write(f"  Mean Dice           : {np.mean(all_dices):.4f}\n")
            f.write(f"  Mean Pixel Accuracy : {np.mean(all_accs):.4f}\n\n")
        f.write("Predicted Class Distribution:\n")
        for cls in range(n_classes):
            pct = 100.0 * class_pixel_counts[cls] / max(total_pixels, 1)
            f.write(f"  {CLASS_NAMES[cls]:<20}: {class_pixel_counts[cls]:>12,} px  ({pct:5.2f}%)\n")
        f.write("\n")
        if has_gt and per_image_rows and 'iou' in per_image_rows[0]:
            f.write("Per-Image Metrics:\n")
            f.write(f"{'Image':<40} {'IoU':>8} {'Dice':>8} {'Acc':>8}\n")
            f.write("-" * 70 + "\n")
            for row in per_image_rows:
                f.write(f"{row['image']:<40} {row['iou']:>8} {row['dice']:>8} {row['pixel_acc']:>8}\n")
        else:
            f.write("Per-Image Dominant Class:\n")
            for row in per_image_rows:
                f.write(f"  {row['image']:<40}  {row.get('dominant_class','N/A')}\n")

    print(f"\nSaved {len(test_images)} predicted masks → {pred_out_dir}")
    print(f"Test metrics → {metrics_path}")
    if has_gt and all_ious:
        print(f"\nTest Results (v3):")
        print(f"  Mean IoU            : {np.mean(all_ious):.4f}")
        print(f"  Mean Dice           : {np.mean(all_dices):.4f}")
        print(f"  Mean Pixel Accuracy : {np.mean(all_accs):.4f}")
    print("\nDone!")

if __name__ == "__main__":
    main()
