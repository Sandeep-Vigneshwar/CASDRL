import os
import glob
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp

# Import our verified architecture
from phase3_train import TriBranchFramework

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4       
EPOCHS_STAGE1 = 50   
MAX_LR = 1e-3        # Much higher peak LR for OneCycleLR

DATA_DIR = "data"

# ==========================================
# 2. PREPROCESSING & AUGMENTATIONS
# ==========================================
def apply_clahe(img_rgb, clip_limit=2.0, tile_grid=(8, 8)):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

train_transform_labeled = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], additional_targets={'vessel': 'mask'})

# ==========================================
# 3. LESION-CENTRIC DATASET
# ==========================================
class RetinalDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. Load Native High-Res Image
        img = cv2.imread(item['img'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = apply_clahe(img)
        native_h, native_w = img.shape[:2]
        
        # 2. Load & Upscale Vessel Prior
        vessel_map = np.load(item['vessel'])
        vessel_map = cv2.resize(vessel_map, (native_w, native_h), interpolation=cv2.INTER_CUBIC)
        vessel_map = np.clip(vessel_map, 0.0, 1.0)
        
        # 3. Stack Native Masks
        gt_mask = np.zeros((native_h, native_w), dtype=np.uint8)
        class_mapping = {'EX': 1, 'HE': 2, 'MA': 3, 'SE': 4}
        
        for lesion, class_idx in class_mapping.items():
            mask_path = item[lesion]
            if mask_path and os.path.exists(mask_path):
                from PIL import Image
                m = np.array(Image.open(mask_path).convert('L'))
                gt_mask[m > 0] = class_idx

        # 4. === FORCED LESION-CENTRIC CROPPING ===
        present_classes = np.unique(gt_mask)
        lesion_classes = [c for c in present_classes if c in [1, 2, 3, 4]]
        
        if len(lesion_classes) > 0:
            # Randomly pick a class that physically exists in this image
            # A 5-pixel MA has the same chance to be picked as a 500-pixel EX
            target_class = random.choice(lesion_classes)
            y_indices, x_indices = np.where(gt_mask == target_class)
            
            # Pick a random center pixel from that specific lesion
            pixel_idx = random.randint(0, len(y_indices) - 1)
            center_y, center_x = y_indices[pixel_idx], x_indices[pixel_idx]
            
            # Calculate safe 512x512 bounds
            y_min = max(0, min(center_y - 256, native_h - 512))
            x_min = max(0, min(center_x - 256, native_w - 512))
        else:
            # Fallback for completely healthy images
            y_min = random.randint(0, native_h - 512)
            x_min = random.randint(0, native_w - 512)
            
        # Extract the exact 512x512 arrays
        img_crop = img[y_min:y_min+512, x_min:x_min+512]
        vessel_crop = vessel_map[y_min:y_min+512, x_min:x_min+512]
        mask_crop = gt_mask[y_min:y_min+512, x_min:x_min+512]
        
        # 5. Apply Pixel/Color Augmentations
        if self.transform:
            augmented = self.transform(image=img_crop, mask=mask_crop, vessel=vessel_crop)
            img_tensor = augmented['image']
            gt_tensor = augmented['mask'].long()
            vessel_tensor = augmented['vessel'].reshape(1, 512, 512)

        return img_tensor, vessel_tensor, gt_tensor

def build_labeled_datalist():
    data_list = []
    
    # 1. DDR Train
    ddr_img_dir = os.path.join(DATA_DIR, "DDR/labeled/images/train")
    ddr_mask_dir = os.path.join(DATA_DIR, "DDR/labeled/annotations/train")
    for img_path in glob.glob(os.path.join(ddr_img_dir, "*.jpg")):
        basename = os.path.basename(img_path).replace(".jpg", ".tif")
        data_list.append({
            'img': img_path,
            'vessel': img_path.replace(".jpg", "_vessel.npy"),
            'EX': os.path.join(ddr_mask_dir, "EX", basename),
            'HE': os.path.join(ddr_mask_dir, "HE", basename),
            'MA': os.path.join(ddr_mask_dir, "MA", basename),
            'SE': os.path.join(ddr_mask_dir, "SE", basename),
        })

    # 2. IDRiD Train
    idrid_img_dir = os.path.join(DATA_DIR, "IDRiD/A. Segmentation/1. Original Images/a. Training Set")
    idrid_mask_base = os.path.join(DATA_DIR, "IDRiD/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set")
    for img_path in glob.glob(os.path.join(idrid_img_dir, "*.jpg")):
        base = os.path.basename(img_path).replace(".jpg", "")
        data_list.append({
            'img': img_path,
            'vessel': img_path.replace(".jpg", "_vessel.npy"),
            'EX': os.path.join(idrid_mask_base, "3. Hard Exudates", f"{base}_EX.tif"),
            'HE': os.path.join(idrid_mask_base, "2. Haemorrhages", f"{base}_HE.tif"),
            'MA': os.path.join(idrid_mask_base, "1. Microaneurysms", f"{base}_MA.tif"),
            'SE': os.path.join(idrid_mask_base, "4. Soft Exudates", f"{base}_SE.tif"),
        })

    return data_list

# ==========================================
# 4. FOCAL TVERSKY LOSS ENGINE
# ==========================================
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, gamma=1.5):
        super().__init__()
        # alpha=FP penalty, beta=FN penalty. beta=0.8 forces aggression!
        self.tversky = smp.losses.TverskyLoss(mode='multiclass', alpha=alpha, beta=beta)
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # SMP Tversky computes (1 - TI). We raise it to gamma to make it focal.
        t_loss = self.tversky(y_pred, y_true)
        t_loss = torch.clamp(t_loss, min=1e-6)
        return torch.pow(t_loss, self.gamma)


def supervised_loss(l_s, l_b, l_a, gt_mask, lambda_ftl=1.0):
    # --- CRITICAL AMP FIX (Float32 Armor) ---
    # Cast logits to float32 to survive the exponential Tversky math
    l_s = l_s.float()
    l_b = l_b.float()
    l_a = l_a.float()
    # ----------------------------------------
    
    dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    ftl = FocalTverskyLoss(alpha=0.2, beta=0.8, gamma=1.5)
    
    # Combined Dice + Focal Tversky guarantees boundary precision AND lesion hunting
    loss_s = dice(l_s, gt_mask) + lambda_ftl * ftl(l_s, gt_mask)
    loss_b = dice(l_b, gt_mask) + lambda_ftl * ftl(l_b, gt_mask)
    loss_a = dice(l_a, gt_mask) + lambda_ftl * ftl(l_a, gt_mask)
    
    total_loss = (loss_s + loss_b + loss_a) / 3.0
    
    # --- BULLETPROOF SHIELD ---
    # Catch any absolute extreme edge cases
    return torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

# ==========================================
# 5. STAGE 1 TRAINING LOOP
# ==========================================
def train_stage_1():
    print(f"--- Building Lesion-Centric Dataset ---")
    data_list = build_labeled_datalist()
    
    dataset = RetinalDataset(data_list, transform=train_transform_labeled)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    print(f"--- Initializing TriBranch Framework ({DEVICE}) ---")
    model = TriBranchFramework(num_classes=5).to(DEVICE)
    
    # AdamW with weight decay to stabilize the Vision Transformer (DINOv2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-4)
    
    # OneCycleLR dynamically pushes the model out of local minima
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=MAX_LR, 
        steps_per_epoch=len(dataloader), 
        epochs=EPOCHS_STAGE1,
        pct_start=0.3 # Spends 30% of time ramping up, 70% decaying
    )
    
    scaler = torch.amp.GradScaler("cuda")
    
    print(f"--- Starting SOTA Stage 1 ({EPOCHS_STAGE1} Epochs) ---")
    
    for epoch in range(EPOCHS_STAGE1):
        model.train()
        epoch_loss = 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS_STAGE1}", leave=False)
        for imgs, vessels, gt_masks in loop:
            imgs, vessels, gt_masks = imgs.to(DEVICE), vessels.to(DEVICE), gt_masks.to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()
            
            with torch.amp.autocast("cuda"):
                l_s, l_b, l_a, _ = model(imgs, vessels, compute_trust=False)
                loss = supervised_loss(l_s, l_b, l_a, gt_masks)
                
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients from the aggressive Tversky loss
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS_STAGE1} | LR: {scheduler.get_last_lr()[0]:.2e} | Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"tribranch_warmup_ep{epoch+1}.pth")

    print("--- Stage 1 Complete. Saving SOTA Base Weights ---")
    torch.save(model.state_dict(), "tribranch_warmup_final.pth")
    return model

if __name__ == "__main__":
    train_stage_1()