import os
os.environ["NO_ALBUMENTATIONS_UPDATE_CHECK"] = "1"
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 100
LR = 3e-4

# Base paths based on your structure
DATA_DIR = "data"
DRIVE_TRAIN_IMG = os.path.join(DATA_DIR, "DRIVE/training/images")
DRIVE_TRAIN_MASK = os.path.join(DATA_DIR, "DRIVE/training/1st_manual")

# ==========================================
# 2. PREPROCESSING UTILITIES
# ==========================================
def apply_clahe(img_rgb, clip_limit=2.0, tile_grid=(8, 8)):
    """Applies CLAHE on the L-channel in LAB color space for heavy illumination variance."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

# Albumentations pipelines
train_transform = A.Compose([
    A.Resize(512, 512, interpolation=cv2.INTER_CUBIC),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

infer_transform = A.Compose([
    A.Resize(512, 512, interpolation=cv2.INTER_CUBIC),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==========================================
# 3. DRIVE DATASET CLASS
# ==========================================
class DRIVEDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.gif")))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Image
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE to match our DDR/IDRiD preprocessing pipeline
        img = apply_clahe(img)

        # Mask (PIL needed for .gif)
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32) # Binarize

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0) # Add channel dim

        return img, mask

# ==========================================
# 4. TRAINING LOOP FOR VESSEL U-NET
# ==========================================
def train_vessel_unet():
    print(f"\n--- Starting Vessel U-Net Pre-training on DRIVE ({DEVICE}) ---")
    
    # Model: Lightweight ResNet34 U-Net is perfect for 8GB VRAM
    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1).to(DEVICE)
    
    dataset = DRIVEDataset(DRIVE_TRAIN_IMG, DRIVE_TRAIN_MASK, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision for RTX 4060
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        
        # Wrap the dataloader with tqdm for a visual progress bar
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for imgs, masks in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss = criterion(preds, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            # Update the progress bar with the real-time loss of the current batch
            loop.set_postfix(loss=loss.item())
            
        # Print the final average loss for the epoch once it completes
        print(f"Epoch {epoch+1}/{EPOCHS} Completed | Avg Loss: {epoch_loss/len(dataloader):.4f}")
            
    print("--- Training Complete. Saving weights... ---")
    torch.save(model.state_dict(), "vessel_unet_drive.pth")
    return model

# ==========================================
# 5. PRIOR GENERATOR PIPELINE
# ==========================================
@torch.no_grad()
def generate_priors(model):
    print("\n--- Generating Static Vessel Maps for DDR and IDRiD ---")
    model.eval()
    
    # Compile a list of all images that need vessel priors
    # We will search the tree and gather all .jpg files inside IDRiD and DDR
    target_images = []
    
    # DDR Train, Val, Test, Unlabeled
    target_images.extend(glob.glob(os.path.join(DATA_DIR, "DDR/**/*.jpg"), recursive=True))
    # IDRiD Original Images (Train & Test)
    target_images.extend(glob.glob(os.path.join(DATA_DIR, "IDRiD/**/*.jpg"), recursive=True))
    
    print(f"Found {len(target_images)} images. Processing in batches...")

    for img_path in tqdm(target_images, desc="Generating Priors"):
        base, _ = os.path.splitext(img_path)
        out_path = f"{base}_vessel.npy"
        
        # Skip if already exists (makes it resumeable)
        #if os.path.exists(out_path):
        #    continue
            
        # Load and preprocess
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = apply_clahe(img)
        
        augmented = infer_transform(image=img)
        tensor_img = augmented['image'].unsqueeze(0).to(DEVICE) # [1, 3, 512, 512]
        
        # Infer and apply sigmoid for probabilities
        with torch.amp.autocast('cuda'):
            logits = model(tensor_img)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy() # [512, 512] float32 array
            
        # Save as float16 to save disk space if preferred, but float32 is safer for deep learning tensors
        np.save(out_path, probs.astype(np.float32))

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Check if we already have the trained weights
    weights_path = "vessel_unet_drive.pth"
    
    if not os.path.exists(weights_path):
        vessel_model = train_vessel_unet()
    else:
        print(f"Found existing weights at {weights_path}. Loading model...")
        vessel_model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
        vessel_model.load_state_dict(torch.load(weights_path))
        
    # Generate the .npy maps
    generate_priors(vessel_model)
    print("\nPhase 2 Complete! All vessel prior maps are ready for the anatomy branch.")
