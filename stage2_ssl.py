import os
import glob
import cv2
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from itertools import cycle
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from the Phase A framework
from phase3_train import TriBranchFramework
from main_train import RetinalDataset, build_labeled_datalist, supervised_loss, apply_clahe, DATA_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS_STAGE2 = 20  
LR_SSL = 2e-5

# ==========================================
# 1. UNLABELED DATASET (Random Crop)
# ==========================================
train_transform_unlabeled = A.Compose([
    A.RandomCrop(height=512, width=512), 
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], additional_targets={'vessel': 'mask'})

class UnlabeledDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        img = cv2.imread(item['img'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = apply_clahe(img)
        native_h, native_w = img.shape[:2]
        
        vessel_map = np.load(item['vessel'])
        vessel_map = cv2.resize(vessel_map, (native_w, native_h), interpolation=cv2.INTER_CUBIC)
        vessel_map = np.clip(vessel_map, 0.0, 1.0)
        
        if self.transform:
            augmented = self.transform(image=img, vessel=vessel_map)
            img = augmented['image']
            vessel_map = augmented['vessel'].float().reshape(1, 512, 512)
            
        return img, vessel_map

def build_unlabeled_datalist():
    data_list = []
    unlabeled_img_dir = os.path.join(DATA_DIR, "DDR/unlabeled")
    for img_path in glob.glob(os.path.join(unlabeled_img_dir, "*.jpg")):
        data_list.append({'img': img_path, 'vessel': img_path.replace(".jpg", "_vessel.npy")})
    return data_list


# ==========================================
# 2. DYNAMIC UNSUPERVISED LOSS
# ==========================================
def unsupervised_loss(logits_s, logits_b, logits_a, optimal_taus):
    # --- CRITICAL AMP FIX ---
    # Cast logits to float32 to prevent FP16 overflow from highly confident predictions
    logits_s = logits_s.float()
    logits_b = logits_b.float()
    logits_a = logits_a.float()
    # ------------------------

    p_s = torch.softmax(logits_s, dim=1)
    p_b = torch.softmax(logits_b, dim=1)
    p_a = torch.softmax(logits_a, dim=1)
    
    consensus_probs = (p_s + p_b + p_a) / 3.0
    max_probs, pseudo_labels = consensus_probs.max(dim=1) 

    # --- INSURANCE POLICY 1 ---
    # Explicitly detach pseudo-labels so Autograd doesn't track them
    pseudo_labels = pseudo_labels.detach() 
    # --------------------------
    
    # Build Class-Adaptive Threshold Map
    tau_map = torch.zeros_like(max_probs)
    tau_map[pseudo_labels == 0] = 0.90 # Strict requirement for background
    
    class_mapping = {1: 'EX', 2: 'HE', 3: 'MA', 4: 'SE'}
    for c_idx, c_name in class_mapping.items():
        if c_name in optimal_taus:
            tau_map[pseudo_labels == c_idx] = optimal_taus[c_name]

    # --- INSURANCE POLICY 1 (Cont.) ---
    # Explicitly detach the mask so it acts purely as a mathematical multiplier
    trust_mask = (max_probs >= tau_map).float().detach()
    # ----------------------------------
            
    # Binary Trust Mask based on Dynamic Taus
    trust_mask = (max_probs >= tau_map).float()
    
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss_s = ce_loss(logits_s, pseudo_labels)
    loss_b = ce_loss(logits_b, pseudo_labels)
    loss_a = ce_loss(logits_a, pseudo_labels)
    
    avg_ce = (loss_s + loss_b + loss_a) / 3.0 
    
    # --- BULLETPROOF SHIELD ---
    # Convert any lingering Infs or NaNs to 0.0 before applying the mask
    avg_ce = torch.nan_to_num(avg_ce, nan=0.0, posinf=0.0, neginf=0.0)
    # --------------------------
    
    mask_sum = trust_mask.sum().clamp(min=1) 
    masked_loss = (avg_ce * trust_mask).sum() / mask_sum
    
    return masked_loss

# ==========================================
# 3. STAGE 2 LOOP
# ==========================================
def train_stage_2():
    print("--- Loading Phase B Safety Gates ---")
    with open("optimal_taus.json", "r") as f:
        optimal_taus = json.load(f)
    print(f"Loaded Gates: {optimal_taus}")
    
    # Import train_transform_labeled directly from main_train module scope if needed, 
    # but here we redefine it implicitly by relying on the RetinalDataset loading it.
    from main_train import train_transform_labeled

    labeled_list = build_labeled_datalist()
    labeled_dataset = RetinalDataset(labeled_list, transform=train_transform_labeled)
    labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE//2, shuffle=True, drop_last=True)
    
    unlabeled_list = build_unlabeled_datalist()
    unlabeled_dataset = UnlabeledDataset(unlabeled_list, transform=train_transform_unlabeled) 
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE//2, shuffle=True, num_workers=2, drop_last=True)
    
    model = TriBranchFramework(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load("tribranch_warmup_final.pth", map_location=DEVICE))
    
    # Standard Adam for SSL to gently guide the weights
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_SSL)
    scaler = torch.amp.GradScaler("cuda")
    
    print(f"\n--- Starting Phase C: Class-Adaptive SSL ({EPOCHS_STAGE2} Epochs) ---")
    labeled_iter = cycle(labeled_loader) 
    
    for epoch in range(EPOCHS_STAGE2):
        model.train()
        epoch_loss = 0
        
        loop = tqdm(unlabeled_loader, desc=f"Stage 2 Epoch {epoch+1}/{EPOCHS_STAGE2}", leave=False)
        for unl_imgs, unl_vessels in loop:
            lbl_imgs, lbl_vessels, lbl_masks = next(labeled_iter)
            
            lbl_imgs, lbl_vessels, lbl_masks = lbl_imgs.to(DEVICE), lbl_vessels.to(DEVICE), lbl_masks.to(DEVICE)
            unl_imgs, unl_vessels = unl_imgs.to(DEVICE), unl_vessels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                # Supervised Step (Focal Tversky is imported with supervised_loss)
                l_s, l_b, l_a, _ = model(lbl_imgs, lbl_vessels, compute_trust=False)
                loss_sup = supervised_loss(l_s, l_b, l_a, lbl_masks)
                
                # Unsupervised Step (Dynamic Thresholding)
                u_s, u_b, u_a, _ = model(unl_imgs, unl_vessels, compute_trust=False)
                loss_unsup = unsupervised_loss(u_s, u_b, u_a, optimal_taus)
                
                lambda_u = 0.5
                total_loss = loss_sup + (lambda_u * loss_unsup)
                
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            loop.set_postfix(L_sup=f"{loss_sup.item():.3f}", L_unsup=f"{loss_unsup.item():.3f}")
            
        print(f"Stage 2 Epoch {epoch+1}/{EPOCHS_STAGE2} | Avg Total Loss: {epoch_loss/len(unlabeled_loader):.4f}")
        torch.save(model.state_dict(), "tribranch_stage2_latest.pth")

    print("\n✅ Stage 2 Training Complete!")
    torch.save(model.state_dict(), "tribranch_final_weights.pth")

if __name__ == "__main__":
    train_stage_2()