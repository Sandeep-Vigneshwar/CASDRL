import os
import glob
import json
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from the Phase A framework
from phase3_train import TriBranchFramework
from main_train import apply_clahe, DATA_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_ERROR_RATE = 0.05  # 5.0% error tolerance

# ==========================================
# 1. NATIVE RESOLUTION VAL DATASET
# ==========================================
class RetinalValDataset(Dataset):
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
        
        gt_mask = np.zeros((native_h, native_w), dtype=np.uint8)
        class_mapping = {'EX': 1, 'HE': 2, 'MA': 3, 'SE': 4}
        
        for lesion, class_idx in class_mapping.items():
            mask_path = item[lesion]
            if mask_path and os.path.exists(mask_path):
                from PIL import Image
                m = np.array(Image.open(mask_path).convert('L'))
                gt_mask[m > 0] = class_idx
        
        if self.transform:
            augmented = self.transform(image=img, mask=gt_mask, vessel=vessel_map)
            img = augmented['image']
            gt_mask = augmented['mask'].long()
            vessel_map = augmented['vessel'].reshape(1, native_h, native_w)

        return img, vessel_map, gt_mask

def build_val_datalist():
    data_list = []
    val_img_dir = os.path.join(DATA_DIR, "DDR/labeled/images/val")
    val_mask_dir = os.path.join(DATA_DIR, "DDR/labeled/annotations/val") 
    
    for img_path in glob.glob(os.path.join(val_img_dir, "*.jpg")):
        basename = os.path.basename(img_path).replace(".jpg", ".tif")
        data_list.append({
            'img': img_path,
            'vessel': img_path.replace(".jpg", "_vessel.npy"),
            'EX': os.path.join(val_mask_dir, "EX", basename),
            'HE': os.path.join(val_mask_dir, "HE", basename),
            'MA': os.path.join(val_mask_dir, "MA", basename),
            'SE': os.path.join(val_mask_dir, "SE", basename),
        })
    return data_list

# ==========================================
# 2. SLIDING WINDOW ENGINE
# ==========================================
def predict_sliding_window(model, img_tensor, vessel_tensor, patch_size=512):
    _, _, h, w = img_tensor.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    img_pad = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    ves_pad = F.pad(vessel_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    out_probs = torch.zeros((1, 5, img_pad.shape[2], img_pad.shape[3]), device=DEVICE)
    
    for y in range(0, img_pad.shape[2], patch_size):
        for x in range(0, img_pad.shape[3], patch_size):
            patch_img = img_pad[:, :, y:y+patch_size, x:x+patch_size]
            patch_ves = ves_pad[:, :, y:y+patch_size, x:x+patch_size]
            
            with torch.amp.autocast("cuda"):
                log_s, log_b, log_a, _ = model(patch_img, patch_ves, compute_trust=False)
                consensus = (torch.softmax(log_s, 1) + torch.softmax(log_b, 1) + torch.softmax(log_a, 1)) / 3.0
                
            out_probs[:, :, y:y+patch_size, x:x+patch_size] = consensus
            
    return out_probs[:, :, :h, :w]

# ==========================================
# 3. CLASS-ADAPTIVE CALIBRATION
# ==========================================
def run_calibration():
    print("--- Phase B: Class-Adaptive Safety Calibration ---")
    
    val_list = build_val_datalist()
    print(f"Total Validation Images: {len(val_list)}")
    
    tensor_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    model = TriBranchFramework(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load("tribranch_warmup_final.pth", map_location=DEVICE))
    model.eval()

    bins = 100
    # Independent histograms for each class
    correct_hist = {1: np.zeros(bins), 2: np.zeros(bins), 3: np.zeros(bins), 4: np.zeros(bins)}
    total_hist = {1: np.zeros(bins), 2: np.zeros(bins), 3: np.zeros(bins), 4: np.zeros(bins)}
    
    with torch.no_grad():
        for item in tqdm(val_list, desc="Extracting Class Distributions"):
            img, vessel_map, gt_mask = RetinalValDataset([item], transform=None)[0]
            
            aug = tensor_transform(image=img)
            img_tensor = aug['image'].unsqueeze(0).to(DEVICE)
            vessel_tensor = torch.tensor(vessel_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            gt_tensor = torch.tensor(gt_mask, dtype=torch.long).to(DEVICE)
            
            stitched_probs = predict_sliding_window(model, img_tensor, vessel_tensor)
            max_probs, preds = torch.max(stitched_probs.squeeze(0), dim=0)
            
            confidences = max_probs.cpu().numpy().flatten()
            predictions = preds.cpu().numpy().flatten()
            ground_truth = gt_tensor.cpu().numpy().flatten()
            
            bin_indices = np.clip(np.digitize(confidences, np.linspace(0, 1, bins+1)) - 1, 0, bins-1)
            
            # Map accuracy specifically per lesion class
            for c in range(1, 5):
                c_mask = (predictions == c)
                c_correct = (ground_truth[c_mask] == c)
                c_bins = bin_indices[c_mask]
                
                np.add.at(correct_hist[c], c_bins, c_correct)
                np.add.at(total_hist[c], c_bins, 1)

    print("\nCalculating Median Confidences for True Positives...")
    
    optimal_taus = {}
    class_names = {1: 'EX', 2: 'HE', 3: 'MA', 4: 'SE'}
    
    for c, name in class_names.items():
        correct_counts = correct_hist[c]
        total_correct = np.sum(correct_counts)
        
        if total_correct == 0:
            optimal_taus[name] = 0.50 # Safe fallback if a class is entirely missing
            print(f"Optimal Safety Gate for {name}: 0.5000 (Fallback)")
            continue
            
        # Find the 50th percentile (Median) confidence of correct predictions
        cumulative = np.cumsum(correct_counts)
        median_idx = np.searchsorted(cumulative, total_correct / 2.0)
        
        # Convert bin index back to confidence threshold
        optimal_tau = median_idx / 100.0
        optimal_taus[name] = optimal_tau
        print(f"Optimal Safety Gate for {name}: {optimal_tau:.4f}")

    # Save dictionary to JSON for Phase C
    with open("optimal_taus.json", "w") as f:
        json.dump(optimal_taus, f, indent=4)
    print("\n✅ Calibration Complete! Independent gates saved to 'optimal_taus.json'.")

if __name__ == "__main__":
    run_calibration()