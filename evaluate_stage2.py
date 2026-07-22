import os
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from phase3_train import TriBranchFramework
from main_train import apply_clahe, DATA_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_test_datalist():
    data_list = []
    # DDR Test
    ddr_img_dir = os.path.join(DATA_DIR, "DDR/labeled/images/test")
    ddr_mask_dir = os.path.join(DATA_DIR, "DDR/labeled/annotations/test")
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

    # IDRiD Test
    idrid_img_dir = os.path.join(DATA_DIR, "IDRiD/A. Segmentation/1. Original Images/b. Testing Set")
    idrid_mask_base = os.path.join(DATA_DIR, "IDRiD/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set")
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

def predict_sliding_window_tta(model, img_tensor, vessel_tensor, patch_size=512):
    _, _, h, w = img_tensor.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    img_pad = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    ves_pad = F.pad(vessel_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    out_probs = torch.zeros((1, 5, img_pad.shape[2], img_pad.shape[3]), device=DEVICE)
    
    for y in range(0, img_pad.shape[2], patch_size):
        for x in range(0, img_pad.shape[3], patch_size):
            p_img = img_pad[:, :, y:y+patch_size, x:x+patch_size]
            p_ves = ves_pad[:, :, y:y+patch_size, x:x+patch_size]
            
            with torch.amp.autocast("cuda"):
                # Pass 1: Normal
                l_s, l_b, l_a, _ = model(p_img, p_ves, compute_trust=False)
                c_norm = (torch.softmax(l_s, 1) + torch.softmax(l_b, 1) + torch.softmax(l_a, 1)) / 3.0
                
                # Pass 2: Horizontal Flip
                l_s, l_b, l_a, _ = model(torch.flip(p_img, [3]), torch.flip(p_ves, [3]), compute_trust=False)
                c_hf = torch.flip((torch.softmax(l_s, 1) + torch.softmax(l_b, 1) + torch.softmax(l_a, 1)) / 3.0, [3])
                
                # Pass 3: Vertical Flip
                l_s, l_b, l_a, _ = model(torch.flip(p_img, [2]), torch.flip(p_ves, [2]), compute_trust=False)
                c_vf = torch.flip((torch.softmax(l_s, 1) + torch.softmax(l_b, 1) + torch.softmax(l_a, 1)) / 3.0, [2])
                
                # Aggregate TTA
                tta_consensus = (c_norm + c_hf + c_vf) / 3.0
                
            out_probs[:, :, y:y+patch_size, x:x+patch_size] = tta_consensus
            
    return out_probs[:, :, :h, :w]

def evaluate_metrics(preds, targets, num_classes=5):
    dsc_per_class = []
    for cls in range(1, num_classes):
        p = (preds == cls).float()
        t = (targets == cls).float()
        
        intersection = (p * t).sum().item()
        dsc = (2. * intersection + 1e-6) / (p.sum().item() + t.sum().item() + 1e-6)
        dsc_per_class.append(dsc)

    macro_dsc = np.mean(dsc_per_class) * 100
    return macro_dsc, dsc_per_class

def run_evaluation():
    print("--- Starting Final TTA Evaluation ---")
    
    test_list = build_test_datalist()
    tensor_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    model = TriBranchFramework(num_classes=5).to(DEVICE)
    weights_path = "tribranch_final_weights.pth"
    if not os.path.exists(weights_path):
        weights_path = "tribranch_stage2_latest.pth"
        
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    all_preds, all_targets = [], []
    # Because Tversky compressed probabilities, extracting at 0.35 yields the best F1 balance
    FOREGROUND_THRESHOLD = 0.008 

    with torch.no_grad():
        for item in tqdm(test_list, desc="TTA Sliding Window"):
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

            aug = tensor_transform(image=img)
            img_tensor = aug['image'].unsqueeze(0).to(DEVICE)
            vessel_tensor = torch.tensor(vessel_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            gt_tensor = torch.tensor(gt_mask, dtype=torch.uint8)
            
            stitched_probs = predict_sliding_window_tta(model, img_tensor, vessel_tensor)
            
            prob_lesions = stitched_probs[:, 1:, :, :]
            max_lesion_prob, lesion_preds = torch.max(prob_lesions, dim=1)
            lesion_preds += 1
            
            preds = torch.where(max_lesion_prob > FOREGROUND_THRESHOLD, lesion_preds, torch.zeros_like(lesion_preds))
            
            all_preds.append(preds.squeeze().cpu().to(torch.uint8).flatten())
            all_targets.append(gt_tensor.flatten())

    flat_preds = torch.cat(all_preds)
    flat_targets = torch.cat(all_targets)
    
    macro_dsc, dsc_lesions = evaluate_metrics(flat_preds, flat_targets)

    print("\n==============================================")
    print("         FINAL SOTA BENCHMARK RESULTS         ")
    print("==============================================")
    print(f"Macro Dice Similarity Coefficient (DSC): {macro_dsc:.2f}%")
    print("----------------------------------------------")
    labels = ['EX', 'HE', 'MA', 'SE']
    for label, score in zip(labels, dsc_lesions):
        print(f"  - {label}: {score*100:.2f}%")
    print("==============================================")

if __name__ == "__main__":
    run_evaluation()