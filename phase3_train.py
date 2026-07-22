import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# ==========================================
# 1. SEMANTIC BRANCH (DINOv2)
# ==========================================
class SemanticBranch(nn.Module):
    def __init__(self, num_classes=5): # Background + EX, HE, MA, SE
        super().__init__()
        # Load lightweight DINOv2 Vision Transformer (ViT-Small)
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', skip_validation=True)
        
        # ViT-S has 12 blocks. We freeze the first 8.
        for name, param in self.backbone.named_parameters():
            if "blocks" in name:
                block_idx = int(name.split(".")[1])
                if block_idx < 8:
                    param.requires_grad = False
            elif "cls_token" in name or "pos_embed" in name or "patch_embed" in name:
                param.requires_grad = False

        d_model = 384 # Embedding dimension for ViT-S
        
        # Decoder: Upsample from 37x37 feature map back to 512x512
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 256, kernel_size=4, stride=4),   # 37 -> 148
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),       # 148 -> 296
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),        # 296 -> 592
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x is [B, 3, 512, 512]. Pad to [B, 3, 518, 518] to divide by 14 evenly.
        x_pad = F.pad(x, (3, 3, 3, 3), mode='reflect')
        
        # Extract features from DINOv2
        features = self.backbone.forward_features(x_pad)['x_norm_patchtokens']
        
        # Reshape sequence back to spatial grid [B, 384, 37, 37]
        B, N, C = features.shape
        grid_size = int(N ** 0.5) # Should be 37
        features = features.transpose(1, 2).reshape(B, C, grid_size, grid_size)
        
        # Decode and crop the padded area out
        out_padded = self.decoder(features) # [B, 5, 592, 592]
        
        # We need to center-crop 512x512 out of the 592x592
        start = (592 - 512) // 2
        out = out_padded[:, :, start:start+512, start:start+512]
        
        return out

# ==========================================
# 2. BOUNDARY BRANCH (DeepLabv3+)
# ==========================================
class BoundaryBranch(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # ASPP handles high-resolution edge preservation
        self.model = smp.DeepLabV3Plus(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)

# ==========================================
# 3. ANATOMY BRANCH (Dual-Stream)
# ==========================================
class AnatomyBranch(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # ---------------------------------------------------------
        # THE HIJACK STRATEGY:
        # Instead of guessing the internal API of UnetDecoder, we
        # instantiate a full U-Net and 'steal' its components. This
        # guarantees 100% compatibility with your SMP version.
        # ---------------------------------------------------------
        base_unet = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
        
        self.rgb_enc = base_unet.encoder
        self.decoder = base_unet.decoder
        self.seg_head = base_unet.segmentation_head
        
        # Stream B: Vessel map encoder (1 channel input)
        self.vessel_enc = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        
        # Cross-Attention at bottleneck (dim=512)
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

    def forward(self, rgb, vessel_map):
        rgb_feats = self.rgb_enc(rgb) 
        vessel_feats = self.vessel_enc(vessel_map) 
        
        # Convert rgb_feats to a list so we can mutate the bottleneck
        if isinstance(rgb_feats, tuple):
            rgb_feats = list(rgb_feats)
            
        bottleneck = rgb_feats[-1]
        B, C, H, W = bottleneck.shape
        
        # Flatten for MultiHeadAttention
        q = bottleneck.flatten(2).transpose(1, 2)
        kv = vessel_feats.flatten(2).transpose(1, 2)
        
        # Cross-Attend: RGB queries Vessel data
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        
        # Add residual connection and reshape back to spatial map
        fused_bottleneck = (q + attn_out).transpose(1, 2).reshape(B, C, H, W)
        
        # Replace original bottleneck with fused bottleneck
        rgb_feats[-1] = fused_bottleneck
        
        # ---------------------------------------------------------
        # DYNAMIC DECODER ROUTING:
        # Gracefully handles the version mismatch for positional arguments
        # ---------------------------------------------------------
        try:
            # Modern SMP versions expect unpacked arguments
            out = self.decoder(*rgb_feats)
        except TypeError:
            # Older SMP versions expect a single list/tuple argument
            out = self.decoder(rgb_feats)
            
        return self.seg_head(out)

# ==========================================
# 4. STRUCTURED DISAGREEMENT MODULE (SDM)
# ==========================================
class SDM(nn.Module):
    def __init__(self, in_channels=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() # Outputs T(x) probabilities between [0, 1]
        )

    def forward(self, x):
        return self.net(x)

def build_disagreement_tensor(p1, p2, p3):
    """
    Constructs the 11-channel tensor from the 3 branch predictions.
    Inputs are SOFTMAX probabilities [B, C, H, W].
    """
    def entropy(p):
        return -(p * torch.log(p + 1e-8)).sum(dim=1, keepdim=True)
    
    # Channels 1-3: Max probability of each branch
    p1_max = p1.max(dim=1, keepdim=True)[0]
    p2_max = p2.max(dim=1, keepdim=True)[0]
    p3_max = p3.max(dim=1, keepdim=True)[0]
    
    # Channels 4-6: Uncertainty (Entropy)
    u1, u2, u3 = entropy(p1), entropy(p2), entropy(p3)
    
    # Channels 7-9: Pairwise absolute differences (summed over classes)
    d12 = (p1 - p2).abs().sum(dim=1, keepdim=True)
    d13 = (p1 - p3).abs().sum(dim=1, keepdim=True)
    d23 = (p2 - p3).abs().sum(dim=1, keepdim=True)
    
    # Channels 10-11: Aggregates
    stack = torch.stack([p1_max, p2_max, p3_max], dim=1) # [B, 3, 1, H, W]
    stack = stack.squeeze(2) # [B, 3, H, W]
    agg_max = stack.max(dim=1, keepdim=True)[0]
    agg_mean = stack.mean(dim=1, keepdim=True)
    
    return torch.cat([p1_max, p2_max, p3_max, u1, u2, u3, d12, d13, d23, agg_max, agg_mean], dim=1)

# ==========================================
# 5. UNIFIED FRAMEWORK
# ==========================================
class TriBranchFramework(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.branch_s = SemanticBranch(num_classes)
        self.branch_b = BoundaryBranch(num_classes)
        self.branch_a = AnatomyBranch(num_classes)
        self.sdm = SDM(in_channels=11)

    def forward(self, rgb, vessel_map, compute_trust=True):
        # 1. Get raw logits
        logits_s = self.branch_s(rgb)
        logits_b = self.branch_b(rgb)
        logits_a = self.branch_a(rgb, vessel_map)
        
        if not compute_trust:
            return logits_s, logits_b, logits_a, None

        # 2. Convert to probabilities for SDM
        p1 = torch.softmax(logits_s, dim=1)
        p2 = torch.softmax(logits_b, dim=1)
        p3 = torch.softmax(logits_a, dim=1)
        
        # 3. Construct disagreement and generate Trust Mask T(x)
        conflict_tensor = build_disagreement_tensor(p1, p2, p3)
        trust_map = self.sdm(conflict_tensor)
        
        return logits_s, logits_b, logits_a, trust_map

# ==========================================
# 6. VRAM DRY-RUN TESTER
# ==========================================
if __name__ == "__main__":
    print("Initializing Tri-Branch Framework...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = TriBranchFramework().to(device)
    model.train() # Set to train to track gradients
    
    print("Simulating a batch forward pass (Batch Size = 4) to test VRAM...")
    # Using batch size 4 for the initial stress test. We can push to 8 if this clears.
    dummy_rgb = torch.randn(4, 3, 512, 512).to(device)
    dummy_vessel = torch.randn(4, 1, 512, 512).to(device)
    
    try:
        with torch.amp.autocast("cuda"):
            l_s, l_b, l_a, trust = model(dummy_rgb, dummy_vessel)
            
            # Simulate a loss backward pass to check gradient memory
            loss = l_s.mean() + l_b.mean() + l_a.mean() + trust.mean()
            loss.backward()
            
        print("✅ SUCCESS: Forward and Backward passes complete.")
        print(f"Output shapes -> Logits: {l_s.shape} | Trust Map: {trust.shape}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ OOM ERROR: We hit the VRAM limit.")
            print("We will need to test branches individually or use gradient accumulation.")
        else:
            raise e