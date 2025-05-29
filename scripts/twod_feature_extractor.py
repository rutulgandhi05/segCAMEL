import torch
from torchvision import transforms
from PIL import Image
import timm

class DINOv2FeatureExtractor:
    def __init__(self, model_name="vit_small_patch14_dinov2.lvd142m", device="cuda"):
        self.device = device
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval().to(device)

        # Get expected input size from model config (default: (3, 518, 518))
        cfg = self.model.default_cfg
        self.img_size = cfg.get("input_size", (3, 518, 518))
        self.H, self.W = self.img_size[1], self.img_size[2]

        # Compose transforms: resize, to tensor, normalize
        self.transform = transforms.Compose([
            transforms.Resize((self.H, self.W), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['mean'], std=cfg['std']),
        ])

    @torch.no_grad()
    def extract_features(self, img):
        """
        img: PIL.Image (arbitrary size)
        Returns: np.ndarray of shape (Hf*Wf, C)
        """
        # Preprocess: resize, normalize
        x = self.transform(img).unsqueeze(0).to(self.device)  # (1,3,H,W)
        feats = self.model.forward_features(x)
        if isinstance(feats, dict):
            feats = feats['x'] if 'x' in feats else next(iter(feats.values()))

        # Handle output shape robustly
        if feats.dim() == 4:
            # (1, C, H, W) → (H*W, C)
            feats = feats.squeeze(0).permute(1, 2, 0).reshape(-1, feats.shape[1])
        elif feats.dim() == 3:
            # (1, N, C) or (1, C, N)
            feats = feats.squeeze(0)
            if feats.shape[0] > feats.shape[1]:
                # (N, C)
                pass
            else:
                # (C, N) → (N, C)
                feats = feats.permute(1, 0)
        elif feats.dim() == 2:
            # (N, C)
            pass
        else:
            raise RuntimeError(f"Unexpected DINO feature shape: {feats.shape}")

        return feats.cpu().numpy()

