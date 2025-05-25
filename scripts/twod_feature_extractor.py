import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class DINOv2FeatureExtractor:
    def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[DEBUG] Using device: {self.device}")
        self.model = timm.create_model(model_name, pretrained=True).to(self.device).eval()
        # DINOv2 base expects 518x518 input
        self.input_size = 518
        self.patch_size = 14
        self.grid_size = self.input_size // self.patch_size  # 37
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image: Image.Image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.forward_features(img_tensor)
            if isinstance(feats, dict):
                feats = feats["x"] if "x" in feats else list(feats.values())[0]
        features = feats.squeeze(0).cpu().numpy()
        print(f"[DEBUG] Extracted features shape: {features.shape}")
        return features

    def get_feature_mask(self, features, n_clusters=5):
        # Automatically infer patch grid size and skip class token if present
        if features.ndim == 2:
            if features.shape[0] == (self.grid_size * self.grid_size) + 1:
                # Remove class token
                features = features[1:]
            hw, c = features.shape
            h = w = int(hw ** 0.5)
            features_flat = features
            mask_shape = (h, w)
        elif features.ndim == 3:
            h, w, c = features.shape
            features_flat = features.reshape(h * w, c)
            mask_shape = (h, w)
        else:
            raise ValueError(f"Unexpected features shape: {features.shape}")
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        labels = kmeans.fit_predict(features_flat)
        mask = labels.reshape(mask_shape)
        return mask

    def visualize_mask(self, image: Image.Image, mask: np.ndarray, alpha=0.5, upsample=True, colormap='tab20'):
        import matplotlib
        if upsample:
            import cv2
            mask_resized = cv2.resize(mask.astype(np.float32), image.size, interpolation=cv2.INTER_LINEAR)
        else:
            mask_resized = np.array(Image.fromarray(mask).resize(image.size, resample=Image.NEAREST))
        # Fix: Use colormap and set bad values for categorical coloring
        cmap = matplotlib.colormaps.get_cmap(colormap)
        n_colors = int(mask.max() + 1)
        color_indices = (mask_resized.astype(int) % n_colors)
        mask_rgb = (cmap(color_indices / max(n_colors - 1, 1))[:, :, :3] * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_rgb)
        blended = Image.blend(image.convert('RGB'), mask_img, alpha=alpha)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title(f'Feature Mask Overlay ({colormap})')
        plt.imshow(blended)
        plt.axis('off')
        plt.show()

    def save_mask(self, mask, out_path):
        # Save the mask as a grayscale PNG (each pixel is the cluster label)
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img.save(out_path)

    def process_and_save_mask(self, image_path, n_clusters=5, out_mask=None):
        image = Image.open(image_path).convert('RGB')
        features = self.extract_features(image)
        mask = self.get_feature_mask(features, n_clusters)
        if out_mask:
            self.save_mask(mask, out_mask)
            print(f"[INFO] Saved mask to {out_mask}")
        else:
            self.visualize_mask(image, mask)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and visualize or save DINOv2 feature masks from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters for segmentation mask.")
    parser.add_argument("--colormap", type=str, default="tab20", help="Colormap for mask visualization.")
    parser.add_argument("--out-mask", type=str, default=None, help="Path to save the mask as PNG (if set, disables visualization)")
    args = parser.parse_args()

    extractor = DINOv2FeatureExtractor()
    extractor.process_and_save_mask(args.image, n_clusters=args.clusters, out_mask=args.out_mask)
