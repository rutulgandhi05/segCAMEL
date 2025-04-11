import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModel
import cv2

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_NAME = 'facebook/dinov2-large'
N_CLUSTERS = 5
MORPH_KERNEL_SIZE = 5
SMOOTH_ITER = 2
# ------------------------------------------------------------

# Load DINOv2 Large model and processor
print(f"Loading {MODEL_NAME}...")
model = AutoModel.from_pretrained(MODEL_NAME)
feature_extractor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model.eval()

def extract_patch_embeddings(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        patch_tokens = outputs.last_hidden_state
        patch_tokens = patch_tokens[:, 1:, :]
        
        num_patches = patch_tokens.shape[1]
        grid_size = int(num_patches ** 0.5)
        assert grid_size * grid_size == num_patches, "Non-square patch grid!"
        
        patch_embeddings = patch_tokens[0].reshape(grid_size, grid_size, -1).cpu().numpy()
        
    print(f"Patch embeddings shape: {patch_embeddings.shape}")
    return patch_embeddings, image.size, image

def segment_patches(patch_embeddings, image_size, n_clusters=N_CLUSTERS):
    h, w, c = patch_embeddings.shape
    patch_embeddings_flat = patch_embeddings.reshape(-1, c)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(patch_embeddings_flat)
    labels = kmeans.labels_.reshape(h, w)
    
    labels_resized = cv2.resize(labels.astype(np.uint8), image_size, interpolation=cv2.INTER_NEAREST)
    
    print(f"Segmented label map shape (resized): {labels_resized.shape}")
    return labels_resized

def refine_segmentation(labels_resized):
    """
    Apply morphological operations and smoothing to clean segmentation map.
    """
    print("Refining segmentation...")
    refined = labels_resized.copy().astype(np.uint8)

    # Morphological operations (opening then closing)
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)

    # Optional: Bilateral filtering for smoothing
    for i in range(SMOOTH_ITER):
        refined = cv2.bilateralFilter(refined, d=9, sigmaColor=75, sigmaSpace=75)

    print("Refinement complete.")
    return refined

def visualize_segmentation(image, segmented_labels, title="Segmented Image"):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_labels, cmap='jet')
    plt.title(title)
    plt.axis('off')
    
    plt.show()

def run_segmentation_pipeline(image_path):
    patch_embeddings, img_size, pil_image = extract_patch_embeddings(image_path)
    
    labels_resized = segment_patches(patch_embeddings, img_size, n_clusters=N_CLUSTERS)
    
    refined_labels = refine_segmentation(labels_resized)
    
    visualize_segmentation(pil_image, refined_labels, title="Refined Segmentation")

if __name__ == "__main__":
    image_path = Path(input("Image Path: ").strip())
    run_segmentation_pipeline(image_path)
