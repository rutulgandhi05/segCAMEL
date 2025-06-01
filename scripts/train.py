# ─── train.py (Unsupervised 2D→3D Distillation, Saving Visualizations Only) ─────────────────

import os
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from models.PTv3.model import PointTransformerV3

import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Visualization utility that saves point-cloud images (no interactive popups)
from utils.visualization import PointCloudVisualizer
visualizer = PointCloudVisualizer()


class MyPointCloudDataset(Dataset):
    """
    A PyTorch Dataset for loading preprocessed point-cloud samples with 2D DINO features.

    Each .pth file in the `train/` or `val/` subdirectory is expected to contain a dictionary
    with the following keys:
        - "coord"     : FloatTensor of shape (N, 3), the XYZ coordinates of N LiDAR points.
        - "feat"      : FloatTensor of shape (N, C1), the raw per-point features (intensity, velocity).
        - "dino_feat" : FloatTensor of shape (N, C2), the frozen 2D DINO features for each point.
        - "grid_size" : A scalar (float) representing the voxel-grid size used by the backbone.

    This dataset does not load any semantic labels; it is designed purely for unsupervised
    distillation of 3D features to match 2D DINO features.

    Args:
        root_dir (str or Path): Path to the dataset root, which must contain "train/" and "val/".
        split (str): One of {"train", "val"}, indicating which subfolder to load from.

    Returns:
        dict: A dictionary with keys "coord", "feat", "dino_feat", and "grid_size".
    """
    def __init__(self, root_dir, split="train"):
        self.base = Path(root_dir) / split
        self.files = sorted(self.base.glob("*.pth"))

    def __len__(self):
        """
        Returns:
            int: Total number of .pth files in this split.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a single point-cloud sample from disk.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            dict: {
                "coord"     : FloatTensor of shape (N, 3),
                "feat"      : FloatTensor of shape (N, C1),
                "dino_feat" : FloatTensor of shape (N, C2),
                "grid_size" : float
            }
        """
        sample = torch.load(self.files[idx])
        return {
            "coord":      sample["coord"],
            "feat":       sample["feat"],
            "dino_feat":  sample["dino_feat"],
            "grid_size":  sample["grid_size"],
        }


def collate_fn(batch):
    """
    Custom collate function for batching unsupervised point-cloud samples.

    Given a list of dictionaries (each with keys "coord", "feat", "dino_feat", "grid_size"),
    this function concatenates all point-level data into single tensors across the batch.

    Steps:
        1. Concatenate "coord" from all batch items into a (ΣN, 3) tensor.
        2. Concatenate "feat" from all batch items into a (ΣN, C1) tensor.
        3. Concatenate "dino_feat" from all batch items into a (ΣN, C2) tensor.
        4. Compute `offset`: a LongTensor of shape (batch_size,) where offset[i] is the
           cumulative number of points up to (and including) the i-th sample.
        5. Assume all samples share the same "grid_size"; take from the first element.

    Args:
        batch (List[dict]): A list of dictionaries, each returned by MyPointCloudDataset.

    Returns:
        Tuple[dict, None]: A 2-tuple where:
            - The first element is a fused `data_dict` with keys:
                - "coord"     : FloatTensor of shape (ΣN, 3)
                - "feat"      : FloatTensor of shape (ΣN, C1)
                - "dino_feat" : FloatTensor of shape (ΣN, C2)
                - "offset"    : LongTensor of shape (batch_size,)
                - "grid_size" : float
            - The second element is always None (no labels in unsupervised mode).
    """
    coords     = torch.cat([d["coord"]      for d in batch], dim=0)
    feats      = torch.cat([d["feat"]       for d in batch], dim=0)
    dino_feats = torch.cat([d["dino_feat"]  for d in batch], dim=0)

    sizes  = [d["coord"].shape[0] for d in batch]
    offset = torch.tensor(sizes, dtype=torch.long).cumsum(0)

    grid_size = batch[0]["grid_size"]

    data_dict = {
        "coord":      coords,       # (ΣN, 3)
        "feat":       feats,        # (ΣN, C1)
        "dino_feat":  dino_feats,   # (ΣN, C2)
        "offset":     offset,       # (batch_size,)
        "grid_size":  grid_size,    # float
    }

    return data_dict, None


class SegmentationNet(nn.Module):
    """
    A segmentation model wrapper that includes a PointTransformerV3 backbone and
    an (unused) classification head. For unsupervised distillation, only the
    backbone’s per-point features are used; the head and CrossEntropyLoss are ignored.

    Args:
        in_channels (int): Number of raw input feature dimensions (C1).
        dino_channels (int): Number of DINO feature dimensions (C2).
        num_classes (int): Number of semantic classes (unused here, but kept for
                           compatibility with future supervised fine-tuning).
    """
    def __init__(self, in_channels, dino_channels, num_classes):
        super().__init__()
        # Backbone: PointTransformerV3 expects `in_channels` and `dino_channels`.
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            dino_channels=dino_channels,
            cls_mode=False,
        )

        # Build a classification head (unused during unsupervised training).
        C_out = self.backbone.dec._modules["dec2"].up.proj_skip._modules["0"].out_features
        self.head = nn.Linear(C_out, num_classes)

    def forward(self, data_dict):
        """
        Forward pass through the backbone and (optionally) the classification head.

        For unsupervised distillation, downstream code will call only `model.backbone(...)`
        to obtain per-point features; `self.head` is not used during distillation.

        Args:
            data_dict (dict): Contains keys:
                - "coord"     : FloatTensor of shape (ΣN, 3)
                - "feat"      : FloatTensor of shape (ΣN, C1)
                - "dino_feat" : FloatTensor of shape (ΣN, C2)  # only for loss
                - "offset"    : LongTensor of shape (batch_size,)
                - "grid_size" : float

        Returns:
            torch.Tensor: `logits` of shape (ΣN, num_classes), produced by applying
                          the head to the backbone’s per-point features. (Unused in unsupervised mode.)
        """
        out_point = self.backbone(data_dict)
        logits = self.head(out_point.feat)
        return logits


def distillation_loss(pred_feats, target_feats):
    """
    Compute the unsupervised distillation loss between predicted 3D features and
    stored 2D DINO features using cosine similarity.

    Let pred_feats ∈ ℝ^(ΣN × D1) be the per-point features from the 3D backbone,
    and target_feats ∈ ℝ^(ΣN × D2) be the corresponding frozen DINO features.
    We first L2-normalize both along the feature dimension, optionally project
    pred_feats to dim D2 if D1 ≠ D2, then compute:

        loss = mean(1 - ⟨pred_norm[i], target_norm[i]⟩) for i ∈ {1, …, ΣN}.

    Args:
        pred_feats (torch.Tensor): Tensor of shape (ΣN, D1).
        target_feats (torch.Tensor): Tensor of shape (ΣN, D2).

    Returns:
        torch.Tensor: A scalar tensor representing the mean cosine-based distillation loss.
    """
    preds_norm = pred_feats / (pred_feats.norm(dim=1, keepdim=True).clamp(min=1e-6))
    targ_norm  = target_feats / (target_feats.norm(dim=1, keepdim=True).clamp(min=1e-6))

    if preds_norm.shape[1] != targ_norm.shape[1]:
        # Linear projection from D1 → D2
        proj = nn.Linear(preds_norm.shape[1], targ_norm.shape[1]).to(preds_norm.device)
        preds_norm = proj(preds_norm)

    cos_sim = (preds_norm * targ_norm).sum(dim=1)  # (ΣN,)
    return (1.0 - cos_sim).mean()


def train_one_epoch(model, loader, optimizer, device, epoch=None):
    """
    Run one epoch of unsupervised training by distilling 3D features to 2D DINO features.

    For each batch:
        1. Move all data in `data_dict` (coord, feat, dino_feat, offset) to `device`.
        2. Forward pass through the backbone: out_point = model.backbone(data_dict)
           → pred_feats = out_point.feat of shape (ΣN, D_backbone).
        3. Retrieve target_feats = data_dict["dino_feat"] of shape (ΣN, C2).
        4. Compute `loss = distillation_loss(pred_feats, target_feats)`.
        5. Backpropagate and update model parameters.

    Logs intermediate losses every 10 batches.

    Args:
        model (nn.Module): The SegmentationNet, but only `model.backbone` is used.
        loader (DataLoader): PyTorch DataLoader yielding (data_dict, None).
        optimizer (torch.optim.Optimizer): Optimizer for updating `model` parameters.
        device (torch.device): Device on which tensors are allocated (e.g., "cuda").
        epoch (int, optional): Current epoch number (for logging).

    Returns:
        float: The average distillation loss over all points in the epoch.
    """
    model.train()
    total_loss = 0.0
    total_pts = 0

    for batch_idx, (data_dict, _) in enumerate(loader):
        # Move all tensors in data_dict to the specified device
        for key in ("coord", "feat", "dino_feat", "offset"):
            data_dict[key] = data_dict[key].to(device)

        # 1) Compute 3D backbone features (ΣN, D_backbone)
        out_point = model.backbone(data_dict)
        pred_feats = out_point.feat  # (ΣN, D_backbone)

        # 2) Retrieve frozen 2D DINO features (ΣN, C2)
        target_feats = data_dict["dino_feat"]  # already on device

        # 3) Compute distillation loss
        loss = distillation_loss(pred_feats, target_feats)

        # 4) Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        npts = data_dict["coord"].shape[0]
        total_loss += loss.item() * npts
        total_pts += npts

        if batch_idx % 10 == 0:
            dino_avg = target_feats.norm(dim=1).mean().item()
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(loader)}]  "
                f"Distill Loss: {loss.item():.4f}  DINO ‖feat‖ Avg: {dino_avg:.4f}"
            )

    return total_loss / total_pts


def validate(model, loader, device, epoch=None):
    """
    Run an unsupervised distillation validation pass (no labels) to monitor loss.

    Similar to `train_one_epoch`, but wrapped in `torch.no_grad()` and without
    optimizer steps. Logs the validation distillation loss at the end of the epoch.
    Also saves the first batch’s point-cloud visualization (colored by DINO and
    by predicted features) to disk (no interactive display).

    Args:
        model (nn.Module): The SegmentationNet, only `model.backbone` is used.
        loader (DataLoader): Validation DataLoader yielding (data_dict, None).
        device (torch.device): Device to run inference on.
        epoch (int, optional): Current epoch number (for logging).

    Returns:
        float: The average distillation loss over all points in the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_pts = 0

    with torch.no_grad():
        for batch_idx, (data_dict, _) in enumerate(loader):
            # Move tensors in data_dict to device
            for key in ("coord", "feat", "dino_feat", "offset"):
                data_dict[key] = data_dict[key].to(device)

            # 1) Compute 3D backbone features
            out_point = model.backbone(data_dict)
            pred_feats = out_point.feat  # (ΣN, D_backbone)

            # 2) Retrieve frozen 2D DINO features
            target_feats = data_dict["dino_feat"]

            # 3) Compute distillation loss
            loss = distillation_loss(pred_feats, target_feats)

            npts = data_dict["coord"].shape[0]
            total_loss += loss.item() * npts
            total_pts += npts

            # Save visualizations for the first batch only
            if batch_idx == 0:
                coords_np     = data_dict["coord"].cpu().numpy()   # (ΣN, 3)
                dino_feats_np = target_feats.cpu().numpy()         # (ΣN, C2)

                # Color points by the first 3 dims of DINO features, normalized to [0,1]
                rgb_dino = dino_feats_np[:, :3]
                rgb_dino = (rgb_dino - rgb_dino.min(0)) / (rgb_dino.max(0) - rgb_dino.min(0) + 1e-6)

                visualizer.show_point_cloud(
                    coords_np,
                    colors=rgb_dino,
                    save_prefix=f"epoch{epoch}_val_dino_rgb",
                    use_open3d=False  # ensures no interactive window, only file saved
                )

                # Color points by the first 3 dims of predicted backbone features (normalized)
                pred_feats_np = pred_feats.cpu().numpy()  # (ΣN, D_backbone)
                rgb_pred = pred_feats_np[:, :3]
                rgb_pred = (rgb_pred - rgb_pred.min(0)) / (rgb_pred.max(0) - rgb_pred.min(0) + 1e-6)

                visualizer.show_point_cloud(
                    coords_np,
                    colors=rgb_pred,
                    save_prefix=f"epoch{epoch}_val_pred_rgb",
                    use_open3d=False  # ensures no interactive window, only file saved
                )

    val_loss = total_loss / total_pts
    logger.info(f"[Epoch {epoch}] Validation Distill Loss: {val_loss:.4f}")
    return val_loss


def main():
    """
    The main entry point for unsupervised DINO→3D distillation training.

    Steps:
        1. Set up logging and device.
        2. Construct train/val datasets and loaders using MyPointCloudDataset + collate_fn.
        3. Instantiate SegmentationNet (backbone + unused head).
        4. Run `num_epochs` of:
             - train_one_epoch(...)
             - validate(...)
           tracking the best validation distillation loss to save a checkpoint.

    The saved checkpoint is `best_unsupervised_model.pth`, containing only the backbone's weights.
    """
    logger.info("Starting unsupervised DINO→3D distillation training.")

    # Adjust the path to where your processed .pth files live:
    root_dir = (Path(__file__).resolve()
                .parent.parent / "data" / "hercules" / "processed" / "Mountain_01_Day")
    logger.info(f"Dataset root: {root_dir}")

    batch_size  = 2
    num_classes = 6      # retained for downstream fine-tuning; not used during distillation
    num_epochs  = 20
    lr          = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create train & validation datasets and loaders
    train_ds = MyPointCloudDataset(root_dir, split="train")
    val_ds   = MyPointCloudDataset(root_dir, split="val")
    logger.info(f"Train set size: {len(train_ds)}  |  Val set size: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Peek at one sample to retrieve channel dimensions
    sample = train_ds[0]
    in_channels   = sample["feat"].shape[1]      # raw input feature dim
    dino_channels = sample["dino_feat"].shape[1] # DINO feature dim
    logger.info(f"in_channels = {in_channels}, dino_channels = {dino_channels}")

    # Instantiate model (backbone + unused head)
    model = SegmentationNet(
        in_channels=in_channels,
        dino_channels=dino_channels,
        num_classes=num_classes
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs} — Training")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        logger.info(f"[Epoch {epoch}] Train Distill Loss: {train_loss:.6f}")

        logger.info(f"Epoch {epoch}/{num_epochs} — Validation")
        val_loss = validate(model, val_loader, device, epoch)
        logger.info(f"[Epoch {epoch}] Val Distill Loss: {val_loss:.6f}")

        # Save the best-performing distillation model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unsupervised_model.pth")
            logger.info(
                f"Saved best model at epoch {epoch} (ValDistillLoss={val_loss:.6f})"
            )


if __name__ == "__main__":
    main()
