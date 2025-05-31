# ─── Imports ─────────────────────────────────────────────────────
import os
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# Adjust this import as needed!
from models.PTv3.model import PointTransformerV3

import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

from utils.visualization import PointCloudVisualizer
visualizer = PointCloudVisualizer()

# ─── Dataset Class ───────────────────────────────────────────────
class MyPointCloudDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.base = Path(root_dir) / split
        self.files = sorted(self.base.glob("*.pth"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return data

# ─── Collate Function ────────────────────────────────────────────
def collate_fn(batch):
    dicts, labels = zip(*batch)
    coords = torch.cat([d["coord"] for d in dicts], dim=0)
    feats  = torch.cat([d["feat"]  for d in dicts], dim=0)
    dino_feats = torch.cat([d["dino_feat"] for d in dicts], dim=0)
    sizes  = [d["coord"].shape[0] for d in dicts]
    offset = torch.tensor(sizes, dtype=torch.long).cumsum(0)
    grid_size = dicts[0]["grid_size"]
    data_dict = {
        "coord": coords,
        "feat": feats,
        "dino_feat": dino_feats,
        "offset": offset,
        "grid_size": grid_size,
    }
    labels = torch.cat(labels, dim=0)
    return data_dict, labels

# ─── Segmentation Model ──────────────────────────────────────────
class SegmentationNet(nn.Module):
    def __init__(self, in_channels, dino_channels, num_classes):
        super().__init__()
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            dino_channels=dino_channels,      # <-- NEW! for DITR fusion
        )
        logger.info(f"Backbone output: {self.backbone.dec._modules.dec2}")
        logger.info(f"Backbone output: {self.backbone.dec._modules['dec2'].up.proj_skip[-1].embed_channels}")
        C_out = self.backbone.dec._modules["dec2"].up.proj_skip[-1].embed_channels
        self.head = nn.Linear(C_out, num_classes)

    def forward(self, data_dict):
        out_point = self.backbone(data_dict)
        logits = self.head(out_point.feat)
        return logits

# ─── Training and Validation ─────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch=None):
    model.train()
    total_loss = 0.0
    total_pts = 0
    
    for batch_idx, (data_dict, labels) in enumerate(loader):
        for k in data_dict:
            data_dict[k] = data_dict[k].to(device)
        labels = labels.to(device)
        preds = model(data_dict)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.numel()
        total_pts  += labels.numel()

        # ---- LOGGING (every 10 batches) ----
        if batch_idx % 10 == 0:
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx}] "
                f"Loss: {loss.item():.4f} "
                f"DINO mean: {data_dict['dino_feat'].mean():.4f}, "
                f"DINO std: {data_dict['dino_feat'].std():.4f}"
            )

            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")

    return total_loss / total_pts

def validate(model, loader, criterion, device, epoch=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_pts = 0

    # Only visualize the first batch of every epoch (to avoid excessive plots)
    visualize_this_epoch = True

    with torch.no_grad():
        for batch_idx, (data_dict, labels) in enumerate(loader):
            for k in data_dict:
                data_dict[k] = data_dict[k].to(device)
            labels = labels.to(device)
            preds = model(data_dict)
            loss = criterion(preds, labels)
            total_loss += loss.item() * labels.numel()
            pred_classes = preds.argmax(dim=1)
            correct += (pred_classes == labels).sum().item()
            total_pts += labels.numel()

            # ---- VISUALIZATION BLOCK ----
            if batch_idx == 0 and visualize_this_epoch:
                coords = data_dict["coord"].cpu().numpy()
                gt_labels = labels.cpu().numpy()
                pred_labels = pred_classes.cpu().numpy()
                dino_feat = data_dict.get("dino_feat", None)
                if dino_feat is not None:
                    dino_feat = dino_feat.cpu().numpy()
                visualizer.show_gt_pred(
                    coords,
                    gt_labels=gt_labels,
                    pred_labels=pred_labels,
                    dino_feat=dino_feat,
                    save_prefix=f"epoch{epoch}_val",
                    use_open3d=False  # Set to True for interactive, else saves images
                )
                visualize_this_epoch = False
            # ---- END VISUALIZATION BLOCK ----

    val_loss = total_loss / total_pts
    val_acc = correct / total_pts
    logger.info(f"Validation [Epoch {epoch}] Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    
    return val_loss, val_acc


# ─── Main Training Script ────────────────────────────────────────
def main():
    logger.info("Starting training...")
    root_dir = Path(__file__).resolve().parent.parent / "data" / "hercules" /"processed" / "Mountain_01_Day"
    logger.info(f"Using dataset root: {root_dir}")

    batch_size = 2
    num_classes = 6           # set this to your number of classes
    num_epochs = 20
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_ds = MyPointCloudDataset(root_dir, split="train")
    logger.info(f"Training dataset size: {len(train_ds)}")
    val_ds = MyPointCloudDataset(root_dir, split="val")
    logger.info(f"Validation dataset size: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    sample = train_ds[0]
    in_channels = sample["feat"].shape[1]
    dino_channels = sample["dino_feat"].shape[1]
    model = SegmentationNet(in_channels=in_channels, dino_channels=dino_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_seg_model.pth")
            logger.info(f"Saved best model at epoch {epoch}")

if __name__ == "__main__":
    main()
