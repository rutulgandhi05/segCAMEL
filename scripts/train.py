from pathlib import Path
from datetime import datetime
from utils.misc import setup_logger
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.PTv3.model import PointTransformerV3


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: Path):
        self.files = sorted(list(Path(root_dir).glob("*.pth")))
        assert len(self.files), f"No .pth files found in {root_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])

def distillation_loss(pred_feat, target_feat):
    pred = F.normalize(pred_feat, dim=1)
    target = F.normalize(target_feat, dim=1)
    return 1 - (pred * target).sum(dim=1).mean()

def train(
    data_dir=Path("data/hercules/Mountain_01_Day/processed_data"),
    epochs=30,
    batch_size=1,
    lr=1e-4,
    save_path=Path("data/checkpoints/best_model_hercules_exp_size100.pth"),
    device="cuda" if torch.cuda.is_available() else "cpu",
    input_mode="vri_dino",   # Options: 'dino_only', 'vri_dino', 'coord_dino', 'coord_vri_dino'
):
    logger = setup_logger()
    logger.info(f"Starting training on device: {device}")
    logger.info(f"Training data: {data_dir}")
    logger.info(f"Input mode: {input_mode}")

    dataset = PointCloudDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Loaded {len(dataset)} preprocessed samples")

    # -- Dynamically infer in_channels from first sample --
    sample = dataset[0]
    coord = sample["coord"]
    feat = sample["feat"]
    dino_feat = sample["dino_feat"]

    if input_mode == "dino_only":
        input_dim = dino_feat.shape[1]
    elif input_mode == "vri_dino":
        input_dim = feat.shape[1] + dino_feat.shape[1]
    elif input_mode == "coord_dino":
        input_dim = coord.shape[1] + dino_feat.shape[1]
    elif input_mode == "coord_vri_dino":
        input_dim = coord.shape[1] + feat.shape[1] + dino_feat.shape[1]
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    logger.info(f"Using input_dim={input_dim}")
    model = PointTransformerV3(in_channels=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        logger.info(f"\n Epoch {epoch + 1}/{epochs}")

        for sample in tqdm(dataloader):

            coord = sample["coord"].to(device)
            feat = sample["feat"].to(device)
            dino_feat = sample["dino_feat"].to(device)
            grid_size = sample["grid_size"].to(device)

            # -- Select features based on input_mode --
            if input_mode == "dino_only":
                input_feat = dino_feat
            elif input_mode == "vri_dino":
                input_feat = torch.cat([feat, dino_feat], dim=1)
            elif input_mode == "coord_dino":
                input_feat = torch.cat([coord, dino_feat], dim=1)
            elif input_mode == "coord_vri_dino":
                input_feat = torch.cat([coord, feat, dino_feat], dim=1)
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}")

            offset = torch.tensor([coord.shape[0]], device=device)
        
            data_dict = {
                "coord": coord,
                "feat": input_feat,
                "grid_size": grid_size,
                "offset": offset,
            }

            logger.info("grid_size used:", data_dict["grid_size"])
            optimizer.zero_grad()
            pred = model(data_dict).feat
            loss = distillation_loss(pred, dino_feat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Avg Loss = {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved to {save_path} (loss = {avg_loss:.6f})")

    logger.info("Training complete.")

if __name__ == "__main__":
    # Set ablation mode here: 'dino_only', 'vri_dino', 'coord_dino', 'coord_vri_dino'
    train(input_mode="dino_only")
    # train(input_mode="vri_dino")
    # train(input_mode="coord_dino")
    # train(input_mode="coord_vri_dino")
