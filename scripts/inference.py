import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from models.PTv3.model import PointTransformerV3


@torch.no_grad()
def infer_one_file(model, proj_head, file_path, device):
    sample = torch.load(file_path, map_location=device)

    raw_coord = sample["coord"].to(device)
    feat = sample["feat"].to(device)  # ✅ use all available features

    # Normalize input
    coord = (raw_coord - raw_coord.mean(dim=0)) / (raw_coord.std(dim=0) + 1e-6)
    feat = (feat - feat.mean(dim=0)) / (feat.std(dim=0) + 1e-6)
    input_feat = torch.cat([coord, feat], dim=1)

    input_dim = input_feat.shape[1]
    print(f"[INFO] Inferred input_feat shape: {input_feat.shape}, input_dim = {input_dim}")

    # Grid-related processing
    grid_size = sample.get("grid_size", 0.05)
    if not torch.is_tensor(grid_size):
        grid_size = torch.tensor(grid_size, device=coord.device)

    coord_min = coord.min(0)[0]
    grid_coord = ((coord - coord_min) / grid_size).floor().int()
    for axis in range(3):
        if len(torch.unique(grid_coord[:, axis])) < 2:
            grid_coord[:, axis] += torch.arange(grid_coord.shape[0], device=grid_coord.device) % 2

    offset = torch.tensor([coord.shape[0]], dtype=torch.long, device=device)
    batch = torch.zeros(coord.shape[0], dtype=torch.long, device=device)

    data_dict = {
        "coord": coord,
        "feat": input_feat,
        "grid_coord": grid_coord,
        "grid_size": grid_size,
        "offset": offset,
        "batch": batch,
    }

    output = model(data_dict)
    proj_feat = proj_head(output.feat)

    return {
        "coord": raw_coord.cpu(),
        "proj_feat": proj_feat.cpu(),
        "input_feat": input_feat.cpu(),
        "dino_feat": sample["dino_feat"].cpu(),
        "mask": sample.get("mask", torch.ones_like(feat[:, 0], dtype=torch.bool)).cpu(),
        "grid_coord": grid_coord.cpu(),
    }


def run_inference(input_dir, output_pth_dir, checkpoint_path, device="cuda"):
    input_dir = Path(input_dir)
    output_pth_dir = Path(output_pth_dir)
    output_pth_dir.mkdir(exist_ok=True, parents=True)

    # Must match training input_dim
    model = PointTransformerV3(in_channels=6).to(device)
    proj_head = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 384)
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    proj_head.load_state_dict(checkpoint["proj_head"])
    model.eval()
    proj_head.eval()

    all_files = sorted(input_dir.glob("*.pth"))
    print(f"Found {len(all_files)} preprocessed .pth files.")

    for file_path in tqdm(all_files, desc="Running inference"):
        try:
            data = infer_one_file(model, proj_head, file_path, device)
            out_path = output_pth_dir / f"{file_path.stem}.pth"
            torch.save(data, out_path)
        except Exception as e:
            print(f"[ERROR] Failed on {file_path.name}: {e}")
            
if __name__ == "__main__":
    input_dir = Path(os.getenv("PREPROCESS_OUTPUT_DIR"))
    output_dir = Path(os.getenv("INFERENCE_OUTPUT_DIR"))
    checkpoint_path = Path(os.getenv("TRAIN_CHECKPOINTS")) / "best_model.pth"
    run_inference(input_dir, output_dir, checkpoint_path)
