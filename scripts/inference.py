import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from models.PTv3.model import PointTransformerV3

@torch.no_grad()
def infer_one_file(model, proj_head, file_path, device):
    sample = torch.load(file_path, map_location=device)

    coord = sample["coord"].to(device)
    feat = sample["feat"][:, :2].to(device)  # Only first 2 dims as in training

    # Match normalization used during training
    coord = (coord - coord.mean(dim=0)) / (coord.std(dim=0) + 1e-6)
    feat = (feat - feat.mean(dim=0)) / (feat.std(dim=0) + 1e-6)

    input_feat = torch.cat([coord, feat], dim=1)

    grid_size = sample.get("grid_size", 0.05)
    if not torch.is_tensor(grid_size):
        grid_size = torch.tensor(grid_size, device=coord.device)

    coord_min = coord.min(0)[0]
    grid_coord = ((coord - coord_min) / grid_size).floor().int()

    # Safety for unique grid coords (as in training)
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
    return coord.cpu().numpy(), proj_feat.cpu().numpy()

def run_inference(input_dir, output_dir, checkpoint_path, device="cuda"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model = PointTransformerV3(in_channels=5).to(device)  # 3 coords + 2 feats
    proj_head = torch.nn.Linear(64, 384).to(device)  # dino_dim=384 from training

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    proj_head.load_state_dict(checkpoint["proj_head"])
    model.eval()
    proj_head.eval()

    all_files = sorted(input_dir.glob("*.pth"))
    print(f"Found {len(all_files)} .pth files to process.")

    for file_path in tqdm(all_files, desc="Running inference"):
        try:
            coord, feat = infer_one_file(model, proj_head, file_path, device)
            out_path = output_dir / f"{file_path.stem}.npz"
            np.savez_compressed(out_path, coord=coord, feat=feat)
        except Exception as e:
            print(f"[ERROR] Failed on {file_path.name}: {e}")

if __name__ == "__main__":
    input_dir = Path(os.getenv("PREPROCESS_OUTPUT_DIR"))
    output_dir = Path(os.getenv("INFERENCE_OUTPUT_DIR"))
    checkpoint_path = Path(os.getenv("CHECKPOINT_PATH"))

    run_inference(input_dir, output_dir, checkpoint_path)
