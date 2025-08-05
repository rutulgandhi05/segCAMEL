import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import os
import warnings

from models.PTv3.model import PointTransformerV3

warnings.filterwarnings("ignore", category=UserWarning)  # e.g., TIMM layer deprecation warnings


class InferenceDataset(Dataset):
    def __init__(self, input_dir):
        self.files = sorted(list(Path(input_dir).glob("*.pth")))
        assert len(self.files), f"No .pth files found in {input_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        sample = torch.load(file_path)
        return {
            "file_path": str(file_path),  # convert to string to avoid collate issues
            "coord": sample["coord"],
            "feat": sample["feat"],
            "grid_size": sample.get("grid_size", 0.05),
        }


@torch.no_grad()
def infer_one_file(model, proj_head, sample, device="cuda"):
    coord = sample["coord"].to(device, non_blocking=True)
    feat = sample["feat"].to(device, non_blocking=True)

    # Normalize both coord and feat
    coord = (coord - coord.mean(dim=0)) / (coord.std(dim=0) + 1e-6)
    feat = (feat - feat.mean(dim=0)) / (feat.std(dim=0) + 1e-6)
    input_feat = torch.cat([coord, feat], dim=1)

    # Build sparse tensor metadata
    grid_size = float(sample["grid_size"])
    coord_min = coord.min(0)[0]
    grid_coord = ((coord - coord_min) / grid_size).floor().int()
    batch_tensor = torch.zeros(coord.shape[0], dtype=torch.long, device=device)
    offset = torch.tensor([coord.shape[0]], dtype=torch.long, device=device)

    data_dict = {
        "coord": coord,
        "feat": input_feat,
        "grid_coord": grid_coord,
        "grid_size": grid_size,
        "offset": offset,
        "batch": batch_tensor
    }

    output = model(data_dict)
    projected_feat = proj_head(output.feat)
    projected_feat = F.normalize(projected_feat, dim=1)

    return coord, projected_feat


def run_inference(input_dir, checkpoint_path, output_dir, batch_size=1, workers=4, device="cuda"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = torch.load(next(input_dir.glob("*.pth")))
    input_dim = sample["coord"].shape[1] + sample["feat"].shape[1]
    dino_dim = sample["dino_feat"].shape[1]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PointTransformerV3(in_channels=5).to(device)
    proj_head = torch.nn.Linear(64, dino_dim).to(device)

    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        print("State dict loading failed. Trying strict=False...")
        model.load_state_dict(checkpoint["model"], strict=False)

    proj_head.load_state_dict(checkpoint["proj_head"])
    model.eval()
    proj_head.eval()

    dataset = InferenceDataset(input_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda x: x[0],  # Fix Path object collation issue
    )

    for batch in tqdm(dataloader, desc="Running inference"):
        coord, projected_feat = infer_one_file(
            model,
            proj_head,
            {
                "coord": batch["coord"],
                "feat": batch["feat"],
                "grid_size": batch["grid_size"],
            },
            device=device
        )

        file_path = Path(batch["file_path"])
        save_path = output_dir / file_path.name
        torch.save({
            "coord": coord.cpu(),
            "projected_feat": projected_feat.cpu()
        }, save_path)

    print(f"Inference complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    INFERENCE_INPUT = Path(os.getenv("PREPROCESS_OUTPUT_DIR"))
    CHECKPOINT_PATH = Path(os.getenv("TRAIN_CHECKPOINTS")) / "best_model.pth"
    INFERENCE_OUTPUT = Path(os.getenv("INFERENCE_OUTPUT_DIR"))

    run_inference(
        input_dir=INFERENCE_INPUT,
        checkpoint_path=CHECKPOINT_PATH,
        output_dir=INFERENCE_OUTPUT,
        batch_size=1,         # Must be 1 due to current single-sample logic
        workers=8,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
