import os
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.PTv3.model_ import PointTransformerV3

from scripts.train_segmentation import (
    PointCloudDataset,
    collate_for_ptv3,
    _load_state
)

# Inference
@torch.no_grad()
def run_inference(
    data_dir: Path,
    checkpoint_path: Path,
    out_dir: Path,
    *,
    voxel_size: float = 0.10,
    multiscale_voxel: bool = False,
    batch_size: int = 2,
    workers: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Runs PTv3 forward (image-free) with the exact preprocessing used in training and
    writes per-sample outputs: ptv3_feat (64-D), coord (normalized), grid_coord, mask, grid_size,
    and proj_dino_pred if the checkpoint contains the projection head.
    """
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"[inference] Data: {data_dir}")
    print(f"[inference] Checkpoint: {checkpoint_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PointCloudDataset(data_dir, voxel_size=voxel_size)
    collate_fn = partial(collate_for_ptv3, multiscale_voxel=multiscale_voxel)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        collate_fn=collate_fn,
        multiprocessing_context="spawn",
    )

    # Probe dims
    probe = dataset[0]
    input_dim = probe["coord"].shape[1] + probe["feat"].shape[1]
   
    # Build model exactly like training
    model = PointTransformerV3(
        in_channels=input_dim,
        enable_flash=False,
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_patch_size=(256, 256, 256, 256),
        enable_rpe=False,
        upcast_attention=True,
        upcast_softmax=True,
    ).to(device).eval()

    # Load checkpoint (model + head if present)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model" in ckpt:
        _load_state(model, ckpt["model"])
    else:
        _load_state(model, ckpt)

    for batch in tqdm(loader, desc="Inference"):

        # Move tensors
        coord = batch["coord"].to(device, non_blocking=True)
        feat  = batch["feat"].to(device, non_blocking=True).float()
        grid_coord = batch["grid_coord"].to(device, non_blocking=True)
        batch_tensor = batch["batch"].to(device, non_blocking=True)
        offset = batch["offset"].to(device, non_blocking=True)
        grid_size_val = batch["grid_size"].mean().item()

        # Same normalization as training (+ clamp to tame outliers)
        coord = (coord - coord.mean(dim=0)) / (coord.std(dim=0) + 1e-6)
        feat  = (feat  -  feat.mean(dim=0)) / ( feat.std(dim=0) + 1e-6)
        input_feat = torch.cat([coord, feat], dim=1).clamp_(-8.0, 8.0)

        data_dict = {
            "coord": coord,
            "feat": input_feat,
            "grid_coord": grid_coord,
            "grid_size": grid_size_val,
            "offset": offset,
            "batch": batch_tensor,
        }

        output = model(data_dict)
        feats = torch.nan_to_num(output.feat, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-20, 20)
   
        # Split and save per sample
        file_stems = batch["file_stems"]
        B = len(file_stems)
        for b in range(B):
            idx_b = (batch_tensor == b).nonzero(as_tuple=False).squeeze(1)
            payload = {
                "file_stem": file_stems[b],
                "ptv3_feat": feats.index_select(0, idx_b).detach().cpu(),   # (Nb,64)
                "coord":      coord.index_select(0, idx_b).detach().cpu(),  # normalized coords used in model
                "grid_coord": grid_coord.index_select(0, idx_b).detach().cpu(),
                "mask":       batch["mask"].index_select(0, idx_b.cpu()).cpu() if batch["mask"].numel() else torch.empty((0,), dtype=torch.bool),
                "grid_size":  torch.tensor(grid_size_val),
            }
            
            save_path = out_dir / f"{file_stems[b]}_inference.pth"
            torch.save(payload, save_path)

        # Cleanup
        del coord, feat, grid_coord, batch_tensor, offset, input_feat, output, feats

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[inference] Done.")


if __name__ == "__main__":
  
    DATA_DIR = Path(os.getenv("PREPROCESS_OUTPUT_DIR"))
    CHECKPOINTS_DIR = Path(os.getenv("TRAIN_CHECKPOINTS"))
    CHECKPOINT_PATH = CHECKPOINTS_DIR / "latest_checkpoint.pth"
    OUT_DIR = Path(os.getenv("INFERENCE_OUTPUT_DIR"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_inference(
        data_dir=DATA_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        out_dir=OUT_DIR,
        voxel_size=0.10,
        multiscale_voxel=False,
        batch_size=4,
        workers=12,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_data_parallel=False,
    )
