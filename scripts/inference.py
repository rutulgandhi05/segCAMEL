import torch
from pathlib import Path
from models.PTv3.model import PointTransformerV3

def save_predicted_features_to_pth(
    model_ckpt_path="data/checkpoints/best_model_hercules_exp_size100.pth",
    data_pth="data/hercules/Mountain_01_Day/processed_data/frame_00000.pth",
    input_mode="dino_only",
    output_key="distilled_feat",
    output_dir="data/checkpoints/hercules_exp_size100_inference",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # --- Load checkpoint ---
    ckpt = torch.load(model_ckpt_path, map_location=device)
    data = torch.load(data_pth)
    coord = data["coord"]
    feat = data["feat"]
    dino_feat = data["dino_feat"]

    if coord.dim() == 3 and coord.shape[0] == 1:
        coord = coord.squeeze(0)
        feat = feat.squeeze(0)
        dino_feat = dino_feat.squeeze(0)

    input_dim = dino_feat.shape[1] if input_mode == "dino_only" else feat.shape[1] + dino_feat.shape[1]
    dino_dim = dino_feat.shape[1]

    model = PointTransformerV3(in_channels=input_dim).to(device)
    proj_head = torch.nn.Linear(64, dino_dim).to(device)  # 64=backbone's output channel

    model.load_state_dict(ckpt["model"])
    proj_head.load_state_dict(ckpt["proj_head"])
    model.eval()
    proj_head.eval()
    
    # --- Prepare input ---
    coord = coord.to(device)
    feat = feat.to(device)

    if input_mode == "dino_only":
        input_feat = dino_feat.to(device)
    elif input_mode == "vri_dino":
        input_feat = torch.cat([feat, dino_feat], dim=1)
    elif input_mode == "coord_dino":
        input_feat = torch.cat([coord, dino_feat], dim=1)
    elif input_mode == "coord_vri_dino":
        input_feat = torch.cat([coord, feat, dino_feat], dim=1)
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    
    grid_size = float(data.get("grid_size", 0.05))
    coord_min = coord.min(0)[0]
    grid_coord = ((coord - coord_min) / grid_size).floor().int()

    for axis in range(3):
        n_unique = len(torch.unique(grid_coord[:, axis]))
        if n_unique < 2:
            grid_coord[:, axis] += torch.arange(grid_coord.shape[0]) % 2
    num_points = coord.shape[0]

    batch = torch.zeros(num_points, dtype=torch.long, device=device)
    offset = torch.tensor([num_points], dtype=torch.long, device=device)

    data_dict = {
        "coord": coord,
        "feat": input_feat,
        "grid_coord": grid_coord,
        "grid_size": grid_size,
        "offset": offset,
        "batch": batch,
    }

    # --- Inference ---
    with torch.no_grad():
        output = model(data_dict)
        pred = output.feat
        pred_proj = proj_head(pred)  # shape: (N, dino_dim)
        # Save as torch.Tensor (not numpy)
        distilled_feat = pred_proj.cpu()

    # --- Save as new .pth ---
    if output_dir is None:
        output_dir = str(Path(data_pth).parent)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(data_pth).stem
    pth_path = output_dir / f"{base_name}_distilled.pth"

    data[output_key] = distilled_feat

    torch.save(data, pth_path)
    print(f"Saved .pth with distilled features to {pth_path}")

if __name__ == "__main__":
    save_predicted_features_to_pth()
