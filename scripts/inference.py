import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from scripts.train import MyPointCloudDataset, collate_fn, SegmentationNet
from utils.visualization import PointCloudVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained segCAMEL model")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to processed dataset directory (containing train/ or val/)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to run inference on",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth). Defaults to <data-root>/best_unsupervised_model.pth",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional directory to save visualizations and features",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=6,
        help="Number of clusters for unsupervised segmentation",
    )
    parser.add_argument(
        "--open3d",
        action="store_true",
        help="Use Open3D for interactive visualization instead of Matplotlib",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = Path(args.data_root)
    ckpt_path = Path(args.ckpt) if args.ckpt is not None else root_dir / "best_unsupervised_model.pth"

    dataset = MyPointCloudDataset(root_dir, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in {root_dir / args.split}")

    sample = dataset[0]
    model = SegmentationNet(
        in_channels=sample["feat"].shape[1],
        dino_channels=sample["dino_feat"].shape[1],
        num_classes=6,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    viz = PointCloudVisualizer()
    out_root = Path(args.out_dir) if args.out_dir else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, (data_dict, _) in enumerate(loader):
            for key in ("coord", "feat", "dino_feat", "offset"):
                data_dict[key] = data_dict[key].to(device)

            out_point = model.backbone(data_dict)
            feats = out_point.feat.cpu().numpy()

            # Unsupervised segmentation via KMeans on predicted features
            kmeans = KMeans(n_clusters=args.num_clusters, n_init=10)
            cluster_labels = kmeans.fit_predict(feats)

            coords_np = data_dict["coord"].cpu().numpy()

            if out_root is not None:
                torch.save(torch.from_numpy(feats), out_root / f"feats_{idx:04d}.pt")

                save_path = out_root / f"seg_{idx:04d}.png"
                viz.show_gt_pred(
                    coords_np,
                    pred_labels=cluster_labels,
                    save_prefix=str(save_path.with_suffix("")),
                    use_open3d=args.open3d,
                )
            else:
                viz.show_gt_pred(
                    coords_np,
                    pred_labels=cluster_labels,
                    use_open3d=args.open3d,
                )


if __name__ == "__main__":
    main()