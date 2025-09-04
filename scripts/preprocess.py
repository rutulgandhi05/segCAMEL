import os
import gc
import time
import torch
from pathlib import Path
from tqdm import tqdm
from threading import Thread
from queue import Queue

from scripts.feature_extractor import Extractor
from scripts.dataloader import HerculesDataset
from torch.utils.data import DataLoader
from scripts.project_2d_to_3d import LidarToImageProjector
from utils.misc import _resolve_default_workers

GRID_SIZE = 0.10


def custom_collate(batch):
    collated = {}
    for key in batch[0]:
        vals = [sample[key] for sample in batch]
        if isinstance(vals[0], torch.Tensor) and all(v.shape == vals[0].shape for v in vals):
            collated[key] = torch.stack(vals, dim=0)
        else:
            collated[key] = vals
    return collated


def _disk_writer(save_queue: Queue):
    while True:
        item = save_queue.get()
        if item is None:
            save_queue.task_done()
            break
        out_path, payload = item
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, out_path)
        finally:
            save_queue.task_done()


def _safe_outpath(base_dir: Path, stem: int | str, suffix: str = ".pth") -> Path:
    p = base_dir / f"{stem}{suffix}"
    if not p.exists():
        return p
    k = 1
    while True:
        q = base_dir / f"{stem}_{k}{suffix}"
        if not q.exists():
            return q
        k += 1


def _safe_stem_from_rel(relpath) -> str:
    try:
        s = Path(str(relpath)).stem
        return s
    except Exception:
        return ""


@torch.no_grad()
def _visible_first_voxel_select(
    xyz: torch.Tensor,
    vis_mask: torch.Tensor,
    voxel_size: float,
    origin: torch.Tensor | None = None,
) -> torch.Tensor:
    if xyz.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    x = xyz.detach().cpu().to(torch.float32)
    v = vis_mask.detach().cpu().to(torch.bool).view(-1)
    if origin is None:
        origin = x.min(dim=0, keepdim=True).values
    else:
        origin = origin.view(1, 3).to(x.dtype)
    g = torch.floor((x - origin) / max(voxel_size, 1e-8)).to(torch.int64)
    max_vals = g.max(dim=0).values + 1
    max_vals = torch.clamp(max_vals, min=1)
    M_yz = max_vals[1] * max_vals[2]
    key = g[:, 0] * M_yz + g[:, 1] * max_vals[2] + g[:, 2]
    flag = (~v).to(torch.int64)
    composite = key * 2 + flag
    perm = torch.argsort(composite, stable=True)
    key_sorted = key[perm]
    first_mask = torch.ones_like(key_sorted, dtype=torch.bool)
    first_mask[1:] = key_sorted[1:] != key_sorted[:-1]
    first_sorted = perm[first_mask]
    sel = first_sorted.sort().values
    return sel.to(torch.long)


def _scale_intrinsics(K: torch.Tensor, raw_w: int, raw_h: int, img_w: int, img_h: int) -> torch.Tensor:
    if raw_w <= 0 or raw_h <= 0:
        return K
    sx = float(img_w) / float(raw_w)
    sy = float(img_h) / float(raw_h)
    Ks = K.clone()
    Ks[0, 0] *= sx
    Ks[1, 1] *= sy
    Ks[0, 2] *= sx
    Ks[1, 2] *= sy
    return Ks


@torch.no_grad()
def preprocess_and_save_hercules(
    root_dir: Path,
    save_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    workers: int = None,
    batch_size: int = 16,
    prefetch_factor: int = 4,
    frame_counter: int = 0,
    *,
    bilinear: bool = False,
    occlusion_eps: float = 0.05,
    save_uv: bool = False,
    border_margin_px: int = 2,
    range_aware_occl: bool = True,
    occl_eps_per_m: float = 0.001,
    occl_eps_min: float = 0.05,
    occl_eps_max: float = 0.20,
    export_train_vox: bool = True,
    train_voxel_size: float = 0.10,
):
    root_dir = Path(root_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"root_dir: {root_dir}   save_dir: {save_dir}")

    extractor = Extractor()

    if workers is None:
        workers = _resolve_default_workers()
    workers = max(1, int(workers))
    print(f"Using {workers} DataLoader workers for preprocessing...")

    dataset = HerculesDataset(
        root_dir,
        transform_factory=extractor.transform_factory,
        max_workers=workers,
        use_right_image=True,
        return_all_fields=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        collate_fn=custom_collate
    )

    print(f"[PREPROC] {root_dir.name}: {len(dataset)} frames | workers={workers} bs={batch_size}")
    print(f"[PREPROC] DINO: {extractor.dino_model}")

    save_queue: Queue = Queue(maxsize=workers * 2)
    writer_thread = Thread(target=_disk_writer, args=(save_queue,), daemon=True)
    writer_thread.start()

    t0 = time.time()

    for batch in tqdm(dataloader, desc="Processing frames", unit="batch"):
        pcs = batch["pointcloud"]
        imgs = batch["image_tensor"]
        intrs = batch["intrinsics"]
        extrs = batch["extrinsics"]
        input_sizes = batch["input_size"]
        fmap_sizes = batch["feature_map_size"]
        stamps = batch.get("timestamps", None)
        rels = batch.get("image_relpath", None)
        cams = batch.get("used_camera", None)
        orig_sizes = batch.get("orig_size", None)  # [PATCH] provided by dataset.py

        if isinstance(imgs, list):
            images_b = torch.stack(imgs, dim=0)
        else:
            images_b = imgs
        images_b = images_b.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            local_feats = extractor.extract_dino_features(images_b)
        lf_flat = local_feats.flat()

        B = len(pcs)
        assert lf_flat.tensor.shape[0] == B

        for b in range(B):
            pc = pcs[b].to(device)
            xyz = pc[:, :3].contiguous()
            feats = pc[:, 3:].contiguous() if pc.shape[1] > 3 else torch.zeros((pc.shape[0], 1), dtype=pc.dtype, device=device)

            K_raw = intrs[b].to(device).to(torch.float32)
            T = extrs[b].to(device).to(torch.float32)
            img_w, img_h = input_sizes[b]

            # [PATCH] scale K using orig_size from dataset.py
            raw_w, raw_h = 0, 0
            if isinstance(orig_sizes, list) and orig_sizes and isinstance(orig_sizes[b], (tuple, list)) and len(orig_sizes[b]) == 2:
                rw, rh = orig_sizes[b]
                raw_w, raw_h = int(rw), int(rh)
            K = _scale_intrinsics(K_raw, raw_w, raw_h, int(img_w), int(img_h))

            Hf = int(getattr(lf_flat, "h", fmap_sizes[b][1] if len(fmap_sizes[b]) == 2 else 0))
            Wf = int(getattr(lf_flat, "w", fmap_sizes[b][0] if len(fmap_sizes[b]) == 2 else 0))
            fm_size = (Wf, Hf)
            patch_feats_flat = lf_flat[b].tensor.squeeze(0).to(torch.float32)

            projector = LidarToImageProjector(
                intrinsic=K,
                extrinsic=T,  # NOTE: do not invert / scale T here
                image_size=(int(img_w), int(img_h)),
                feature_map_size=fm_size,
                patch_features=patch_feats_flat,
            )

            if range_aware_occl:
                r = torch.linalg.norm(xyz.to(torch.float32), dim=1)
                eps_vec = torch.clamp(occl_eps_per_m * r, min=occl_eps_min, max=occl_eps_max)
                try:
                    point_feats, vis_mask = projector.assign_features(
                        lidar_xyz=xyz.to(torch.float32),
                        bilinear=bilinear,
                        occlusion_eps=eps_vec,
                    )
                except TypeError:
                    eps_scalar = float(eps_vec.median().item())
                    point_feats, vis_mask = projector.assign_features(
                        lidar_xyz=xyz.to(torch.float32),
                        bilinear=bilinear,
                        occlusion_eps=eps_scalar,
                    )
            else:
                point_feats, vis_mask = projector.assign_features(
                    lidar_xyz=xyz.to(torch.float32),
                    bilinear=bilinear,
                    occlusion_eps=occlusion_eps,
                )

            # Debug stats every 50 frames
            if (frame_counter % 50) == 0:
                try:
                    uvs, z_cam, _ = projector.project_points(xyz.to(torch.float32))
                    u, v = uvs[:, 0], uvs[:, 1]
                    in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
                    z_pos = z_cam > 0
                    cov_img = float(in_img.float().mean().item()) * 100.0
                    cov_zpos = float(z_pos.float().mean().item()) * 100.0
                    ui = u.clamp(0, img_w - 1).round().to(torch.long)
                    vi = v.clamp(0, img_h - 1).round().to(torch.long)
                    pix = (vi * img_w + ui)[in_img & z_pos]
                    zbuf_pix = int(pix.unique().numel())
                    print(f"[PREPROC][dbg] in-image: {cov_img:.2f}% | z>0: {cov_zpos:.2f}% | zbuf_pix≈{zbuf_pix}")
                except Exception:
                    pass

            # Border margin suppression
            if border_margin_px and border_margin_px > 0:
                try:
                    uvs, _, _ = projector.project_points(xyz.to(torch.float32))
                    u = uvs[:, 0]; v = uvs[:, 1]
                    in_bounds = (u >= border_margin_px) & (u < (img_w - border_margin_px)) & \
                                (v >= border_margin_px) & (v < (img_h - border_margin_px))
                    vis_mask = vis_mask & in_bounds.to(vis_mask.device)
                    if point_feats is not None and point_feats.numel() == xyz.shape[0] * point_feats.shape[1]:
                        point_feats = point_feats.clone()
                        point_feats[~vis_mask] = 0.0
                except Exception:
                    pass

            if frame_counter % 50 == 0:
                cov = float(vis_mask.float().mean().item()) * 100.0
                print(f"[PREPROC] visible points: {cov:.2f}%")

            uv_payload = None
            if save_uv:
                try:
                    uvs, _, _ = projector.project_points(xyz.to(torch.float32))
                    uv_payload = uvs.detach().cpu()
                except Exception:
                    uv_payload = None

            lid_ts = int(stamps[b][0]) if stamps is not None else frame_counter
            stem = lid_ts

            out_path = _safe_outpath(save_dir, stem, suffix=".pth")
            payload = {
                "coord": xyz.float().cpu(),
                "feat": feats.float().cpu(),
                "dino_feat": point_feats.half().cpu(),
                "mask": vis_mask.cpu(),
                "grid_size": torch.tensor(float(train_voxel_size)),
                "intrinsics": K.cpu(),
                "extrinsics": T.cpu(),
                "input_size": (int(img_w), int(img_h)),
                "feature_map_size": (int(Wf), int(Hf)),
                "timestamps": stamps[b] if stamps is not None else None,
                "image_relpath": rels[b] if rels is not None else None,
                "used_camera": cams[b] if cams is not None else None,
                "orig_size": (int(raw_w), int(raw_h)) if (raw_w > 0 and raw_h > 0) else None,
            }
            if save_uv and uv_payload is not None:
                payload["proj_uv"] = uv_payload
            save_queue.put((out_path, payload))

            if export_train_vox:
                try:
                    sel = _visible_first_voxel_select(
                        xyz=xyz,
                        vis_mask=vis_mask,
                        voxel_size=float(train_voxel_size),
                        origin=None,
                    )
                    if sel.numel() > 0:
                        sel_dev = sel.to(xyz.device)
                        vox_coord = xyz.index_select(0, sel_dev).float().cpu()
                        vox_feat  = feats.index_select(0, sel_dev).float().cpu()
                        vox_mask  = vis_mask.index_select(0, sel_dev).cpu()
                        vox_dino  = point_feats.index_select(0, sel_dev).half().cpu()
                        trainvox_payload = {
                            "vox_coord": vox_coord,
                            "vox_feat":  vox_feat,
                            "vox_mask":  vox_mask,
                            "vox_dino":  vox_dino,
                            "grid_size": torch.tensor(float(train_voxel_size)),
                            "lidar_stem": str(stem),
                            "image_stem": _safe_stem_from_rel(rels[b]) if rels is not None else "",
                            "image_relpath": rels[b] if rels is not None else None,
                            "used_camera": cams[b] if cams is not None else None,
                            "timestamps": stamps[b] if stamps is not None else None,
                        }
                        trainvox_path = _safe_outpath(save_dir, stem, suffix="_trainvox.pth")
                        save_queue.put((trainvox_path, trainvox_payload))
                except Exception as e:
                    print(f"[WARN][trainvox] {stem}: {e}")

            frame_counter += 1

        del images_b, local_feats, lf_flat
        if device == "cuda":
            torch.cuda.empty_cache()

    save_queue.join()
    save_queue.put(None)
    writer_thread.join()

    try:
        gc.collect()
        del extractor
        del dataloader
        del dataset
        torch.cuda.empty_cache()
    except Exception:
        pass
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    dt = (time.time() - t0) / 60.0
    print(f"[PREPROC] Done {root_dir.name}: {frame_counter} total frames ({dt:.1f} min)")
    return frame_counter


if __name__ == "__main__":
    data_root = os.getenv("TMP_HERCULES_DATASET")
    save_dir = os.getenv("PREPROCESS_OUTPUT_DIR")
    if not data_root:
        raise EnvironmentError("TMP_HERCULES_DATASET must be set.")
    if not save_dir:
        raise EnvironmentError("PREPROCESS_OUTPUT_DIR environment variable not set.")

    data_root = Path(str(data_root))
    save_dir = Path(str(save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    folders_env = os.getenv("PREPROCESS_FOLDERS")
    if not folders_env:
        raise EnvironmentError("PREPROCESS_FOLDERS environment variable not set.")
    folders = [f.strip() for f in folders_env.split(",") if f.strip()]

    counter = 0
    for folder in folders:
        root_dir = data_root / folder
        print(f"Processing folder: {folder}")
        counter = preprocess_and_save_hercules(
            root_dir=root_dir,
            save_dir=save_dir,
            workers=None,
            batch_size=8,
            prefetch_factor=2,
            frame_counter=counter,
            bilinear=False,
            occlusion_eps=0.05,  
            save_uv=False,
            border_margin_px=2,
            range_aware_occl=True,
            occl_eps_per_m=0.0015,
            occl_eps_min=0.03,
            occl_eps_max=0.20,
            export_train_vox=True,
            train_voxel_size=0.10,
        )
    print(f"[PREPROC] Total frames processed: {counter}")
