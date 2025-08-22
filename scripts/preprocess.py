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

GRID_SIZE = 0.05


def custom_collate(batch):
    # Custom collate for variable‐size fields (pointcloud etc.)
    collated = {}
    for key in batch[0]:
        vals = [sample[key] for sample in batch]
        if isinstance(vals[0], torch.Tensor) and all(v.shape == vals[0].shape for v in vals):
            collated[key] = torch.stack(vals, dim=0)
        else:
            collated[key] = vals
    return collated


def _disk_writer(save_queue: Queue):
    """Simple writer thread."""
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

def _safe_outpath(base_dir: Path, stem: int | str) -> Path:
    p = base_dir / f"{stem}.pth"
    if not p.exists():
        return p
    k = 1
    while True:
        q = base_dir / f"{stem}_{k}.pth"
        if not q.exists():
            return q
        k += 1

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
    # Projector knobs (safe defaults):
    bilinear: bool = False,          # bilinear sampling on the token lattice
    occlusion_eps: float = 0.05,     # >0 enables front-most filtering per token (meters in camera Z)
    save_uv: bool = False           # save per-point projected (u,v) for debugging
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

    # --- set up background writer thread ---
    save_queue: Queue = Queue(maxsize=workers * 2)
    writer_thread = Thread(target=_disk_writer, args=(save_queue,), daemon=True)
    writer_thread.start()

    t0 = time.time()

    # --- main loop ---
    for batch in tqdm(dataloader, desc="Processing frames", unit="batch"):
        pcs = batch["pointcloud"]                  # list[Tensor]
        imgs = batch["image_tensor"]               # list[Tensor] or Tensor
        intrs = batch["intrinsics"]                # list[Tensor]
        extrs = batch["extrinsics"]                # list[Tensor]
        input_sizes = batch["input_size"]          # list[(W,H)]
        fmap_sizes = batch["feature_map_size"]     # list[(Wf,Hf)]
        stamps = batch.get("timestamps", None)     # optional
        rels = batch.get("image_relpath", None)    # optional
        cams = batch.get("used_camera", None)      # optional


        if isinstance(imgs, list):
            images_b = torch.stack(imgs, dim=0)
        else:
            images_b = imgs
        images_b = images_b.to(device, non_blocking=True)

        # extract DINO features
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            local_feats = extractor.extract_dino_features(images_b)
        lf_flat = local_feats.flat()

        B = len(pcs)
        assert lf_flat.tensor.shape[0] == B, "Batch size mismatch between images and extracted features."


        # process each sample in the batch
        for b in range(B):
            pc = pcs[b].to(device)                      # (N, C)
            xyz = pc[:, :3].contiguous()
            feats = pc[:, 3:].contiguous() if pc.shape[1] > 3 else torch.zeros((pc.shape[0], 1), dtype=pc.dtype, device=device)

            K = intrs[b].to(device).to(torch.float32)
            T = extrs[b].to(device).to(torch.float32)
            img_w, img_h = input_sizes[b]                  # (W,H) after TF resize

            # Derive feature-map size directly from LocalFeatures metadata to avoid W/H swap:
            # LocalFeatures stores (h,w)
            Hf = int(getattr(lf_flat, "h", fmap_sizes[b][1] if len(fmap_sizes[b]) == 2 else 0))
            Wf = int(getattr(lf_flat, "w", fmap_sizes[b][0] if len(fmap_sizes[b]) == 2 else 0))
            fm_size = (Wf, Hf)

            # Flattened per-sample patch features (Hf*Wf, D) -> float32
            patch_feats_flat = lf_flat[b].tensor.squeeze(0).to(torch.float32)  # (Hf*Wf, D)


            # project & assign
            projector = LidarToImageProjector(
                intrinsic=K,
                extrinsic=T,
                image_size=(img_w, img_h),
                feature_map_size=fm_size,
                patch_features=patch_feats_flat,  # accepts (Hf*Wf, D) or (Hf,Wf,D)
            )
            point_feats, vis_mask = projector.assign_features(
                lidar_xyz=xyz.to(torch.float32),
                bilinear=bilinear,
                occlusion_eps=occlusion_eps,
            )

            if frame_counter % 50 == 0:
                cov = float(vis_mask.float().mean().item()) * 100.0
                print(f"[PREPROC] visible points: {cov:.2f}%")

            # Optional: save projected (u,v) for debugging/overlays
            uv_payload = None
            if save_uv:
                uvs, _, _ = projector.project_points(xyz.to(torch.float32))
                uv_payload = uvs.detach().cpu()



            lid_ts = int(stamps[b][0]) if stamps is not None else frame_counter
            out_path = _safe_outpath(save_dir, lid_ts)

            payload = {
                "coord": xyz.float().cpu(),               # (N,3)
                "feat": feats.float().cpu(),              # (N,F)
                "dino_feat": point_feats.half().cpu(),    # (N,D) save fp16 to cut storage
                "mask": vis_mask.cpu(),                   # (N,) True where Z>0 & in-image (& occlusion-kept)
                "grid_size": torch.tensor(GRID_SIZE),     # scalar
                # light metadata for future debugging
                "intrinsics": K.cpu(),
                "extrinsics": T.cpu(),
                "input_size": (int(img_w), int(img_h)),
                "feature_map_size": (int(Wf), int(Hf)),
                "timestamps": stamps[b] if stamps is not None else None,
                "image_relpath": rels[b] if rels is not None else None,
                "used_camera": cams[b] if cams is not None else None,
            }

            if save_uv and uv_payload is not None:
                payload["proj_uv"] = uv_payload  # (N,2)

            assert "image_tensor" not in payload
            save_queue.put((out_path, payload))
            frame_counter += 1


        del images_b, local_feats, lf_flat
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- end of main loop ---
    # wait for all writes to finish
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
    data_root = os.getenv("HERCULES_DATASET")
    save_dir = os.getenv("PREPROCESS_OUTPUT_DIR")
    pipeline_mode = os.getenv("PIPELINE_MODE")  # default to "inference"
    if not data_root:
        raise EnvironmentError("HERCULES_DATASET environment variable not set.")
    if not save_dir:
        raise EnvironmentError("PREPROCESS_OUTPUT_DIR environment variable not set.")

    data_root = Path(str(data_root))
    save_dir = Path(str(save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    if pipeline_mode == "inference":
        folders = [ "Sports_complex_03_Day"] #inference only
    elif pipeline_mode == "preprocess":
        folders = [ "Library_01_Day", "Sports_complex_01_Day", "Mountain_01_Day"]

    counter = 0
    for folder in folders:
        root_dir = data_root / folder
        print(f"Processing folder: {folder}")

        counter   = preprocess_and_save_hercules(
            root_dir=root_dir,
            save_dir=save_dir,
            workers=None,
            batch_size=8,     
            prefetch_factor=2,  # tune based on your I/O vs CPU/GPU balance
            frame_counter=counter,
            bilinear=False,
            occlusion_eps=0.05,     # set e.g. 0.05–0.10 to enable front-most filtering per token
            save_uv=False, 
        )

    print(f"[PREPROC] Total frames processed: {counter}")