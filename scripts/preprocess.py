import time
import torch
from pathlib import Path
from tqdm import tqdm
from threading import Thread
from queue import Queue

from scripts.feature_extractor import Extractor
from scripts.dataloader import HerculesDataset
from torch.utils.data import DataLoader
from utils.misc import setup_logger
from scripts.project_2d_to_3d import LidarToImageProjector

logger = setup_logger("preprocess")

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

def preprocess_and_save_hercules(
    root_dir: Path,
    save_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    workers: int = 8,
    batch_size: int = 8,
    prefetch_factor: int = 2,
    frame_counter: int = 0
):
    root_dir = Path(root_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"root_dir: {root_dir}   save_dir: {save_dir}")
    # --- extractor and dataset ---
    extractor = Extractor()
    dataset = HerculesDataset(root_dir, transform=extractor.transform_factory)
    dataset_len = len(dataset)
    print(f"Dataset length: {dataset_len}")
    # --- DataLoader with persistent workers & prefetching ---
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

    print(f"Using DINO model: {extractor.dino_model}")

    # --- set up background writer thread ---
    save_queue: Queue = Queue(maxsize=workers * 2)

    def writer():
        while True:
            item = save_queue.get()
            if item is None:
                break
            path, data = item
            torch.save(data, path)
            save_queue.task_done()

    writer_thread = Thread(target=writer, daemon=True)
    writer_thread.start()

    start_time = time.time()
    frame_counter = frame_counter

    # --- main loop ---
    for batch in tqdm(dataloader, desc="Processing frames", unit="batch"):
        # prepare images
        image_tensors = batch["image_tensor"]
        if isinstance(image_tensors, list):
            image_tensors = torch.stack(image_tensors, dim=0)
        image_tensors = image_tensors.to(device, non_blocking=True)

        # extract DINO features
        with torch.no_grad():
            features_batch = extractor.extract_dino_features(image_tensors)

        # now move features to device once
        # process each sample in the batch
        for i in range(image_tensors.shape[0]):
            # flat DINO patch features
            features = features_batch[i]
            dino_feat = features.flat().tensor.to(device)

            # lidar xyz + extra feats
            pc = batch["pointcloud"][i].to(device)
            xyz = pc[:, :3].to(device)
            feats = pc[:, 3:].to(device) if pc.shape[1] > 3 else torch.zeros((xyz.shape[0], 1), dtype=pc.dtype, device=device)

            # intrinsics / extrinsics
            intr = batch["intrinsics"][i].to(device)
            ext  = batch["extrinsics"][i].to(device)

            # project & assign
            projector = LidarToImageProjector(
                intrinsic=intr,
                extrinsic=ext,
                image_size=batch["input_size"][i],
                feature_map_size=batch["feature_map_size"][i],
                patch_features=dino_feat
            )
            assigned_feats, mask = projector.assign_features(xyz.float())

            # — now move everything to CPU once —
            coord_cpu     = xyz.float().cpu()
            feat_cpu      = feats.float().cpu()
            dino_cpu      = assigned_feats.float().cpu()
            mask_cpu      = mask.cpu()
            img_cpu       = image_tensors[i].cpu()
            grid_size_cpu = torch.tensor(0.05, dtype=torch.float32)

            # build save dict
            save_data = {
                "coord": coord_cpu,
                "feat": feat_cpu,
                "dino_feat": dino_cpu,
                "grid_size": grid_size_cpu,
                "mask": mask_cpu,
                "image_tensor": img_cpu,
            }

            # enqueue write
            save_path = save_dir / f"frame_{frame_counter:05d}.pth"
            save_queue.put((save_path, save_data))

            frame_counter += 1
            local_counter += 1

            # optional progress log
            if local_counter % 10 == 0 or local_counter == dataset_len:
                print(f"[Queued] {save_path}")

    # wait for all writes to finish
    save_queue.join()
    save_queue.put(None)
    writer_thread.join()

    total_time = time.time() - start_time
    print(f"Preprocessing completed in {total_time:.1f}s ({local_counter} frames)")

    return frame_counter

if __name__ == "__main__":
    import os
    
    data_root = os.getenv("HERCULES_DATASET")
    
    if not data_root:
        raise EnvironmentError("HERCULES_DATASET environment variable not set.")
    
    data_root = Path(str(data_root))
    folders = ["Mountain_01_Day", "Library_01_Day", "Sports_complex_01_Day"]
    save_dir = data_root / "processed_data"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    frame_counter = 0
    for folder in folders:
        root_dir = data_root / folder
        print(f"Processing folder: {folder}")
        print(f"Root directory: {root_dir}")
        print(f"Save directory: {save_dir}")

        frame_counter  = preprocess_and_save_hercules(
            root_dir=root_dir,
            save_dir=save_dir,
            workers=8,
            batch_size=16,     
            prefetch_factor=4,  # tune based on your I/O vs CPU/GPU balance
            frame_counter=frame_counter
        )

        print(f"Processed {frame_counter } frames in {folder}")

    print(f"Total frames processed: {frame_counter}")