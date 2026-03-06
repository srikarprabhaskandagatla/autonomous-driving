# train.py

import torch
import wandb
from ultralytics import YOLO
from pathlib import Path


def train():
    # Hard stop - if no GPU, fail immediately rather than silently CPU-train
    assert torch.cuda.is_available(), "CUDA not available. Check SLURM GPU allocation."
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f}GB")

    wandb.init(
        project="autonomous-driving-perception",
        name="yolov9c-bdd100k-run1",
        config={
            "model":          "yolov9c",
            "dataset":        "BDD100K-8K",
            "epochs":         100,
            "batch_size":     32,
            "image_size":     640,
            "optimizer":      "AdamW",
            "lr0":            0.001,
            "augmentation":   "mosaic+mixup+copy_paste",
        }
    )

    model = YOLO('weights/yolov9c.pt')

    results = model.train(
        data='data/processed/bdd100k.yaml',

        # Core hyperparameters
        epochs=100,
        imgsz=640,
        batch=32,           # A100 80GB handles 32 at 640px comfortably
        device=0,

        # Optimizer
        optimizer='AdamW',
        lr0=0.001,          # Initial LR
        lrf=0.01,           # Final LR = lr0 * lrf (cosine target)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,

        # Augmentation — each one justified
        mosaic=1.0,         # Combines 4 images; best single augmentation for mAP
        mixup=0.15,         # Blends 2 images; improves boundary detection
        copy_paste=0.3,     # Copies objects across images; helps rare classes
        degrees=10.0,       # Rotation; important for non-canonical driving angles
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,        # Hue shift for lighting variation
        hsv_s=0.7,
        hsv_v=0.4,

        # Runtime
        workers=8,
        amp=True,           # Automatic mixed precision — FP16 forward, FP32 grads

        # Logging
        project='runs/train',
        name='yolov9c_bdd100k',
        save_period=10,     # Checkpoint every 10 epochs
        val=True,
        plots=True,
        verbose=True,
    )

    best_map = results.results_dict.get('metrics/mAP50(B)', 0)
    print(f"\nFinal mAP@50: {best_map:.4f}")
    wandb.log({"final_mAP50": best_map})
    wandb.finish()


if __name__ == '__main__':
    train()