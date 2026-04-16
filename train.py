import torch
import wandb
from ultralytics import YOLO
from pathlib import Path


def train():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Check SLURM GPU allocation.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f}GB")

    wandb.init(
        project="autonomous-driving-perception",
        name="yolov9c-bdd100k-run1",
        config={
            "model": "yolov9c",
            "dataset": "BDD100K-8K",
            "epochs": 100,
            "batch_size": 32,
            "image_size": 640,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "augmentation": "mosaic+mixup+copy_paste",
        }
    )

    model = YOLO('weights/yolov9c.pt')

    results = model.train(
        data='data/processed/bdd100k.yaml',
        epochs=100,
        imgsz=640,
        batch=32,
        device=0,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        workers=8,
        amp=True,
        project='runs/train',
        name='yolov9c_bdd100k',
        save_period=10,
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
