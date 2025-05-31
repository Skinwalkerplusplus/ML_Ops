# model = YOLO("yolov8s-seg.pt")  # Modelo seleccionado
# model.train(data="train/food_seg.yaml", epochs=5, verbose=False)

import glob
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
from collections import defaultdict

from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
    ShiftScaleRotate, GaussianBlur, ElasticTransform, GridDistortion
)

from ultralytics import YOLO
import wandb

import yaml

with open("train/yolo_sweep.yaml", "r") as f:
    config = yaml.safe_load(f)
print(config)

wandb.init(
    project="yolov8-training",
    config={
        "lr0": 0.01,
        "batch": 16,
        "mask_ratio": 4,
        "overlap_mask": True,
    }
)

sweep_config = {
    "method": "bayes",
    "metric": {"name": "metrics/mAP50-95(M)", "goal": "maximize"},
    "parameters": {
        "lr0": {"min": 0.0001, "max": 0.01},
        "mask_ratio": {"values": [2, 4, 8]},
        "overlap_mask": {"values": [True, False]},
        "batch": {"values": [8, 16, 32]}
    }
}

def train():
    model = YOLO("yolov8s-seg.pt")  # Load segmentation model
    model.train(
        data="train/food_seg.yaml",
        epochs=5,
        lr0=wandb.config.lr0,
        mask_ratio=wandb.config.mask_ratio,
        overlap_mask=wandb.config.overlap_mask,
        batch=wandb.config.batch,
        project="yolov8-training",
    )

sweep_id = wandb.sweep(sweep=sweep_config, project="yolov8-training")
wandb.agent(sweep_id, function=train, count=20)

metrics = model.val()

# Resultados
print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5-0.95: {metrics.box.map}")
print(f"Mask mAP: {metrics.seg.map}")  # Metrica especifica para la segmentacion