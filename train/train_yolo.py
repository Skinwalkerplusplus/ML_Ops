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

# # Conseguimos todos los labels
# label_files = glob.glob("/content/food_dataset/train/labels/*.txt")
# class_ids = []

# for file in label_files:
#     with open(file, 'r') as f:
#         for line in f:
#             class_id = int(line.split()[0])
#             class_ids.append(class_id)

# unique_classes = np.unique(class_ids)
# print("Class IDs in dataset:", unique_classes)

wandb.init(project="yolov8-training", entity="ziroldjr-upm")

def rgb_mask_to_yolo_txt(mask_path, output_txt_path):
    # RGB
    mask = cv2.imread(str(mask_path))
    height, width = mask.shape[:2]

    # Las clases estan en la R
    class_mask = mask[:, :, 2]  # OpenCv lee como BGR, asi que la 2

    # Contornos para cada clase
    with open(output_txt_path, 'w') as f:
        for class_id in np.unique(class_mask):
            if class_id == 0:  # No queremos el fondo
                continue
            # Mascara binaria
            binary_mask = (class_mask == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 10:  # Saltamos las regiones pequenas
                    # Normalizamos coordenadas
                    polygon = cnt.flatten().astype(float)
                    polygon[::2] /= width   # Coordenadas x
                    polygon[1::2] /= height  # Coordenadas y
                    # Escribimos: <class_id> <x1> <y1> <x2> <y2> ...
                    f.write(f"{class_id} " + " ".join(map(str, polygon)) + "\n")

# Procesamos las imagenes
mask_dir = Path("UECFoodPIXCOMPLETE/mask_train")
test_dir = Path("UECFoodPIXCOMPLETE/mask_test")
output_label_dir = Path("UECFoodPIXCOMPLETE/train/labels")  # Output label
output_label_dir.mkdir(parents=True, exist_ok=True)
output_test_dir = Path("UECFoodPIXCOMPLETE/test/labels")  # Output test
output_test_dir.mkdir(parents=True, exist_ok=True)

for png_path in mask_dir.glob("*.png"):
    txt_path = output_label_dir / f"{png_path.stem}.txt"
    rgb_mask_to_yolo_txt(png_path, txt_path)

for png_path in test_dir.glob("*.png"):
    txt_path = output_test_dir / f"{png_path.stem}.txt"
    rgb_mask_to_yolo_txt(png_path, txt_path)



image_dir = Path("UECFoodPIXCOMPLETE/train/images")
label_dir = Path("UECFoodPIXCOMPLETE/train/labels")
mask_dir = Path("UECFoodPIXCOMPLETE/mask_train")

# Cuentas de clase
class_counts = defaultdict(int)
for label_file in label_dir.glob("*.txt"):
    with open(label_file, 'r') as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

# Aumentacion para pares
def get_augmentations():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.3),
        Rotate(limit=30, p=0.5),
        RandomBrightnessContrast(p=0.4),
        GaussianBlur(blur_limit=(3, 5), p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
    ], additional_targets={'mask': 'mask'})

min_samples = 200  # Minimo por clase
max_aug_per_image = 8  # Maximo por imagen

# Mapping de clases
class_to_images = defaultdict(list)
for image_file in image_dir.glob("*.jpg"):
    label_file = label_dir / f"{image_file.stem}.txt"
    mask_file = mask_dir / f"{image_file.stem}.png"
    if mask_file.exists():
        with open(label_file, 'r') as f:
            classes_in_image = [int(line.split()[0]) for line in f]
        for class_id in set(classes_in_image):
            class_to_images[class_id].append(image_file)

# Aumentacion
augmented_count = 0
for class_id, count in class_counts.items():
    if count < min_samples:
        needed = min_samples - count
        available_images = class_to_images[class_id]
        num_images = len(available_images)

        if num_images == 0:
            continue

        augs_per_image = min(max_aug_per_image, max(1, needed // num_images + 1))

        print(f"Augmenting class {class_id} (has {count} samples, needs {needed} more)")

        for image_file in available_images:
            if needed <= 0:
                break

            label_file = label_dir / f"{image_file.stem}.txt"
            mask_file = mask_dir / f"{image_file.stem}.png"

            # Cargamos componentes
            img = cv2.imread(str(image_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

            mask = np.squeeze(mask)

            if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
                mask = mask.T

            # Leemos labels originales
            with open(label_file, 'r') as f:
                original_labels = [line.strip() for line in f]

            # Aumentaciones
            for i in range(augs_per_image):
                if needed <= 0:
                    break

                # Mantenemos la relacion imagen-mascara
                augmented = get_augmentations()(
                    image=img,
                    mask=mask
                )


                new_stem = f"{image_file.stem}_aug{augmented_count}"

                cv2.imwrite(
                    str(image_dir / f"{new_stem}.jpg"),
                    cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                )

                cv2.imwrite(
                    str(mask_dir / f"{new_stem}.png"),
                    augmented['mask']
                )

                # Copiamos labels
                with open(label_dir / f"{new_stem}.txt", 'w') as f:
                    f.write("\n".join(original_labels))

                augmented_count += 1
                needed -= 1
                class_counts[class_id] += 1

print(f"Created {augmented_count} new augmented samples")
print("Updated class counts:")
for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id}: {count} objects")



image_dir = "UECFoodPIXCOMPLETE/train/images"
label_dir = "UECFoodPIXCOMPLETE/train/labels"
val_ratio = 0.1  # Porcentaje para la validacion

# Conseguimos todos los directorios
image_files = [f.split(".")[0] for f in os.listdir(image_dir) if f.endswith(".jpg")]
train_files, val_files = train_test_split(image_files, test_size=val_ratio, random_state=42)

# Creamos carpetas de validacion
os.makedirs("UECFoodPIXCOMPLETE/val/images", exist_ok=True)
os.makedirs("UECFoodPIXCOMPLETE/val/labels", exist_ok=True)

# Movemos a carpetas de validacion
for file in val_files:
    # Imagenes
    os.rename(
        f"{image_dir}/{file}.jpg",
        f"UECFoodPIXCOMPLETE/val/images/{file}.jpg"
    )
    # Labels
    if os.path.exists(f"{label_dir}/{file}.txt"):
        os.rename(
            f"{label_dir}/{file}.txt",
            f"UECFoodPIXCOMPLETE/val/labels/{file}.txt"
        )

print(f"Train: {len(train_files)} images | Val: {len(val_files)} images")

# model = YOLO("yolov8s-seg.pt")  # Modelo seleccionado
# model.train(data="train/food_seg.yaml", epochs=5, verbose=False)

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

sweep_id = wandb.sweep(sweep="train/yolo_sweep.yaml", project="yolov8-training")
wandb.agent(sweep_id, function=train, count=20)

metrics = model.val()

# Resultados
print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5-0.95: {metrics.box.map}")
print(f"Mask mAP: {metrics.seg.map}")  # Metrica especifica para la segmentacion