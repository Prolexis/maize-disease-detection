import os
import shutil
import random
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

# Ruta a tu dataset original (donde están todas las imágenes por clase)
original_dataset_dir = '/content/drive/MyDrive/maize-leaf-disease/Data'

# Nueva ruta para almacenar el dataset dividido
base_dir = '/content/drive/MyDrive/maize-leaf-disease/Data2'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Crear carpetas de salida
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# Parámetros de división científica: 70% train, 15% val, 15% test
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Procesar cada clase
classes = os.listdir(original_dataset_dir)

for class_name in classes:
    class_path = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.seed(42)  # Garantizar reproducibilidad en la partición
    random.shuffle(images)

    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Crear carpetas de clase
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Copiar imágenes
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

    print(f'Clase {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test')

print("✅ División científica en tres conjuntos (Train/Val/Test) completada.")