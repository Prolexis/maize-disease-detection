import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_pre
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from google.colab import drive

# ------------------------
# 📂 Rutas y configuración general
# ------------------------
drive.mount('/content/drive')
base_path = "/content/drive/MyDrive/maize-leaf-disease"
train_dir = f"{base_path}/Data2/train"
val_dir = f"{base_path}/Data2/val"
test_dir = f"{base_path}/Data2/test"
report_path = f"{base_path}/Reports"
model_path = f"{base_path}/Models"
os.makedirs(report_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# ------------------------
# ⚙️ Carga de datos
# ------------------------
img_size = 224  # Aumentado de 128 a 224 para conservar morfología foliar de las lesiones (Norma Q1)
batch_size = 32

# Cargar nombres de clases
temp_gen = ImageDataGenerator(preprocessing_function=mobilenet_pre)
temp_data = temp_gen.flow_from_directory(train_dir, target_size=(img_size, img_size))
class_names = list(temp_data.class_indices.keys())
num_classes = len(class_names)

# ------------------------
# 🔄 Generadores de datos
# ------------------------
def get_data_generators(preprocess_fn):
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=25,
        zoom_range=0.25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,  # Útil para patología foliar
        brightness_range=(0.8, 1.2),
        fill_mode='nearest'
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_data = train_gen.flow_from_directory(
        train_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical', shuffle=True
    )
    val_data = val_gen.flow_from_directory(
        val_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    test_data = test_gen.flow_from_directory(
        test_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical', shuffle=False
    )

    return train_data, val_data, test_data

# ------------------------
# ⚖️ Construcción de modelos
# ------------------------
def build_model(base_model_fn, preprocess_fn, name):
    base_model = base_model_fn(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')
    
    # Congelar capas inferiores inicialmente para estabilizar transfer learning
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output, name=name)
    return model

# Helper para calcular especificidad multiclase
def calculate_specificity(cm, num_classes):
    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    return specificities

# ------------------------
# 🔢 Entrenamiento y evaluación
# ------------------------
def train_and_evaluate(model_fn, preprocess_fn, model_name):
    print(f"\n======== Entrenando {model_name} ========")
    model = build_model(model_fn, preprocess_fn, model_name)
    
    train_data, val_data, test_data = get_data_generators(preprocess_fn)
    
    # Scheduler de tasa de aprendizaje: Decaimiento Coseno (CosineDecay)
    total_steps = len(train_data) * 20  # 20 Épocas máximas
    lr_schedule = CosineDecay(initial_learning_rate=1e-4, decay_steps=total_steps, alpha=0.01)
    
    # Optimizador desacoplado AdamW con decaimiento de peso para prevenir overfitting
    opt = AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=3, factor=0.3, verbose=1, monitor='val_loss')
    ]

    start = time.time()
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        callbacks=callbacks
    )
    duration = time.time() - start
    print(f"⏱️ Tiempo de entrenamiento: {duration:.2f} segundos")

    # Guardar modelo en formato moderno .keras (Recomendación científica)
    model.save(os.path.join(model_path, f"{model_name}.keras"))

    # Graficar curvas de aprendizaje y pérdida
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Precisión - {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Exactitud')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Pérdida - {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(report_path, f"{model_name}_learning_curves.png"))
    plt.close()

    # Evaluación rigurosa en el Test Set independiente (Holdout set)
    print(f"\n======== Evaluando en Conjunto de Prueba Independiente (Test Set) ========")
    test_data.reset()
    start_infer = time.time()
    preds = model.predict(test_data)
    infer_duration = (time.time() - start_infer) / len(test_data.filenames) * 1000 # ms por imagen
    
    y_pred = np.argmax(preds, axis=1)
    y_true = test_data.classes

    # Calcular matriz de confusión y métricas avanzadas (Sensibilidad/Especificidad)
    conf_matrix = confusion_matrix(y_true, y_pred)
    specs = calculate_specificity(conf_matrix, num_classes)
    
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Calcular curvas Precision-Recall y AUC multiclase
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    pr_auc_values = []
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], preds[:, i])
        pr_auc = auc(recall, precision)
        pr_auc_values.append(pr_auc)
        plt.plot(recall, precision, label=f'{class_names[i]} (PR-AUC = {pr_auc:.3f})')
    plt.xlabel('Sensibilidad (Recall)')
    plt.ylabel('Precisión')
    plt.title(f'Curvas Precision-Recall - {model_name}')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(report_path, f"{model_name}_pr_curves.png"))
    plt.close()

    # Escribir reporte exhaustivo formal Q1 en txt
    report_file_path = os.path.join(report_path, f"{model_name}_report.txt")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(f"REPORTE EVALUACIÓN EXHAUSTIVA DE DESEMPEÑO: {model_name}\n")
        f.write("="*60 + "\n")
        f.write(f"Tiempo de entrenamiento: {duration:.2f} segundos\n")
        f.write(f"Tiempo de inferencia unitario: {infer_duration:.2f} ms/imagen\n")
        f.write(f"Número de parámetros: {model.count_params():,}\n")
        f.write("="*60 + "\n\n")
        
        # Reporte de clasificación tradicional
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write("\n" + "="*60 + "\n")
        f.write("MÉTRICAS DIAGNÓSTICAS AVANZADAS POR CLASE:\n")
        for idx, c_name in enumerate(class_names):
            f.write(f"Clase: {c_name}\n")
            f.write(f"  - Sensibilidad (Sensitivity / Recall): {report_dict[c_name]['recall']:.4f}\n")
            f.write(f"  - Especificidad (Specificity): {specs[idx]:.4f}\n")
            f.write(f"  - Precision-Recall AUC (PR-AUC): {pr_auc_values[idx]:.4f}\n")
            f.write("-"*40 + "\n")

    # Guardar matriz de confusión de alta resolución
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Greens")
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(os.path.join(report_path, f"{model_name}_confusion_matrix.png"), dpi=200)
    plt.close()

# ------------------------
# 📈 Entrenar todos los modelos
# ------------------------
modelos = [
    (MobileNetV2, mobilenet_pre, "MobileNetV2"),
    (ResNet50, resnet_pre, "ResNet50"),
    (EfficientNetB0, efficientnet_pre, "EfficientNetB0")
]

for model_fn, preprocess_fn, name in modelos:
    train_and_evaluate(model_fn, preprocess_fn, name)

print("\n📅 Entrenamiento y evaluación científica completados con éxito.")
