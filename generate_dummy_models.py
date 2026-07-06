import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

def create_dummy_model(name):
    # Crear un modelo secuencial simple con la forma de entrada y salida esperada por la app
    model = Sequential([
        Flatten(input_shape=(128, 128, 3)),
        Dense(16, activation='relu'),
        Dense(4, activation='softmax')  # 4 clases de salida
    ], name=name)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    model_names = ["MobileNetV2", "ResNet50", "EfficientNetB0"]
    
    print("Iniciando la creación de modelos de prueba (dummy)...")
    for name in model_names:
        file_path = os.path.join(models_dir, f"{name}.h5")
        if not os.path.exists(file_path):
            print(f"Generando {name}.h5...")
            model = create_dummy_model(name)
            model.save(file_path)
            print(f"¡Modelo de prueba {name}.h5 guardado en {file_path}!")
        else:
            print(f"El modelo {name}.h5 ya existe, omitiendo.")
            
    print("\nTodos los modelos de prueba creados exitosamente.")

if __name__ == "__main__":
    main()
