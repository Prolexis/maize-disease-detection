# -*- coding: utf-8 -*-
import h5py
import json
import os

def patch_h5_file(filepath):
    print(f"Procesando {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: The file {filepath} does not exist.")
        return False
        
    try:
        # Abrir en modo lectura/escritura ('r+')
        with h5py.File(filepath, 'r+') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                
                # Decodificar de bytes si es necesario
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                config = json.loads(config_str)
                
                # Función recursiva para buscar y borrar 'quantization_config'
                def remove_quantization(obj):
                    modified = False
                    if isinstance(obj, dict):
                        if 'quantization_config' in obj:
                            del obj['quantization_config']
                            modified = True
                        for k, v in list(obj.items()):
                            if remove_quantization(v):
                                modified = True
                    elif isinstance(obj, list):
                        for item in obj:
                            if remove_quantization(item):
                                modified = True
                    return modified
                
                was_modified = remove_quantization(config)
                
                if was_modified:
                    # Guardar la configuración modificada
                    new_config_str = json.dumps(config)
                    f.attrs['model_config'] = new_config_str.encode('utf-8')
                    print("Patched successfully! Incompatible quantization fields removed.")
                else:
                    print("File did not contain references to quantization_config or was already clean.")
                return True
            else:
                print("Warning: model_config attribute not found in H5 file.")
                return False
    except Exception as e:
        print(f"Error processing H5 file: {e}")
        return False

def main():
    models_dir = "models"
    model_names = ["MobileNetV2", "ResNet50", "EfficientNetB0"]
    
    print("=== INICIANDO PARCHEO DE MODELOS ENTRENADOS ===")
    for name in model_names:
        path = os.path.join(models_dir, f"{name}.h5")
        patch_h5_file(path)
    print("=== PROCESO FINALIZADO ===")

if __name__ == "__main__":
    main()
