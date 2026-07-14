import os
import h5py
import json

def remove_key_recursive(d, key_to_remove):
    if isinstance(d, dict):
        if key_to_remove in d:
            print(f"Removing {key_to_remove} from dict key: {d.get('name', 'unnamed')}")
            del d[key_to_remove]
        for k, v in list(d.items()):
            remove_key_recursive(v, key_to_remove)
    elif isinstance(d, list):
        for item in d:
            remove_key_recursive(item, key_to_remove)

def fix_model_file(filepath):
    print(f"\nAnalyzing model: {filepath}")
    try:
        with h5py.File(filepath, 'r+') as f:
            if 'model_config' not in f.attrs:
                print("No model_config attribute found.")
                return
            
            config_raw = f.attrs['model_config']
            if isinstance(config_raw, bytes):
                config_str = config_raw.decode('utf-8')
            else:
                config_str = config_raw
                
            config_json = json.loads(config_str)
            
            # Count elements before
            remove_key_recursive(config_json, 'quantization_config')
            
            # Write back
            new_config_str = json.dumps(config_json)
            if isinstance(config_raw, bytes):
                f.attrs['model_config'] = new_config_str.encode('utf-8')
            else:
                f.attrs['model_config'] = new_config_str
                
            print(f"Successfully cleaned and saved: {filepath}")
    except Exception as e:
        print(f"Error fixing {filepath}: {str(e)}")

def main():
    models_dir = 'models'
    for name in ["MobileNetV2", "ResNet50", "EfficientNetB0"]:
        filepath = os.path.join(models_dir, f"{name}.h5")
        if os.path.exists(filepath):
            fix_model_file(filepath)
        else:
            print(f"Model file not found: {filepath}")

if __name__ == "__main__":
    main()
