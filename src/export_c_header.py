import os
import tensorflow as tf

def export_model_to_c_header(keras_model_path, output_header_path, variable_name="maize_model"):
    """
    Convierte un modelo de Keras (.h5 o .keras) a formato TensorFlow Lite (.tflite)
    y luego genera un archivo de cabecera de C (.h) conteniendo el modelo como un 
    arreglo de bytes (unsigned char) para microcontroladores y sistemas embebidos.
    """
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"No se encontró el modelo en la ruta: {keras_model_path}")
        
    print(f"Cargando modelo de Keras desde {keras_model_path}...")
    model = tf.keras.models.load_model(keras_model_path)
    
    print("Convirtiendo modelo a formato TensorFlow Lite (.tflite)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Opciones de optimización recomendadas para sistemas embebidos (cuantización a float16)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    print(f"Escribiendo cabecera de C (.h) en {output_header_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_header_path)), exist_ok=True)
    
    with open(output_header_path, 'w', encoding='utf-8') as f:
        # Escribir Guardas de Inclusión
        f.write(f"// Archivo generado automaticamente para implementacion embebida de IA\n")
        f.write(f"// Modelo: {os.path.basename(keras_model_path)}\n\n")
        f.write(f"#ifndef {variable_name.upper()}_DATA_H\n")
        f.write(f"#define {variable_name.upper()}_DATA_H\n\n")
        
        # Escribir soporte para diferentes alineaciones según el compilador
        f.write("#if defined(__GNUC__) || defined(__clang__)\n")
        f.write("#define ALIGNMENT_16 __attribute__((aligned(16)))\n")
        f.write("#elif defined(_MSC_VER)\n")
        f.write("#define ALIGNMENT_16 __declspec(align(16))\n")
        f.write("#elif defined(__cplusplus) && __cplusplus >= 201103L\n")
        f.write("#define ALIGNMENT_16 alignas(16)\n")
        f.write("#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L\n")
        f.write("#define ALIGNMENT_16 _Alignas(16)\n")
        f.write("#else\n")
        f.write("#define ALIGNMENT_16\n")
        f.write("#endif\n\n")

        # Escribir tamaño del arreglo
        f.write(f"const unsigned int {variable_name}_tflite_len = {len(tflite_model)};\n\n")
        
        # Escribir el arreglo de bytes alineado
        f.write(f"ALIGNMENT_16 const unsigned char {variable_name}_tflite[] = {{\n  ")
        
        for idx, val in enumerate(tflite_model):
            f.write(f"0x{val:02x}")
            if idx < len(tflite_model) - 1:
                f.write(", ")
            # Romper líneas cada 12 bytes para mantener legibilidad del archivo
            if (idx + 1) % 12 == 0 and idx < len(tflite_model) - 1:
                f.write("\n  ")
                
        f.write("\n};\n\n")
        f.write("#endif // " + variable_name.upper() + "_DATA_H\n")
        
    print(f"✅ Exportación completada con éxito. Tamaño TFLite: {len(tflite_model)} bytes.")

if __name__ == "__main__":
    # Ejemplo de uso local del script
    # Convierte el MobileNetV2.h5 local si existe a C header
    local_model = "models/MobileNetV2.h5"
    if os.path.exists(local_model):
        try:
            export_model_to_c_header(local_model, "models/maize_mobilenet_v2.h")
        except Exception as e:
            print(f"Error al exportar: {e}")
