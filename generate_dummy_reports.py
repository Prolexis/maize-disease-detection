import os
from PIL import Image

def main():
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    expected_files = [
        "modelos_comparacion_completa.png",
        "matrices_confusion_todos.png",
        "matriz_confusion_mobilenetv2.png",
        "matriz_confusion_resnet50.png",
        "matriz_confusion_efficientnetb0.png",
        "metricas_detalladas_por_clase.png",
        "mcnemar_analysis.png"
    ]
    
    print("Iniciando la creación de imágenes de reporte de prueba...")
    for filename in expected_files:
        file_path = os.path.join(reports_dir, filename)
        if not os.path.exists(file_path):
            # Crear una imagen pequeña de color verde para probar
            img = Image.new('RGB', (200, 200), color='#2E8B57')
            img.save(file_path)
            print(f"Imagen creada: {file_path}")
            
    # Crear reporte de texto de prueba
    txt_path = os.path.join(reports_dir, "reporte_completo.txt")
    if not os.path.exists(txt_path):
        with open(txt_path, 'w') as f:
            f.write("Reporte de prueba completo para la detección de enfermedades de maíz.\n")
            f.write("MobileNetV2: 93.52% accuracy\n")
            f.write("ResNet50: 94.12% accuracy\n")
            f.write("EfficientNetB0: 95.88% accuracy\n")
        print(f"Archivo de texto creado: {txt_path}")
        
    print("\nTodos los reportes de prueba creados exitosamente.")

if __name__ == "__main__":
    main()
