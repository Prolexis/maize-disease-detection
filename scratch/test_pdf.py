import numpy as np
import sys
import os

# Asegurar que podemos importar app.py
sys.path.append(os.getcwd())

import app

def test_pdf():
    print("Iniciando prueba unitaria de generación de PDF...")
    
    # Crear datos dummy
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    
    predictions = {
        "MobileNetV2": {
            'class': "Sano",
            'confidence': 0.95,
            'probabilities': [0.01, 0.02, 0.02, 0.95]
        },
        "ResNet50": {
            'class': "Sano",
            'confidence': 0.93,
            'probabilities': [0.02, 0.03, 0.02, 0.93]
        },
        "EfficientNetB0": {
            'class': "Sano",
            'confidence': 0.97,
            'probabilities': [0.01, 0.01, 0.01, 0.97]
        }
    }
    
    try:
        pdf_bytes = app.generate_pdf_report(
            image=image,
            predictions=predictions,
            uploaded_filename="test_image.png",
            consensus_reached=True,
            consensus_diagnosis="Sano"
        )
        
        print("\n--- RESULTADO DE LA PRUEBA ---")
        print(f"Tipo retornado: {type(pdf_bytes)}")
        print(f"Longitud en bytes: {len(pdf_bytes)}")
        
        assert isinstance(pdf_bytes, bytes), f"ERROR: Se esperaba tipo 'bytes', pero se obtuvo '{type(pdf_bytes)}'"
        assert len(pdf_bytes) > 0, "ERROR: El PDF está vacío (0 bytes)"
        
        print("\n✅ ¡PRUEBA EXITOSA! El PDF se generó como un objeto 'bytes' válido.")
        
    except Exception as e:
        print("\n❌ ¡PRUEBA FALLIDA! Ocurrió una excepción durante la generación del PDF:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_pdf()
