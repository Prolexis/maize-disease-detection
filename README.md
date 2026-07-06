# 🌽 Sistema Inteligente de Diagnóstico Fitosanitario para el Cultivo de Maíz

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compatible-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Este proyecto implementa una solución de diagnóstico automatizado de enfermedades foliares en plantas de maíz mediante técnicas avanzadas de **Deep Learning** y **Transfer Learning**. El sistema integra tres arquitecturas de redes neuronales convolucionales (CNN) entrenadas de forma independiente para proporcionar una predicción robusta basada en un sistema de consenso democrático, reduciendo falsos positivos y garantizando diagnósticos altamente confiables.

---

## 📋 Características Principales

* **Interfaz Web Interactiva**: Desarrollada en Streamlit con un diseño Next-Gen glassmorphic adaptado automáticamente para modo claro y oscuro.
* **Sistema de Consenso Tri-Modelo**: La aplicación clasifica imágenes utilizando tres modelos independientes y reporta si se ha alcanzado un consenso unánime.
* **Generación Automática de Reportes PDF**: Descarga reportes completos con la imagen analizada, gráficos de probabilidad detallados por modelo, análisis de consenso y recomendaciones técnicas y fitosanitarias.
* **Panel de Estadísticas y Reportes**: Incluye visualización de curvas de aprendizaje, matrices de confusión detalladas, análisis comparativo de F1-Score y pruebas estadísticas de McNemar ejecutadas sobre el conjunto de evaluación.

---

## 🧠 Arquitectura y Rendimiento de Modelos

El sistema evalúa las muestras utilizando tres arquitecturas CNN entrenadas con el dataset *PlantVillage Corn Leaf Disease*:

| Modelo | Arquitectura Base | Épocas | Tiempo de Entrenamiento | Precisión de Validación |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | MobileNetV2 | 10 | ~46.97 min | **93.52%** |
| **ResNet50** | ResNet50 | 10 | ~162.80 min | **94.12%** |
| **EfficientNetB0** | EfficientNetB0 | 10 | ~55.61 min | **95.88%** |

*Nota: EfficientNetB0 presenta el mejor balance de velocidad y precisión del sistema, mientras que MobileNetV2 destaca por su rapidez en tiempo de inferencia.*

---

## 🍃 Clases de Enfermedades Detectables

1. **Sano (Saludable)**: Hojas sin signos visibles de patologías foliares.
2. **Tizón del Norte** (*Exserohilum turcicum*): Caracterizado por lesiones alargadas y elípticas en forma de "cigarro", de color grisáceo o marrón.
3. **Roya Común** (*Puccinia sorghi*): Pequeñas pústulas circulares o alargadas de color marrón-rojizo que se presentan en ambas caras de la hoja.
4. **Mancha Gris** (*Cercospora zeae-maydis*): Lesiones rectangulares y alargadas delimitadas directamente por las venas de la hoja.

---

## 📂 Estructura del Proyecto

```directory
maize-disease-detection/
├── app.py                     # Aplicación principal de Streamlit
├── entrenamiento.py           # Script de entrenamiento y guardado de modelos
├── reportes.py                # Script de generación de reportes y matrices estadísticas
├── división_data.py           # Utilidad para separación de conjuntos de datos (Train/Val)
├── requirements.txt           # Dependencias de librerías de Python
├── Dockerfile                 # Configuración de la imagen Docker de producción
├── docker-compose.yml         # Orquestación de contenedores y mapeo de volúmenes
├── models/                    # Directorio local para almacenar modelos (.h5)
│   └── .gitkeep
├── reports/                   # Directorio local para guardar los reportes gráficos generados
│   ├── modelos_comparacion_completa.png
│   ├── matrices_confusion_todos.png
│   ├── metricas_detalladas_por_clase.png
│   └── reporte_completo.txt
└── README.md                  # Documentación del proyecto
```

---

## 🚀 Instalación y Despliegue

### Requisito Previo
Asegúrate de haber descargado los archivos de modelos entrenados (`MobileNetV2.h5`, `ResNet50.h5` y `EfficientNetB0.h5`) y haberlos guardado en la carpeta `models/` de la raíz del proyecto. *Si no tienes los modelos aún, puedes autogenerar unos de prueba ejecutando el script `python generate_dummy_models.py`*.

---

### Método A: Despliegue con Docker y Docker Compose (Recomendado)

Docker automatiza la instalación de librerías del sistema para OpenCV y TensorFlow.

1. Abre tu terminal en la raíz del proyecto.
2. Construye e inicia el contenedor ejecutando:
   ```bash
   docker compose up --build
   ```
3. Accede a la interfaz web en tu navegador:
   👉 **[http://localhost:8501](http://localhost:8501)**

*El volumen mapeado en `docker-compose.yml` permite que los PDF y reportes generados dentro del contenedor se guarden inmediatamente en tu carpeta local `reports/`.*

---

### Método B: Despliegue Local (Python)

Si prefieres correr el proyecto directamente en tu sistema operativo:

1. Crea y activa un entorno virtual de Python:
   ```powershell
   # En Windows:
   python -m venv .venv
   .venv\Scripts\activate

   # En Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Instala los requerimientos e dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta la aplicación de Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Abre la dirección web mostrada en la terminal (usualmente `http://localhost:8501`).

---

## 📄 Generación de Reportes de Diagnóstico
Al cargar la imagen de una hoja y procesarla, podrás presionar el botón **📥 Generar Reporte PDF**. Este reporte generará un documento formal listo para impresión que incluye:
* Metadatos del análisis (fecha, hora, resolución de imagen).
* La imagen original analizada embebida.
* Gráficos comparativos de probabilidad para cada modelo.
* Diagnóstico de consenso y justificación técnica.
* Recomendaciones de tratamiento específicas basadas en la enfermedad diagnosticada.
