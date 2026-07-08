# CHANGELOG: Proyecto Fitosanitario y AutoML de Maíz

Este documento registra el estado y evolución de los componentes del software en el proceso de actualización integral.

---

## [0.0.0] - Estado Inicial (Existente)
Los siguientes componentes ya estaban implementados y se mantienen **sin modificaciones** para garantizar que lo que ya funciona no sea alterado:

* **Modelos Convolucionales Preentrenados**:
  * `models/MobileNetV2.h5`
  * `models/ResNet50.h5`
  * `models/EfficientNetB0.h5`
* **Lógica de Inferencia de Imágenes**:
  * Carga y preprocesamiento de imágenes de hojas de maíz (`128x128` píxeles, preprocesamiento específico por arquitectura).
  * Consenso de modelos para emitir diagnósticos basados en unanimidad democrática.
* **Script de Entrenamiento en la Nube**:
  * [entrenamiento.py](file:///c:/Users/Usuario/Desktop/maize-disease-detection/entrenamiento.py) (diseñado para Google Colab con Google Drive montado).
* **Script de Generación de Reportes Estáticos**:
  * [reportes.py](file:///c:/Users/Usuario/Desktop/maize-disease-detection/reportes.py).
* **Sistema de Despliegue**:
  * [Dockerfile](file:///c:/Users/Usuario/Desktop/maize-disease-detection/Dockerfile) y [docker-compose.yml](file:///c:/Users/Usuario/Desktop/maize-disease-detection/docker-compose.yml).

---

## [1.0.0] - Versión Mejorada y Nuevas Funcionalidades
Las siguientes mejoras y componentes han sido desarrollados e integrados desde cero:

### ⚙️ Mejorado (Mejoras sobre el código existente)
* **Dashboard Principal ([app.py](file:///c:/Users/Usuario/Desktop/maize-disease-detection/app.py))**:
  * Integración de una **pantalla de autenticación (Login)** segura para validar credenciales del usuario antes de desplegar el panel.
  * Incorporación de un **Selector de Modo de Panel** en el sidebar para alternar fluidamente entre el *Diagnóstico por Imágenes* existente y la nueva *Plataforma AutoML Tabular*.
  * Limpieza estática de bloques CSS y optimización del rendimiento de renderizado en Streamlit.

### 🌟 Creado desde Cero (Nuevas funcionalidades)
* **Dataset Tabular Fitosanitario ([data/maize_crop_data.csv](file:///c:/Users/Usuario/Desktop/maize-disease-detection/data/maize_crop_data.csv))**:
  * Conjunto de datos agrícola que simula mediciones meteorológicas y químicas de suelos para predecir el estado de salud de cultivos de maíz (`temperatura`, `humedad`, `ph_suelo`, `nitrogeno`, `fosforo`, `potasio`, `lluvia_mm`, `variedad`, `estado`).
* **Módulos de Machine Learning (`src/`)**:
  * [config.py](file:///c:/Users/Usuario/Desktop/maize-disease-detection/src/config.py): Semillas globales de reproducibilidad (numpy, tensorflow, random) y configuraciones del sistema.
  * [eda.py](file:///c:/Users/Usuario/Desktop/maize-disease-detection/src/eda.py): Limpieza automatizada, estadísticos descriptivos detallados por clase, análisis de desbalance de clases, mapas de calor de correlación y visualización de distribuciones. Cada fase genera **interpretación en texto automática**.
  * [training.py](file:///c:/Users/Usuario/Desktop/maize-disease-detection/src/training.py): Entrenamiento de **3 modelos clásicos** (Regresión Logística, Random Forest, MLP) y **2 híbridos** (Voting RF+MLP, Stacking con Gradient Boosting). Métricas exhaustivas de evaluación, curvas ROC/PR y matriz de confusión por modelo. Guardado serializado con metadatos.
  * [cross_validation.py](file:///c:/Users/Usuario/Desktop/maize-disease-detection/src/cross_validation.py): Stratified K-Fold con boxplot de variabilidad entre folds.
  * [tuning.py](file:///c:/Users/Usuario/Desktop/maize-disease-detection/src/tuning.py): Grid Search / Random Search para optimizar hiperparámetros.
  * [stats_tests.py](file:///c:/Users/Usuario/Desktop/maize-disease-detection/src/stats_tests.py): Pruebas estadísticas avanzadas: Shapiro-Wilk y Levene (supuestos), ANOVA + Tukey o Friedman + Nemenyi, Wilcoxon y McNemar, acompañadas de interpretaciones en lenguaje natural.
  * [reporting.py](file:///c:/Users/Usuario/Desktop/maize-disease-detection/src/reporting.py): Exportación automática a reportes **PDF**, **Word (.docx)** y **Excel (.xlsx)** con hojas dedicadas para cada fase de análisis.
