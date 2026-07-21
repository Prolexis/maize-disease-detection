# 🌽 Sistema Inteligente de Diagnóstico Fitosanitario y Plataforma AutoML para el Cultivo de Maíz

[![Next.js](https://img.shields.io/badge/Next.js-16.2.10-black.svg?style=flat-square&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B.svg?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?style=flat-square&logo=docker)](https://www.docker.com/)

Plataforma empresarial de diagnóstico fitosanitario, analítica agronómica y MLOps para la detección de enfermedades foliares y predicción de salud en cultivos de maíz. Integra un **Frontend Web en Next.js 16** con arquitectura multi-idioma (Español, Inglés, Portugués), un **Backend RESTful en FastAPI (v1)** con seguridad JWT y WebSockets de transmisión en tiempo real, un **Dashboard MLOps en Streamlit**, y tuberías de **AutoML Tabular y Visión Computacional**.

---

## 📋 Arquitectura General del Sistema

```
                         ┌──────────────────────────────────────────┐
                         │   Navegador Web del Usuario              │
                         └────────────────────┬─────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│   Frontend Next.js 16 (React / TS)    │   │   Streamlit MLOps Dashboard           │
│   - Puerto: 3000                      │   │   - Puerto: 8501                      │
│   - Multi-idioma (es, en, pt)         │   │   - Paneles interactivos glassmorphism│
│   - Transmisión WebSocket             │   │   - EDA y Generación de Reportes      │
└───────────────────┬───────────────────┘   └───────────────────┬───────────────────┘
                    │                                           │
                    └─────────────────────────┬─────────────────┘
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│   Backend FastAPI v1 (Python 3.11)                                                │
│   - Puerto: 8000                                                                  │
│   - Autenticación JWT / Encriptación Passlib                                      │
│   - WebSockets de Entrenamiento (`/api/v1/training/ws/{client_id}`)               │
│   - Rutas API REST: /auth, /model, /dataset, /training, /reports, /stats, /chat   │
└─────────────────────────────────────┬─────────────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           ▼                          ▼                          ▼
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ Modelos Deep       │   │ Modelos Tabulares  │   │ Base de Datos      │
│ Learning (.h5)     │   │ AutoML (.pkl)      │   │ SQLite (maize.db)  │
│ - MobileNetV2      │   │ - Random Forest    │   │ - Usuarios         │
│ - ResNet50         │   │ - XGBoost          │   │ - Historial MLOps  │
│ - EfficientNetB0   │   │ - Stacking/Voting  │   │   Experimentos     │
└────────────────────┘   └────────────────────┘   └────────────────────┘
```

---

## ✨ Características Principales

### 🌿 1. Visión Computacional y Detección Foliares
- **Consenso Tri-Modelo:** Clasificación de imágenes con ensamble de tres redes neuronales convolucionales (**MobileNetV2**, **ResNet50**, **EfficientNetB0**).
- **Diagnóstico Unánime:** Detección de patologías como *Sano*, *Tizón del Norte*, *Roya Común* y *Mancha Gris*.

### 📊 2. AutoML Tabular Agronómico
- **Ingeniería de Características Automatizada:** Imputación inteligente de nulos, tratamiento de outliers y escalado estandarizado.
- **Modelado Híbrido:** Entrenamiento competitivo entre clasificadores clásicos (Random Forest, XGBoost) y ensambles (Voting, Stacking).
- **Pruebas Estadísticas Inferenciales:** Pruebas de Wilcoxon, análisis de intervalos de confianza Bootstrap e interpretabilidad fitosanitaria.
- **Predicción Tabular en Tiempo Real:** Inferencia en vivo ajustando variables ambientales (temperatura, humedad, pH del suelo, nitrógeno, fósforo, potasio, precipitaciones).

### 🌐 3. Aplicación Web Next.js 16 Multi-idioma
- **Internacionalización (i18n):** Soporte completo para Español (`es`), Inglés (`en`) y Portugués (`pt`).
- **WebSockets de Progreso:** Monitoreo en tiempo real con sistema de respaldo dual (HTTP Polling automático ante micro-desconexiones).
- **Modo Oscuro / Claro:** Tema dinámico adaptativo.

### 📜 4. MLOps y Reportabilidad Multiformato
- **Historial de Experimentos:** Registro persistente en SQLite (`experiments`) de métricas MLOps (*Accuracy*, *F1-Score*, *Split*, *Semilla*, *CV Folds*, *Tuning*).
- **Descarga de Informes Automatizada:** Generación en un clic de reportes fitosanitarios en tres formatos:
  - 🔴 **PDF** (Informes ejecutivos listos para impresión)
  - 🔵 **Word (.docx)** (Documentación editable)
  - 🟢 **Excel (.xlsx)** (Tablas y métricas estructuradas)

---

## 🛠️ Tecnologías Utilizadas

* **Frontend:** Next.js 16 (Turbopack/Webpack), React 18, TypeScript, Tailwind CSS, Lucide Icons, Recharts, next-intl.
* **Backend:** FastAPI, Uvicorn, WebSockets, PyJWT, Passlib (Bcrypt), Pydantic v2.
* **Machine Learning & AutoML:** TensorFlow 2.15, Scikit-Learn, XGBoost, Pandas, NumPy, SciPy.
* **Reportes:** ReportLab (PDF), python-docx (Word), OpenPyXL (Excel), Matplotlib, Seaborn.
* **Base de Datos & MLOps:** SQLite3.
* **Contenerización:** Docker & Docker Compose.

---

## 🚀 Instalación y Despliegue

### Requisitos Previos
- Docker Desktop instalado y corriendo.
- (Opcional para desarrollo local) Python 3.10 o 3.11 y Node.js 18+.

---

### Opción A: Despliegue con Docker Compose (Recomendado)

Construye y levanta toda la pila (Backend, Frontend Next.js y Dashboard Streamlit) en contenedores aislados:

```bash
docker compose up --build
```

Una vez completada la inicialización, accede a los servicios:
- 🌐 **Aplicación Web Principal (Next.js):** [http://localhost:3000](http://localhost:3000)
- 📊 **Dashboard MLOps (Streamlit):** [http://localhost:8501](http://localhost:8501)
- ⚡ **Documentación Interactiva API (FastAPI Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Opción B: Ejecución Local para Desarrollo

#### 1. Backend (FastAPI)
```bash
# Crear entorno virtual e instalar dependencias
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt

# Iniciar servidor Uvicorn
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```
Accede a [http://localhost:3000](http://localhost:3000).

#### 3. Dashboard (Streamlit)
```bash
streamlit run app.py
```
Accede a [http://localhost:8501](http://localhost:8501).

---

## 🧪 Pruebas Automatizadas

El proyecto incluye una suite completa de pruebas unitarias e integración para el backend:

```bash
python -m pytest backend/tests
```

Para validar el build de producción del cliente web Next.js:

```bash
cd frontend
npm run build
```

---

## 🔒 Credenciales por Defecto

* **Usuario Administrador:** `admin`
* **Contraseña:** `admin123`

---

## 📜 Licencia

Proyecto desarrollado bajo la licencia MIT.
