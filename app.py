# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
import time
import base64
from fpdf import FPDF
import io
from datetime import datetime
import pytz

# Importar componentes de la plataforma AutoML tabular
from src.config import set_seed, apply_theme_to_plot
from src.chatbot import get_chatbot_response
from src.eda import clean_data, get_descriptive_stats, interpret_eda, plot_eda_charts
from src.training import train_and_evaluate_all, plot_training_charts, interpret_training, save_best_model
from src.cross_validation import run_cross_validation, plot_cv_dispersion, interpret_cv
from src.tuning import run_hyperparameter_tuning, interpret_tuning
from src.stats_tests import run_statistical_tests, interpret_stats
from src.reporting import generate_xlsx_report, generate_docx_report, generate_tabular_pdf_report, generate_image_docx_report, generate_image_xlsx_report

# Inicialización de multi-idioma (Español, English, Português)
if 'lang' not in st.session_state:
    st.session_state.lang = 'es'

def t(key):
    try:
        from src.translation import TRANSLATIONS
        lang = st.session_state.get('lang', 'es')
        return TRANSLATIONS[lang].get(key, key)
    except:
        return key

# Configuración de la página
st.set_page_config(
    page_title="🌽 Detector de Enfermedades en Hojas de Maíz",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Configuración de fuentes globales en elementos personalizados */
    .main-header {
        font-family: 'Outfit', -apple-system, sans-serif;
    }
    .model-card, .prediction-result, .nav-tab, button, [data-baseweb="tab"], p, li, h2, h3, [data-testid="metric-container"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* Animación de carga progresiva (fadeInUp) */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Aplicar animación suave a componentes clave al cargarse */
    .stMarkdown, .model-card, [data-testid="stFileUploader"], [data-testid="stDataFrame"], .stAlert, [data-testid="metric-container"] {
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    /* Título Animado Estilo 2026 */
    @keyframes gradient-flow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main-header {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(-45deg, #10b981, #06b6d4, #3b82f6, #059669);
        background-size: 300% 300%;
        animation: gradient-flow 10s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -1.5px;
        text-shadow: 0 10px 30px rgba(16, 185, 129, 0.1);
    }

    /* Tarjetas de Modelos con Glassmorphism */
    .model-card {
        background: color-mix(in srgb, var(--secondary-background-color) 75%, transparent) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: var(--text-color);
        padding: 1.8rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        border-left: 6px solid #10b981;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.04);
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease, border-color 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
        border-left-color: #06b6d4;
    }

    .model-card h3 {
        margin-top: 0;
        margin-bottom: 0.8rem;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: var(--text-color) !important;
    }
    
    .prediction-result {
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.9rem 1.3rem;
        border-radius: 10px;
        margin-top: 0.5rem;
        display: inline-block;
        width: 100%;
        box-sizing: border-box;
        letter-spacing: -0.2px;
    }
    
    /* Alertas de Diagnóstico - Adaptadas con color-mix dinámico */
    .healthy {
        background-color: rgba(16, 185, 129, 0.1) !important;
        color: color-mix(in srgb, #10b981 85%, var(--text-color)) !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
        box-shadow: inset 0 0 10px rgba(16, 185, 129, 0.05);
    }
    
    .diseased {
        background-color: rgba(244, 63, 94, 0.1) !important;
        color: color-mix(in srgb, #f43f5e 85%, var(--text-color)) !important;
        border: 1px solid rgba(244, 63, 94, 0.25) !important;
        box-shadow: inset 0 0 10px rgba(244, 63, 94, 0.05);
        animation: pulse-glow 2s infinite alternate;
    }
    
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 5px rgba(244, 63, 94, 0.1); }
        100% { box-shadow: 0 0 15px rgba(244, 63, 94, 0.25); }
    }
    
    /* Adaptaciones automáticas en Modo Oscuro (Si se prefiere por sistema) */
    @media (prefers-color-scheme: dark) {
        .model-card {
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        }
        .diseased {
            animation: pulse-glow-dark 2s infinite alternate;
        }
    }
    
    @keyframes pulse-glow-dark {
        0% { box-shadow: 0 0 5px rgba(248, 113, 113, 0.1); }
        100% { box-shadow: 0 0 20px rgba(248, 113, 113, 0.3); }
    }

    /* Rediseño de Botones de Streamlit a Cápsulas Modernas */
    div.stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.35) !important;
        border-color: transparent !important;
    }

    div.stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Estilización del Botón de Descarga */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100%;
    }

    div.stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.35) !important;
        border-color: transparent !important;
    }

    /* Zona de Carga de Archivos (Uploader) */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(16, 185, 129, 0.3) !important;
        border-radius: 16px !important;
        background-color: rgba(16, 185, 129, 0.02) !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #10b981 !important;
        background-color: rgba(16, 185, 129, 0.05) !important;
    }

    /* Pestañas de Navegación Estilo Pill Segmented */
    button[data-baseweb="tab"] {
        border-radius: 30px !important;
        padding: 0.5rem 1.5rem !important;
        margin: 0 0.3rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
        transition: all 0.3s ease !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(16, 185, 129, 0.12) !important;
        color: #10b981 !important;
        border-color: rgba(16, 185, 129, 0.25) !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #10b981 !important;
    }

    /* Ocultar barra de selección y borde de pestañas nativas de Streamlit */
    div[data-baseweb="tab-highlight"] {
        background-color: transparent !important;
        height: 0px !important;
    }
    
    div[data-baseweb="tab-border"] {
        background-color: transparent !important;
    }

    /* Frosted glass universal para la barra lateral basado en color-mix */
    section[data-testid="stSidebar"] {
        background-color: color-mix(in srgb, var(--secondary-background-color) 88%, transparent) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.15) !important;
    }

    /* Estilización de los botones del Sidebar */
    section[data-testid="stSidebar"] button {
        background: rgba(255, 255, 255, 0.03) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
        box-shadow: none !important;
        padding: 0.4rem 1.2rem !important;
        width: 100% !important;
    }
    
    section[data-testid="stSidebar"] button:hover {
        background: rgba(16, 185, 129, 0.08) !important;
        border-color: #10b981 !important;
        color: #10b981 !important;
        transform: translateY(-1px) !important;
    }

    /* Estilo del botón de logout en el sidebar */
    button[key="btn_logout"] {
        border: 1px solid rgba(220, 38, 38, 0.25) !important;
        color: #ef4444 !important;
        margin-top: 1rem !important;
    }
    button[key="btn_logout"]:hover {
        background: rgba(220, 38, 38, 0.08) !important;
        border-color: #dc2626 !important;
        color: #b91c1c !important;
        transform: translateY(-1px) !important;
    }

    /* Estilización de las tarjetas de métricas */
    [data-testid="metric-container"] {
        background: color-mix(in srgb, var(--secondary-background-color) 75%, transparent) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        padding: 1.2rem 1.6rem !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.03) !important;
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease, border-color 0.3s ease !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08) !important;
        border-color: rgba(16, 185, 129, 0.25) !important;
    }

    /* Estilos sutiles y bordes suaves para las alertas de Streamlit */
    div.stAlert, [data-testid="stNotification"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.01) !important;
    }

    /* Esquinas redondeadas y estética limpia para el contenedor de tablas de datos */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

def inject_custom_css():
    theme = st.session_state.get('theme', 'Oscuro')
    is_dark = theme in ['Oscuro', 'Dark', 'escuro', 'Escuro']
    if is_dark:
        theme_css = """
        <style>
            :root {
                --background-color: #0f172a !important;
                --text-color: #f8fafc !important;
                --secondary-background-color: #1e293b !important;
            }
            .stApp {
                background-color: #0f172a !important;
                color: #f8fafc !important;
            }
            /* Títulos y textos generales */
            h1, h2, h3, h4, h5, h6, p, label, li, span, div.stMarkdown {
                color: #f8fafc !important;
            }
            [data-testid="stHeader"] {
                background-color: #0f172a !important;
            }
            [data-testid="stSidebar"] {
                background-color: #1e293b !important;
                border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
            }
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
            [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6,
            [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
                color: #f8fafc !important;
            }
            .model-card, [data-testid="metric-container"] {
                background: #1e293b !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                color: #f8fafc !important;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
            }
            .model-card h3, [data-testid="metric-container"] p, [data-testid="metric-container"] div {
                color: #f8fafc !important;
            }
            /* Botones del Sidebar */
            section[data-testid="stSidebar"] button {
                background: rgba(255, 255, 255, 0.05) !important;
                color: #f8fafc !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
            }
            section[data-testid="stSidebar"] button:hover {
                background: rgba(16, 185, 129, 0.15) !important;
                border-color: #10b981 !important;
                color: #10b981 !important;
            }
            /* Tabs */
            button[data-baseweb="tab"] {
                color: #94a3b8 !important;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                color: #10b981 !important;
                background-color: rgba(16, 185, 129, 0.15) !important;
                border-color: rgba(16, 185, 129, 0.25) !important;
            }
            /* Inputs y Selectboxes */
            [data-testid="stSelectbox"] div,
            [data-testid="stTextInput"] div,
            [data-testid="stTextInput"] input,
            [data-testid="stFileUploader"] div,
            [data-testid="stFileUploader"] button,
            div[data-baseweb="select"],
            div[data-baseweb="select"] *,
            div[data-baseweb="menu"],
            div[data-baseweb="menu"] *,
            div[role="listbox"],
            div[role="listbox"] *,
            li[role="option"],
            li[role="option"] * {
                background-color: #1e293b !important;
                color: #f8fafc !important;
                border-color: rgba(255, 255, 255, 0.1) !important;
            }
            li[role="option"]:hover, li[role="option"]:hover * {
                background-color: rgba(16, 185, 129, 0.15) !important;
                color: #10b981 !important;
            }
            /* Chat messages */
            [data-testid="stChatMessage"] {
                background-color: #1e293b !important;
                border: 1px solid rgba(255, 255, 255, 0.05) !important;
                border-radius: 12px !important;
            }
        </style>
        """
    else:
        theme_css = """
        <style>
            :root {
                --background-color: #f8fafc !important;
                --text-color: #0f172a !important;
                --secondary-background-color: #ffffff !important;
            }
            .stApp {
                background-color: #f8fafc !important;
                color: #0f172a !important;
            }
            /* Títulos y textos generales */
            h1, h2, h3, h4, h5, h6, p, label, li, span, div.stMarkdown {
                color: #0f172a !important;
            }
            [data-testid="stHeader"] {
                background-color: #f8fafc !important;
            }
            [data-testid="stSidebar"] {
                background-color: #f1f5f9 !important;
                border-right: 1px solid #cbd5e1 !important;
            }
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
            [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6,
            [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
                color: #0f172a !important;
            }
            .model-card, [data-testid="metric-container"] {
                background: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                color: #0f172a !important;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
            }
            .model-card h3, [data-testid="metric-container"] p, [data-testid="metric-container"] div {
                color: #0f172a !important;
            }
            /* Botones del Sidebar */
            section[data-testid="stSidebar"] button {
                background: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #cbd5e1 !important;
            }
            section[data-testid="stSidebar"] button:hover {
                background: rgba(16, 185, 129, 0.08) !important;
                border-color: #10b981 !important;
                color: #10b981 !important;
            }
            /* Tabs */
            button[data-baseweb="tab"] {
                color: #475569 !important;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                color: #10b981 !important;
                background-color: rgba(16, 185, 129, 0.1) !important;
                border-color: rgba(16, 185, 129, 0.2) !important;
            }
            /* Inputs y Selectboxes */
            [data-testid="stSelectbox"] div,
            [data-testid="stTextInput"] div,
            [data-testid="stTextInput"] input,
            [data-testid="stFileUploader"] div,
            [data-testid="stFileUploader"] button,
            div[data-baseweb="select"],
            div[data-baseweb="select"] *,
            div[data-baseweb="menu"],
            div[data-baseweb="menu"] *,
            div[role="listbox"],
            div[role="listbox"] *,
            li[role="option"],
            li[role="option"] * {
                background-color: #ffffff !important;
                color: #0f172a !important;
                border-color: #cbd5e1 !important;
            }
            li[role="option"]:hover, li[role="option"]:hover * {
                background-color: rgba(16, 185, 129, 0.08) !important;
                color: #10b981 !important;
            }
            /* Chat messages */
            [data-testid="stChatMessage"] {
                background-color: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 12px !important;
            }
        </style>
        """
    st.markdown(theme_css, unsafe_allow_html=True)
    
    # Inyectar meta tag para evitar traducción automática por parte del navegador y añadir clase notranslate
    lang_code = st.session_state.get('lang', 'es')
    meta_html = f"""
    <script>
        try {{
            const doc = window.parent.document;
            // 1. Añadir meta tag notranslate
            if (!doc.querySelector('meta[name="google"][content="notranslate"]')) {{
                const meta = doc.createElement('meta');
                meta.name = "google";
                meta.content = "notranslate";
                doc.head.appendChild(meta);
            }}
            // 2. Forzar idioma en elemento html
            doc.documentElement.lang = '{lang_code}';
            
            // 3. Añadir clase notranslate al contenedor de la app
            const app = doc.querySelector('.stApp');
            if (app && !app.classList.contains('notranslate')) {{
                app.classList.add('notranslate');
            }}
        }} catch(e) {{
            console.error("Blocker failed:", e);
        }}
    </script>
    """
    st.components.v1.html(meta_html, height=0, width=0)


# Configuración de rutas - AJUSTA ESTAS RUTAS SEGÚN TU ESTRUCTURA
MODEL_PATH = "models" if os.path.exists("models") else "/content/drive/MyDrive/maize-leaf-disease/Models"
REPORTS_PATH = "reports" if os.path.exists("reports") else "/content/drive/MyDrive/maize-leaf-disease/Reports2"

IMG_SIZE = 128

# Nombres de clases (ajusta según tus clases reales)
CLASS_NAMES = [
    "Mancha gris",
    "Roña común",
    "Tizón del norte",
    "Sano"
]

@st.cache_resource
def load_models():
    """Carga los modelos entrenados"""
    models = {}
    model_files = {
        "MobileNetV2": "MobileNetV2.h5",
        "ResNet50": "ResNet50.h5",
        "EfficientNetB0": "EfficientNetB0.h5"
    }

    for name, filename in model_files.items():
        model_path = os.path.join(MODEL_PATH, filename)
        if os.path.exists(model_path):
            try:
                models[name] = load_model(model_path)
                st.success(t("model_load_success").format(name=name))
            except Exception as e:
                st.error(t("model_load_error").format(name=name, err=str(e)))
        else:
            st.warning(t("model_file_not_found").format(path=model_path))

    return models

def check_report_files():
    """Verifica la existencia de archivos de reportes"""
    reports_path = Path(REPORTS_PATH)

    expected_files = {
        "Comparación General": "modelos_comparacion_completa.png",
        "Matrices Combinadas": "matrices_confusion_todos.png",
        "Matriz MobileNetV2": "matriz_confusion_mobilenetv2.png",
        "Matriz ResNet50": "matriz_confusion_resnet50.png",
        "Matriz EfficientNetB0": "matriz_confusion_efficientnetb0.png",
        "Métricas Detalladas": "metricas_detalladas_por_clase.png",
        "Análisis McNemar": "mcnemar_analysis.png",
        "Reporte Completo": "reporte_completo.txt"
    }

    existing_files = {}
    for name, filename in expected_files.items():
        file_path = reports_path / filename
        existing_files[name] = file_path.exists()

    return existing_files, reports_path

def preprocess_image(image, model_name):
    """Preprocesa la imagen según el modelo conservando la relación de aspecto con padding"""
    h, w = image.shape[:2]
    desired_size = IMG_SIZE
    
    # Calcular factor de escala para ajustar al tamaño deseado sin deformar
    scale = desired_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Redimensionar conservando la relación de aspecto
    image_resized = cv2.resize(image, (new_w, new_h))
    
    # Crear un lienzo negro cuadrado (128x128x3)
    padded_image = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
    
    # Calcular coordenadas para centrar la imagen redimensionada en el lienzo
    dy = (desired_size - new_h) // 2
    dx = (desired_size - new_w) // 2
    
    # Copiar la imagen redimensionada al centro del lienzo
    padded_image[dy:dy+new_h, dx:dx+new_w] = image_resized
    
    # Convertir a array y expandir dimensiones
    image_array = np.array(padded_image, dtype=np.float32)
    image_expanded = np.expand_dims(image_array, axis=0)

    # Aplicar preprocesamiento específico del modelo
    if model_name == "MobileNetV2":
        return mobilenet_preprocess(image_expanded)
    elif model_name == "ResNet50":
        return resnet_preprocess(image_expanded)
    elif model_name == "EfficientNetB0":
        return efficientnet_preprocess(image_expanded)
    else:
        return image_expanded / 255.0

def predict_disease(image, models):
    """Realiza predicciones con todos los modelos"""
    predictions = {}

    for model_name, model in models.items():
        # Preprocesar imagen
        processed_image = preprocess_image(image, model_name)

        # Realizar predicción
        pred = model.predict(processed_image, verbose=0)
        pred_class_idx = np.argmax(pred[0])
        pred_class = CLASS_NAMES[pred_class_idx]
        confidence = float(pred[0][pred_class_idx])

        predictions[model_name] = {
            'class': pred_class,
            'confidence': confidence,
            'probabilities': pred[0]
        }

    return predictions

def get_peru_time():
    """Obtiene la fecha y hora actual en zona horaria de Perú"""
    peru_tz = pytz.timezone('America/Lima')
    peru_time = datetime.now(peru_tz)
    return peru_time

def clean_text_for_pdf(text):
    """Limpia el texto eliminando caracteres especiales incompatibles con latin-1"""
    import unicodedata

    # Reemplazos específicos
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N',
        'ü': 'u', 'Ü': 'U',
        '°': ' grados', '–': '-', '—': '-',
        ''': "'", ''': "'", '"': '"', '"': '"',
        '€': 'EUR', '£': 'GBP', '¥': 'YEN',
        '©': '(c)', '®': '(R)', '™': '(TM)',
        # Emojis comunes por si quedan algunos
        '🌽': '[MAIZ]', '📊': '[GRAFICO]', '📋': '[INFO]',
        '🔍': '[BUSCAR]', '⚠️': '[ALERTA]', '✅': '[OK]',
        '❌': '[ERROR]', '🟢': '[VERDE]', '🔴': '[ROJO]',
        '💡': '[IDEA]', '📷': '[IMAGEN]', '🤖': '[ROBOT]'
    }

    # Aplicar reemplazos
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalizar y convertir a ASCII
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

def generate_pdf_report(image, predictions, uploaded_filename, consensus_reached, consensus_diagnosis, lang='es'):
    """Genera un reporte PDF optimizado sin espacios vacíos innecesarios."""
    from src.translation import t_lang, translate_class_lang
    
    def r_t(key):
        return t_lang(key, lang)
        
    peru_time = get_peru_time()

    # Limpiar texto de entrada
    uploaded_filename = clean_text_for_pdf(uploaded_filename)
    if consensus_diagnosis:
        translated_diagnosis = translate_class_lang(consensus_diagnosis, lang)
        consensus_diagnosis_cleaned = clean_text_for_pdf(translated_diagnosis)
    else:
        consensus_diagnosis_cleaned = ""

    # Usar el listado de nombres de clases traducidas para graficar y tabular
    translated_class_names = [clean_text_for_pdf(translate_class_lang(c, lang)) for c in ["Mancha gris", "Roña común", "Tizón del norte", "Sano"]]

    # Diccionario local de textos traducidos para el reporte PDF
    pdf_text_dict = {
        "es": {
            "title": "DIAGNOSTICO FITOSANITARIO - MAIZ",
            "subtitle": "Sistema de Deteccion Automatica de Enfermedades",
            "header": "REPORTE DE DIAGNÓSTICO FITOSANITARIO (IMÁGENES)",
            "page": "Pagina",
            "generated": "Generado el",
            "peru_time": "(Hora Peru)",
            "info_title": "INFORMACION DEL ANALISIS",
            "file": "Archivo",
            "datetime": "Fecha y hora",
            "models_used": "Modelos utilizados",
            "resolution": "Resolucion de procesamiento",
            "diag_title": "DIAGNOSTICO PRINCIPAL",
            "img_title": "IMAGEN ANALIZADA",
            "img_details": "Detalles de la imagen:",
            "orig_size": "Tamano original",
            "format": "Formato",
            "channels": "Canales de color",
            "det_title": "RESULTADOS DETALLADOS",
            "chart_title": "Predicciones del Modelo",
            "prob_ylabel": "Probabilidad",
            "model_card_title": "Resultado del Modelo",
            "pred_label": "Prediccion",
            "conf_label": "Confianza",
            "state_label": "Estado",
            "state_ok": "[OK] Saludable",
            "state_warn": "[!] Enfermedad detectada",
            "prob_per_class": "Probabilidades por clase:",
            "comp_title": "ANALISIS COMPARATIVO",
            "comp_summary": "Resumen de predicciones:",
            "comp_table_header": "Modelo                Prediccion           Confianza    Estado",
            "state_healthy_lbl": "[OK] Sana",
            "state_diseased_lbl": "[!] Enferma",
            "consensus_reached_title": "[OK] Consenso Alcanzado",
            "consensus_reached_body": "Los tres modelos coinciden en el diagnostico: {diagnosis}\nEsto indica alta confiabilidad en el resultado.\nNivel de acuerdo: 100% (3/3 modelos)",
            "no_consensus_title": "[!] Sin Consenso",
            "no_consensus_body_start": "Los modelos presentan diferentes diagnosticos:\n",
            "no_consensus_body_end": "Se recomienda analisis adicional para confirmar.",
            "rec_title": "RECOMENDACIONES",
            "rec_healthy": [
                "- Continuar con las practicas de manejo actuales",
                "- Realizar monitoreos preventivos regulares cada 7-10 dias",
                "- Mantener condiciones optimas de cultivo (riego, fertilizacion)",
                "- Implementar rotacion de cultivos para prevenir enfermedades",
                "- Vigilar plantas circundantes por posibles sintomas"
            ],
            "rec_diseased": [
                "- Consultar inmediatamente con un especialista en fitopatologia",
                "- Aislar las plantas afectadas si es posible",
                "- Implementar medidas de control especificas para la enfermedad",
                "- Monitorear la extension de la enfermedad en el cultivo",
                "- Considerar tratamientos preventivos en plantas cercanas",
                "- Documentar la evolucion con fotografias regulares",
                "- Revisar condiciones ambientales que favorecen la enfermedad"
            ],
            "rec_no_consensus": [
                "- Tomar una nueva imagen con mejor calidad e iluminacion",
                "- Asegurar que la hoja este bien centrada y enfocada",
                "- Consultar con un especialista para confirmacion visual",
                "- Realizar analisis de laboratorio si persisten sintomas",
                "- Considerar multiples muestras de diferentes partes de la planta"
            ],
            "disease_info_title": "INFORMACION ESPECIFICA",
            "disease_header": "Enfermedad",
            "symptoms_header": "Sintomas caracteristicos:",
            "conditions_header": "Condiciones favorables:",
            "treatments_header": "Estrategias de manejo:",
            "tech_title": "INFORMACION TECNICA",
            "tech_spec_header": "Especificaciones del sistema:",
            "tech_spec_list": [
                "- Modelos basados en transfer learning con redes neuronales convolucionales",
                "- Dataset de entrenamiento: PlantVillage Corn Leaf Disease",
                "- Arquitecturas: MobileNetV2, ResNet50, EfficientNetB0",
                "- Precision promedio en validacion: >95%",
                "- Resolucion de procesamiento: {size}x{size} pixeles",
                "- Preprocesamiento especifico por modelo aplicado",
                "- Analisis basado en caracteristicas visuales de la hoja"
            ],
            "disclaimer_title": "[!] IMPORTANTE - LIMITACIONES Y DISCLAIMER",
            "disclaimer_text": "- Este analisis automatizado debe ser validado por un profesional\n- La precision del diagnostico depende de la calidad de la imagen\n- Se recomienda tomar multiples muestras para mayor certeza\n- Este sistema es una herramienta de apoyo, no un sustituto del diagnostico profesional\n- En caso de dudas, consulte con un fitopatologo certificado\n- Los resultados pueden variar segun condiciones de iluminacion y enfoque",
            "system_info_title": "Informacion del sistema:",
            "system_info_name": "Sistema de Deteccion Automatica de Enfermedades en Maiz",
            "system_info_version": "Version: 2.0 | Fecha de generacion: {date}",
            "system_info_tech": "Desarrollado con tecnologia de Deep Learning",
            "diseases": {
                "Tizón del norte": {
                    "descripcion": "Enfermedad fungica causada por Exserohilum turcicum que afecta principalmente las hojas del maiz.",
                    "sintomas": [
                        "- Lesiones alargadas en forma de cigarro",
                        "- Color marron grisaceo con bordes definidos",
                        "- Pueden alcanzar varios centimetros de longitud",
                        "- Amarillamiento prematuro de hojas",
                        "- En casos severos, marchitez de la planta"
                    ],
                    "condiciones": "Favorecido por alta humedad (>90%) y temperaturas de 18-27C",
                    "tratamiento": [
                        "- Aplicacion de fungicidas especificos (azoles, estrobilurinas)",
                        "- Uso de variedades resistentes",
                        "- Rotacion de cultivos con especies no susceptibles",
                        "- Manejo de residuos de cosecha",
                        "- Espaciamiento adecuado para mejorar ventilacion"
                    ]
                },
                "Roña común": {
                    "descripcion": "Enfermedad fungica causada por Puccinia sorghi que produce pustulas caracteristicas en las hojas.",
                    "sintomas": [
                        "- Pustulas pequenas y circulares de color marron-rojizo",
                        "- Aparecen en ambas caras de la hoja",
                        "- Pueden coalescer formando areas grandes",
                        "- Amarillamiento prematuro del follaje",
                        "- Reduccion en el vigor de la planta"
                    ],
                    "condiciones": "Temperaturas moderadas (16-25C) y presencia de rocio matutino",
                    "tratamiento": [
                        "- Fungicidas preventivos antes de la aparicion de sintomas",
                        "- Variedades con genes de resistencia",
                        "- Eliminacion de hospederos alternativos",
                        "- Monitoreo temprano y control oportuno",
                        "- Aplicacion foliar de productos cupricos"
                    ]
                },
                "Mancha gris": {
                    "descripcion": "Enfermedad fungica causada por Cercospora zeae-maydis que produce manchas caracteristicas en las hojas.",
                    "sintomas": [
                        "- Manchas rectangulares de color gris a marron",
                        "- Delimitadas por las venas de las hojas",
                        "- Pueden desarrollar un halo amarillento",
                        "- Coalescencia causa muerte de tejido foliar",
                        "- Afecta principalmente hojas inferiores"
                    ],
                    "condiciones": "Alta humedad relativa y temperaturas calidas (25-30C)",
                    "tratamiento": [
                        "- Rotacion con cultivos no gramineas",
                        "- Aplicacion de fungicidas sistemicos",
                        "- Manejo de densidad de siembra",
                        "- Eliminacion de residuos infectados",
                        "- Mejoramiento de drenaje del suelo"
                    ]
                }
            }
        },
        "en": {
            "title": "MAIZE PHYTOSANITARY DIAGNOSIS",
            "subtitle": "Automatic Disease Detection System",
            "header": "PHYTOSANITARY DIAGNOSIS REPORT (IMAGES)",
            "page": "Page",
            "generated": "Generated on",
            "peru_time": "(Peru Time)",
            "info_title": "ANALYSIS INFORMATION",
            "file": "File",
            "datetime": "Date and time",
            "models_used": "Models used",
            "resolution": "Processing resolution",
            "diag_title": "PRIMARY DIAGNOSIS",
            "img_title": "ANALYZED IMAGE",
            "img_details": "Image details:",
            "orig_size": "Original size",
            "format": "Format",
            "channels": "Color channels",
            "det_title": "DETAILED RESULTS",
            "chart_title": "Model Predictions",
            "prob_ylabel": "Probability",
            "model_card_title": "Model Result",
            "pred_label": "Prediction",
            "conf_label": "Confidence",
            "state_label": "State",
            "state_ok": "[OK] Healthy",
            "state_warn": "[!] Disease detected",
            "prob_per_class": "Probabilities per class:",
            "comp_title": "COMPARATIVE ANALYSIS",
            "comp_summary": "Predictions summary:",
            "comp_table_header": "Model                Prediction           Confidence    State",
            "state_healthy_lbl": "[OK] Healthy",
            "state_diseased_lbl": "[!] Diseased",
            "consensus_reached_title": "[OK] Consensus Reached",
            "consensus_reached_body": "All three models agree on the diagnosis: {diagnosis}\nThis indicates high reliability in the result.\nAgreement level: 100% (3/3 models)",
            "no_consensus_title": "[!] No Consensus",
            "no_consensus_body_start": "Models present different diagnoses:\n",
            "no_consensus_body_end": "Additional analysis is recommended for confirmation.",
            "rec_title": "RECOMMENDATIONS",
            "rec_healthy": [
                "- Continue with current management practices",
                "- Perform regular preventive monitoring every 7-10 days",
                "- Maintain optimal crop conditions (irrigation, fertilization)",
                "- Implement crop rotation to prevent diseases",
                "- Monitor surrounding plants for potential symptoms"
            ],
            "rec_diseased": [
                "- Consult immediately with a phytopathology specialist",
                "- Isolate affected plants if possible",
                "- Implement specific disease control measures",
                "- Monitor the extension of the disease in the crop",
                "- Consider preventive treatments in nearby plants",
                "- Document the evolution with regular photographs",
                "- Review environmental conditions that favor the disease"
            ],
            "rec_no_consensus": [
                "- Take a new image with better quality and lighting",
                "- Ensure the leaf is well centered and focused",
                "- Consult with a specialist for visual confirmation",
                "- Perform laboratory analysis if symptoms persist",
                "- Consider multiple samples from different parts of the plant"
            ],
            "disease_info_title": "SPECIFIC INFORMATION",
            "disease_header": "Disease",
            "symptoms_header": "Characteristic symptoms:",
            "conditions_header": "Favorable conditions:",
            "treatments_header": "Management strategies:",
            "tech_title": "TECHNICAL INFORMATION",
            "tech_spec_header": "System specifications:",
            "tech_spec_list": [
                "- Models based on transfer learning with convolutional neural networks",
                "- Training dataset: PlantVillage Corn Leaf Disease",
                "- Architectures: MobileNetV2, ResNet50, EfficientNetB0",
                "- Average validation accuracy: >95%",
                "- Processing resolution: {size}x{size} pixels",
                "- Specific preprocessing applied per model",
                "- Analysis based on foliar visual features"
            ],
            "disclaimer_title": "[!] IMPORTANT - LIMITATIONS AND DISCLAIMER",
            "disclaimer_text": "- This automated analysis should be validated by a professional\n- The diagnostic accuracy depends on the quality of the image\n- It is recommended to take multiple samples for higher certainty\n- This system is a support tool, not a substitute for professional diagnosis\n- In case of doubt, consult a certified phytopathologist\n- Results may vary depending on lighting and focus conditions",
            "system_info_title": "System information:",
            "system_info_name": "Automatic Disease Detection System in Maize",
            "system_info_version": "Version: 2.0 | Generation date: {date}",
            "system_info_tech": "Developed with Deep Learning technology",
            "diseases": {
                "Tizón del norte": {
                    "descripcion": "Fungal disease caused by Exserohilum turcicum that mainly affects maize leaves.",
                    "sintomas": [
                        "- Elongated cigar-shaped lesions",
                        "- Grayish-brown color with defined borders",
                        "- Can reach several centimeters in length",
                        "- Premature yellowing of leaves",
                        "- In severe cases, wilting of the plant"
                    ],
                    "condiciones": "Favored by high humidity (>90%) and temperatures of 18-27C",
                    "tratamiento": [
                        "- Application of specific fungicides (azoles, strobilurins)",
                        "- Use of resistant varieties",
                        "- Crop rotation with non-susceptible species",
                        "- Crop residue management",
                        "- Proper spacing to improve ventilation"
                    ]
                },
                "Roña común": {
                    "descripcion": "Fungal disease caused by Puccinia sorghi that produces characteristic pustules on leaves.",
                    "sintomas": [
                        "- Small and circular reddish-brown pustules",
                        "- Appear on both sides of the leaf",
                        "- Can coalesce forming large areas",
                        "- Premature yellowing of foliage",
                        "- Reduction in plant vigor"
                    ],
                    "condiciones": "Moderate temperatures (16-25C) and presence of morning dew",
                    "tratamiento": [
                        "- Preventive fungicides before symptoms appear",
                        "- Varieties with resistance genes",
                        "- Elimination of alternative hosts",
                        "- Early monitoring and timely control",
                        "- Foliar application of copper products"
                    ]
                },
                "Mancha gris": {
                    "descripcion": "Fungal disease caused by Cercospora zeae-maydis that produces characteristic spots on leaves.",
                    "sintomas": [
                        "- Rectangular gray to brown spots",
                        "- Delimited by leaf veins",
                        "- Can develop a yellowish halo",
                        "- Coalescence causes death of foliar tissue",
                        "- Mainly affects lower leaves"
                    ],
                    "condiciones": "High relative humidity and warm temperatures (25-30C)",
                    "tratamiento": [
                        "- Rotation with non-grass crops",
                        "- Application of systemic fungicides",
                        "- Crop density management",
                        "- Elimination of infected residues",
                        "- Soil drainage improvement"
                    ]
                }
            }
        },
        "pt": {
            "title": "DIAGNOSTICO FITOSSANITARIO - MILHO",
            "subtitle": "Sistema de Deteccao Automatica de Doencas",
            "header": "RELATORIO DE DIAGNÓSTICO FITOSSANITÁRIO (IMAGENS)",
            "page": "Pagina",
            "generated": "Gerado em",
            "peru_time": "(Hora Peru)",
            "info_title": "INFORMACAO DA ANALISE",
            "file": "Arquivo",
            "datetime": "Data e hora",
            "models_used": "Modelos utilizados",
            "resolution": "Resolucao de processamento",
            "diag_title": "DIAGNOSTICO PRINCIPAL",
            "img_title": "IMAGEM ANALISADA",
            "img_details": "Detalhes da imagem:",
            "orig_size": "Tamanho original",
            "format": "Formato",
            "channels": "Canais de cor",
            "det_title": "RESULTADOS DETALHADOS",
            "chart_title": "Previsoes do Modelo",
            "prob_ylabel": "Probabilidade",
            "model_card_title": "Resultado do Modelo",
            "pred_label": "Previsao",
            "conf_label": "Confianca",
            "state_label": "Estado",
            "state_ok": "[OK] Saudavel",
            "state_warn": "[!] Doenca detectada",
            "prob_per_class": "Probabilidades por classe:",
            "comp_title": "ANALISE COMPARATIVA",
            "comp_summary": "Resumo das previsoes:",
            "comp_table_header": "Modelo                Previsao           Confianca    Estado",
            "state_healthy_lbl": "[OK] Saudavel",
            "state_diseased_lbl": "[!] Doente",
            "consensus_reached_title": "[OK] Consenso Alcancado",
            "consensus_reached_body": "Os tres modelos coincidem no diagnostico: {diagnosis}\nIsto indica alta confiabilidade no resultado.\nNivel de acordo: 100% (3/3 modelos)",
            "no_consensus_title": "[!] Sem Consenso",
            "no_consensus_body_start": "Os modelos apresentam diferentes diagnosticos:\n",
            "no_consensus_body_end": "Recomenda-se analise adicional para confirmar.",
            "rec_title": "RECOMENDACOES",
            "rec_healthy": [
                "- Continuar com as praticas de manejo atuais",
                "- Realizar monitoramentos preventivos regulares a cada 7-10 dias",
                "- Manter condicoes optimas de cultivo (irrigacao, fertilizacao)",
                "- Implementar rotacao de culturas para prevenir doencas",
                "- Vigiar plantas vizinhas por possiveis sintomas"
            ],
            "rec_diseased": [
                "- Consultar imediatamente com um especialista em fitopatologia",
                "- Isolar as plantas afetadas se for possivel",
                "- Implementar medidas de controle especificas para a doenca",
                "- Monitorar a extensao da doenca no cultivo",
                "- Considerar tratamentos preventivos em plantas proximas",
                "- Documentar a evolucao com fotografias regulares",
                "- Revisar condicoes ambientais que favorecem a doenca"
            ],
            "rec_no_consensus": [
                "- Tirar uma nova imagem com melhor qualidade e iluminacao",
                "- Assegurar que a folha esteja bem centrada e focada",
                "- Consultar com um especialista para confirmacao visual",
                "- Realizar analise de laboratorio se os sintomas persistirem",
                "- Considerar multiplas amostras de diferentes partes da planta"
            ],
            "disease_info_title": "INFORMACAO ESPECIFICA",
            "disease_header": "Doenca",
            "symptoms_header": "Sintomas caracteristicos:",
            "conditions_header": "Condicoes favoraveis:",
            "treatments_header": "Estrategias de manejo:",
            "tech_title": "INFORMACAO TECNICA",
            "tech_spec_header": "Especificacoes do sistema:",
            "tech_spec_list": [
                "- Modelos baseados em transfer learning com redes neurais convolucionais",
                "- Dataset de treinamento: PlantVillage Corn Leaf Disease",
                "- Arquiteturas: MobileNetV2, ResNet50, EfficientNetB0",
                "- Acuracia media de validacao: >95%",
                "- Resolucao de processamento: {size}x{size} pixels",
                "- Pre-processamento especifico aplicado por modelo",
                "- Analise baseada em caracteristicas visuais da folha"
            ],
            "disclaimer_title": "[!] IMPORTANTE - LIMITACOES E DISCLAIMER",
            "disclaimer_text": "- Esta analise automatizada deve ser validada por um profissional\n- A precisao do diagnostico depende da qualidade da imagem\n- Recomenda-se colher multiplas amostras para maior certeza\n- Este sistema e uma ferramenta de apoio, nao um substituto do diagnostico profissional\n- Em caso de duvida, consulte um fitopatologo certificado\n- Os resultados podem variar segundo condicoes de iluminacao e foco",
            "system_info_title": "Informacao do sistema:",
            "system_info_name": "Sistema de Deteccao Automatica de Doencas em Milho",
            "system_info_version": "Versao: 2.0 | Data de geracao: {date}",
            "system_info_tech": "Desenvolvido com tecnologia de Deep Learning",
            "diseases": {
                "Tizón del norte": {
                    "descripcion": "Doenca fungica causada por Exserohilum turcicum que afeta principalmente as folhas do milho.",
                    "sintomas": [
                        "- Lesoes alongadas em forma de charuto",
                        "- Cor marrom acinzentada com bordes definidos",
                        "- Podem atingir varios centimetros de comprimento",
                        "- Amarelecimento precoce das folhas",
                        "- Em casos graves, murcha da planta"
                    ],
                    "condiciones": "Favorecido por alta umidade (>90%) e temperaturas de 18-27C",
                    "tratamiento": [
                        "- Aplicacao de fungicidas especificos (azoes, estrobilurinas)",
                        "- Uso de variedades resistentes",
                        "- Rotacao de culturas com especies nao suscetiveis",
                        "- Manejo de residuos da colheita",
                        "- Espacamento adequado para melhorar a ventilacao"
                    ]
                },
                "Roña común": {
                    "descripcion": "Doenca fungica causada por Puccinia sorghi que produz pustulas caracteristicas nas folhas.",
                    "sintomas": [
                        "- Pustulas pequenas e circulares de cor marrom-avermelhada",
                        "- Aparecem em ambas as faces da folha",
                        "- Podem coalescer formando grandes areas",
                        "- Amarelecimento precoce da folhagem",
                        "- Reducao no vigor da planta"
                    ],
                    "condiciones": "Temperaturas moderadas (16-25C) e presenca de orvalho matinal",
                    "tratamiento": [
                        "- Fungicidas preventivos antes do surgimento dos sintomas",
                        "- Variedades com genes de resistencia",
                        "- Eliminacao de hospederos alternativos",
                        "- Monitoramento precoce e controle oportuno",
                        "- Aplicacao foliar de produtos cupricos"
                    ]
                },
                "Mancha gris": {
                    "descripcion": "Doenca fungica causada por Cercospora zeae-maydis que produz manchas caracteristicas nas folhas.",
                    "sintomas": [
                        "- Manchas retangulares de cor cinza a marrom",
                        "- Delimitadas pelas nervuras das folhas",
                        "- Podem desenvolver um halo amarelado",
                        "- Coalescencia causa morte do tecido foliar",
                        "- Afeta principalmente as folhas inferiores"
                    ],
                    "condiciones": "Alta umidade relativa e temperaturas quentes (25-30C)",
                    "tratamiento": [
                        "- Rotacao com culturas nao gramineas",
                        "- Aplicacao de fungicidas sistemicos",
                        "- Manejo da densidade de plantio",
                        "- Eliminacao de residuos infectados",
                        "- Melhoria da drenagem do solo"
                    ]
                }
            }
        }
    }

    lang_key = lang if lang in ["es", "en", "pt"] else "es"
    tx = pdf_text_dict[lang_key]

    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=15)

        def header(self):
            if self.page_no() == 1:
                # Portada/Primera pagina header grande
                self.set_font('Arial', 'B', 18)
                self.set_text_color(46, 139, 87)
                self.cell(0, 15, clean_text_for_pdf(tx["title"]), 0, 1, 'C')
                self.set_font('Arial', 'I', 11)
                self.set_text_color(100, 100, 100)
                self.cell(0, 8, clean_text_for_pdf(tx["subtitle"]), 0, 1, 'C')
                self.set_draw_color(46, 139, 87)
                self.line(10, 35, 200, 35)
                self.ln(10)
            else:
                # Paginas siguientes header compacto para ahorrar espacio
                self.set_font('Arial', 'B', 9)
                self.set_text_color(46, 139, 87)
                self.cell(0, 6, clean_text_for_pdf(tx["header"]), 0, 0, 'L')
                self.set_font('Arial', 'I', 8)
                self.set_text_color(128, 128, 128)
                self.cell(0, 6, f'{clean_text_for_pdf(tx["file"])}: {uploaded_filename}', 0, 1, 'R')
                self.set_draw_color(200, 200, 200)
                self.line(10, 17, 200, 17)
                self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            date_str = tx["generated"] + f" {peru_time.strftime('%Y-%m-%d %H:%M:%S')} " + tx["peru_time"]
            self.cell(0, 10, clean_text_for_pdf(f'{tx["page"]} {self.page_no()} | {date_str}'), 0, 0, 'C')

        def check_and_add_page(self, needed_height):
            # Agregar pagina si el elemento excede el limite
            if self.get_y() + needed_height > 265:
                self.add_page()

        def chapter_title(self, title, icon=""):
            self.check_and_add_page(25)
            self.ln(3)
            self.set_font('Arial', 'B', 14)
            self.set_text_color(46, 139, 87)
            self.cell(0, 10, f'{icon} {title}', 0, 1, 'L')
            self.set_draw_color(46, 139, 87)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def section_title(self, title, icon=""):
            self.check_and_add_page(15)
            self.ln(2)
            self.set_font('Arial', 'B', 11)
            self.set_text_color(70, 70, 70)
            self.cell(0, 7, f'{icon} {title}', 0, 1, 'L')
            self.ln(1)

        def normal_text(self, text, bold=False):
            self.check_and_add_page(8)
            self.set_font('Arial', 'B' if bold else '', 9.5)
            self.set_text_color(0, 0, 0)
            self.cell(0, 5.5, text, 0, 1, 'L')

        def info_box(self, title, content, bg_color=(240, 248, 255)):
            lines = content.split('\n')
            needed = len(lines) * 5 + 15
            self.check_and_add_page(needed)
            x, y = self.get_x(), self.get_y()
            self.set_fill_color(*bg_color)
            self.rect(x, y, 190, needed, 'F')

            self.set_font('Arial', 'B', 10.5)
            self.set_text_color(25, 25, 112)
            self.cell(0, 7, title, 0, 1, 'L')

            self.set_font('Arial', '', 9)
            self.set_text_color(0, 0, 0)
            for line in lines:
                if line.strip():
                    self.cell(0, 4.5, f"  {line.strip()}", 0, 1, 'L')
            self.ln(3)

        def add_consensus_result(self, consensus_reached, consensus_diagnosis):
            self.check_and_add_page(20)
            if consensus_reached:
                if consensus_diagnosis == "Sano":
                    bg_color = (212, 237, 218)
                    title = f"[OK] {tx['diag_title']}: {tx['state_healthy_lbl'].upper()}"
                else:
                    bg_color = (248, 215, 218)
                    title = f"[!] {tx['diag_title']}: {tx['state_diseased_lbl'].upper()} ({consensus_diagnosis_cleaned.upper()})"
            else:
                bg_color = (255, 243, 205)
                title = f"[?] {tx['diag_title']}: {tx['pred_consensus_warning'].upper()}"

            self.set_fill_color(*bg_color)
            self.rect(10, self.get_y(), 190, 12, 'F')

            self.set_font('Arial', 'B', 12)
            self.set_text_color(0, 0, 0)
            self.cell(0, 12, title, 0, 1, 'C')
            self.ln(4)

    pdf = PDF()
    pdf.add_page()

    # 1. INFORMACIÓN GENERAL
    pdf.chapter_title(clean_text_for_pdf(tx["info_title"]), "[INFO]")
    pdf.normal_text(clean_text_for_pdf(f"{tx['file']}: {uploaded_filename}"), bold=True)
    pdf.normal_text(clean_text_for_pdf(f"{tx['datetime']}: {peru_time.strftime('%Y-%m-%d %H:%M:%S')} {tx['peru_time']}"))
    pdf.normal_text(clean_text_for_pdf(f"{tx['models_used']}: MobileNetV2, ResNet50, EfficientNetB0"))
    pdf.normal_text(clean_text_for_pdf(f"{tx['resolution']}: {IMG_SIZE}x{IMG_SIZE} px"))

    # 2. DIAGNÓSTICO PRINCIPAL
    pdf.chapter_title(clean_text_for_pdf(tx["diag_title"]), "[DIAG]")
    pdf.add_consensus_result(consensus_reached, consensus_diagnosis)

    # 3. IMAGEN ANALIZADA
    pdf.chapter_title(clean_text_for_pdf(tx["img_title"]), "[IMG]")
    try:
        image_pil = Image.fromarray(image)
        temp_img_path = f"temp_analysis_img_{int(peru_time.timestamp())}.png"
        image_pil.save(temp_img_path, format='PNG')

        img_width = 80
        page_width = 190
        x_position = (page_width - img_width) / 2 + 10

        # Calcular altura real de la imagen según relación de aspecto
        img_w, img_h = image_pil.size
        aspect = img_h / img_w
        pdf_img_height = img_width * aspect

        pdf.check_and_add_page(pdf_img_height + 25)
        pdf.image(temp_img_path, x=x_position, w=img_width)
        pdf.ln(pdf_img_height + 3)

        pdf.section_title(clean_text_for_pdf(tx["img_details"]), "[i]")
        pdf.normal_text(clean_text_for_pdf(f"- {tx['orig_size']}: {image_pil.size[0]}x{image_pil.size[1]} px"))
        pdf.normal_text(clean_text_for_pdf(f"- {tx['format']}: {image_pil.format if hasattr(image_pil, 'format') else 'Unknown'}"))
        pdf.normal_text(clean_text_for_pdf(f"- {tx['channels']}: RGB"))

        try:
            os.remove(temp_img_path)
        except:
            pass
    except Exception as e:
        pdf.normal_text(clean_text_for_pdf(f"[Error al procesar la imagen: {e}]"))
        pdf.ln(5)

    # 4. RESULTADOS DETALLADOS POR MODELO
    pdf.chapter_title(clean_text_for_pdf(tx["det_title"]), "[MODELS]")

    temp_graph_paths = []
    try:
        for i, (model_name, pred) in enumerate(predictions.items()):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#2E8B57']
            bars = ax.bar(translated_class_names, pred['probabilities'], color=colors, alpha=0.8)
            ax.set_title(clean_text_for_pdf(f"{tx['chart_title']} {model_name}"), fontsize=11, fontweight='bold', pad=10)
            ax.set_ylabel(clean_text_for_pdf(tx["prob_ylabel"]), fontsize=9)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.2, axis='y')

            max_idx = np.argmax(pred['probabilities'])
            bars[max_idx].set_color('#2E8B57')
            bars[max_idx].set_alpha(1.0)

            for j, v in enumerate(pred['probabilities']):
                ax.text(j, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')

            plt.xticks(rotation=30, ha='right', fontsize=8)
            plt.tight_layout()

            temp_graph_path = f"temp_graph_{model_name}_{int(peru_time.timestamp())}.png"
            plt.savefig(temp_graph_path, dpi=120, bbox_inches='tight')
            temp_graph_paths.append(temp_graph_path)
            plt.close()

        # Añadir las gráficas y tablas
        for i, (model_name, pred) in enumerate(predictions.items()):
            pdf.section_title(clean_text_for_pdf(f"{r_t('pred_col_model')} {model_name}"), "[M]")
            confidence_level = "ALTA" if pred['confidence'] > 0.8 else "MEDIA" if pred['confidence'] > 0.6 else "BAJA"

            translated_pred_class = translate_class_lang(pred['class'], lang)
            status_text = tx["state_ok"] if pred['class'] == 'Sano' else tx["state_warn"]
            pdf.info_box(
                clean_text_for_pdf(f"{tx['model_card_title']} {model_name}"),
                f"{tx['pred_label']}: {clean_text_for_pdf(translated_pred_class)}\n"
                f"{tx['conf_label']}: {pred['confidence']:.2%} ({confidence_level})\n"
                f"{tx['state_label']}: {clean_text_for_pdf(status_text)}"
            )

            # Insertar gráfico con altura dinámica calculada
            chart_width = 130
            chart_height = chart_width * 0.58
            
            if i < len(temp_graph_paths) and os.path.exists(temp_graph_paths[i]):
                pdf.check_and_add_page(chart_height + 5)
                pdf.image(temp_graph_paths[i], x=40, w=chart_width)
                pdf.ln(chart_height + 2)

            pdf.section_title(clean_text_for_pdf(tx["prob_per_class"]), "[DATA]")
            for j, class_name in enumerate(["Mancha gris", "Roña común", "Tizón del norte", "Sano"]):
                prob = pred['probabilities'][j]
                marker = "=>" if j == np.argmax(pred['probabilities']) else "  "
                translated_cn = translate_class_lang(class_name, lang)
                pdf.normal_text(clean_text_for_pdf(f"{marker} {translated_cn}: {prob:.2%}"))
            pdf.ln(4)

    except Exception as e:
        pdf.normal_text(clean_text_for_pdf(f"Error generando gráficas: {e}"))
    finally:
        for temp_path in temp_graph_paths:
            try:
                os.remove(temp_path)
            except:
                pass

    # 5. ANÁLISIS COMPARATIVO
    pdf.chapter_title(clean_text_for_pdf(tx["comp_title"]), "[COMP]")
    pdf.section_title(clean_text_for_pdf(tx["comp_summary"]), "[SUM]")
    pdf.normal_text(clean_text_for_pdf(tx["comp_table_header"]))
    pdf.normal_text("-" * 65)

    for model_name, pred in predictions.items():
        status = tx["state_healthy_lbl"] if pred['class'] == 'Sano' else tx["state_diseased_lbl"]
        clean_class = clean_text_for_pdf(translate_class_lang(pred['class'], lang))
        line = f"{model_name:<15} {clean_class:<15} {pred['confidence']:>8.1%}    {status}"
        pdf.normal_text(clean_text_for_pdf(line))
    pdf.ln(4)

    if consensus_reached:
        consensus_reached_text = tx["consensus_reached_body"].format(diagnosis=consensus_diagnosis_cleaned)
        pdf.info_box(
            clean_text_for_pdf(tx["consensus_reached_title"]),
            clean_text_for_pdf(consensus_reached_text)
        )
    else:
        predictions_list = [translate_class_lang(pred['class'], lang) for pred in predictions.values()]
        unique_predictions = list(set(predictions_list))
        consensus_text = clean_text_for_pdf(tx["no_consensus_body_start"])
        for pred in unique_predictions:
            count = predictions_list.count(pred)
            clean_pred = clean_text_for_pdf(pred)
            consensus_text += f"- {clean_pred}: {count} modelo(s)\n"
        consensus_text += clean_text_for_pdf(tx["no_consensus_body_end"])
        pdf.info_box(clean_text_for_pdf(tx["no_consensus_title"]), consensus_text)

    # 6. RECOMENDACIONES
    pdf.chapter_title(clean_text_for_pdf(tx["rec_title"]), "[REC]")
    if consensus_reached:
        if consensus_diagnosis == "Sano":
            recommendations = tx["rec_healthy"]
        else:
            recommendations = tx["rec_diseased"]
    else:
        recommendations = tx["rec_no_consensus"]
        
    for rec in recommendations:
        pdf.normal_text(clean_text_for_pdf(rec))

    # 7. INFORMACIÓN SOBRE ENFERMEDADES
    if consensus_reached and consensus_diagnosis != "Sano":
        pdf.chapter_title(clean_text_for_pdf(tx["disease_info_title"]), "[DISEASE]")
        details_lang = tx["diseases"]
        if consensus_diagnosis in details_lang:
            details = details_lang[consensus_diagnosis]
            pdf.section_title(clean_text_for_pdf(f"{tx['disease_header']}: {consensus_diagnosis_cleaned}"), "[PATHOGEN]")
            pdf.normal_text(clean_text_for_pdf(details['descripcion']))
            pdf.ln(2)

            pdf.section_title(clean_text_for_pdf(tx["symptoms_header"]), "[SYMP]")
            for sintoma in details['sintomas']:
                pdf.normal_text(clean_text_for_pdf(sintoma))
            pdf.ln(2)

            pdf.section_title(clean_text_for_pdf(tx["conditions_header"]), "[ENV]")
            pdf.normal_text(clean_text_for_pdf(details['condiciones']))
            pdf.ln(2)

            pdf.section_title(clean_text_for_pdf(tx["treatments_header"]), "[TREAT]")
            for tratamiento in details['tratamiento']:
                pdf.normal_text(clean_text_for_pdf(tratamiento))

    # 8. INFORMACIÓN TÉCNICA Y DISCLAIMER
    pdf.chapter_title(clean_text_for_pdf(tx["tech_title"]), "[TECH]")
    pdf.section_title(clean_text_for_pdf(tx["tech_spec_header"]), "[SPEC]")
    
    tech_info_list = [clean_text_for_pdf(item.format(size=IMG_SIZE)) for item in tx["tech_spec_list"]]
    for tech in tech_info_list:
        pdf.normal_text(tech)

    pdf.ln(4)
    pdf.info_box(
        clean_text_for_pdf(tx["disclaimer_title"]),
        clean_text_for_pdf(tx["disclaimer_text"])
    )

    # 9. PIE DE PÁGINA
    pdf.section_title(clean_text_for_pdf(tx["system_info_title"]), "[SYS]")
    pdf.normal_text(clean_text_for_pdf(tx["system_info_name"]))
    pdf.normal_text(clean_text_for_pdf(tx["system_info_version"].format(date=peru_time.strftime('%Y-%m-%d %H:%M:%S'))))
    pdf.normal_text(clean_text_for_pdf(tx["system_info_tech"]))

    # Generar PDF final
    try:
        pdf_output = pdf.output()
        if isinstance(pdf_output, str):
            pdf_bytes = pdf_output.encode('latin-1')
        else:
            pdf_bytes = bytes(pdf_output)
    except Exception as e:
        temp_pdf_path = f"temp_report_{int(peru_time.timestamp())}.pdf"
        pdf.output(temp_pdf_path)
        with open(temp_pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        try:
            os.remove(temp_pdf_path)
        except:
            pass
    return bytes(pdf_bytes)
def plot_predictions(predictions):
    """Crea gráficos de las predicciones"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (model_name, pred) in enumerate(predictions.items()):
        ax = axes[idx]

        # Crear gráfico de barras
        bars = ax.bar(CLASS_NAMES, pred['probabilities'])
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Probabilidad')
        ax.set_ylim(0, 1)

        # Colorear la barra de la predicción más alta
        max_idx = np.argmax(pred['probabilities'])
        bars[max_idx].set_color('#2E8B57')

        # Rotar etiquetas del eje x
        ax.tick_params(axis='x', rotation=45)

        # Añadir valores en las barras
        for i, v in enumerate(pred['probabilities']):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig

def show_prediction_interface(models):
    """Muestra la interfaz de predicción"""
    # Función auxiliar para traducir clases de salida
    def translate_class(class_name):
        class_map = {
            "Mancha gris": "class_gray_spot",
            "Roña común": "class_common_rust",
            "Tizón del norte": "class_northern_blight",
            "Sano": "class_healthy"
        }
        return t(class_map.get(class_name, class_name))

    st.markdown(f"## {t('pred_upload_header')}")
    uploaded_file = st.file_uploader(
        t("pred_upload_label"),
        type=['png', 'jpg', 'jpeg'],
        help=t("pred_upload_help")
    )

    if uploaded_file is not None:
        # Mostrar imagen cargada
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption=t("pred_loaded_caption"), use_column_width=True)

            # Información de la imagen
            st.markdown(f"### {t('pred_info_header')}")
            st.write(f"**{t('pred_info_name')}:** {uploaded_file.name}")
            st.write(f"**{t('pred_info_size')}:** {image.size}")
            st.write(f"**{t('pred_info_format')}:** {image.format}")

        with col2:
            # Convertir a array numpy para procesamiento
            image_array = np.array(image.convert('RGB'))

            # Realizar predicciones
            st.markdown(f"## {t('pred_running')}")

            with st.spinner(t('pred_spinner')):
                predictions = predict_disease(image_array, models)

            # Mostrar resultados
            st.markdown(f"## {t('pred_results_header')}")

            # Crear tarjetas de resultados
            for model_name, pred in predictions.items():
                is_healthy = pred['class'] == 'Sano'
                card_class = "healthy" if is_healthy else "diseased"
                translated_cls = translate_class(pred['class'])

                st.markdown(f"""
                <div class="model-card">
                    <h3>🤖 {model_name}</h3>
                    <div class="prediction-result {card_class}">
                        {t('pred_col_prediction')}: {translated_cls} ({pred['confidence']:.2%} {t('pred_col_confidence').lower()})
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Gráficos de probabilidades
            st.markdown(f"## {t('pred_probs_header')}")
            fig = plot_predictions(predictions)
            st.pyplot(fig)

            # Tabla resumen
            st.markdown(f"## {t('pred_summary_header')}")
            summary_data = []
            for model_name, pred in predictions.items():
                translated_cls = translate_class(pred['class'])
                summary_data.append({
                    t('pred_col_model'): model_name,
                    t('pred_col_prediction'): translated_cls,
                    t('pred_col_confidence'): f"{pred['confidence']:.2%}",
                    t('pred_col_state'): t('state_healthy') if pred['class'] == 'Sano' else t('state_diseased')
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # Consenso de modelos
            st.markdown(f"## {t('pred_consensus_header')}")
            predictions_list = [pred['class'] for pred in predictions.values()]
            unique_predictions = list(set(predictions_list))

            consensus_reached = len(unique_predictions) == 1
            consensus_diagnosis = unique_predictions[0] if consensus_reached else None

            if consensus_reached:
                translated_diagnosis = translate_class(consensus_diagnosis)
                st.success(t("pred_consensus_success").format(diagnosis=translated_diagnosis))
            else:
                st.warning(t("pred_consensus_warning"))
                for pred in unique_predictions:
                    count = predictions_list.count(pred)
                    translated_pred = translate_class(pred)
                    st.write(f"- {translated_pred}: {t('pred_models_count').format(count=count)}")

            # Botón para generar reportes
            st.markdown(t("generate_reports_header"))

            col_rep1, col_rep2, col_rep3 = st.columns(3)

            with col_rep1:
                if st.button(t("pred_gen_pdf"), type="primary", use_container_width=True, key="btn_img_pdf"):
                    with st.spinner(t("verifying_pipeline")):
                        try:
                            pdf_bytes = generate_pdf_report(
                                image=image_array,
                                predictions=predictions,
                                uploaded_filename=uploaded_file.name,
                                consensus_reached=consensus_reached,
                                consensus_diagnosis=consensus_diagnosis,
                                lang=st.session_state.get('lang', 'es')
                            )
                            peru_time = get_peru_time()
                            timestamp = peru_time.strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                label=t("pred_download_pdf"),
                                data=pdf_bytes,
                                file_name=f"reporte_maiz_{timestamp}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            st.success(t("pred_pdf_ready"))
                        except Exception as e:
                            st.error(t("pred_err").format(err=str(e)))

            with col_rep2:
                if st.button(t("pred_gen_docx"), type="primary", use_container_width=True, key="btn_img_docx"):
                    with st.spinner(t("verifying_pipeline")):
                        try:
                            peru_time = get_peru_time()
                            timestamp = peru_time.strftime("%Y%m%d_%H%M%S")
                            filepath = f"reports/reporte_maiz_{timestamp}.docx"
                            generate_image_docx_report(
                                image=image_array,
                                predictions=predictions,
                                uploaded_filename=uploaded_file.name,
                                consensus_reached=consensus_reached,
                                consensus_diagnosis=consensus_diagnosis,
                                filepath=filepath,
                                lang=st.session_state.get('lang', 'es')
                            )
                            with open(filepath, "rb") as f:
                                docx_bytes = f.read()
                            st.download_button(
                                label=t("pred_download_docx"),
                                data=docx_bytes,
                                file_name=f"reporte_maiz_{timestamp}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True
                            )
                            st.success(t("pred_docx_ready"))
                        except Exception as e:
                            st.error(t("pred_err").format(err=str(e)))
 
            with col_rep3:
                if st.button(t("pred_gen_xlsx"), type="primary", use_container_width=True, key="btn_img_xlsx"):
                    with st.spinner(t("verifying_pipeline")):
                        try:
                            peru_time = get_peru_time()
                            timestamp = peru_time.strftime("%Y%m%d_%H%M%S")
                            filepath = f"reports/reporte_maiz_{timestamp}.xlsx"
                            generate_image_xlsx_report(
                                predictions=predictions,
                                uploaded_filename=uploaded_file.name,
                                consensus_reached=consensus_reached,
                                consensus_diagnosis=consensus_diagnosis,
                                filepath=filepath,
                                lang=st.session_state.get('lang', 'es')
                            )
                            with open(filepath, "rb") as f:
                                xlsx_bytes = f.read()
                            st.download_button(
                                label=t("pred_download_xlsx"),
                                data=xlsx_bytes,
                                file_name=f"reporte_maiz_{timestamp}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            st.success(t("pred_xlsx_ready"))
                        except Exception as e:
                            st.error(t("pred_err").format(err=str(e)))
                            st.info(t("install_libraries_info"))

            st.info(t("info_reports_included"))

            # Información adicional sobre el diagnóstico
            if consensus_reached:
                st.markdown("## 💡 Información sobre el Diagnóstico")

                if consensus_diagnosis == "Sano":
                    st.success("""
                    **Hoja Saludable Detectada**

                    La hoja analizada no presenta signos visibles de enfermedad.
                    Continúe con las prácticas de manejo actuales y mantenga
                    un monitoreo preventivo regular.
                    """)
                else:
                    disease_info = {
                        "Tizón del norte": {
                            "description": "Enfermedad fúngica que causa lesiones alargadas de color marrón.",
                            "recommendations": "Aplicar fungicidas, mejorar ventilación, evitar humedad excesiva."
                        },
                        "Roña común": {
                            "description": "Enfermedad fúngica que produce pústulas de color marrón-rojizo.",
                            "recommendations": "Usar variedades resistentes, aplicar fungicidas preventivos."
                        },
                        "Mancha gris": {
                            "description": "Enfermedad que causa manchas grises rectangulares en las hojas.",
                            "recommendations": "Rotación de cultivos, manejo de residuos, fungicidas específicos."
                        }
                    }

                    if consensus_diagnosis in disease_info:
                        info = disease_info[consensus_diagnosis]
                        st.warning(f"""
                        **{consensus_diagnosis} Detectado**

                        **Descripción:** {info['description']}

                        **Recomendaciones:** {info['recommendations']}

                        ⚠️ *Consulte con un especialista en fitopatología para confirmar el diagnóstico y obtener un plan de tratamiento específico.*
                        """)
            else:
                st.info("""
                **🔍 Análisis Adicional Recomendado**

                Los modelos no alcanzaron consenso. Esto puede deberse a:
                - Calidad de la imagen
                - Estadio temprano de la enfermedad
                - Condiciones de iluminación

                Recomendamos tomar una nueva fotografía con mejor iluminación
                o consultar con un especialista.
                """)

def show_training_reports():
    """Muestra los reportes de entrenamiento"""
    st.header("📊 Reportes de Entrenamiento")
    st.markdown("Visualización completa de todos los reportes generados durante el entrenamiento y evaluación de los modelos.")

    existing_files, reports_path = check_report_files()

    # Mostrar estado de archivos
    with st.expander("📁 Estado de Archivos de Reportes"):
        for file, exists in existing_files.items():
            if exists:
                st.success(f"✅ {file}")
            else:
                st.error(f"❌ {file} - No encontrado")

    # Sección de tiempos de entrenamiento
    st.subheader("⏱️ Tiempos de Entrenamiento")

    # Crear métricas de tiempo
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="🚀 MobileNetV2",
            value="46.97 min",
            delta="Más rápido",
            delta_color="normal"
        )
        st.caption("2,818 segundos total")

    with col2:
        st.metric(
            label="⚡ EfficientNetB0",
            value="55.61 min",
            delta="Moderado",
            delta_color="normal"
        )
        st.caption("3,337 segundos total")

    with col3:
        st.metric(
            label="🎯 ResNet50",
            value="162.8 min",
            delta="Más lento",
            delta_color="inverse"
        )
        st.caption("9,768 segundos total")

    # Gráfico de tiempos
    st.markdown("#### 📊 Comparación Visual de Tiempos")

    # Datos de tiempo
    time_data = {
        'Modelo': ['MobileNetV2', 'EfficientNetB0', 'ResNet50'],
        'Tiempo (min)': [46.97, 55.61, 162.8],
        'Eficiencia': ['Alta', 'Media-Alta', 'Baja']
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E8B57', '#FFA500', '#DC143C']  # Verde, naranja, rojo
    bars = ax.bar(time_data['Modelo'], time_data['Tiempo (min)'], color=colors, alpha=0.7)

    ax.set_ylabel('Tiempo de Entrenamiento (minutos)')
    ax.set_title('Tiempo de Entrenamiento por Modelo')
    ax.grid(True, alpha=0.3, axis='y')

    # Añadir valores en las barras
    for bar, tiempo in zip(bars, time_data['Tiempo (min)']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{tiempo:.1f} min', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    # Análisis de eficiencia
    st.markdown("""
    **📋 Análisis de Eficiencia:**
    - **MobileNetV2**: Entrenamiento más rápido, ideal para prototipado
    - **EfficientNetB0**: Buen balance tiempo/rendimiento
    - **ResNet50**: Entrenamiento más lento pero mayor precisión final
    """)

    # 1. Comparación General de Modelos
    st.subheader("🏆 Comparación General de Modelos")
    comparison_file = reports_path / "modelos_comparacion_completa.png"
    if comparison_file.exists():
        st.image(str(comparison_file), caption="Comparación de precisión y pérdida entre los tres modelos")
    else:
        st.error("Archivo de comparación no encontrado")

    # 2. Matrices de Confusión Combinadas
    st.subheader("🔍 Matrices de Confusión - Vista Comparativa")
    matrices_file = reports_path / "matrices_confusion_todos.png"
    if matrices_file.exists():
        st.image(str(matrices_file), caption="Matrices de confusión de los tres modelos lado a lado")
    else:
        st.error("Archivo de matrices combinadas no encontrado")

    # 3. Matrices Individuales
    st.subheader("🔎 Matrices de Confusión - Detalle Individual")

    matrix_files = {
        "MobileNetV2": "matriz_confusion_mobilenetv2.png",
        "CNN ResNet50": "matriz_confusion_resnet50.png",
        "CNN EfficientNetB0": "matriz_confusion_efficientnetb0.png"
    }

    cols = st.columns(3)
    for idx, (model_name, filename) in enumerate(matrix_files.items()):
        with cols[idx]:
            matrix_path = reports_path / filename
            if matrix_path.exists():
                st.image(str(matrix_path), caption=f"Matriz - {model_name}")
            else:
                st.error(f"Matriz de {model_name} no encontrada")

    # 4. Métricas Detalladas
    st.subheader("📈 Métricas Detalladas por Clase")
    metrics_file = reports_path / "metricas_detalladas_por_clase.png"
    if metrics_file.exists():
        st.image(str(metrics_file), caption="Análisis detallado de Precision, Recall y F1-Score por clase y modelo")
    else:
        st.error("Archivo de métricas detalladas no encontrado")

    # 4. Métricas Detalladas
    st.subheader("📈 Pruebas de McNemar")
    metrics_file = reports_path / "mcnemar_analysis.png"
    if metrics_file.exists():
        st.image(str(metrics_file), caption="Pruebas de mcnemar y tablas de contingencia")
    else:
        st.error("Archivo de analisis de mcnemar no encontrado")

    # 5. Reporte de Texto
    st.subheader("📄 Reporte Detallado en Texto")
    text_report_path = reports_path / "reporte_completo.txt"
    if text_report_path.exists():
        with open(text_report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        st.text_area("Reporte Completo", report_content, height=400)

        # Botón de descarga
        st.download_button(
            label="📥 Descargar Reporte Completo",
            data=report_content,
            file_name="reporte_maiz_completo.txt",
            mime="text/plain"
        )
    else:
        st.error("Archivo de reporte de texto no encontrado")

def show_model_comparison():
    """Muestra la comparación entre modelos"""
    lang = st.session_state.get('lang', 'es')
    lang_key = lang if lang in ['es', 'en', 'pt'] else 'es'
    
    # Translations dict
    tx = {
        'es': {
            'header': "🔬 Comparación de Modelos",
            'models_header': "### 🤖 Modelos Implementados",
            'models_desc': (
                "**MobileNetV2:**\n"
                "- Arquitectura optimizada para dispositivos móviles\n"
                "- Menos parámetros y mayor velocidad\n"
                "- Ideal para aplicaciones en tiempo real\n\n"
                "**ResNet50:**\n"
                "- Arquitectura con conexiones residuales\n"
                "- Excelente para tareas de clasificación complejas\n"
                "- Mayor precisión en datasets desafiantes\n\n"
                "**EfficientNetB0:**\n"
                "- Arquitectura optimizada para eficiencia\n"
                "- Balance entre precisión y velocidad\n"
                "- Escalamiento uniforme de ancho, profundidad y resolución"
            ),
            'table_title': "📊 Tabla Comparativa",
            'col_feature': "Característica",
            'feat_params': "Parámetros (aprox.)",
            'feat_time': "Tiempo de entrenamiento",
            'feat_speed': "Velocidad de inferencia",
            'feat_acc': "Precisión final",
            'feat_val_acc': "Val Accuracy final",
            'feat_mem': "Uso de memoria",
            'feat_best': "Mejor para",
            'val_very_fast': "Muy rápida",
            'val_fast': "Rápida",
            'val_moderate': "Moderada",
            'val_low': "Bajo",
            'val_high': "Alto",
            'val_mod': "Moderado",
            'val_mobile': "Aplicaciones móviles",
            'val_max': "Precisión máxima",
            'val_balance': "Balance eficiencia/precisión",
            'efficiency_title': "⏱️ Análisis de Eficiencia Temporal",
            'eff_col1': (
                "**🥇 Mejor eficiencia tiempo/precisión:**\n"
                "- **MobileNetV2**: Entrenamiento más rápido con buena precisión\n"
                "- Ideal para desarrollo iterativo rápido\n\n"
                "**🏆 Mejor precisión absoluta:**\n"
                "- **ResNet50**: Máxima precisión de validación (98.83%)\n"
                "- Tiempo considerable pero resultados superiores"
            ),
            'eff_col2': (
                "**⚖️ Mejor balance:**\n"
                "- **EfficientNetB0**: Buen balance tiempo/precisión\n"
                "- Precisión alta con tiempo moderado\n\n"
                "**📊 Ratio eficiencia:**\n"
                "- MobileNetV2: 33.2% precisión/minuto\n"
                "- EfficientNetB0: 17.7% precisión/minuto\n"
                "- ResNet50: 6.1% precisión/minuto"
            ),
            'plot_title_time': "📈 Tiempo de Entrenamiento vs Precisión",
            'plot_time_lbl': "Tiempo de Entrenamiento (minutos)",
            'plot_val_acc': "Validation Accuracy (%)",
            'plot_title_vs': "Tiempo vs Precisión de Validación",
            'plot_model_lbl': "Modelos",
            'plot_acc_lbl': "Accuracy (%)",
            'plot_title_comp': "Comparación de Precisiones",
            'plot_lbl_train': "Precisión de Entrenamiento",
            'plot_lbl_val': "Precisión de Validación"
        },
        'en': {
            'header': "🔬 Model Comparison",
            'models_header': "### 🤖 Implemented Models",
            'models_desc': (
                "**MobileNetV2:**\n"
                "- Architecture optimized for mobile devices\n"
                "- Fewer parameters and faster speed\n"
                "- Ideal for real-time applications\n\n"
                "**ResNet50:**\n"
                "- Architecture with residual connections\n"
                "- Excellent for complex classification tasks\n"
                "- Higher accuracy on challenging datasets\n\n"
                "**EfficientNetB0:**\n"
                "- Architecture optimized for efficiency\n"
                "- Balance between accuracy and speed\n"
                "- Uniform scaling of width, depth, and resolution"
            ),
            'table_title': "📊 Comparative Table",
            'col_feature': "Feature",
            'feat_params': "Parameters (approx.)",
            'feat_time': "Training Time",
            'feat_speed': "Inference Speed",
            'feat_acc': "Final Accuracy",
            'feat_val_acc': "Final Val Accuracy",
            'feat_mem': "Memory Usage",
            'feat_best': "Best for",
            'val_very_fast': "Very fast",
            'val_fast': "Fast",
            'val_moderate': "Moderate",
            'val_low': "Low",
            'val_high': "High",
            'val_mod': "Moderate",
            'val_mobile': "Mobile applications",
            'val_max': "Maximum precision",
            'val_balance': "Efficiency/accuracy balance",
            'efficiency_title': "⏱️ Temporal Efficiency Analysis",
            'eff_col1': (
                "**🥇 Best time/accuracy efficiency:**\n"
                "- **MobileNetV2**: Fastest training with good accuracy\n"
                "- Ideal for fast iterative development\n\n"
                "**🏆 Best absolute precision:**\n"
                "- **ResNet50**: Maximum validation accuracy (98.83%)\n"
                "- Significant time but superior results"
            ),
            'eff_col2': (
                "**⚖️ Best balance:**\n"
                "- **EfficientNetB0**: Good time/accuracy balance\n"
                "- High precision with moderate time\n\n"
                "**📊 Efficiency ratio:**\n"
                "- MobileNetV2: 33.2% accuracy/minute\n"
                "- EfficientNetB0: 17.7% accuracy/minute\n"
                "- ResNet50: 6.1% accuracy/minute"
            ),
            'plot_title_time': "📈 Training Time vs Accuracy",
            'plot_time_lbl': "Training Time (minutes)",
            'plot_val_acc': "Validation Accuracy (%)",
            'plot_title_vs': "Time vs Validation Accuracy",
            'plot_model_lbl': "Models",
            'plot_acc_lbl': "Accuracy (%)",
            'plot_title_comp': "Accuracy Comparison",
            'plot_lbl_train': "Training Accuracy",
            'plot_lbl_val': "Validation Accuracy"
        },
        'pt': {
            'header': "🔬 Comparação de Modelos",
            'models_header': "### 🤖 Modelos Implementados",
            'models_desc': (
                "**MobileNetV2:**\n"
                "- Arquitetura otimizada para dispositivos móveis\n"
                "- Menos parâmetros e maior velocidade\n"
                "- Ideal para aplicações em tempo real\n\n"
                "**ResNet50:**\n"
                "- Arquitetura com conexões residuais\n"
                "- Excelente para tarefas complexas de classificação\n"
                "- Maior acurácia em conjuntos de dados desafiadores\n\n"
                "**EfficientNetB0:**\n"
                "- Arquitetura otimizada para eficiência\n"
                "- Equilíbrio entre acurácia e velocidade\n"
                "- Escalonamento uniforme de largura, profundidade e resolução"
            ),
            'table_title': "📊 Tabela Comparativa",
            'col_feature': "Característica",
            'feat_params': "Parâmetros (aprox.)",
            'feat_time': "Tempo de Treinamento",
            'feat_speed': "Velocidade de Inferência",
            'feat_acc': "Acurácia final",
            'feat_val_acc': "Val Accuracy final",
            'feat_mem': "Uso de memória",
            'feat_best': "Melhor para",
            'val_very_fast': "Muito rápida",
            'val_fast': "Rápida",
            'val_moderate': "Moderada",
            'val_low': "Baixo",
            'val_high': "Alto",
            'val_mod': "Moderado",
            'val_mobile': "Aplicações móveis",
            'val_max': "Precisão máxima",
            'val_balance': "Equilíbrio eficiência/acurácia",
            'efficiency_title': "⏱️ Análise de Eficiência Temporal",
            'eff_col1': (
                "**🥇 Melhor eficiência tempo/acurácia:**\n"
                "- **MobileNetV2**: Treinamento mais rápido com boa acurácia\n"
                "- Ideal para desenvolvimento iterativo rápido\n\n"
                "**🏆 Melhor acurácia absoluta:**\n"
                "- **ResNet50**: Acurácia máxima de validação (98.83%)\n"
                "- Tempo considerável, mas resultados superiores"
            ),
            'eff_col2': (
                "**⚖️ Melhor equilíbrio:**\n"
                "- **EfficientNetB0**: Bom equilíbrio tempo/acurácia\n"
                "- Acurácia alta com tempo moderado\n\n"
                "**📊 Razão de eficiência:**\n"
                "- MobileNetV2: 33.2% acurácia/minuto\n"
                "- EfficientNetB0: 17.7% acurácia/minuto\n"
                "- ResNet50: 6.1% acurácia/minuto"
            ),
            'plot_title_time': "📈 Tempo de Treinamento vs Acurácia",
            'plot_time_lbl': "Tempo de Treinamento (minutos)",
            'plot_val_acc': "Validation Accuracy (%)",
            'plot_title_vs': "Tempo vs Acurácia de Validação",
            'plot_model_lbl': "Modelos",
            'plot_acc_lbl': "Acurácia (%)",
            'plot_title_comp': "Comparação de Acurácias",
            'plot_lbl_train': "Acurácia de Treinamento",
            'plot_lbl_val': "Acurácia de Validação"
        }
    }
    
    t_data = tx[lang_key]

    st.header(t_data['header'])
    st.markdown(t_data['models_header'])
    st.markdown(t_data['models_desc'])

    # Crear tabla comparativa con tiempos de entrenamiento
    st.subheader(t_data['table_title'])
    comparison_data = {
        t_data['col_feature']: [
            t_data['feat_params'],
            t_data['feat_time'],
            t_data['feat_speed'],
            t_data['feat_acc'],
            t_data['feat_val_acc'],
            t_data['feat_mem'],
            t_data['feat_best']
        ],
        "MobileNetV2": [
            "3.5M",
            "46.97 min (2,818 seg)",
            t_data['val_very_fast'],
            "99.21%",
            "93.52%",
            t_data['val_low'],
            t_data['val_mobile']
        ],
        "ResNet50": [
            "25M",
            "162.8 min (9,768 seg)",
            t_data['val_moderate'],
            "99.54%",
            "98.83%",
            t_data['val_high'],
            t_data['val_max']
        ],
        "EfficientNetB0": [
            "5.3M",
            "55.61 min (3,337 seg)",
            t_data['val_fast'],
            "98.42%",
            "98.19%",
            t_data['val_mod'],
            t_data['val_balance']
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # Análisis de rendimiento por tiempo
    st.subheader(t_data['efficiency_title'])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(t_data['eff_col1'])
    with col2:
        st.markdown(t_data['eff_col2'])

    # Gráfico de tiempo vs precisión
    st.subheader(t_data['plot_title_time'])

    # Datos para el gráfico
    models_data = {
        'Modelo': ['MobileNetV2', 'EfficientNetB0', 'ResNet50'],
        'Tiempo (minutos)': [46.97, 55.61, 162.8],
        'Val Accuracy (%)': [93.52, 98.19, 98.83],
        'Training Accuracy (%)': [99.21, 98.42, 99.54]
    }

    # Aplicar el tema actual antes de graficar
    apply_theme_to_plot(st.session_state.get('theme', 'Oscuro'))

    # Crear gráfico con matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Tiempo vs Val Accuracy
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax1.scatter(models_data['Tiempo (minutos)'], models_data['Val Accuracy (%)'],
               c=colors, s=200, alpha=0.7)
    ax1.set_xlabel(t_data['plot_time_lbl'])
    ax1.set_ylabel(t_data['plot_val_acc'])
    ax1.set_title(t_data['plot_title_vs'])
    ax1.grid(True, alpha=0.3)

    # Añadir etiquetas
    for i, model in enumerate(models_data['Modelo']):
        ax1.annotate(model,
                    (models_data['Tiempo (minutos)'][i], models_data['Val Accuracy (%)'][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Gráfico 2: Comparación de barras
    x = np.arange(len(models_data['Modelo']))
    width = 0.35

    ax2.bar(x - width/2, models_data['Training Accuracy (%)'], width,
           label=t_data['plot_lbl_train'], color='lightcoral', alpha=0.8)
    ax2.bar(x + width/2, models_data['Val Accuracy (%)'], width,
           label=t_data['plot_lbl_val'], color='skyblue', alpha=0.8)

    ax2.set_xlabel(t_data['plot_model_lbl'])
    ax2.set_ylabel(t_data['plot_acc_lbl'])
    ax2.set_title(t_data['plot_title_comp'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_data['Modelo'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

def check_login():
    """Valida credenciales e inyecta la pantalla de login con estilos cargados desde assets."""
    # Sincronizar el idioma desde el selector de widgets antes de renderizar las columnas
    if "login_lang_selector" in st.session_state:
        lang_choice = st.session_state.login_lang_selector
        if lang_choice == "Español":
            st.session_state.lang = "es"
        elif lang_choice == "English":
            st.session_state.lang = "en"
        elif lang_choice == "Português":
            st.session_state.lang = "pt"

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    if not st.session_state.authenticated:
        import base64
        
        # Cargar imagen de fondo en base64 de forma segura
        bg_image_base64 = ""
        img_path = "data/login_leaf_background.png"
        if os.path.exists(img_path):
            with open(img_path, "rb") as img_file:
                bg_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
        # Cargar estilos CSS desde el archivo externo assets/login_styles.css
        css_path = "assets/login_styles.css"
        if os.path.exists(css_path):
            with open(css_path, "r", encoding="utf-8") as f:
                login_css = f.read()
            # Inyectar base64 dinámico de la imagen
            login_css = login_css.replace("__BG_IMAGE_BASE64__", bg_image_base64)
            
            # Aplicar tema visual al login
            theme = st.session_state.get('theme', 'Oscuro')
            is_dark = theme in ['Oscuro', 'Dark', 'escuro', 'Escuro']
            if is_dark:
                login_css += """
                html, body, [data-testid="stAppViewContainer"] {
                    background-color: #0f172a !important;
                }
                div[data-testid="stHorizontalBlock"] {
                    background-color: #1e293b !important;
                    border: 1px solid rgba(255, 255, 255, 0.08) !important;
                }
                .login-right-form h2, .login-right-form p, label {
                    color: #f8fafc !important;
                }
                input[id^="login_username"], input[id^="login_password"] {
                    background-color: #1e293b !important;
                    color: #f8fafc !important;
                    border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
                }
                div.st-key-btn_login_info button, button.ewrlt5x2,
                div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div.element-container button {
                    background-color: #1e293b !important;
                    border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
                    color: #f8fafc !important;
                }
                div.st-key-btn_login_info button:hover, button.ewrlt5x2:hover,
                div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div.element-container button:hover {
                    background-color: rgba(255, 255, 255, 0.05) !important;
                    color: #f8fafc !important;
                    border-color: rgba(255, 255, 255, 0.25) !important;
                }
                """
            else:
                login_css += """
                html, body, [data-testid="stAppViewContainer"] {
                    background-color: #f8fafc !important;
                }
                div[data-testid="stHorizontalBlock"] {
                    background-color: #ffffff !important;
                    border: 1px solid #e2e8f0 !important;
                }
                .login-right-form h2, .login-right-form p, label {
                    color: #0f172a !important;
                }
                input[id^="login_username"], input[id^="login_password"] {
                    background-color: #f8fafc !important;
                    color: #0f172a !important;
                    border: 1.5px solid #cbd5e1 !important;
                }
                div.st-key-btn_login_info button, button.ewrlt5x2,
                div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div.element-container button {
                    background-color: #ffffff !important;
                    border: 1.5px solid #cbd5e1 !important;
                    color: #0f172a !important;
                }
                div.st-key-btn_login_info button:hover, button.ewrlt5x2:hover,
                div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div.element-container button:hover {
                    background-color: #f8fafc !important;
                    color: #0f172a !important;
                }
                """
            st.markdown(f"<style>{login_css}</style>", unsafe_allow_html=True)
            
            # Evitar traducción automática en el login
            meta_html = """
            <script>
                if (window.parent && window.parent.document) {
                    if (!window.parent.document.querySelector('meta[name="google"][content="notranslate"]')) {
                        const meta = window.parent.document.createElement('meta');
                        meta.name = "google";
                        meta.content = "notranslate";
                        window.parent.document.head.appendChild(meta);
                    }
                }
            </script>
            """
            st.components.v1.html(meta_html, height=0, width=0)
        else:
            # Fallback en caso de que no exista el archivo
            st.warning(t("login_styles_not_found"))
            
        col1, col2 = st.columns([45, 55])
        
        with col1:
            st.markdown(f"""<div style="height: 100%; display: flex; flex-direction: column; justify-content: space-between; font-family: 'Poppins', sans-serif;">
<div>
<!-- Logo Circular -->
<div style="width: 36px; height: 36px; border-radius: 50%; border: 1.5px solid #4ADE80; display: flex; align-items: center; justify-content: center; background-color: rgba(74, 222, 128, 0.1);">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4ADE80" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.58.1 8A7 7 0 0 1 11 20z" />
<path d="M19 2c-2.26 4.33-5.27 7.14-8 10" />
</svg>
</div>
<h1 style="color: #ffffff; font-size: 1.5rem; font-weight: 800; line-height: 1.25; margin-top: 1.2rem; margin-bottom: 0.5rem; font-family: 'Poppins', sans-serif; letter-spacing: -0.5px;">
{t('left_title_part1')}<br><span style="color: #4ADE80;">{t('left_title_part2')}</span><br>{t('left_title_part3')}
</h1>
<p style="color: #e2e8f0; font-size: 0.8rem; line-height: 1.35; max-width: 320px; font-weight: 300; margin-bottom: 0.3rem;">
{t('left_desc')}
</p>
</div>

<!-- Viewfinder Visor -->
<div class="viewfinder-container">
<div class="viewfinder-box">
<div class="laser-scanner"></div>
<div class="viewfinder-corner top-left"></div>
<div class="viewfinder-corner top-right"></div>
<div class="viewfinder-corner bottom-left"></div>
<div class="viewfinder-corner bottom-right"></div>
<div class="viewfinder-target">
<div class="viewfinder-circle-outer"></div>
<div class="viewfinder-circle-inner"></div>
<div class="viewfinder-crosshair-h"></div>
<div class="viewfinder-crosshair-v"></div>
</div>
</div>
</div>

<!-- Beneficios -->
<div style="background-color: rgba(255, 255, 255, 0.08); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); border-radius: 12px; padding: 0.8rem 1rem; border: 1px solid rgba(255, 255, 255, 0.12); margin-top: auto;">
<div class="benefit-item">
<div style="width: 24px; height: 24px; border-radius: 50%; background-color: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.3); display: flex; align-items: center; justify-content: center; margin-right: 0.55rem; flex-shrink: 0;">
<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#4ADE80" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
  <rect x="9" y="9" width="6" height="6"></rect>
  <line x1="9" y1="1" x2="9" y2="4"></line>
  <line x1="15" y1="1" x2="15" y2="4"></line>
  <line x1="9" y1="20" x2="9" y2="23"></line>
  <line x1="15" y1="20" x2="15" y2="23"></line>
  <line x1="20" y1="9" x2="23" y2="9"></line>
  <line x1="20" y1="15" x2="23" y2="15"></line>
  <line x1="1" y1="9" x2="4" y2="9"></line>
  <line x1="1" y1="15" x2="4" y2="15"></line>
</svg>
</div>
<div>
<h4 style="color: #ffffff; margin: 0; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.3px;">{t('benefit1_title')}</h4>
<p style="color: #cbd5e1; margin: 0; font-size: 0.72rem; font-weight: 400; line-height: 1.1;">{t('benefit1_desc')}</p>
</div>
</div>
<div class="benefit-item">
<div style="width: 24px; height: 24px; border-radius: 50%; background-color: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.3); display: flex; align-items: center; justify-content: center; margin-right: 0.55rem; flex-shrink: 0;">
<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#4ADE80" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
  <path d="m9 11 2 2 4-4"></path>
</svg>
</div>
<div>
<h4 style="color: #ffffff; margin: 0; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.3px;">{t('benefit2_title')}</h4>
<p style="color: #cbd5e1; margin: 0; font-size: 0.72rem; font-weight: 400; line-height: 1.1;">{t('benefit2_desc')}</p>
</div>
</div>
<div class="benefit-item">
<div style="width: 24px; height: 24px; border-radius: 50%; background-color: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.3); display: flex; align-items: center; justify-content: center; margin-right: 0.55rem; flex-shrink: 0;">
<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#4ADE80" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <line x1="18" y1="20" x2="18" y2="10"></line>
  <line x1="12" y1="20" x2="12" y2="4"></line>
  <line x1="6" y1="20" x2="6" y2="14"></line>
</svg>
</div>
<div>
<h4 style="color: #ffffff; margin: 0; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.3px;">{t('benefit3_title')}</h4>
<p style="color: #cbd5e1; margin: 0; font-size: 0.72rem; font-weight: 400; line-height: 1.1;">{t('benefit3_desc')}</p>
</div>
</div>
</div>
</div>""", unsafe_allow_html=True)
            
        with col2:
            # Selector de idioma discreto en la parte superior derecha de la tarjeta de Login
            lang_col1, lang_col2 = st.columns([3, 1.2])
            with lang_col2:
                lang_choice = st.selectbox(
                    "🌐", 
                    ["Español", "English", "Português"],
                    index=0 if st.session_state.get('lang', 'es') == 'es' else (1 if st.session_state.get('lang', 'es') == 'en' else 2),
                    key="login_lang_selector",
                    label_visibility="collapsed"
                )
                if lang_choice == "Español":
                    st.session_state.lang = "es"
                elif lang_choice == "English":
                    st.session_state.lang = "en"
                else:
                    st.session_state.lang = "pt"
            
            st.markdown(f"""<div class="login-right-form" style="text-align: center; margin-bottom: 0.8rem; font-family: 'Poppins', sans-serif;">
<div style="width: 38px; height: 38px; border-radius: 50%; border: 1.5px solid #22C55E; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.4rem auto; background-color: rgba(34, 197, 94, 0.05);">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1B5E20" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.58.1 8A7 7 0 0 1 11 20z" />
<path d="M19 2c-2.26 4.33-5.27 7.14-8 10" />
</svg>
</div>
<h2 style="color: #1E293B; font-size: 1.4rem; font-weight: 700; margin: 0; font-family: 'Poppins', sans-serif; letter-spacing: -0.3px;">{t('welcome')}</h2>
<p style="color: #64748B; font-size: 0.78rem; margin-top: 0.1rem; margin-bottom: 0; font-family: 'Poppins', sans-serif;">{t('login_desc')}</p>
</div>""", unsafe_allow_html=True)
            
            # Formulario
            username = st.text_input(t("email"), placeholder=t("email_placeholder"), key="login_username")
            password = st.text_input(t("password"), type="password", placeholder="******", key="login_password")
            
            st.markdown(f'<p class="forgot-link" style="text-align: right; margin: -5px 0 10px 0;"><a href="#">{t("forgot_pwd")}</a></p>', unsafe_allow_html=True)
            
            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            
            if st.button(t("login_btn"), type="primary", use_container_width=True, key="btn_login_submit"):
                if username in ["admin", "admin@maiz.com"] and password == "admin123":
                    with st.spinner(t("verifying_credentials")):
                        import time
                        time.sleep(0.65)
                    st.success(t("login_success"))
                    time.sleep(0.4)
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error(t("login_error"))
                    
            st.markdown("""<div class="login-divider">
<span class="divider-line"></span>
<span class="divider-text">o</span>
<span class="divider-line"></span>
</div>""", unsafe_allow_html=True)
            
            if st.button(t("more_info_btn"), use_container_width=True, key="btn_login_info"):
                st.info(t("automl_system_info"))
                
            st.markdown(f"""
            <div class="login-footer" style="text-align: center; color: #94A3B8; font-size: 0.65rem; margin-top: 1.2rem; font-family: 'Poppins', sans-serif; line-height: 1.5;">
                &copy; 2026 {t('title')}<br>
                <span style="font-weight: 600; color: #64748B;">Versión 1.0.0</span>
            </div>
            """, unsafe_allow_html=True)
            
        return False
    return True

def show_fitosanitario_panel():
    """Muestra el panel de diagnóstico fitosanitario por imágenes original."""
    # Navegación con tabs
    tab1, tab2, tab3 = st.tabs([t("tab_prediction"), t("tab_performance"), t("tab_comparison")])

    with tab1:
        st.markdown(t("model_desc_list"))

        # Cargar modelos
        st.markdown(t("loading_models"))
        models = load_models()

        if not models:
            st.error(t("models_loaded_error"))
        else:
            st.success(f"✅ {len(models)} {t('models_loaded_success')}")
            show_prediction_interface(models)

    with tab2:
        st.warning(t("tab_perf_warning"))
        show_training_reports()

    with tab3:
        show_model_comparison()

    # Sidebar con información
    st.sidebar.markdown(f"## {t('sb_app_info')}")
    st.sidebar.markdown(t('sb_detectable_classes'))

    st.sidebar.markdown(f"## {t('sb_instructions_title')}")
    st.sidebar.markdown(t('sb_instructions_body'))

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {t('sb_config')}")
    if st.sidebar.button(t("sb_btn_reload")):
        st.cache_resource.clear()
        st.rerun()

    if st.sidebar.button(t("sb_btn_verify")):
        existing_files, _ = check_report_files()
        files_found = sum(existing_files.values())
        total_files = len(existing_files)
        st.sidebar.success(f"{t('sb_files_found')}: {files_found}/{total_files}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(t('sb_notes_title'))
    st.sidebar.markdown(t('sb_notes_body'))

def show_automl_panel():
    """Muestra la plataforma AutoML tabular modular."""
    lang = st.session_state.get('lang', 'es')
    lang_key = lang if lang in ['es', 'en', 'pt'] else 'es'
    
    # Translations inside show_automl_panel
    am_tx = {
        'es': {
            'eda_stats': "### Estadísticos Descriptivos Globales",
            'eda_balance': "#### Balance y Distribución de Clases",
            'eda_corr': "#### Mapa de Calor de Correlación",
            'eda_dist': "#### Distribución de Variables por Clase",
            'eda_interpret': "#### 💡 Interpretación del EDA",
            'train_table': "### Tabla Comparativa de Rendimiento (Test Set)",
            'train_roc': "#### Curvas ROC Comparativas",
            'train_learning': "#### Curvas de Aprendizaje (Loss Evolution)",
            'train_cm': "#### Matrices de Confusión por Modelo",
            'train_cm_caption': "Matriz de Confusión - {name}",
            'train_interpret': "#### 💡 Interpretación de Modelado",
            'cv_title': "### Resultados de Validación Cruzada",
            'cv_model': "Modelo",
            'cv_mean_acc': "Precisión Media (Mean Accuracy)",
            'cv_std_acc': "Desviación Estándar (Std Accuracy)",
            'cv_mean_f1': "F1-Score Medio (Mean F1-Score)",
            'cv_std_f1': "Desviación Estándar F1 (Std F1-Score)",
            'cv_interpret': "#### 💡 Interpretación de Validación Cruzada",
            'tuning_title': "### Resultados de Optimización (Random Forest)",
            'tuning_params': "**Mejores Hiperparámetros:**",
            'tuning_before': "Precisión Antes del Tuning",
            'tuning_after': "Precisión Después del Tuning",
            'tuning_interpret': "#### 💡 Interpretación del Tuning",
            'stats_title': "### Resultados de las Pruebas Estadísticas",
            'stats_hypothesis': "**Prueba de Hipótesis Utilizada:**",
            'stats_interpret': "#### 💡 Interpretación Estadística Avanzada"
        },
        'en': {
            'eda_stats': "### Global Descriptive Statistics",
            'eda_balance': "#### Class Balance and Distribution",
            'eda_corr': "#### Correlation Heatmap",
            'eda_dist': "#### Feature Distribution by Class",
            'eda_interpret': "#### 💡 EDA Interpretation",
            'train_table': "### Performance Comparison Table (Test Set)",
            'train_roc': "#### Comparative ROC Curves",
            'train_learning': "#### Learning Curves (Loss Evolution)",
            'train_cm': "#### Confusion Matrices by Model",
            'train_cm_caption': "Confusion Matrix - {name}",
            'train_interpret': "#### 💡 Modeling Interpretation",
            'cv_title': "### Cross-Validation Results",
            'cv_model': "Model",
            'cv_mean_acc': "Mean Accuracy",
            'cv_std_acc': "Std Accuracy",
            'cv_mean_f1': "Mean F1-Score",
            'cv_std_f1': "Std F1-Score",
            'cv_interpret': "#### 💡 Cross-Validation Interpretation",
            'tuning_title': "### Optimization Results (Random Forest)",
            'tuning_params': "**Best Hyperparameters:**",
            'tuning_before': "Accuracy Before Tuning",
            'tuning_after': "Accuracy After Tuning",
            'tuning_interpret': "#### 💡 Tuning Interpretation",
            'stats_title': "### Statistical Test Results",
            'stats_hypothesis': "**Hypothesis Test Used:**",
            'stats_interpret': "#### 💡 Advanced Statistical Interpretation"
        },
        'pt': {
            'eda_stats': "### Estatísticas Descritivas Globais",
            'eda_balance': "#### Equilíbrio e Distribuição de Classes",
            'eda_corr': "#### Mapa de Calor de Correlação",
            'eda_dist': "#### Distribuição de Variáveis por Classe",
            'eda_interpret': "#### 💡 Interpretação do EDA",
            'train_table': "### Tabela Comparativa de Desempenho (Test Set)",
            'train_roc': "#### Curvas ROC Comparativas",
            'train_learning': "#### Curvas de Aprendizado (Loss Evolution)",
            'train_cm': "#### Matrizes de Confusão por Modelo",
            'train_cm_caption': "Matriz de Confusão - {name}",
            'train_interpret': "#### 💡 Interpretação de Modelagem",
            'cv_title': "### Resultados de Validação Cruzada",
            'cv_model': "Modelo",
            'cv_mean_acc': "Acurácia Média (Mean Accuracy)",
            'cv_std_acc': "Desvio Padrão (Std Accuracy)",
            'cv_mean_f1': "F1-Score Médio (Mean F1-Score)",
            'cv_std_f1': "Desvio Padrão F1 (Std F1-Score)",
            'cv_interpret': "#### 💡 Interpretação de Validação Cruzada",
            'tuning_title': "### Resultados de Otimização (Random Forest)",
            'tuning_params': "**Melhores Hiperparámetros:**",
            'tuning_before': "Acurácia Antes do Tuning",
            'tuning_after': "Acurácia Depois do Tuning",
            'tuning_interpret': "#### 💡 Interpretação do Tuning",
            'stats_title': "### Resultados dos Testes Estatísticos",
            'stats_hypothesis': "**Teste de Hipótese Utilizado:**",
            'stats_interpret': "#### 💡 Interpretação Estatística Avançada"
        }
    }
    
    t_am = am_tx[lang_key]

    st.markdown(t("automl_desc"))
    
    # 1. Cargar datos
    if 'automl_df' not in st.session_state:
        st.session_state.automl_df = None
        
    uploaded_file = st.file_uploader(t("upload_csv"), type=["csv"])
    
    if uploaded_file is not None:
        try:
            st.session_state.automl_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"{t('error_reading_file')}: {e}")
    else:
        if st.session_state.automl_df is None and os.path.exists("data/maize_crop_data.csv"):
            st.info(t("test_dataset_detected"))
            if st.button(t("btn_load_test"), use_container_width=True):
                st.session_state.automl_df = pd.read_csv("data/maize_crop_data.csv")
                st.rerun()
                
    df = st.session_state.automl_df
                
    if df is None:
        st.warning(t("upload_csv_warning"))
        return
        
    if st.button(t("clean_data_btn")):
        st.session_state.automl_df = None
        st.session_state.pipeline_executed = False
        st.rerun()
        
    # Mostrar vista previa
    st.markdown(t("preview_dataset"))
    st.dataframe(df.head(5), use_container_width=True)
    
    # Seleccionar la columna objetivo (Target)
    target_col = st.selectbox(t("select_target"), df.columns.tolist(), index=len(df.columns)-1)
    
    # Configuración de hiperparámetros
    st.markdown("---")
    st.markdown(t("experiment_config"))
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        cv_folds = st.slider(t("cross_val_label"), 3, 10, 5)
        split_ratio = st.slider(t("train_pct_label"), 60, 90, 80) / 100.0
    with col_c2:
        seed = st.number_input(t("seed_label"), value=42, step=1)
        alpha = st.selectbox(t("alpha_label"), [0.01, 0.05, 0.10], index=1)
        tuning_method = st.radio(t("search_method_label"), ["grid", "random"], index=0, horizontal=True)
        
    # Inicializar semillas
    set_seed(seed)
    
    # Botón de ejecución
    if st.button(t("btn_run_pipeline"), type="primary", use_container_width=True):
        with st.spinner(t("verifying_pipeline")):
            # 1. EDA
            df_cleaned, num_duplicates, imputed_nulls, outliers_detected = clean_data(df, target_col)
            df_eda, class_stats = get_descriptive_stats(df_cleaned, target_col)
            eda_charts = plot_eda_charts(df_cleaned, target_col, lang=st.session_state.get('lang', 'es'))
            eda_interpret = interpret_eda(df_cleaned, target_col, num_duplicates, imputed_nulls, outliers_detected, df_eda, lang=st.session_state.get('lang', 'es'))
            
            # 2. Entrenamiento
            results, X_train, X_test, y_train, y_test, classes = train_and_evaluate_all(
                df_cleaned, target_col, split_ratio=split_ratio, seed=seed
            )
            roc_chart, learning_chart = plot_training_charts(results, X_test, y_test, classes, lang=st.session_state.get('lang', 'es'))
            training_interpret = interpret_training(results, lang=st.session_state.get('lang', 'es'))
            
            # 3. CV
            cv_results = run_cross_validation(df_cleaned, target_col, cv_folds=cv_folds, seed=seed)
            cv_chart = plot_cv_dispersion(cv_results, lang=st.session_state.get('lang', 'es'))
            cv_interpret = interpret_cv(cv_results, lang=st.session_state.get('lang', 'es'))
            
            # 4. Tuning (Random Forest)
            tuning_results = run_hyperparameter_tuning(
                df_cleaned, target_col, method=tuning_method, seed=seed
            )
            tuning_interpret = interpret_tuning(tuning_results, lang=st.session_state.get('lang', 'es'))
            
            # 5. Stats - extraer predicciones del mejor clásico y mejor híbrido
            classic_names = ['Regresión Logística (Clásico)', 'Random Forest (Clásico)', 'Red Neuronal MLP (Clásico)']
            hybrid_names  = ['Híbrido Votación (RF+MLP)', 'Híbrido Stacking (Meta-GB)']
            best_classic_name = max(
                [n for n in classic_names if n in results],
                key=lambda n: results[n]['accuracy'],
                default=list(results.keys())[0]
            )
            best_hybrid_name = max(
                [n for n in hybrid_names if n in results],
                key=lambda n: results[n]['accuracy'],
                default=list(results.keys())[-1]
            )
            y_pred_classic = results[best_classic_name]['y_pred']
            y_pred_hybrid  = results[best_hybrid_name]['y_pred']
            
            stats_results = run_statistical_tests(
                cv_results, y_test, y_pred_classic, y_pred_hybrid,
                alpha=alpha, lang=st.session_state.get('lang', 'es')
            )
            stats_chart = stats_results.get('stats_chart', '')
            stats_interpret = interpret_stats(stats_results, alpha=alpha, lang=st.session_state.get('lang', 'es'))
            
            # Recopilar rutas de gráficos para el reporte
            image_paths = {
                'balance':       eda_charts.get('balance', ''),
                'correlation':   eda_charts.get('correlation', ''),
                'distributions': eda_charts.get('distributions', ''),
                'boxplots':      eda_charts.get('boxplots', ''),
                'roc':           roc_chart if isinstance(roc_chart, str) else '',
                'learning':      learning_chart if isinstance(learning_chart, str) else '',
                'cv':            cv_chart if isinstance(cv_chart, str) else '',
                'stats':         stats_chart if isinstance(stats_chart, str) else '',
            }
            
            # Construir dict de interpretaciones ANTES de los reportes
            interpretations = {
                'eda': eda_interpret,
                'training': training_interpret,
                'cv': cv_interpret,
                'tuning': tuning_interpret,
                'stats': stats_interpret
            }

            # Construir DataFrame de entrenamiento para reportes
            df_training_rep = pd.DataFrame({
                name: {
                    'Accuracy': res['accuracy'],
                    'Precision': res['precision'],
                    'Recall': res['recall'],
                    'F1-Score': res['f1-score'],
                    'AUC': res['auc'],
                    'Tiempo de Entrenamiento (s)': res['train_time'],
                    'Tiempo de Inferencia (s)': res['inference_time'],
                    'No. Parámetros': res['param_count'],
                    'Tamaño (KB)': res['model_size_kb'],
                }
                for name, res in results.items()
            }).T

            # Generar Reportes en PDF, Word y Excel
            xlsx_report = generate_xlsx_report(
                df_eda, df_training_rep, cv_results, tuning_results, stats_results,
                filepath="reports/reporte_automl.xlsx", lang=st.session_state.get('lang', 'es')
            )
            docx_report = generate_docx_report(
                df_eda, df_training_rep, cv_results, tuning_results, stats_results,
                interpretations, image_paths,
                filepath="reports/reporte_automl.docx", lang=st.session_state.get('lang', 'es')
            )
            pdf_report = generate_tabular_pdf_report(
                df_eda, df_training_rep, cv_results, tuning_results, stats_results,
                interpretations, image_paths,
                filepath="reports/reporte_automl.pdf", lang=st.session_state.get('lang', 'es')
            )
            
            st.session_state.pipeline_executed = True
            st.session_state.df_eda = df_eda
            st.session_state.df_training = pd.DataFrame({
                name: {
                    'Accuracy': f"{res['accuracy']:.4%}",
                    'Precision': f"{res['precision']:.4%}",
                    'Recall': f"{res['recall']:.4%}",
                    'F1-Score': f"{res['f1-score']:.4f}",
                    'AUC': f"{res['auc']:.4f}",
                    'Train Time (s)': f"{res['train_time']:.4f}",
                    'Params': res['param_count'],
                    'Size (KB)': f"{res['model_size_kb']:.1f}",
                }
                for name, res in results.items()
            }).T
            st.session_state.cv_results = cv_results
            st.session_state.tuning_results = tuning_results
            st.session_state.stats_results = stats_results
            st.session_state.interpretations = interpretations
            st.session_state.image_paths = image_paths
            st.session_state.xlsx_report = xlsx_report
            st.session_state.docx_report = docx_report
            st.session_state.pdf_report = pdf_report
            
            st.success(t("pipeline_completed_success"))

    # 4. Mostrar resultados guardados en session_state
    if st.session_state.get('pipeline_executed', False):
        st.markdown("---")
        st.markdown(t("generate_reports_header"))
        
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            with open(st.session_state.pdf_report, "rb") as f:
                st.download_button(t("download_pdf"), f.read(), file_name="reporte_fitosanitario_automl.pdf", mime="application/pdf", use_container_width=True)
        with col_d2:
            with open(st.session_state.docx_report, "rb") as f:
                st.download_button(t("download_docx"), f.read(), file_name="reporte_fitosanitario_automl.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        with col_d3:
            with open(st.session_state.xlsx_report, "rb") as f:
                st.download_button(t("download_xlsx"), f.read(), file_name="reporte_fitosanitario_automl.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                
        # Mostrar fases con tabs
        tab_eda, tab_train, tab_cv, tab_tuning, tab_stats = st.tabs([
            t("tab_eda"), 
            t("tab_train"), 
            t("tab_cv"), 
            t("tab_tuning"), 
            t("tab_stats")
        ])
        
        with tab_eda:
            st.markdown(t_am['eda_stats'])
            st.dataframe(st.session_state.df_eda, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(t_am['eda_balance'])
                st.image(st.session_state.image_paths['balance'])
            with col2:
                if st.session_state.image_paths.get('correlation'):
                    st.markdown(t_am['eda_corr'])
                    st.image(st.session_state.image_paths['correlation'])
                    
            st.markdown(t_am['eda_dist'])
            st.image(st.session_state.image_paths['distributions'])
            st.image(st.session_state.image_paths['boxplots'])
            
            st.markdown(t_am['eda_interpret'])
            st.info(st.session_state.interpretations['eda'])
            
        with tab_train:
            st.markdown(t_am['train_table'])
            st.dataframe(st.session_state.df_training, use_container_width=True)
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown(t_am['train_roc'])
                st.image(st.session_state.image_paths['roc'])
            with col_t2:
                st.markdown(t_am['train_learning'])
                st.image(st.session_state.image_paths['learning'])
                
            st.markdown(t_am['train_cm'])
            for model_name in st.session_state.df_training.index:
                filename = os.path.join("reports", f"confusion_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
                if os.path.exists(filename):
                    st.image(filename, caption=t_am['train_cm_caption'].format(name=model_name), width=400)
                    
            st.markdown(t_am['train_interpret'])
            st.info(st.session_state.interpretations['training'])
            
        with tab_cv:
            st.markdown(t_am['cv_title'])
            cv_disp_data = []
            for name, res in st.session_state.cv_results.items():
                cv_disp_data.append({
                    t_am['cv_model']: name,
                    t_am['cv_mean_acc']: f"{res['mean_accuracy']:.4%}",
                    t_am['cv_std_acc']: f"{res['std_accuracy']:.4%}",
                    t_am['cv_mean_f1']: f"{res['mean_f1']:.4f}",
                    t_am['cv_std_f1']: f"{res['std_f1']:.4f}"
                })
            st.dataframe(pd.DataFrame(cv_disp_data), use_container_width=True)
            st.image(st.session_state.image_paths['cv'])
            
            st.markdown(t_am['cv_interpret'])
            st.info(st.session_state.interpretations['cv'])
            
        with tab_tuning:
            st.markdown(t_am['tuning_title'])
            t_res = st.session_state.tuning_results
            st.markdown(f"{t_am['tuning_params']} `{t_res['best_params']}`")
            st.metric(t_am['tuning_before'], f"{t_res['accuracy_before']:.2%}")
            st.metric(t_am['tuning_after'], f"{t_res['accuracy_after']:.2%}", delta=f"{t_res['accuracy_after'] - t_res['accuracy_before']:+.2%}")
            
            st.markdown(t_am['tuning_interpret'])
            st.info(st.session_state.interpretations['tuning'])
            
        with tab_stats:
            st.markdown(t_am['stats_title'])
            s_res = st.session_state.stats_results
            st.markdown(f"{t_am['stats_hypothesis']} `{s_res['test_type']}`")
            st.image(st.session_state.image_paths['stats'])
            
            st.markdown(t_am['stats_interpret'])
            st.info(st.session_state.interpretations['stats'])

def main():
    # Validar credenciales
    if not check_login():
        return

    # Inicializar theme en st.session_state si no existe
    if 'theme' not in st.session_state:
        st.session_state.theme = 'Oscuro'
        
    inject_custom_css()

    # Selector de Idioma en el Sidebar para sincronización
    st.sidebar.markdown(t("language_selector_title"))
    language = st.sidebar.selectbox(
        t("language_selector_label"),
        ["Español", "English", "Português"],
        index=0 if st.session_state.get('lang', 'es') == 'es' else (1 if st.session_state.get('lang', 'es') == 'en' else 2),
        key="main_lang_selector"
    )
    if language == "Español":
        st.session_state.lang = "es"
    elif language == "English":
        st.session_state.lang = "en"
    else:
        st.session_state.lang = "pt"

    # Selector de Tema en el Sidebar
    st.sidebar.markdown(f"### {t('theme_label')}")
    theme_opts = [t('theme_dark'), t('theme_light')]
    theme_sel = st.sidebar.selectbox(
        t('theme_label'),
        theme_opts,
        index=0 if st.session_state.get('theme', 'Oscuro') == 'Oscuro' else 1,
        key="main_theme_selector",
        label_visibility="collapsed"
    )
    if theme_sel == t('theme_dark'):
        st.session_state.theme = 'Oscuro'
    else:
        st.session_state.theme = 'Claro'
        
    inject_custom_css()
    apply_theme_to_plot(st.session_state.theme)

    # Encabezado principal
    st.markdown(f'<h1 class="main-header">{t("title")}</h1>',
                unsafe_allow_html=True)

    # Selector de Modo en el Sidebar
    st.sidebar.markdown(t("panel_selector_title"))
    app_mode = st.sidebar.selectbox(
        t("panel_selector_label"),
        [t("fitosanitario_panel"), t("automl_panel")]
    )
    
    # Chatbot en el Sidebar
    st.sidebar.markdown("---")
    
    # Inicialización del chat
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    st.sidebar.markdown(f"### {t('cb_title')}")
    
    # Mostrar historial en un contenedor con scroll
    chat_container = st.sidebar.container(height=220)
    with chat_container:
        with st.chat_message("assistant"):
            st.write(t("cb_intro"))
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.write(msg['text'])
                
    # Callback para procesar chat
    def process_chat():
        query = st.session_state.get('cb_user_input', '')
        if query:
            st.session_state.chat_history.append({'role': 'user', 'text': query})
            bot_response = get_chatbot_response(query, lang=st.session_state.lang)
            st.session_state.chat_history.append({'role': 'assistant', 'text': bot_response})
            st.session_state.speech_text = bot_response
            st.session_state.speech_spoken = False
            # Limpiar caja de entrada
            st.session_state.cb_user_input = ""

    # Entrada de texto con callback
    user_query = st.sidebar.text_input(t("cb_placeholder"), key="cb_user_input", label_visibility="collapsed", on_change=process_chat)
    
    col_cb1, col_cb2 = st.sidebar.columns([3, 1])
    with col_cb1:
        st.button(t("cb_send"), use_container_width=True, key="btn_cb_send", on_click=process_chat)
                
    with col_cb2:
        # Renderizar componente de micrófono
        lang_code = {'es': 'es-ES', 'en': 'en-US', 'pt': 'pt-BR'}.get(st.session_state.lang, 'es-ES')
        mic_html = f"""
        <body style="margin:0; padding:0; background:transparent; overflow:hidden;">
        <button id="mic_btn" style="
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 50%;
            width: 38px;
            height: 38px;
            font-size: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
            transition: transform 0.2s;
            outline: none;
        " onclick="startRecognition()">🎙️</button>

        <script>
            function startRecognition() {{
                const btn = document.getElementById('mic_btn');
                btn.style.transform = 'scale(0.9)';
                btn.style.background = '#ef4444';
                
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (!SpeechRecognition) {{
                    alert("Speech recognition not supported in this browser.");
                    btn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
                    return;
                }}
                
                const recognition = new SpeechRecognition();
                recognition.lang = '{lang_code}';
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;
                
                recognition.onresult = (event) => {{
                    const text = event.results[0][0].transcript;
                    try {{
                        const parentDoc = window.parent.document;
                        const input = parentDoc.querySelector('section[data-testid="stSidebar"] input[type="text"]');
                        if (input) {{
                            input.value = text;
                            input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                            
                            setTimeout(() => {{
                                const buttons = Array.from(parentDoc.querySelectorAll('button'));
                                const sendBtn = buttons.find(b => b.textContent.includes('Envia') || b.textContent.includes('Send') || b.textContent.includes('Enviar'));
                                if (sendBtn) {{
                                    sendBtn.click();
                                }}
                            }}, 500);
                        }}
                    }} catch (e) {{
                        console.error("Parent document access failed:", e);
                    }}
                }};
                
                recognition.onspeechend = () => {{
                    recognition.stop();
                    btn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
                    btn.style.transform = 'scale(1)';
                }};
                
                recognition.onerror = (event) => {{
                    console.error("Speech recognition error:", event.error);
                    btn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
                    btn.style.transform = 'scale(1)';
                }};
                
                recognition.start();
            }}
        </script>
        </body>
        """
        st.components.v1.html(mic_html, height=38, width=38)

    # Speech synthesis player
    if 'speech_text' in st.session_state and not st.session_state.get('speech_spoken', True):
        clean_speech = st.session_state.speech_text.replace("*", "").replace("`", "").replace("#", "").replace("'", "\\'").replace("\n", " ")
        lang_code = {'es': 'es-ES', 'en': 'en-US', 'pt': 'pt-BR'}.get(st.session_state.lang, 'es-ES')
        tts_html = f"""
        <script>
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance('{clean_speech}');
                utterance.lang = '{lang_code}';
                setTimeout(() => {{
                    const voices = window.speechSynthesis.getVoices();
                    const voice = voices.find(v => v.lang.startsWith('{lang_code.split("-")[0]}'));
                    if (voice) utterance.voice = voice;
                    window.speechSynthesis.speak(utterance);
                }}, 200);
            }}
        </script>
        """
        st.components.v1.html(tts_html, height=0, width=0)
        st.session_state.speech_spoken = True

    st.sidebar.markdown("---")
    
    # Botón de cerrar sesión
    if st.sidebar.button(t("logout_btn"), use_container_width=True, key="btn_logout"):
        st.session_state.authenticated = False
        st.success(t("success_logout"))
        st.rerun()

    # Mapear modo
    if app_mode == t("fitosanitario_panel"):
        show_fitosanitario_panel()
    else:
        show_automl_panel()

    # Footer común
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        {t('footer_text')}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()