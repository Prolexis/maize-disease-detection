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
from src.config import set_seed
from src.eda import clean_data, get_descriptive_stats, interpret_eda, plot_eda_charts
from src.training import train_and_evaluate_all, plot_training_charts, interpret_training, save_best_model
from src.cross_validation import run_cross_validation, plot_cv_dispersion, interpret_cv
from src.tuning import run_hyperparameter_tuning, interpret_tuning
from src.stats_tests import run_statistical_tests, interpret_stats
from src.reporting import generate_xlsx_report, generate_docx_report, generate_tabular_pdf_report, generate_image_docx_report, generate_image_xlsx_report

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
                st.success(f"✅ Modelo {name} cargado exitosamente")
            except Exception as e:
                st.error(f"❌ Error cargando {name}: {str(e)}")
        else:
            st.warning(f"⚠️ No se encontró el archivo: {model_path}")

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
    """Preprocesa la imagen según el modelo"""
    # Redimensionar imagen
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Convertir a array y expandir dimensiones
    image_array = np.array(image_resized, dtype=np.float32)
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
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text


def generate_pdf_report(image, predictions, uploaded_filename, consensus_reached, consensus_diagnosis):
    """Genera un reporte PDF optimizado sin espacios vacíos innecesarios."""
    peru_time = get_peru_time()

    # Limpiar texto de entrada
    uploaded_filename = clean_text_for_pdf(uploaded_filename)
    if consensus_diagnosis:
        consensus_diagnosis = clean_text_for_pdf(consensus_diagnosis)

    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=15)

        def header(self):
            if self.page_no() == 1:
                # Portada/Primera pagina header grande
                self.set_font('Arial', 'B', 18)
                self.set_text_color(46, 139, 87)
                self.cell(0, 15, 'DIAGNOSTICO FITOSANITARIO - MAIZ', 0, 1, 'C')
                self.set_font('Arial', 'I', 11)
                self.set_text_color(100, 100, 100)
                self.cell(0, 8, 'Sistema de Deteccion Automatica de Enfermedades', 0, 1, 'C')
                self.set_draw_color(46, 139, 87)
                self.line(10, 35, 200, 35)
                self.ln(10)
            else:
                # Paginas siguientes header compacto para ahorrar espacio
                self.set_font('Arial', 'B', 9)
                self.set_text_color(46, 139, 87)
                self.cell(0, 6, 'REPORTE DE DIAGNÓSTICO FITOSANITARIO (IMÁGENES)', 0, 0, 'L')
                self.set_font('Arial', 'I', 8)
                self.set_text_color(128, 128, 128)
                self.cell(0, 6, f'Archivo: {uploaded_filename}', 0, 1, 'R')
                self.set_draw_color(200, 200, 200)
                self.line(10, 17, 200, 17)
                self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Pagina {self.page_no()} | Generado el {peru_time.strftime("%Y-%m-%d %H:%M:%S")} (Hora Peru)', 0, 0, 'C')

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
                    title = "[OK] DIAGNOSTICO: HOJA SALUDABLE"
                else:
                    bg_color = (248, 215, 218)
                    title = f"[!] DIAGNOSTICO: {consensus_diagnosis.upper()}"
            else:
                bg_color = (255, 243, 205)
                title = "[?] SIN CONSENSO ENTRE MODELOS"

            self.set_fill_color(*bg_color)
            self.rect(10, self.get_y(), 190, 12, 'F')

            self.set_font('Arial', 'B', 12)
            self.set_text_color(0, 0, 0)
            self.cell(0, 12, title, 0, 1, 'C')
            self.ln(4)

    pdf = PDF()
    pdf.add_page()

    # 1. INFORMACIÓN GENERAL
    pdf.chapter_title("INFORMACION DEL ANALISIS", "[INFO]")
    pdf.normal_text(f"Archivo: {uploaded_filename}", bold=True)
    pdf.normal_text(f"Fecha y hora: {peru_time.strftime('%Y-%m-%d %H:%M:%S')} (Hora Peru)")
    pdf.normal_text(f"Modelos utilizados: MobileNetV2, ResNet50, EfficientNetB0")
    pdf.normal_text(f"Resolucion de procesamiento: {IMG_SIZE}x{IMG_SIZE} pixeles")

    # 2. DIAGNÓSTICO PRINCIPAL
    pdf.chapter_title("DIAGNOSTICO PRINCIPAL", "[DIAG]")
    pdf.add_consensus_result(consensus_reached, consensus_diagnosis)

    # 3. IMAGEN ANALIZADA
    pdf.chapter_title("IMAGEN ANALIZADA", "[IMG]")
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

        pdf.section_title("Detalles de la imagen:", "[i]")
        pdf.normal_text(f"- Tamano original: {image_pil.size[0]}x{image_pil.size[1]} pixeles")
        pdf.normal_text(f"- Formato: {image_pil.format if hasattr(image_pil, 'format') else 'Unknown'}")
        pdf.normal_text(f"- Canales de color: RGB")

        try:
            os.remove(temp_img_path)
        except:
            pass
    except Exception as e:
        pdf.normal_text(f"[Error al procesar la imagen: {e}]")
        pdf.ln(5)

    # 4. RESULTADOS DETALLADOS POR MODELO (Se remueve add_page para flujo continuo)
    pdf.chapter_title("RESULTADOS DETALLADOS", "[MODELS]")

    temp_graph_paths = []
    try:
        for i, (model_name, pred) in enumerate(predictions.items()):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#2E8B57']
            bars = ax.bar(CLASS_NAMES, pred['probabilities'], color=colors, alpha=0.8)
            ax.set_title(f'Predicciones del Modelo {model_name}', fontsize=11, fontweight='bold', pad=10)
            ax.set_ylabel('Probabilidad', fontsize=9)
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
            pdf.section_title(f"Modelo {model_name}", "[M]")
            confidence_level = "ALTA" if pred['confidence'] > 0.8 else "MEDIA" if pred['confidence'] > 0.6 else "BAJA"

            pdf.info_box(
                f"Resultado del Modelo {model_name}",
                f"Prediccion: {clean_text_for_pdf(pred['class'])}\n"
                f"Confianza: {pred['confidence']:.2%} ({confidence_level})\n"
                f"Estado: {'[OK] Saludable' if pred['class'] == 'Sano' else '[!] Enfermedad detectada'}"
            )

            # Insertar gráfico con altura dinámica calculada (figsize 6x3.5 -> aspect ratio 3.5/6 = 0.58)
            chart_width = 130
            chart_height = chart_width * 0.58
            
            if i < len(temp_graph_paths) and os.path.exists(temp_graph_paths[i]):
                pdf.check_and_add_page(chart_height + 5)
                # Centrar gráfico
                pdf.image(temp_graph_paths[i], x=40, w=chart_width)
                pdf.ln(chart_height + 2)

            pdf.section_title("Probabilidades por clase:", "[DATA]")
            for j, class_name in enumerate(CLASS_NAMES):
                prob = pred['probabilities'][j]
                marker = "=>" if j == np.argmax(pred['probabilities']) else "  "
                pdf.normal_text(f"{marker} {clean_text_for_pdf(class_name)}: {prob:.2%}")
            pdf.ln(4)

    except Exception as e:
        pdf.normal_text(f"Error generando gráficas: {e}")
    finally:
        for temp_path in temp_graph_paths:
            try:
                os.remove(temp_path)
            except:
                pass

    # 5. ANÁLISIS COMPARATIVO
    pdf.chapter_title("ANALISIS COMPARATIVO", "[COMP]")
    pdf.section_title("Resumen de predicciones:", "[SUM]")
    pdf.normal_text("Modelo                Prediccion           Confianza    Estado")
    pdf.normal_text("-" * 65)

    for model_name, pred in predictions.items():
        status = "[OK] Sana" if pred['class'] == 'Sano' else "[!] Enferma"
        clean_class = clean_text_for_pdf(pred['class'])
        line = f"{model_name:<15} {clean_class:<15} {pred['confidence']:>8.1%}    {status}"
        pdf.normal_text(line)
    pdf.ln(4)

    if consensus_reached:
        pdf.info_box(
            "[OK] Consenso Alcanzado",
            f"Los tres modelos coinciden en el diagnostico: {consensus_diagnosis}\n"
            f"Esto indica alta confiabilidad en el resultado.\n"
            f"Nivel de acuerdo: 100% (3/3 modelos)"
        )
    else:
        predictions_list = [pred['class'] for pred in predictions.values()]
        unique_predictions = list(set(predictions_list))
        consensus_text = "Los modelos presentan diferentes diagnosticos:\n"
        for pred in unique_predictions:
            count = predictions_list.count(pred)
            clean_pred = clean_text_for_pdf(pred)
            consensus_text += f"- {clean_pred}: {count} modelo(s)\n"
        consensus_text += "Se recomienda analisis adicional para confirmar."
        pdf.info_box("[!] Sin Consenso", consensus_text)

    # 6. RECOMENDACIONES
    pdf.chapter_title("RECOMENDACIONES", "[REC]")
    if consensus_reached:
        if consensus_diagnosis == "Sano":
            recommendations = [
                "- Continuar con las practicas de manejo actuales",
                "- Realizar monitoreos preventivos regulares cada 7-10 dias",
                "- Mantener condiciones optimas de cultivo (riego, fertilizacion)",
                "- Implementar rotacion de cultivos para prevenir enfermedades",
                "- Vigilar plantas circundantes por posibles sintomas"
            ]
        else:
            recommendations = [
                "- Consultar inmediatamente con un especialista en fitopatologia",
                "- Aislar las plantas afectadas si es posible",
                "- Implementar medidas de control especificas para la enfermedad",
                "- Monitorear la extension de la enfermedad en el cultivo",
                "- Considerar tratamientos preventivos en plantas cercanas",
                "- Documentar la evolucion con fotografias regulares",
                "- Revisar condiciones ambientales que favorecen la enfermedad"
            ]
    else:
        recommendations = [
            "- Tomar una nueva imagen con mejor calidad e iluminacion",
            "- Asegurar que la hoja este bien centrada y enfocada",
            "- Consultar con un especialista para confirmacion visual",
            "- Realizar analisis de laboratorio si persisten sintomas",
            "- Considerar multiples muestras de diferentes partes de la planta"
        ]
    for rec in recommendations:
        pdf.normal_text(rec)

    # 7. INFORMACIÓN SOBRE ENFERMEDADES
    if consensus_reached and consensus_diagnosis != "Sano":
        pdf.chapter_title("INFORMACION ESPECIFICA", "[DISEASE]")
        disease_details = {
            "Tizon del norte": {
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
            "Rona común": {
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

        if consensus_diagnosis in disease_details:
            details = disease_details[consensus_diagnosis]
            pdf.section_title(f"Enfermedad: {consensus_diagnosis}", "[PATHOGEN]")
            pdf.normal_text(details['descripcion'])
            pdf.ln(2)

            pdf.section_title("Sintomas caracteristicos:", "[SYMP]")
            for sintoma in details['sintomas']:
                pdf.normal_text(sintoma)
            pdf.ln(2)

            pdf.section_title("Condiciones favorables:", "[ENV]")
            pdf.normal_text(details['condiciones'])
            pdf.ln(2)

            pdf.section_title("Estrategias de manejo:", "[TREAT]")
            for tratamiento in details['tratamiento']:
                pdf.normal_text(tratamiento)

    # 8. INFORMACIÓN TÉCNICA Y DISCLAIMER
    pdf.chapter_title("INFORMACION TECNICA", "[TECH]")
    pdf.section_title("Especificaciones del sistema:", "[SPEC]")
    tech_info = [
        "- Modelos basados en transfer learning con redes neuronales convolucionales",
        "- Dataset de entrenamiento: PlantVillage Corn Leaf Disease",
        "- Arquitecturas: MobileNetV2, ResNet50, EfficientNetB0",
        "- Precision promedio en validacion: >95%",
        "- Resolucion de procesamiento: 128x128 pixeles",
        "- Preprocesamiento especifico por modelo aplicado",
        "- Analisis basado en caracteristicas visuales de la hoja"
    ]
    for info in tech_info:
        pdf.normal_text(info)
    pdf.ln(4)

    pdf.info_box(
        "[!] IMPORTANTE - LIMITACIONES Y DISCLAIMER",
        "- Este analisis automatizado debe ser validado por un profesional\n"
        "- La precision del diagnostico depende de la calidad de la imagen\n"
        "- Se recomienda tomar multiples muestras para mayor certeza\n"
        "- Este sistema es una herramienta de apoyo, no un sustituto del diagnostico profesional\n"
        "- En caso de dudas, consulte con un fitopatologo certificado\n"
        "- Los resultados pueden variar segun condiciones de iluminacion y enfoque"
    )

    # 9. PIE DE PÁGINA
    pdf.section_title("Informacion del sistema:", "[SYS]")
    pdf.normal_text("Sistema de Deteccion Automatica de Enfermedades en Maiz")
    pdf.normal_text(f"Version: 2.0 | Fecha de generacion: {peru_time.strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.normal_text("Desarrollado con tecnologia de Deep Learning")

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
    st.markdown("## 📤 Cargar Imagen")
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de una hoja de maíz",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos soportados: PNG, JPG, JPEG"
    )

    if uploaded_file is not None:
        # Mostrar imagen cargada
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_column_width=True)

            # Información de la imagen
            st.markdown("### 📋 Información de la imagen")
            st.write(f"**Nombre:** {uploaded_file.name}")
            st.write(f"**Tamaño:** {image.size}")
            st.write(f"**Formato:** {image.format}")

        with col2:
            # Convertir a array numpy para procesamiento
            image_array = np.array(image.convert('RGB'))

            # Realizar predicciones
            st.markdown("## 🔍 Realizando Predicciones...")

            with st.spinner('Procesando imagen con los modelos...'):
                predictions = predict_disease(image_array, models)

            # Mostrar resultados
            st.markdown("## 📊 Resultados de Predicción")

            # Crear tarjetas de resultados
            for model_name, pred in predictions.items():
                is_healthy = pred['class'] == 'Sano'
                card_class = "healthy" if is_healthy else "diseased"

                st.markdown(f"""
                <div class="model-card">
                    <h3>🤖 {model_name}</h3>
                    <div class="prediction-result {card_class}">
                        Predicción: {pred['class']} ({pred['confidence']:.2%} confianza)
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Gráficos de probabilidades
            st.markdown("## 📈 Probabilidades por Modelo")
            fig = plot_predictions(predictions)
            st.pyplot(fig)

            # Tabla resumen
            st.markdown("## 📋 Resumen de Resultados")
            summary_data = []
            for model_name, pred in predictions.items():
                summary_data.append({
                    'Modelo': model_name,
                    'Predicción': pred['class'],
                    'Confianza': f"{pred['confidence']:.2%}",
                    'Estado': '🟢 Sana' if pred['class'] == 'Sano' else '🔴 Enferma'
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # Consenso de modelos
            st.markdown("## 🎯 Consenso de Modelos")
            predictions_list = [pred['class'] for pred in predictions.values()]
            unique_predictions = list(set(predictions_list))

            consensus_reached = len(unique_predictions) == 1
            consensus_diagnosis = unique_predictions[0] if consensus_reached else None

            if consensus_reached:
                st.success(f"✅ **Consenso alcanzado:** Todos los modelos predicen '{consensus_diagnosis}'")
            else:
                st.warning("⚠️ **Sin consenso:** Los modelos tienen predicciones diferentes")
                for pred in unique_predictions:
                    count = predictions_list.count(pred)
                    st.write(f"- {pred}: {count} modelo(s)")

            # Botón para generar reportes
            st.markdown("## 📥 Generar y Descargar Reportes")

            col_rep1, col_rep2, col_rep3 = st.columns(3)

            with col_rep1:
                if st.button("📥 Generar Reporte PDF", type="primary", use_container_width=True, key="btn_img_pdf"):
                    with st.spinner("Generando PDF..."):
                        try:
                            pdf_bytes = generate_pdf_report(
                                image=image_array,
                                predictions=predictions,
                                uploaded_filename=uploaded_file.name,
                                consensus_reached=consensus_reached,
                                consensus_diagnosis=consensus_diagnosis
                            )
                            peru_time = get_peru_time()
                            timestamp = peru_time.strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                label="📥 Descargar PDF",
                                data=pdf_bytes,
                                file_name=f"reporte_maiz_{timestamp}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            st.success("✅ PDF listo!")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

            with col_rep2:
                if st.button("📥 Generar Reporte Word (.docx)", type="primary", use_container_width=True, key="btn_img_docx"):
                    with st.spinner("Generando Word..."):
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
                                filepath=filepath
                            )
                            with open(filepath, "rb") as f:
                                docx_bytes = f.read()
                            st.download_button(
                                label="📥 Descargar Word",
                                data=docx_bytes,
                                file_name=f"reporte_maiz_{timestamp}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True
                            )
                            st.success("✅ Word listo!")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

            with col_rep3:
                if st.button("📥 Generar Reporte Excel (.xlsx)", type="primary", use_container_width=True, key="btn_img_xlsx"):
                    with st.spinner("Generando Excel..."):
                        try:
                            peru_time = get_peru_time()
                            timestamp = peru_time.strftime("%Y%m%d_%H%M%S")
                            filepath = f"reports/reporte_maiz_{timestamp}.xlsx"
                            generate_image_xlsx_report(
                                predictions=predictions,
                                uploaded_filename=uploaded_file.name,
                                consensus_reached=consensus_reached,
                                consensus_diagnosis=consensus_diagnosis,
                                filepath=filepath
                            )
                            with open(filepath, "rb") as f:
                                xlsx_bytes = f.read()
                            st.download_button(
                                label="📥 Descargar Excel",
                                data=xlsx_bytes,
                                file_name=f"reporte_maiz_{timestamp}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            st.success("✅ Excel listo!")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                            st.info("💡 Asegúrate de que las librerías estén instaladas: `pip install fpdf2 pytz`")

            st.info("""
            **📋 Los reportes descargables incluyen:**
            - Imagen analizada
            - Diagnóstico y confianza de cada modelo
            - Tabla de probabilidades por clase
            - Análisis de consenso y recomendaciones específicas
            """)

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
    st.header("🔬 Comparación de Modelos")

    # Información general sobre los modelos
    st.markdown("""
    ### 🤖 Modelos Implementados

    **MobileNetV2:**
    - Arquitectura optimizada para dispositivos móviles
    - Menos parámetros y mayor velocidad
    - Ideal para aplicaciones en tiempo real

    **ResNet50:**
    - Arquitectura con conexiones residuales
    - Excelente para tareas de clasificación complejas
    - Mayor precisión en datasets desafiantes

    **EfficientNetB0:**
    - Arquitectura optimizada para eficiencia
    - Balance entre precisión y velocidad
    - Escalamiento uniforme de ancho, profundidad y resolución
    """)

    # Crear tabla comparativa con tiempos de entrenamiento
    st.subheader("📊 Tabla Comparativa")
    comparison_data = {
        "Característica": [
            "Parámetros (aprox.)",
            "Tiempo de entrenamiento",
            "Velocidad de inferencia",
            "Precisión final",
            "Val Accuracy final",
            "Uso de memoria",
            "Mejor para"
        ],
        "MobileNetV2": [
            "3.5M",
            "46.97 min (2,818 seg)",
            "Muy rápida",
            "99.21%",
            "93.52%",
            "Bajo",
            "Aplicaciones móviles"
        ],
        "ResNet50": [
            "25M",
            "162.8 min (9,768 seg)",
            "Moderada",
            "99.54%",
            "98.83%",
            "Alto",
            "Precisión máxima"
        ],
        "EfficientNetB0": [
            "5.3M",
            "55.61 min (3,337 seg)",
            "Rápida",
            "98.42%",
            "98.19%",
            "Moderado",
            "Balance eficiencia/precisión"
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # Análisis de rendimiento por tiempo
    st.subheader("⏱️ Análisis de Eficiencia Temporal")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🥇 Mejor eficiencia tiempo/precisión:**
        - **MobileNetV2**: Entrenamiento más rápido con buena precisión
        - Ideal para desarrollo iterativo rápido

        **🏆 Mejor precisión absoluta:**
        - **ResNet50**: Máxima precisión de validación (98.83%)
        - Tiempo considerable pero resultados superiores
        """)

    with col2:
        st.markdown("""
        **⚖️ Mejor balance:**
        - **EfficientNetB0**: Buen balance tiempo/precisión
        - Precisión alta con tiempo moderado

        **📊 Ratio eficiencia:**
        - MobileNetV2: 33.2% precisión/minuto
        - EfficientNetB0: 17.7% precisión/minuto
        - ResNet50: 6.1% precisión/minuto
        """)

    # Gráfico de tiempo vs precisión
    st.subheader("📈 Tiempo de Entrenamiento vs Precisión")

    # Datos para el gráfico
    models_data = {
        'Modelo': ['MobileNetV2', 'EfficientNetB0', 'ResNet50'],
        'Tiempo (minutos)': [46.97, 55.61, 162.8],
        'Val Accuracy (%)': [93.52, 98.19, 98.83],
        'Training Accuracy (%)': [99.21, 98.42, 99.54]
    }

    # Crear gráfico con matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Tiempo vs Val Accuracy
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax1.scatter(models_data['Tiempo (minutos)'], models_data['Val Accuracy (%)'],
               c=colors, s=200, alpha=0.7)
    ax1.set_xlabel('Tiempo de Entrenamiento (minutos)')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Tiempo vs Precisión de Validación')
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
           label='Training Accuracy', color='lightcoral', alpha=0.8)
    ax2.bar(x + width/2, models_data['Val Accuracy (%)'], width,
           label='Validation Accuracy', color='skyblue', alpha=0.8)

    ax2.set_xlabel('Modelos')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Comparación de Precisiones')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_data['Modelo'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

def check_login():
    """Valida credenciales e inyecta la pantalla de login con estilos cargados desde assets."""
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
            st.markdown(f"<style>{login_css}</style>", unsafe_allow_html=True)
        else:
            # Fallback en caso de que no exista el archivo
            st.warning("⚠️ Estilos de inicio de sesión no encontrados en assets/login_styles.css")
            
        col1, col2 = st.columns([45, 55])
        
        with col1:
            st.markdown("""<div style="height: 100%; display: flex; flex-direction: column; justify-content: space-between; font-family: 'Poppins', sans-serif;">
<div>
<!-- Logo Circular -->
<div style="width: 36px; height: 36px; border-radius: 50%; border: 1.5px solid #4ADE80; display: flex; align-items: center; justify-content: center; background-color: rgba(74, 222, 128, 0.1);">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4ADE80" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.58.1 8A7 7 0 0 1 11 20z" />
<path d="M19 2c-2.26 4.33-5.27 7.14-8 10" />
</svg>
</div>
<h1 style="color: #ffffff; font-size: 1.5rem; font-weight: 800; line-height: 1.25; margin-top: 1.2rem; margin-bottom: 0.5rem; font-family: 'Poppins', sans-serif; letter-spacing: -0.5px;">
DETECTOR DE<br><span style="color: #4ADE80;">ENFERMEDADES</span><br>EN HOJAS DE MAÍZ
</h1>
<p style="color: #e2e8f0; font-size: 0.8rem; line-height: 1.35; max-width: 320px; font-weight: 300; margin-bottom: 0.3rem;">
Inteligencia Artificial para identificar enfermedades y proteger tu cultivo
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
<h4 style="color: #ffffff; margin: 0; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.3px;">IA AVANZADA</h4>
<p style="color: #cbd5e1; margin: 0; font-size: 0.72rem; font-weight: 400; line-height: 1.1;">Modelos entrenados para mayor precisión</p>
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
<h4 style="color: #ffffff; margin: 0; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.3px;">ANÁLISIS CONFIABLE</h4>
<p style="color: #cbd5e1; margin: 0; font-size: 0.72rem; font-weight: 400; line-height: 1.1;">Diagnósticos rápidos y precisos</p>
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
<h4 style="color: #ffffff; margin: 0; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.3px;">MEJORES DECISIONES</h4>
<p style="color: #cbd5e1; margin: 0; font-size: 0.72rem; font-weight: 400; line-height: 1.1;">Información clara para un cultivo más saludable</p>
</div>
</div>
</div>
</div>""", unsafe_allow_html=True)
            
        with col2:
            st.markdown("""<div class="login-right-form" style="text-align: center; margin-bottom: 0.8rem; font-family: 'Poppins', sans-serif;">
<div style="width: 38px; height: 38px; border-radius: 50%; border: 1.5px solid #22C55E; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.4rem auto; background-color: rgba(34, 197, 94, 0.05);">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1B5E20" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.58.1 8A7 7 0 0 1 11 20z" />
<path d="M19 2c-2.26 4.33-5.27 7.14-8 10" />
</svg>
</div>
<h2 style="color: #1E293B; font-size: 1.4rem; font-weight: 700; margin: 0; font-family: 'Poppins', sans-serif; letter-spacing: -0.3px;">Bienvenido</h2>
<p style="color: #64748B; font-size: 0.78rem; margin-top: 0.1rem; margin-bottom: 0; font-family: 'Poppins', sans-serif;">Inicie sesión para acceder al sistema.</p>
</div>""", unsafe_allow_html=True)
            
            # Formulario
            username = st.text_input("Correo electrónico", placeholder="ejemplo@correo.com", key="login_username")
            password = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña", key="login_password")
            
            st.markdown('<p class="forgot-link" style="text-align: right; margin: -5px 0 10px 0;"><a href="#">¿Olvidaste tu contraseña?</a></p>', unsafe_allow_html=True)
            
            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            
            if st.button("→  INICIAR SESIÓN", type="primary", use_container_width=True, key="btn_login_submit"):
                if username in ["admin", "admin@maiz.com"] and password == "admin123":
                    with st.spinner("Verificando credenciales..."):
                        import time
                        time.sleep(0.65)
                    st.success("✅ ¡Ingreso exitoso!")
                    time.sleep(0.4)
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Correo o contraseña incorrectos")
                    
            st.markdown("""<div class="login-divider">
<span class="divider-line"></span>
<span class="divider-text">o</span>
<span class="divider-line"></span>
</div>""", unsafe_allow_html=True)
            
            if st.button("Más información del sistema", use_container_width=True, key="btn_login_info"):
                st.info("Sistema inteligente de diagnóstico fitosanitario y AutoML para la optimización de cultivos de maíz.")
                
            st.markdown("""
            <div class="login-footer" style="text-align: center; color: #94A3B8; font-size: 0.65rem; margin-top: 1.2rem; font-family: 'Poppins', sans-serif; line-height: 1.5;">
                &copy; 2024 Detector de Enfermedades en Hojas de Maíz<br>
                <span style="font-weight: 600; color: #64748B;">Versión 1.0.0</span>
            </div>
            """, unsafe_allow_html=True)
            
        return False
    return True

def show_fitosanitario_panel():
    """Muestra el panel de diagnóstico fitosanitario por imágenes original."""
    # Navegación con tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Predicción", "📈 Rendimiento Histórico del Entrenamiento (Estático)", "🔬 Comparación de Modelos"])

    with tab1:
        st.markdown("""
        Esta aplicación utiliza tres modelos de deep learning para detectar enfermedades en hojas de maíz:
        - **MobileNetV2**: Modelo eficiente y rápido
        - **ResNet50**: Modelo robusto con conexiones residuales
        - **EfficientNetB0**: Modelo optimizado para eficiencia
        """)

        # Cargar modelos
        st.markdown("## 🤖 Cargando Modelos...")
        models = load_models()

        if not models:
            st.error("❌ No se pudieron cargar los modelos. Verifica las rutas.")
        else:
            st.success(f"✅ {len(models)} modelo(s) cargado(s) exitosamente")
            show_prediction_interface(models)

    with tab2:
        st.warning("ℹ️ **Nota de Uso:** Este panel muestra las métricas fijas y las curvas de aprendizaje del entrenamiento original de los modelos. No cambia al cargar una nueva imagen. Para generar e imprimir el reporte de diagnóstico de tu hoja cargada, utiliza los botones de descarga de PDF, Word o Excel al final de la pestaña **'Predicción'**.")
        show_training_reports()

    with tab3:
        show_model_comparison()

    # Sidebar con información
    st.sidebar.markdown("## 📊 Información de la App")
    st.sidebar.markdown("""
    **Clases detectables:**
    - 🟢 Sano
    - 🔴 Tizón del norte
    - 🟠 Roña común
    - 🟡 Mancha gris
    """)

    st.sidebar.markdown("## 📋 Instrucciones")
    st.sidebar.markdown("""
    1. Ve a la pestaña "Predicción"
    2. Carga una imagen de una hoja de maíz
    3. Espera a que se procese
    4. Revisa las predicciones de los tres modelos
    5. Analiza las probabilidades
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Configuración")
    if st.sidebar.button("🔄 Recargar Modelos"):
        st.cache_resource.clear()
        st.rerun()

    if st.sidebar.button("📁 Verificar Archivos"):
        existing_files, _ = check_report_files()
        files_found = sum(existing_files.values())
        total_files = len(existing_files)
        st.sidebar.success(f"Archivos encontrados: {files_found}/{total_files}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Notas")
    st.sidebar.markdown("""
    - Sube imágenes claras de hojas
    - Resolución recomendada: 224x224px
    - Formatos: JPG, PNG
    - Para mejores resultados, centra la hoja en la imagen
    """)

def show_automl_panel():
    """Muestra la plataforma AutoML tabular modular."""
    st.markdown("""
    Esta plataforma permite cargar un dataset tabular de variables agrícolas y ejecutar de extremo a extremo un pipeline de Machine Learning (3 modelos clásicos + 2 híbridos).
    """)
    
    # 1. Cargar datos
    if 'automl_df' not in st.session_state:
        st.session_state.automl_df = None
        
    uploaded_file = st.file_uploader("Cargar archivo de datos (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            st.session_state.automl_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
    else:
        if st.session_state.automl_df is None and os.path.exists("data/maize_crop_data.csv"):
            st.info("💡 Se ha detectado el conjunto de datos de prueba pregenerado `maize_crop_data.csv` en el servidor local.")
            if st.button("📊 Cargar Dataset de Prueba Fitosanitario", use_container_width=True):
                st.session_state.automl_df = pd.read_csv("data/maize_crop_data.csv")
                st.rerun()
                
    df = st.session_state.automl_df
                
    if df is None:
        st.warning("⚠️ Cargue un archivo CSV para iniciar el análisis.")
        return
        
    if st.button("🗑️ Limpiar Datos cargados"):
        st.session_state.automl_df = None
        st.session_state.pipeline_executed = False
        st.rerun()
        
    # Mostrar vista previa
    st.markdown("### 📋 Vista Previa del Dataset")
    st.dataframe(df.head(5), use_container_width=True)
    
    # Seleccionar la columna objetivo (Target)
    target_col = st.selectbox("Seleccione la variable objetivo (Target):", df.columns.tolist(), index=len(df.columns)-1)
    
    # Configuración de hiperparámetros
    st.markdown("---")
    st.markdown("### ⚙️ Configuración del Experimento")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        cv_folds = st.slider("Validación Cruzada (K-Folds)", 3, 10, 5)
        split_ratio = st.slider("Porcentaje de Entrenamiento (%)", 60, 90, 80) / 100.0
    with col_c2:
        seed = st.number_input("Semilla de Reproducibilidad", value=42, step=1)
        alpha = st.selectbox("Nivel de Significancia (α)", [0.01, 0.05, 0.10], index=1)
        tuning_method = st.radio("Método de Búsqueda de Hiperparámetros", ["grid", "random"], index=0, horizontal=True)
        
    # Inicializar semillas
    set_seed(seed)
    
    # Botón de ejecución
    if st.button("🚀 Ejecutar Pipeline de Machine Learning", type="primary", use_container_width=True):
        with st.spinner("Ejecutando Pipeline... por favor espere."):
            # 1. EDA
            df_cleaned, num_duplicates, imputed_nulls, outliers_detected = clean_data(df, target_col)
            df_eda, class_stats = get_descriptive_stats(df_cleaned, target_col)
            eda_charts = plot_eda_charts(df_cleaned, target_col)
            eda_interpret = interpret_eda(df_cleaned, target_col, num_duplicates, imputed_nulls, outliers_detected, df_eda)
            
            # 2. Entrenamiento
            results, X_train, X_test, y_train, y_test, classes = train_and_evaluate_all(
                df_cleaned, target_col, split_ratio=split_ratio, seed=seed
            )
            roc_chart, learning_chart = plot_training_charts(results, X_test, y_test, classes)
            training_interpret = interpret_training(results)
            
            # Convertir resultados de entrenamiento a dataframe
            training_metrics = []
            for model_name, res in results.items():
                training_metrics.append({
                    'Accuracy': res['accuracy'],
                    'Precision': res['precision'],
                    'Recall': res['recall'],
                    'F1-Score': res['f1-score'],
                    'AUC': res['auc'],
                    'Tiempo de Entrenamiento (s)': res['train_time'],
                    'Tiempo de Inferencia (s)': res['inference_time'],
                    'No. Parámetros': res['param_count'],
                    'Tamaño (KB)': res['model_size_kb']
                })
            df_training = pd.DataFrame(training_metrics, index=list(results.keys()))
            
            # Guardar mejor modelo
            best_model_path, meta_path = save_best_model(results)
            
            # 3. Cross Validation
            cv_results = run_cross_validation(df_cleaned, target_col, cv_folds=cv_folds, seed=seed)
            cv_chart = plot_cv_dispersion(cv_results)
            cv_interpret = interpret_cv(cv_results)
            
            # 4. Tuning
            tuning_results = run_hyperparameter_tuning(df_cleaned, target_col, method=tuning_method, seed=seed)
            tuning_interpret = interpret_tuning(tuning_results)
            
            # 5. Pruebas estadísticas
            # Obtener predicciones del mejor clásico y mejor híbrido
            classics_names = ['Regresión Logística (Clásico)', 'Random Forest (Clásico)', 'Red Neuronal MLP (Clásico)']
            hybrids_names = ['Híbrido Votación (RF+MLP)', 'Híbrido Stacking (Meta-GB)']
            best_classic = max(classics_names, key=lambda n: results[n]['accuracy'])
            best_hybrid = max(hybrids_names, key=lambda n: results[n]['accuracy'])
            y_pred_classic = results[best_classic]['y_pred']
            y_pred_hybrid = results[best_hybrid]['y_pred']
            
            stats_results = run_statistical_tests(cv_results, y_test, y_pred_classic, y_pred_hybrid, alpha=alpha)
            stats_interpret = interpret_stats(stats_results, alpha=alpha)
            
            # 6. Guardar interpretaciones agrupadas
            interpretations = {
                'eda': eda_interpret,
                'training': training_interpret,
                'cv': cv_interpret,
                'tuning': tuning_interpret,
                'stats': stats_interpret
            }
            
            # 7. Generar reportes
            image_paths = {
                'balance': eda_charts['balance'],
                'correlation': eda_charts.get('correlation', ''),
                'distributions': eda_charts['distributions'],
                'boxplots': eda_charts['boxplots'],
                'roc': roc_chart,
                'learning': learning_chart,
                'cv': cv_chart,
                'stats': stats_results['stats_chart']
            }
            
            xlsx_report = generate_xlsx_report(df_eda, df_training, cv_results, tuning_results, stats_results)
            docx_report = generate_docx_report(df_eda, df_training, cv_results, tuning_results, stats_results, interpretations, image_paths)
            pdf_report = generate_tabular_pdf_report(df_eda, df_training, cv_results, tuning_results, stats_results, interpretations, image_paths)
            
            # Almacenar en session_state
            st.session_state.pipeline_executed = True
            st.session_state.df_eda = df_eda
            st.session_state.df_training = df_training
            st.session_state.cv_results = cv_results
            st.session_state.tuning_results = tuning_results
            st.session_state.stats_results = stats_results
            st.session_state.interpretations = interpretations
            st.session_state.image_paths = image_paths
            st.session_state.xlsx_report = xlsx_report
            st.session_state.docx_report = docx_report
            st.session_state.pdf_report = pdf_report
            
            st.success("✅ ¡Pipeline completado con éxito! Revisa los resultados abajo.")

    # 4. Mostrar resultados guardados en session_state
    if st.session_state.get('pipeline_executed', False):
        st.markdown("---")
        st.markdown("## 📥 Descarga de Reportes Integrales")
        
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            with open(st.session_state.pdf_report, "rb") as f:
                st.download_button("📥 Descargar Reporte PDF", f.read(), file_name="reporte_fitosanitario_automl.pdf", mime="application/pdf", use_container_width=True)
        with col_d2:
            with open(st.session_state.docx_report, "rb") as f:
                st.download_button("📥 Descargar Reporte Word (.docx)", f.read(), file_name="reporte_fitosanitario_automl.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        with col_d3:
            with open(st.session_state.xlsx_report, "rb") as f:
                st.download_button("📥 Descargar Reporte Excel (.xlsx)", f.read(), file_name="reporte_fitosanitario_automl.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                
        # Mostrar fases con tabs
        tab_eda, tab_train, tab_cv, tab_tuning, tab_stats = st.tabs([
            "🔍 Análisis Exploratorio (EDA)", 
            "🤖 Modelado y Entrenamiento", 
            "🔁 Validación Cruzada", 
            "🔧 Tuning de Hiperparámetros", 
            "🔬 Pruebas Estadísticas"
        ])
        
        with tab_eda:
            st.markdown("### Estadísticos Descriptivos Globales")
            st.dataframe(st.session_state.df_eda, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Balance y Distribución de Clases")
                st.image(st.session_state.image_paths['balance'])
            with col2:
                if st.session_state.image_paths.get('correlation'):
                    st.markdown("#### Mapa de Calor de Correlación")
                    st.image(st.session_state.image_paths['correlation'])
                    
            st.markdown("#### Distribución de Variables por Clase")
            st.image(st.session_state.image_paths['distributions'])
            st.image(st.session_state.image_paths['boxplots'])
            
            st.markdown("#### 💡 Interpretación del EDA")
            st.info(st.session_state.interpretations['eda'])
            
        with tab_train:
            st.markdown("### Tabla Comparativa de Rendimiento (Test Set)")
            st.dataframe(st.session_state.df_training, use_container_width=True)
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("#### Curvas ROC Comparativas")
                st.image(st.session_state.image_paths['roc'])
            with col_t2:
                st.markdown("#### Curvas de Aprendizaje (Loss Evolution)")
                st.image(st.session_state.image_paths['learning'])
                
            st.markdown("#### Matrices de Confusión por Modelo")
            for model_name in st.session_state.df_training.index:
                filename = os.path.join("reports", f"confusion_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
                if os.path.exists(filename):
                    st.image(filename, caption=f"Matriz de Confusión - {model_name}", width=400)
                    
            st.markdown("#### 💡 Interpretación de Modelado")
            st.info(st.session_state.interpretations['training'])
            
        with tab_cv:
            st.markdown("### Resultados de Validación Cruzada")
            cv_disp_data = []
            for name, res in st.session_state.cv_results.items():
                cv_disp_data.append({
                    'Modelo': name,
                    'Mean Accuracy': f"{res['mean_accuracy']:.4%}",
                    'Std Accuracy': f"{res['std_accuracy']:.4%}",
                    'Mean F1-Score': f"{res['mean_f1']:.4f}",
                    'Std F1-Score': f"{res['std_f1']:.4f}"
                })
            st.dataframe(pd.DataFrame(cv_disp_data), use_container_width=True)
            st.image(st.session_state.image_paths['cv'])
            
            st.markdown("#### 💡 Interpretación de Validación Cruzada")
            st.info(st.session_state.interpretations['cv'])
            
        with tab_tuning:
            st.markdown("### Resultados de Optimización (Random Forest)")
            t_res = st.session_state.tuning_results
            st.markdown(f"**Mejores Hiperparámetros:** `{t_res['best_params']}`")
            st.metric("Precisión Antes del Tuning", f"{t_res['accuracy_before']:.2%}")
            st.metric("Precisión Después del Tuning", f"{t_res['accuracy_after']:.2%}", delta=f"{t_res['accuracy_after'] - t_res['accuracy_before']:+.2%}")
            
            st.markdown("#### 💡 Interpretación del Tuning")
            st.info(st.session_state.interpretations['tuning'])
            
        with tab_stats:
            st.markdown("### Resultados de las Pruebas Estadísticas")
            s_res = st.session_state.stats_results
            st.markdown(f"**Prueba de Hipótesis Utilizada:** `{s_res['test_type']}`")
            st.image(st.session_state.image_paths['stats'])
            
            st.markdown("#### 💡 Interpretación Estadística Avanzada")
            st.info(st.session_state.interpretations['stats'])

def main():
    # Validar credenciales
    if not check_login():
        return

    # Encabezado principal
    st.markdown('<h1 class="main-header">🌽 Detector de Enfermedades en Hojas de Maíz</h1>',
                unsafe_allow_html=True)

    # Selector de Modo en el Sidebar
    st.sidebar.markdown("# 🗺️ Selector de Panel")
    app_mode = st.sidebar.selectbox(
        "Seleccione el panel de trabajo:",
        ["🌽 Diagnóstico Fitosanitario (Imágenes)", "📊 AutoML Pipeline Analítico (Tabular)"]
    )
    
    st.sidebar.markdown("---")
    
    # Botón de cerrar sesión
    if st.sidebar.button("🔒 Cerrar Sesión", use_container_width=True, key="btn_logout"):
        st.session_state.authenticated = False
        st.success("Sesión cerrada correctamente.")
        st.rerun()

    if app_mode == "🌽 Diagnóstico Fitosanitario (Imágenes)":
        show_fitosanitario_panel()
    else:
        show_automl_panel()

    # Footer común
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        🌽 Desarrollado para el análisis de enfermedades en cultivos de maíz<br>
        Integración de diagnóstico de imágenes y pipeline analítico de Machine Learning
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()