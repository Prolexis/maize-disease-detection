# -*- coding: utf-8 -*-
import streamlit as st
import os
import base64

# Configuración de página
st.set_page_config(
    page_title="Iniciar Sesión - Detector Inteligente de Maíz",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inicializar sesión
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def show_login():
    # Cargar imagen de fondo en base64 de forma segura
    bg_image_base64 = ""
    img_path = "data/login_leaf_background.png"
    if os.path.exists(img_path):
        with open(img_path, "rb") as img_file:
            bg_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    # Estilos CSS Estilo 2026 adaptados con Google Font Poppins y sin Scroll
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Ocultar elementos nativos de Streamlit */
    #MainMenu {{display: none !important;}}
    footer {{display: none !important;}}
    header {{display: none !important;}}
    div[data-testid="stSidebar"] {{display: none !important;}}
    div[data-testid="stHeader"] {{display: none !important;}}
    div[data-testid="stToolbar"] {{display: none !important;}}
    
    /* Reset de fuentes e inyección de fondo */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Poppins', sans-serif !important;
        background-color: #F8FAFC !important;
        height: 100vh !important;
        overflow: hidden !important;
    }}
    
    section.main {{
        padding: 0 !important;
        height: 100vh !important;
        overflow: hidden !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    
    /* Centrar contenedor vertical y horizontalmente */
    div.block-container {{
        max-width: 1000px !important;
        padding: 0 !important;
        height: auto !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        overflow: hidden !important;
    }}
    
    /* Contenedor tipo tarjeta dividida */
    div[data-testid="stHorizontalBlock"] {{
        background-color: #ffffff;
        border-radius: 24px;
        box-shadow: 0 20px 40px rgba(30, 41, 59, 0.08);
        overflow: hidden;
        border: 1px solid #e2e8f0;
        display: flex !important;
        flex-direction: row !important;
        height: 540px !important; /* Altura optimizada */
    }}
    
    /* Columnas a 50% */
    div[data-testid="stHorizontalBlock"] > div {{
        padding: 0 !important;
        margin: 0 !important;
        flex: 1 !important;
    }}
    
    /* Columna izquierda (45%) */
    div[data-testid="stHorizontalBlock"] > div:first-child {{
        background-image: linear-gradient(rgba(20, 83, 45, 0.85), rgba(20, 83, 45, 0.95)), url('data:image/png;base64,{bg_image_base64}');
        background-size: cover;
        background-position: center;
        padding: 2.5rem 2.2rem !important;
        color: #ffffff;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100% !important;
        max-width: 45% !important;
        min-width: 45% !important;
        flex: 0 0 45% !important;
    }}
    
    /* Columna derecha (55%) */
    div[data-testid="stHorizontalBlock"] > div:last-child {{
        background-color: #ffffff;
        padding: 2.5rem 3.5rem !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100% !important;
        max-width: 55% !important;
        min-width: 55% !important;
        flex: 0 0 55% !important;
    }}
    
    /* Visor de escaneo en la imagen (Viewfinder) de menor tamaño */
    .viewfinder-container {{
        position: relative;
        width: 100%;
        height: 140px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }}
    .viewfinder-box {{
        position: relative;
        width: 100px;
        height: 100px;
    }}
    .viewfinder-corner {{
        position: absolute;
        width: 16px;
        height: 16px;
        border-color: #16A34A;
        border-style: solid;
        border-width: 0;
        filter: drop-shadow(0 0 3px #16A34A);
    }}
    .top-left {{
        top: 0;
        left: 0;
        border-top-width: 3px;
        border-left-width: 3px;
        border-top-left-radius: 6px;
    }}
    .top-right {{
        top: 0;
        right: 0;
        border-top-width: 3px;
        border-right-width: 3px;
        border-top-right-radius: 6px;
    }}
    .bottom-left {{
        bottom: 0;
        left: 0;
        border-bottom-width: 3px;
        border-left-width: 3px;
        border-bottom-left-radius: 6px;
    }}
    .bottom-right {{
        bottom: 0;
        right: 0;
        border-bottom-width: 3px;
        border-right-width: 3px;
        border-bottom-right-radius: 6px;
    }}
    .viewfinder-target {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .viewfinder-circle-outer {{
        position: absolute;
        width: 22px;
        height: 22px;
        border: 1.5px solid rgba(22, 163, 74, 0.6);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }}
    .viewfinder-circle-inner {{
        position: absolute;
        width: 4px;
        height: 4px;
        background-color: #16A34A;
        border-radius: 50%;
        box-shadow: 0 0 6px #16A34A;
    }}
    .viewfinder-crosshair-h {{
        position: absolute;
        width: 12px;
        height: 1px;
        background-color: rgba(22, 163, 74, 0.6);
    }}
    .viewfinder-crosshair-v {{
        position: absolute;
        width: 1px;
        height: 12px;
        background-color: rgba(22, 163, 74, 0.6);
    }}
    @keyframes pulse {{
        0% {{ transform: scale(0.9); opacity: 0.5; }}
        50% {{ transform: scale(1.1); opacity: 1; }}
        100% {{ transform: scale(0.9); opacity: 0.5; }}
    }}
    
    /* Estilo para los inputs de Streamlit con iconos en el background */
    input[id^="login_username"] {{
        padding-left: 42px !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%239CA3AF' stroke-width='2'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: 14px center;
        background-size: 16px;
        border-radius: 10px !important;
        border: 1.5px solid #CBD5E1 !important;
        background-color: #F8FAFC !important;
        height: 40px !important;
        font-family: 'Poppins', sans-serif !important;
        color: #1E293B !important;
    }}
    input[id^="login_password"] {{
        padding-left: 42px !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%239CA3AF' stroke-width='2'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: 14px center;
        background-size: 16px;
        border-radius: 10px !important;
        border: 1.5px solid #CBD5E1 !important;
        background-color: #F8FAFC !important;
        height: 40px !important;
        font-family: 'Poppins', sans-serif !important;
        color: #1E293B !important;
    }}
    
    div[data-baseweb="input"]:focus-within {{
        border-color: #16A34A !important;
        box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.1) !important;
    }}
    
    /* Botón de ingreso */
    button[key="btn_login_submit"] {{
        background-color: #16A34A !important;
        border-color: #16A34A !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.55rem 0 !important;
        letter-spacing: 0.5px;
        font-size: 0.9rem !important;
        font-family: 'Poppins', sans-serif !important;
        box-shadow: 0 4px 10px rgba(22, 163, 74, 0.15) !important;
        transition: all 0.2s !important;
    }}
    button[key="btn_login_submit"]:hover {{
        background-color: #14532D !important;
        border-color: #14532D !important;
        box-shadow: 0 6px 14px rgba(20, 83, 45, 0.25) !important;
    }}
    
    /* Estilos del label de inputs */
    label {{
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin-bottom: 2px !important;
        font-family: 'Poppins', sans-serif !important;
    }}
    
    /* Enlace de olvido de contraseña */
    .forgot-link a {{
        color: #16A34A;
        text-decoration: none;
        font-size: 0.78rem;
        font-weight: 600;
        transition: color 0.2s;
    }}
    .forgot-link a:hover {{
        color: #14532D;
    }}
    
    /* Checkbox layout */
    div[data-testid="stCheckbox"] label {{
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #475569 !important;
    }}
    
    /* Responsive styling */
    @media (max-width: 768px) {{
        div[data-testid="stAppViewContainer"] {{
            overflow: auto !important;
            height: auto !important;
        }}
        div.block-container {{
            height: auto !important;
            padding: 1.5rem !important;
        }}
        div[data-testid="stHorizontalBlock"] {{
            flex-direction: column !important;
            height: auto !important;
        }}
        div[data-testid="stHorizontalBlock"] > div {{
            width: 100% !important;
            max-width: 100% !important;
            min-width: 100% !important;
        }}
        div[data-testid="stHorizontalBlock"] > div:first-child {{
            min-height: 250px !important;
            padding: 1.8rem 1.5rem !important;
        }}
        div[data-testid="stHorizontalBlock"] > div:last-child {{
            padding: 1.8rem 1.5rem !important;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([0.45, 0.55])
    
    with col1:
        st.markdown("""<div style="height: 100%; display: flex; flex-direction: column; justify-content: space-between; font-family: 'Poppins', sans-serif;">
<div>
<!-- Logo Circular -->
<div style="width: 36px; height: 36px; border-radius: 50%; border: 1.5px solid rgba(255,255,255,0.3); display: flex; align-items: center; justify-content: center;">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 3C6.5 3 2 7.5 2 13C2 18.5 6.5 23 12 23C17.5 23 22 18.5 22 13V3H12ZM12 21C7.6 21 4 17.4 4 13C4 8.6 7.6 5 12 5C16.4 5 20 8.6 20 13V21H12Z" fill="#ffffff"/>
<path d="M12 7C9.8 7 8 8.8 8 11C8 13.2 9.8 15 12 15C14.2 15 16 13.2 16 11C16 8.8 14.2 7 12 7Z" fill="#16A34A"/>
</svg>
</div>
<h1 style="color: #ffffff; font-size: 1.5rem; font-weight: 800; line-height: 1.25; margin-top: 1.2rem; margin-bottom: 0.5rem; font-family: 'Poppins', sans-serif; letter-spacing: -0.5px;">
DETECTOR<br>DE <span style="color: #16A34A;">ENFERMEDADES</span><br>EN HOJAS DE MAÍZ
</h1>
<p style="color: #e2e8f0; font-size: 0.8rem; line-height: 1.35; max-width: 320px; font-weight: 300; margin-bottom: 0.3rem;">
Inteligencia Artificial para detectar enfermedades del cultivo de maíz mediante Deep Learning y Visión Computacional.
</p>
</div>

<!-- Viewfinder Visor -->
<div class="viewfinder-container">
<div class="viewfinder-box">
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
<div style="display: flex; align-items: center; margin-bottom: 0.4rem;">
<span style="font-size: 0.95rem; margin-right: 0.55rem;">🌱</span>
<span style="color: #ffffff; font-size: 0.78rem; font-weight: 600; font-family: 'Poppins', sans-serif;">IA Avanzada</span>
</div>
<div style="display: flex; align-items: center; margin-bottom: 0.4rem;">
<span style="font-size: 0.95rem; margin-right: 0.55rem;">⚡</span>
<span style="color: #ffffff; font-size: 0.78rem; font-weight: 600; font-family: 'Poppins', sans-serif;">Resultados en segundos</span>
</div>
<div style="display: flex; align-items: center;">
<span style="font-size: 0.95rem; margin-right: 0.55rem;">📈</span>
<span style="color: #ffffff; font-size: 0.78rem; font-weight: 600; font-family: 'Poppins', sans-serif;">Precisión superior al 95%</span>
</div>
</div>
</div>""", unsafe_allow_html=True)
        
    with col2:
        st.markdown("""<div class="login-right-form" style="text-align: center; margin-bottom: 0.8rem; font-family: 'Poppins', sans-serif;">
<div style="width: 38px; height: 38px; border-radius: 50%; border: 1.5px solid rgba(22, 163, 74, 0.15); display: flex; align-items: center; justify-content: center; margin: 0 auto 0.4rem auto; background-color: rgba(22, 163, 74, 0.05);">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 3C6.5 3 2 7.5 2 13C2 18.5 6.5 23 12 23C17.5 23 22 18.5 22 13V3H12ZM12 21C7.6 21 4 17.4 4 13C4 8.6 7.6 5 12 5C16.4 5 20 8.6 20 13V21H12Z" fill="#16A34A"/>
<path d="M12 7C9.8 7 8 8.8 8 11C8 13.2 9.8 15 12 15C14.2 15 16 13.2 16 11C16 8.8 14.2 7 12 7Z" fill="#14532D"/>
</svg>
</div>
<h2 style="color: #1E293B; font-size: 1.4rem; font-weight: 700; margin: 0; font-family: 'Poppins', sans-serif; letter-spacing: -0.3px;">Bienvenido</h2>
<p style="color: #64748B; font-size: 0.78rem; margin-top: 0.1rem; margin-bottom: 0; font-family: 'Poppins', sans-serif;">Inicie sesión para acceder al sistema.</p>
</div>""", unsafe_allow_html=True)
        
        # Formulario
        username = st.text_input("Correo electrónico", placeholder="ejemplo@correo.com", key="login_username")
        password = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña", key="login_password")
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            st.checkbox("Recordarme", value=True, key="login_remember")
        with col_opt2:
            st.markdown('<p class="forgot-link" style="text-align: right; margin: 0; padding-top: 2px;"><a href="#">¿Olvidaste tu contraseña?</a></p>', unsafe_allow_html=True)
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        
        if st.button("👉 INICIAR SESIÓN", type="primary", use_container_width=True, key="btn_login_submit"):
            if username in ["admin", "admin@maiz.com"] and password == "admin123":
                st.session_state.authenticated = True
                st.success("✅ ¡Ingreso exitoso!")
                st.rerun()
            else:
                st.error("❌ Credenciales inválidas. Inténtelo de nuevo.")
                
        # Pie de página
        st.markdown("""
        <div style="text-align: center; color: #94A3B8; font-size: 0.65rem; margin-top: 1.5rem; font-family: 'Poppins', sans-serif; line-height: 1.5;">
            Detector Inteligente de Enfermedades en Hojas de Maíz<br>
            <span style="font-weight: 600; color: #64748B;">Versión 1.0</span>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    show_login()
