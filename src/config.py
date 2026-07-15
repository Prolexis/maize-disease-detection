import os
import random
import numpy as np

def set_seed(seed=42):
    """Establecer semillas de reproducibilidad para todos los motores de computación"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Forzar ejecución determinista si es posible
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except ImportError:
        pass

# Configuraciones por defecto
DEFAULT_CONFIG = {
    'seed': 42,
    'split_ratio': 0.8,
    'cv_folds': 5,
    'alpha': 0.05,
    'tuning_metric': 'accuracy',
    'tuning_method': 'grid'
}

def apply_theme_to_plot(theme='Oscuro'):
    """
    Modifica dinámicamente los parámetros de estilo globales de matplotlib (rcParams)
    para adaptarse al tema visual activo (Claro u Oscuro).
    """
    import matplotlib.pyplot as plt
    
    is_dark = theme in ['Oscuro', 'Dark', 'dark', 'escuro', 'Escuro']
    
    if is_dark:
        bg_color = '#1E293B'  # Slate 800
        text_color = '#F8FAFC'  # Slate 50
        grid_color = '#475569'  # Slate 600
        spine_color = '#334155'  # Slate 700
        
        plt.rcParams.update({
            'figure.facecolor': bg_color,
            'axes.facecolor': bg_color,
            'text.color': text_color,
            'axes.labelcolor': text_color,
            'xtick.color': text_color,
            'ytick.color': text_color,
            'grid.color': grid_color,
            'axes.edgecolor': spine_color,
            'legend.facecolor': bg_color,
            'legend.edgecolor': spine_color,
        })
    else:
        bg_color = '#FFFFFF'
        text_color = '#0F172A'  # Slate 900
        grid_color = '#E2E8F0'  # Slate 200
        spine_color = '#CBD5E1'  # Slate 300
        
        plt.rcParams.update({
            'figure.facecolor': bg_color,
            'axes.facecolor': bg_color,
            'text.color': text_color,
            'axes.labelcolor': text_color,
            'xtick.color': text_color,
            'ytick.color': text_color,
            'grid.color': grid_color,
            'axes.edgecolor': spine_color,
            'legend.facecolor': bg_color,
            'legend.edgecolor': spine_color,
        })
