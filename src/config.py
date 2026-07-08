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
