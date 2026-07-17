# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from src.eda import clean_data, get_descriptive_stats

def clean_and_analyze_dataset(dataset_path: str, target_col: str, lang: str = "es") -> dict:
    """Ejecuta la limpieza de datos avanzada y extrae los descriptivos estadísticos como JSON compatible"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No se encontró el dataset en la ruta: {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    
    # 1. Ejecutar limpieza modular
    df_cleaned, num_duplicates, imputed_nulls, outliers_detected = clean_data(df, target_col)
    
    # Obtener atributos inyectados
    transformed_cols = df_cleaned.attrs.get('transformed_cols', [])
    multivariate_outliers = df_cleaned.attrs.get('multivariate_outliers', 0)
    
    # 2. Obtener descriptivos
    global_stats, class_stats = get_descriptive_stats(df_cleaned, target_col)
    
    # Sanitizar descriptivos de valores NaN / Inf para evitar errores de serialización JSON
    def sanitize_dict(d: dict) -> dict:
        sanitized = {}
        for k, v in d.items():
            if isinstance(v, dict):
                sanitized[k] = sanitize_dict(v)
            elif isinstance(v, (float, np.float64, np.float32)):
                if np.isnan(v) or np.isinf(v):
                    sanitized[k] = None
                else:
                    sanitized[k] = float(v)
            elif isinstance(v, (int, np.int64, np.int32)):
                sanitized[k] = int(v)
            else:
                sanitized[k] = v
        return sanitized
        
    sanitized_stats = sanitize_dict(global_stats.to_dict(orient='index'))
    
    # Sanitizar los detalles de nulos imputados
    clean_imputed = {}
    for col, val in imputed_nulls.items():
        clean_imputed[col] = {
            "count": int(val[0]),
            "method": str(val[1])
        }
        
    # Calcular interpretación usando el módulo compartido
    from src.interpretation import interpretar_eda
    interpretation = interpretar_eda(
        df_cleaned, target_col, num_duplicates, imputed_nulls, outliers_detected, global_stats, lang=lang
    )
        
    return {
        "num_duplicates": int(num_duplicates),
        "imputed_nulls": clean_imputed,
        "outliers_detected": {k: int(v) for k, v in outliers_detected.items()},
        "multivariate_outliers": int(multivariate_outliers),
        "transformed_cols": transformed_cols,
        "descriptive_stats": sanitized_stats,
        "interpretation": interpretation
    }

