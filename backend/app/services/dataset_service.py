# -*- coding: utf-8 -*-
import os
import hashlib
import pandas as pd
from typing import Dict, Any, List

DATA_DIR = "data"
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, "maize_crop_data.csv")

def save_dataset(file_content: bytes, filename: str = "maize_crop_data.csv") -> Dict[str, Any]:
    """Guarda el archivo CSV cargado, calcula su hash MD5 y extrae metadatos de las columnas"""
    os.makedirs(DATA_DIR, exist_ok=True)
    target_path = os.path.join(DATA_DIR, filename)
    
    # Escribir archivo en disco
    with open(target_path, "wb") as f:
        f.write(file_content)
        
    # Calcular hash MD5
    md5_hash = hashlib.md5(file_content).hexdigest()
    
    # Leer datos para metadatos básicos
    df = pd.read_csv(target_path)
    row_count = len(df)
    columns = df.columns.tolist()
    
    return {
        "filename": filename,
        "path": target_path,
        "dataset_hash": md5_hash,
        "rows": row_count,
        "columns": columns
    }

def get_default_dataset_info() -> Dict[str, Any]:
    """Obtiene metadatos del dataset por defecto si existe"""
    if not os.path.exists(DEFAULT_DATASET_PATH):
        raise FileNotFoundError("No se encontró el dataset por defecto en 'data/maize_crop_data.csv'")
        
    with open(DEFAULT_DATASET_PATH, "rb") as f:
        content = f.read()
        
    md5_hash = hashlib.md5(content).hexdigest()
    df = pd.read_csv(DEFAULT_DATASET_PATH)
    
    return {
        "filename": "maize_crop_data.csv",
        "path": DEFAULT_DATASET_PATH,
        "dataset_hash": md5_hash,
        "rows": len(df),
        "columns": df.columns.tolist()
    }
