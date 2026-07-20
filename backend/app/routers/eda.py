# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException, status, Query
import pandas as pd
import os
from app.core.dependencies import get_current_user
from app.models.schemas import EdaResponse
from app.services import eda_service

router = APIRouter()

@router.get("/analyze", response_model=EdaResponse)
def analyze_dataset(
    target_col: str = Query(None, description="Nombre de la columna objetivo. Si no se provee, se usará la última columna."),
    lang: str = Query("es", description="Idioma de la interpretación (es, en, pt)"),
    username: str = Depends(get_current_user)
):
    dataset_path = "data/maize_crop_data.csv"
    if not os.path.exists(dataset_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se encontró ningún dataset cargado. Por favor, suba un archivo CSV primero."
        )
        
    try:
        # Si no se define target_col, usar el último por defecto
        if not target_col:
            df_temp = pd.read_csv(dataset_path, nrows=1)
            target_col = df_temp.columns[-1]
            
        result = eda_service.clean_and_analyze_dataset(dataset_path, target_col, lang=lang)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el análisis exploratorio: {e}"
        )

@router.get("/summary")
def get_eda_summary(username: str = Depends(get_current_user)):
    """Obtiene el resumen estadístico descriptivo del dataset cargado"""
    dataset_path = "data/maize_crop_data.csv"
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset no encontrado.")
    try:
        df = pd.read_csv(dataset_path)
        numeric_df = df.select_dtypes(include=['number'])
        return {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "descriptive_stats": numeric_df.describe().to_dict(),
            "null_counts": df.isnull().sum().to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/correlations")
def get_eda_correlations(username: str = Depends(get_current_user)):
    """Calcula la matriz de correlación de Pearson para características numéricas"""
    dataset_path = "data/maize_crop_data.csv"
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset no encontrado.")
    try:
        df = pd.read_csv(dataset_path)
        numeric_df = df.select_dtypes(include=['number'])
        corr = numeric_df.corr().fillna(0).to_dict()
        return {"correlation_matrix": corr}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/distributions")
def get_eda_distributions(username: str = Depends(get_current_user)):
    """Obtiene distribuciones de frecuencia de las variables"""
    dataset_path = "data/maize_crop_data.csv"
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset no encontrado.")
    try:
        import numpy as np
        df = pd.read_csv(dataset_path)
        distributions = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                counts, bin_edges = np.histogram(df[col].dropna(), bins=10)
                distributions[col] = {
                    "bins": [float(b) for b in bin_edges],
                    "counts": [int(c) for c in counts]
                }
            else:
                distributions[col] = df[col].value_counts().to_dict()
        return {"distributions": distributions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/outliers")
def get_eda_outliers(username: str = Depends(get_current_user)):
    """Detecta valores atípicos usando la regla del Rango Intercuartílico (IQR)"""
    dataset_path = "data/maize_crop_data.csv"
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset no encontrado.")
    try:
        df = pd.read_csv(dataset_path)
        outliers_by_col = {}
        for col in df.select_dtypes(include=['number']).columns:
            q1 = float(df[col].quantile(0.25))
            q3 = float(df[col].quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((df[col] < lower) | (df[col] > upper)).sum())
            outliers_by_col[col] = {"outlier_count": count, "lower_bound": lower, "upper_bound": upper}
        return {"outliers": outliers_by_col}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

