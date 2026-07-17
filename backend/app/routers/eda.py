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

