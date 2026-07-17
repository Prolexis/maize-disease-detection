# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
import os
from app.core.dependencies import get_current_user

router = APIRouter()

@router.get("/download/{report_type}")
def download_report(
    report_type: str,
    username: str = Depends(get_current_user)
):
    """Descarga los reportes bilingües generados (pdf, docx, xlsx)"""
    if report_type not in ["pdf", "docx", "xlsx"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tipo de reporte inválido. Debe ser: pdf, docx o xlsx."
        )
        
    filepath = f"reports/reporte_automl.{report_type}"
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontró el reporte {report_type}. Por favor ejecute el pipeline primero."
        )
        
    media_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }
    
    return FileResponse(
        path=filepath,
        filename=f"reporte_fitosanitario_automl.{report_type}",
        media_type=media_types[report_type]
    )
