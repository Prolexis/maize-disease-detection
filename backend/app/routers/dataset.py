# -*- coding: utf-8 -*-
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from ..core.dependencies import get_current_user
from ..services import dataset_service

router = APIRouter()

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    username: str = Depends(get_current_user)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se admiten archivos CSV."
        )
        
    try:
        content = await file.read()
        info = dataset_service.save_dataset(content, file.filename)
        return info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al guardar el dataset: {e}"
        )

@router.get("/info")
def get_info(username: str = Depends(get_current_user)):
    try:
        info = dataset_service.get_default_dataset_info()
        return info
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al leer la información: {e}"
        )
