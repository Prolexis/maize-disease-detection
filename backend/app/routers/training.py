# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, BackgroundTasks, status
from app.core.dependencies import get_current_user
from app.models.schemas import TrainingConfig
from app.jobs.job_manager import run_training_pipeline_async, LATEST_RUN_RESULT
from app.core.database import get_db_connection

router = APIRouter()

@router.post("/train", status_code=status.HTTP_202_ACCEPTED)
def train_model(
    client_id: str,
    payload: TrainingConfig,
    background_tasks: BackgroundTasks,
    username: str = Depends(get_current_user)
):
    """Lanza el entrenamiento de AutoML en segundo plano y asocia el progreso al client_id"""
    config_dict = payload.dict()
    background_tasks.add_task(run_training_pipeline_async, client_id, config_dict)
    return {
        "status": "training_started",
        "client_id": client_id,
        "message": "Entrenamiento iniciado en segundo plano. Escuche el WebSocket de progreso."
    }

@router.get("/latest")
def get_latest_training_result(
    lang: str = "es",
    username: str = Depends(get_current_user)
):
    """Obtiene los resultados de métricas e interpretación del último run ejecutado desde el disco"""
    import os
    import json
    
    latest_path = "models/latest_results.json"
    if not os.path.exists(latest_path):
        return {"status": "no_runs_completed_yet"}
        
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        lang_key = lang.lower() if lang in ["es", "en", "pt"] else "es"
        
        # Inyectar interpretaciones por pestaña del idioma correspondiente
        if "interpretations_by_tab" in data:
            lang_data = data["interpretations_by_tab"].get(lang_key, data["interpretations_by_tab"].get("es", {}))
            data["interpretations"] = lang_data.get("training", data.get("interpretations", ""))
            data["eda_interpretation"] = lang_data.get("eda", "")
            data["cv_interpretation"] = lang_data.get("cv", "")
            data["tuning_interpretation"] = lang_data.get("tuning", "")
            data["stats_interpretation"] = lang_data.get("stats", "")
            
        return data
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Error al leer los resultados del entrenamiento: {e}"
        )


@router.get("/history")
def get_training_history(username: str = Depends(get_current_user)):
    """
    Retorna el historial completo de experimentos guardado en la tabla experiments de SQLite.
    Cada fila incluye: fecha, hash del dataset, mejor modelo, accuracy, f1, configuración de CV y tuning.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, run_date, dataset_hash, best_model_name, accuracy, f1_score,
                   split_ratio, seed, cv_folds, alpha, tuning_method
            FROM experiments
            ORDER BY id DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        experiments = [dict(row) for row in rows]
        return {"count": len(experiments), "experiments": experiments}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Error al obtener historial: {e}")

