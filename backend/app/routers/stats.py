# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException, status
from app.core.dependencies import get_current_user
from app.jobs.job_manager import LATEST_RUN_RESULT
from app.models.schemas import StatsResponse

router = APIRouter()

@router.get("/results", response_model=StatsResponse)
def get_statistical_results(lang: str = "es", username: str = Depends(get_current_user)):
    """Obtiene los resultados detallados de significancia estadística y Bootstrap CIs"""
    import os
    import json
    
    latest_path = "models/latest_results.json"
    if not os.path.exists(latest_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se han ejecutado análisis estadísticos aún. Por favor complete el entrenamiento AutoML."
        )
        
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if "stats" not in data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No se han encontrado resultados de pruebas estadísticas en el archivo de resultados."
            )
            
        stats_data = data["stats"]
        
        # Obtener las interpretaciones del idioma correspondiente
        lang_key = lang.lower() if lang in ["es", "en", "pt"] else "es"
        from src.interpretation import interpretar_stats
        interpretations = interpretar_stats(stats_data, alpha=0.05, lang=lang_key)
        
        return {
            "test_type": stats_data.get("test_type", "Desconocido"),
            "overall_stat": float(stats_data.get("overall_stat", 0.0)),
            "overall_pval": float(stats_data.get("overall_pval", 1.0)),
            "use_parametric": bool(stats_data.get("use_parametric", False)),
            "shapiro_pvals": {k: float(v) for k, v in stats_data.get("shapiro_pvals", {}).items()},
            "levene_pval": float(stats_data.get("levene_pval", 1.0)),
            "bootstrap_ci": {k: [float(val) for val in v] for k, v in stats_data.get("bootstrap_ci", {}).items()},
            "bootstrap_ci_f1": {k: [float(val) for val in v] for k, v in stats_data.get("bootstrap_ci_f1", {}).items()} if "bootstrap_ci_f1" in stats_data else None,
            "wilcoxon": stats_data.get("wilcoxon", {}),
            "mcnemar": stats_data.get("mcnemar", {}),
            "pairwise_comparisons": stats_data.get("pairwise_comparisons", {}),
            "posthoc_results": stats_data.get("posthoc_results", {}),
            "interpretations": interpretations
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al leer los resultados estadísticos: {e}"
        )

