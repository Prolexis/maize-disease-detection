# -*- coding: utf-8 -*-
import asyncio
import os
import hashlib
from datetime import datetime
import sqlite3
import pandas as pd

from app.websockets.training_ws import ws_manager
from app.core.database import get_db_connection
from src.eda import clean_data, get_descriptive_stats
from src.training import train_and_evaluate_all, save_best_model, interpret_training
from src.cross_validation import run_cross_validation
from src.tuning import run_hyperparameter_tuning
from src.stats_tests import run_statistical_tests, interpret_stats
from src.reporting import generate_xlsx_report, generate_docx_report, generate_tabular_pdf_report

# Asegurar tabla de tracking en la base de datos SQLite
def init_tracking_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_date TEXT NOT NULL,
        dataset_hash TEXT NOT NULL,
        best_model_name TEXT NOT NULL,
        accuracy REAL NOT NULL,
        f1_score REAL NOT NULL,
        split_ratio REAL NOT NULL,
        seed INTEGER NOT NULL,
        cv_folds INTEGER NOT NULL,
        alpha REAL NOT NULL,
        tuning_method TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

# Inicializar tabla al cargar módulo
init_tracking_db()

# Almacenar el estado del último experimento entrenado en memoria para consultas rápidas de la API
LATEST_RUN_RESULT = {}

async def run_training_pipeline_async(client_id: str, config: dict):
    global LATEST_RUN_RESULT
    
    loop = asyncio.get_event_loop()
    dataset_path = "data/maize_crop_data.csv"
    target_col = "Enfermedad"
    
    try:
        # --- Fase 0: Validaciones iniciales ---
        if not os.path.exists(dataset_path):
            await ws_manager.send_personal_message({
                "type": "error",
                "message": "No se encontró ningún dataset cargado en 'data/maize_crop_data.csv'"
            }, client_id)
            return
            
        await ws_manager.send_personal_message({
            "type": "progress",
            "percent": 5,
            "message": "Iniciando pipeline de AutoML..."
        }, client_id)
        
        # --- Fase 1: Carga y hashing del dataset (Versión) ---
        await ws_manager.send_personal_message({
            "type": "progress",
            "percent": 15,
            "message": "Cargando dataset y calculando hash de versión..."
        }, client_id)
        
        with open(dataset_path, "rb") as f:
            file_bytes = f.read()
        dataset_hash = hashlib.md5(file_bytes).hexdigest()
        
        # Leer dataframe
        df = pd.read_csv(dataset_path)
        if target_col not in df.columns:
            target_col = df.columns[-1] # Fallback al último
            
        # --- Fase 2: Limpieza de datos y EDA ---
        await ws_manager.send_personal_message({
            "type": "progress",
            "percent": 25,
            "message": "Ejecutando limpieza de datos y preparación de variables..."
        }, client_id)
        
        df_cleaned, num_duplicates, imputed_nulls, outliers_detected = await loop.run_in_executor(
            None, clean_data, df, target_col
        )
        
        df_eda, class_stats = await loop.run_in_executor(
            None, get_descriptive_stats, df_cleaned, target_col
        )
        
        # --- Fase 3: Entrenamiento de los 5 modelos ---
        await ws_manager.send_personal_message({
            "type": "progress",
            "percent": 45,
            "message": "Entrenando y evaluando 5 modelos de aprendizaje automático..."
        }, client_id)
        
        results, X_train, X_test, y_train, y_test, classes = await loop.run_in_executor(
            None, train_and_evaluate_all, df_cleaned, target_col, config['split_ratio'], config['seed']
        )
        
        sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        best_model_name, best_res = sorted_models[0]
        
        # --- Fase 4: Validación Cruzada ---
        await ws_manager.send_personal_message({
            "type": "progress",
            "percent": 65,
            "message": f"Ejecutando validación cruzada de {config['cv_folds']} pliegues..."
        }, client_id)
        
        cv_results = await loop.run_in_executor(
            None, run_cross_validation, df_cleaned, target_col, config['cv_folds'], config['seed']
        )
        
        # --- Fase 5: Ajuste de Hiperparámetros (Tuning) ---
        await ws_manager.send_personal_message({
            "type": "progress",
            "percent": 75,
            "message": f"Optimizando hiperparámetros del modelo base mediante {config['tuning_method']} search..."
        }, client_id)
        
        tuning_results = await loop.run_in_executor(
            None, run_hyperparameter_tuning, df_cleaned, target_col, config['tuning_method'], config['seed']
        )
        
        # --- Fase 6: Pruebas Estadísticas Robustas (con Bootstrap CI) ---
        await ws_manager.send_personal_message({
            "type": "progress",
            "percent": 85,
            "message": "Ejecutando validaciones y pruebas de significancia estadística..."
        }, client_id)
        
        classic_names = ['Regresión Logística (Clásico)', 'Random Forest (Clásico)', 'Red Neuronal MLP (Clásico)']
        hybrid_names  = ['Híbrido Votación (RF+MLP)', 'Híbrido Stacking (Meta-GB)']
        best_classic_name = max(
            [n for n in classic_names if n in results],
            key=lambda n: results[n]['accuracy'],
            default=list(results.keys())[0]
        )
        best_hybrid_name = max(
            [n for n in hybrid_names if n in results],
            key=lambda n: results[n]['accuracy'],
            default=list(results.keys())[-1]
        )
        y_pred_classic = results[best_classic_name]['y_pred']
        y_pred_hybrid  = results[best_hybrid_name]['y_pred']
        
        stats_results = await loop.run_in_executor(
            None, run_statistical_tests, cv_results, y_test, y_pred_classic, y_pred_hybrid, config['alpha'], "reports", "es", results
        )
        
        # --- Fase 7: Serialización y Guardado del mejor modelo + metadata ---
        await ws_manager.send_personal_message({
            "type": "progress",
            "percent": 90,
            "message": "Guardando el mejor pipeline entrenado y registrando metadatos..."
        }, client_id)
        
        model_path, meta_path = await loop.run_in_executor(
            None, save_best_model, results, "best_tabular_model.pkl", "metadata.json",
            tuning_results.get('best_params'), dataset_hash
        )
        
        # --- Fase 8: Generación de Reportes (PDF/Word/Excel) ---
        await ws_manager.send_personal_message({
            "type": "progress",
            "percent": 95,
            "message": "Generando reportes bilingües estructurados (PDF, Excel, Word)..."
        }, client_id)
        
        df_training_rep = pd.DataFrame({
            name: {
                'Accuracy': res['accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'F1-Score': res['f1-score'],
                'AUC': res['auc'],
                'Tiempo de Entrenamiento (s)': res['train_time'],
                'Tiempo de Inferencia (s)': res['inference_time'],
                'No. Parámetros': res['param_count'],
                'Tamaño (KB)': res['model_size_kb'],
            }
            for name, res in results.items()
        }).T
        
        # Generación síncrona/hilo de los reportes
        image_paths = {
            'balance': 'reports/eda_balance_clases.png',
            'correlation': 'reports/eda_correlation_matrix.png',
            'distributions': 'reports/eda_feature_distribution.png',
            'boxplots': 'reports/eda_boxplots.png',
            'roc': 'reports/train_roc_curve.png',
            'learning': 'reports/train_learning_curves.png',
            'cv': 'reports/cv_dispersion.png',
            'stats': 'reports/stats_significance.png',
        }
        interpretations = {
            'eda': "Análisis EDA completado.",
            'training': interpret_training(results),
            'cv': "Validación cruzada exitosa.",
            'tuning': "Hiperparámetros optimizados.",
            'stats': "Significancia calculada."
        }
        
        xlsx_report = await loop.run_in_executor(
            None, generate_xlsx_report, df_eda, df_training_rep, cv_results, tuning_results, stats_results, "reports/reporte_automl.xlsx", "es"
        )
        docx_report = await loop.run_in_executor(
            None, generate_docx_report, df_eda, df_training_rep, cv_results, tuning_results, stats_results, interpretations, image_paths, "reports/reporte_automl.docx", "es"
        )
        pdf_report = await loop.run_in_executor(
            None, generate_tabular_pdf_report, df_eda, df_training_rep, cv_results, tuning_results, stats_results, interpretations, image_paths, "reports/reporte_automl.pdf", "es"
        )
        
        # --- Fase 9: Guardar en el histórico de experimentos (SQLite) ---
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO experiments (
                run_date, dataset_hash, best_model_name, accuracy, f1_score,
                split_ratio, seed, cv_folds, alpha, tuning_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset_hash,
            best_model_name,
            float(best_res['accuracy']),
            float(best_res['f1-score']),
            config['split_ratio'],
            config['seed'],
            config['cv_folds'],
            config['alpha'],
            config['tuning_method']
        ))
        conn.commit()
        conn.close()
        
        # Guardar en memoria el resultado del último run para que lo use la API
        # Preparar metricas para React
        metrics_dict = {}
        for model_name, res in results.items():
            metrics_dict[model_name] = {
                "accuracy": res['accuracy'],
                "precision": res['precision'],
                "recall": res['recall'],
                "f1": res['f1-score']
            }
        
        # Preparar resultados de CV para React
        cv_react_results = {}
        for model_name, cv_res in cv_results.items():
            cv_react_results[model_name] = cv_res['accuracies']
        
        LATEST_RUN_RESULT = {
            "best_model": best_model_name,
            "best_model_metrics": {
                "accuracy": float(best_res['accuracy']),
                "precision": float(best_res['precision']),
                "recall": float(best_res['recall']),
                "f1": float(best_res['f1-score'])
            },
            "accuracy": float(best_res['accuracy']),
            "f1_score": float(best_res['f1-score']),
            "dataset_hash": dataset_hash,
            "training_time": float(best_res['train_time']),
            "confusion_matrix": best_res['confusion_matrix'].tolist(),
            "roc_curve_path": "reports/train_roc_curve.png",
            "learning_curve_path": "reports/train_learning_curves.png",
            "interpretations": interpret_training(results),
            "metrics": metrics_dict,
            "cv_results": cv_react_results,
            "tuning_results": tuning_results,
            "stats": stats_results,
            "pdf_report": pdf_report,
            "docx_report": docx_report,
            "xlsx_report": xlsx_report,
            # Interpretaciones por idioma (es/en/pt)
            "interpretations_by_tab": {
                "es": {
                    "training": interpret_training(results),
                    "eda": "Análisis EDA completado exitosamente.",
                    "cv": "Validación cruzada ejecutada con éxito.",
                    "tuning": "Optimización de hiperparámetros completada.",
                    "stats": "Pruebas estadísticas calculadas correctamente."
                },
                "en": {
                    "training": interpret_training(results),
                    "eda": "EDA analysis completed successfully.",
                    "cv": "Cross-validation executed successfully.",
                    "tuning": "Hyperparameter optimization completed.",
                    "stats": "Statistical tests calculated correctly."
                },
                "pt": {
                    "training": interpret_training(results),
                    "eda": "Análise EDA concluída com sucesso.",
                    "cv": "Validação cruzada executada com sucesso.",
                    "tuning": "Otimização de hiperparâmetros concluída.",
                    "stats": "Testes estatísticos calculados corretamente."
                }
            }
        }

        # --- Persistir resultados en disco para que la API los lea incluso tras reinicio ---
        import json as _json
        os.makedirs("models", exist_ok=True)
        results_to_save = {k: v for k, v in LATEST_RUN_RESULT.items() if k not in ("pdf_report", "docx_report", "xlsx_report")}
        with open("models/latest_results.json", "w", encoding="utf-8") as f:
            _json.dump(results_to_save, f, ensure_ascii=False, indent=2, default=str)
        
        # Enviar completado
        await ws_manager.send_personal_message({
            "type": "completed",
            "percent": 100,
            "message": "¡Entrenamiento de AutoML completado con éxito!",
            "payload": {
                "best_model_name": best_model_name,
                "accuracy": float(best_res['accuracy']),
                "f1_score": float(best_res['f1-score']),
                "dataset_hash": dataset_hash
            }
        }, client_id)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        await ws_manager.send_personal_message({
            "type": "error",
            "message": f"Error crítico en el pipeline: {str(e)}"
        }, client_id)
