# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Add the project root directory to sys.path so we can import from 'src'
project_root = Path(__file__).resolve().parent.parent.parent  # backend/app/main.py -> backend -> parent (maize-disease-detection)
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.core.database import init_db
from app.routers import auth, dataset, eda, training, model, stats, reports, chat
from app.websockets.training_ws import websocket_endpoint

app = FastAPI(
    title="Maize Disease Detector & AutoML API",
    description="API de producción para diagnóstico fitosanitario de maíz y AutoML tabular",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.on_event("startup")
def startup_event():
    init_db()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro de rutas de la API (v1 y legacy /api/)
for prefix_base in ["/api/v1", "/api"]:
    app.include_router(auth.router, prefix=f"{prefix_base}/auth", tags=["Autenticación"])
    app.include_router(dataset.router, prefix=f"{prefix_base}/dataset", tags=["Dataset"])
    app.include_router(eda.router, prefix=f"{prefix_base}/eda", tags=["Análisis Exploratorio (EDA)"])
    app.include_router(training.router, prefix=f"{prefix_base}/training", tags=["Entrenamiento"])
    app.include_router(model.router, prefix=f"{prefix_base}/model", tags=["Modelos"])
    app.include_router(stats.router, prefix=f"{prefix_base}/stats", tags=["Pruebas Estadísticas"])
    app.include_router(reports.router, prefix=f"{prefix_base}/reports", tags=["Reportes"])
    app.include_router(chat.router, prefix=f"{prefix_base}/chat", tags=["Chatbot"])

# Registro de WebSocket de progreso de entrenamiento
app.add_api_websocket_route("/api/v1/training/ws/{client_id}", websocket_endpoint)
app.add_api_websocket_route("/api/training/ws/{client_id}", websocket_endpoint)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Maize Disease Detector API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
