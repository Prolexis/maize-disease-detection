# -*- coding: utf-8 -*-
import os

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./maize_backend.db")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "super_secret_jwt_key_2026_maize_disease")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 horas (1 día)
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
        "*"
    ]

settings = Settings()
