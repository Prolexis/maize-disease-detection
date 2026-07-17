# -*- coding: utf-8 -*-
import sqlite3
import os
from app.core.config import settings

def get_db_connection():
    """Obtiene una conexión directa a la base de datos SQLite"""
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Inicializa el esquema de la base de datos y crea el usuario por defecto"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Crear tabla de usuarios si no existe
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)
    conn.commit()
    
    # Crear usuarios por defecto (admin y admin@maiz.com) con contraseña segura admin123
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        from app.core.security import get_password_hash
        hashed = get_password_hash("admin123")
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", hashed))
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin@maiz.com", hashed))
        conn.commit()
        
    conn.close()
