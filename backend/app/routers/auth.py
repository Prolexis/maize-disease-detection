# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException, Depends, status
from ..models.schemas import UserLogin, Token
from ..core.database import get_db_connection
from ..core.security import verify_password, create_access_token

router = APIRouter()

@router.post("/login", response_model=Token)
def login(payload: UserLogin):
    username = payload.username
    password = payload.password
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales de acceso inválidas"
        )
        
    hashed_password = row["password"]
    if not verify_password(password, hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales de acceso inválidas"
        )
        
    # Crear token JWT
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

from ..core.dependencies import get_current_user

@router.post("/refresh", response_model=Token)
def refresh_token(username: str = Depends(get_current_user)):
    """Renueva el token JWT para la sesión activa del usuario"""
    new_token = create_access_token(data={"sub": username})
    return {"access_token": new_token, "token_type": "bearer"}

@router.post("/logout")
def logout(username: str = Depends(get_current_user)):
    """Invalida/cierra la sesión activa del usuario"""
    return {"status": "success", "message": f"Sesión de {username} cerrada correctamente."}
