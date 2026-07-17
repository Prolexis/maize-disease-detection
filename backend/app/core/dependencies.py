from fastapi import Header, Query, HTTPException, status
from app.core.security import decode_access_token

def get_current_user(
    authorization: str = Header(None),
    token: str = Query(None)
):
    """Obtiene y valida el usuario actual mediante el token JWT de la cabecera Authorization o query string"""
    jwt_token = None
    
    if authorization:
        try:
            token_type, parsed_token = authorization.split(" ")
            if token_type.lower() == "bearer":
                jwt_token = parsed_token
        except ValueError:
            pass
            
    if not jwt_token and token:
        jwt_token = token
        
    if not jwt_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Falta cabecera de Autorización (token JWT) o parámetro de consulta 'token'"
        )
        
    payload = decode_access_token(jwt_token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado"
        )
    return payload.get("sub")
