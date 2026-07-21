# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException
from ..core.dependencies import get_current_user
from ..models.schemas import ChatRequest, ChatResponse
from src.chatbot import get_chatbot_response

router = APIRouter()

@router.post("/", response_model=ChatResponse)
def ask_chatbot(
    payload: ChatRequest,
    username: str = Depends(get_current_user)
):
    """Realiza una pregunta de diagnóstico fitosanitario o redes neuronales a MaizIA"""
    try:
        response = get_chatbot_response(payload.message, lang=payload.lang)
        return {"response": response}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en el asistente MaizIA: {e}"
        )
