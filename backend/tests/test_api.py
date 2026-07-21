# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.main import app
from backend.app.core.database import init_db

init_db()
client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data

def test_auth_login():
    response = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_unauthorized_access():
    response = client.get("/api/v1/model/metadata")
    assert response.status_code == 401

def test_model_metadata_authorized():
    # Login first
    login_resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    token = login_resp.json()["access_token"]
    
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/v1/model/metadata", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "vision_model" in data
    assert "tabular_model" in data
