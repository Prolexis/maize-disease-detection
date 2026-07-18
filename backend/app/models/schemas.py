# -*- coding: utf-8 -*-
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    message: str
    lang: str = "es"

class ChatResponse(BaseModel):
    response: str

class EdaResponse(BaseModel):
    num_duplicates: int
    imputed_nulls: Dict[str, Any]
    outliers_detected: Dict[str, int]
    multivariate_outliers: int
    transformed_cols: List[str]
    descriptive_stats: Dict[str, Any]
    interpretation: str

class StatsResponse(BaseModel):
    test_type: str
    overall_stat: float
    overall_pval: float
    use_parametric: bool
    shapiro_pvals: Dict[str, float]
    levene_pval: float
    bootstrap_ci: Dict[str, List[float]]
    bootstrap_ci_f1: Optional[Dict[str, List[float]]] = None
    wilcoxon: Optional[Dict[str, Any]] = None
    mcnemar: Optional[Dict[str, Any]] = None
    pairwise_comparisons: Optional[Dict[str, Any]] = None
    posthoc_results: Optional[Dict[str, Any]] = None
    interpretations: Optional[Union[str, Dict[str, Any], List[Any]]] = None

class TrainingConfig(BaseModel):
    split_ratio: float = 0.8
    seed: int = 42
    cv_folds: int = 5
    alpha: float = 0.05
    tuning_method: str = "random"
