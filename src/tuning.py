import os
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from src.training import get_preprocessor

def run_hyperparameter_tuning(df, target_col, method='grid', seed=42):
    """
    Optimiza el modelo Red Neuronal MLP.
    Compara resultados antes y después del tuning.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=seed, stratify=y)
    
    preprocessor = get_preprocessor(df, target_col)
    
    # 1. Modelo Base (Sin tuning)
    base_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=seed)
    base_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', base_mlp)
    ])
    
    base_pipeline.fit(X_train, y_train)
    base_preds = base_pipeline.predict(X_test)
    base_acc = accuracy_score(y_test, base_preds)
    
    # 2. Espacio de hiperparámetros
    param_grid = {
        'classifier__hidden_layer_sizes': [(64, 32), (32, 16)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__alpha': [0.0001, 0.001]
    }
    
    tuned_mlp = MLPClassifier(max_iter=1000, random_state=seed)
    tuned_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', tuned_mlp)
    ])
    
    start_time = time.time()
    
    if method == 'grid':
        search = GridSearchCV(tuned_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    else:
        search = RandomizedSearchCV(tuned_pipeline, param_grid, cv=3, n_iter=6, scoring='accuracy', random_state=seed, n_jobs=-1)
        
    search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    best_pipeline = search.best_estimator_
    tuned_preds = best_pipeline.predict(X_test)
    tuned_acc = accuracy_score(y_test, tuned_preds)
    
    # Limpiar prefijos del pipeline para los mejores parámetros
    best_params = {k.replace('classifier__', ''): v for k, v in search.best_params_.items()}
    
    return {
        'method': method,
        'search_time': search_time,
        'best_params': best_params,
        'accuracy_before': base_acc,
        'accuracy_after': tuned_acc,
        'best_pipeline': best_pipeline
    }

def interpret_tuning(tuning_results, lang='es'):
    """
    Genera interpretación automática del proceso de ajuste de hiperparámetros.
    """
    from src.interpretation import interpretar_tuning
    return interpretar_tuning(tuning_results, lang=lang)

