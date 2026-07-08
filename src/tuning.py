import os
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from src.training import get_preprocessor

def run_hyperparameter_tuning(df, target_col, method='grid', seed=42):
    """
    Optimiza el modelo Random Forest como ejemplo de tuning para evitar sobrecargas de tiempo.
    Compara resultados antes y después del tuning.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=seed, stratify=y)
    
    preprocessor = get_preprocessor(df, target_col)
    
    # 1. Modelo Base (Sin tuning)
    base_rf = RandomForestClassifier(random_state=seed)
    base_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', base_rf)
    ])
    
    base_pipeline.fit(X_train, y_train)
    base_preds = base_pipeline.predict(X_test)
    base_acc = accuracy_score(y_test, base_preds)
    
    # 2. Espacio de hiperparámetros
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [5, 8, 12],
        'classifier__min_samples_split': [2, 5]
    }
    
    tuned_rf = RandomForestClassifier(random_state=seed)
    tuned_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', tuned_rf)
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

def interpret_tuning(tuning_results):
    """
    Genera interpretación automática del proceso de ajuste de hiperparámetros.
    """
    diff = tuning_results['accuracy_after'] - tuning_results['accuracy_before']
    best_params_str = ", ".join([f"{k}={v}" for k, v in tuning_results['best_params'].items()])
    
    interpretations = [
        f"**Ajuste de Hiperparámetros ({tuning_results['method'].upper()}):** El proceso de optimización tomó **{tuning_results['search_time']:.3f} segundos**.",
        f"Los mejores parámetros encontrados para Random Forest son: **{best_params_str}**.",
        f"La precisión en el conjunto de prueba pasó de **{tuning_results['accuracy_before']:.2%}** a **{tuning_results['accuracy_after']:.2%}** (un incremento neto de **{diff*100:+.2f}%**)."
    ]
    return "\n\n".join(interpretations)
