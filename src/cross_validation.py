import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from src.training import get_preprocessor

def run_cross_validation(df, target_col, cv_folds=5, seed=42, save_path="reports"):
    """
    Ejecuta validación cruzada estratificada sobre los 5 modelos.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Inicializar preprocesador y modelos
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    
    preprocessor = get_preprocessor(df, target_col)
    
    clf_lr = LogisticRegression(max_iter=1000, random_state=seed)
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed)
    clf_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=seed)
    clf_h1 = VotingClassifier(estimators=[('rf', clf_rf), ('mlp', clf_mlp)], voting='soft')
    clf_h2 = StackingClassifier(
        estimators=[('lr', clf_lr), ('rf', clf_rf), ('mlp', clf_mlp)],
        final_estimator=GradientBoostingClassifier(n_estimators=50, random_state=seed)
    )
    
    models = {
        'Regresión Logística': clf_lr,
        'Random Forest': clf_rf,
        'Red Neuronal MLP': clf_mlp,
        'Híbrido Votación': clf_h1,
        'Híbrido Stacking': clf_h2
    }
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    cv_results = {}
    
    for name, clf in models.items():
        accuracies = []
        f1_scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clonar estimador para evitar fuga de datos
            model_clone = clone(clf)
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model_clone)
            ])
            
            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_val)
            
            accuracies.append(accuracy_score(y_val, preds))
            f1_scores.append(f1_score(y_val, preds, average='weighted', zero_division=0))
            
        cv_results[name] = {
            'accuracies': accuracies,
            'f1_scores': f1_scores,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores)
        }
        
    return cv_results

def plot_cv_dispersion(cv_results, save_path="reports", lang="es"):
    """
    Dibuja y guarda un boxplot mostrando la dispersión de precisión en los k-folds por modelo.
    """
    os.makedirs(save_path, exist_ok=True)
    
    lang_key = lang if lang in ['es', 'en', 'pt'] else 'es'
    
    # Translations
    title_lbl = {
        'es': 'Variabilidad de Precisión (Accuracy) en Stratified K-Fold',
        'en': 'Accuracy Variability in Stratified K-Fold',
        'pt': 'Variabilidade de Acurácia (Accuracy) em Stratified K-Fold'
    }.get(lang_key)
    
    model_lbl = {
        'es': 'Modelo',
        'en': 'Model',
        'pt': 'Modelo'
    }.get(lang_key)
    
    acc_lbl = {
        'es': 'Accuracy (Precisión)',
        'en': 'Accuracy',
        'pt': 'Acurácia (Accuracy)'
    }.get(lang_key)

    # Construir dataframe plano para seaborn
    data = []
    for name, res in cv_results.items():
        for acc in res['accuracies']:
            data.append({
                model_lbl: name,
                acc_lbl: acc
            })
            
    df_plot = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=model_lbl, y=acc_lbl, data=df_plot, palette="Set3")
    sns.stripplot(x=model_lbl, y=acc_lbl, data=df_plot, color='black', size=5, jitter=0.2, alpha=0.6)
    plt.title(title_lbl)
    plt.xlabel(model_lbl)
    plt.ylabel(acc_lbl)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    cv_chart = os.path.join(save_path, 'cv_dispersion.png')
    plt.savefig(cv_chart, dpi=150)
    plt.close()
    
    return cv_chart

def interpret_cv(cv_results, lang='es'):
    """
    Genera interpretación automática sobre la estabilidad y sobreajuste de los modelos.
    """
    from src.interpretation import interpretar_cv
    return interpretar_cv(cv_results, lang=lang)
