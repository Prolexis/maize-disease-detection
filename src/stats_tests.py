import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def compute_bootstrap_ci(y_true, y_pred, metric='accuracy', n_bootstraps=1000, confidence_level=0.95):
    """
    Calcula el intervalo de confianza por bootstrap para la métrica especificada
    (accuracy o f1-score) del modelo sobre el conjunto de prueba.
    """
    from sklearn.metrics import accuracy_score, f1_score
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    bootstrapped_metrics = []
    
    rng = np.random.default_rng(42)
    n_samples = len(y_true)
    
    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, n_samples)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        if n_samples > 0:
            if metric == 'f1-score':
                val = f1_score(sample_true, sample_pred, average='macro', zero_division=0)
            else:
                val = accuracy_score(sample_true, sample_pred)
        else:
            val = 0.0
        bootstrapped_metrics.append(val)
        
    sorted_metrics = np.sort(bootstrapped_metrics)
    lower_pct = (1 - confidence_level) / 2
    upper_pct = 1 - lower_pct
    
    lower_idx = int(lower_pct * n_bootstraps)
    upper_idx = int(upper_pct * n_bootstraps)
    
    return [float(sorted_metrics[lower_idx]), float(sorted_metrics[upper_idx])]

def run_statistical_tests(cv_results, y_test, y_pred_classic=None, y_pred_hybrid=None, alpha=0.05, save_path="reports", lang="es", results=None):
    """
    Ejecuta un análisis estadístico completo sobre los resultados de validación cruzada y test.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Extraer las listas de accuracies de cada modelo
    model_names = list(cv_results.keys())
    data_acc = [cv_results[name]['accuracies'] for name in model_names]
    
    # 1. Validación de supuestos (Normalidad e Homocedasticidad)
    # Shapiro-Wilk para cada grupo
    shapiro_pvals = {}
    normality_holds = True
    for name, accs in cv_results.items():
        try:
            vals = accs['accuracies'] if isinstance(accs, dict) and 'accuracies' in accs else accs
            _, p_val = stats.shapiro(vals)
            import math
            if math.isnan(p_val):
                p_val = 1.0
        except Exception:
            p_val = 1.0  # asumir normalidad si el test falla (pocas muestras)
        shapiro_pvals[name] = p_val
        if p_val < alpha:
            normality_holds = False
            
    # Levene para homogeneidad de varianza
    _, levene_pval = stats.levene(*data_acc)
    variance_holds = levene_pval >= alpha
    
    # 2. ANOVA o Friedman
    use_parametric = normality_holds and variance_holds
    overall_stat = 0.0
    overall_pval = 0.0
    posthoc_results = {}
    
    if use_parametric:
        # ANOVA de una vía
        f_stat, p_val = stats.f_oneway(*data_acc)
        overall_stat = f_stat
        overall_pval = p_val
        test_type = "ANOVA de una vía (Paramétrico)"
        
        # Tukey HSD
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        flat_data = []
        flat_groups = []
        for name in model_names:
            flat_data.extend(cv_results[name]['accuracies'])
            flat_groups.extend([name] * len(cv_results[name]['accuracies']))
        tukey = pairwise_tukeyhsd(endog=flat_data, groups=flat_groups, alpha=alpha)
        # .summary() is a method — must be called with parentheses
        for row in tukey.summary().data[1:]:
            g1, g2, meandiff, p_adj, lower, upper, reject = row
            key = f"{g1} vs {g2}"
            posthoc_results[key] = {
                'diff': meandiff,
                'p_val': p_adj,
                'significant': reject
            }
    else:
        # Friedman no paramétrico
        f_stat, p_val = stats.friedmanchisquare(*data_acc)
        overall_stat = f_stat
        overall_pval = p_val
        test_type = "Friedman (No paramétrico)"
        
        # Test post-hoc de Nemenyi manual utilizando diferencias de rangos
        matrix_accs = np.array(data_acc).T # Shape: (n_folds, n_models)
        ranks = np.array([stats.rankdata(-row) for row in matrix_accs]) # Rango 1 para el mejor valor
        mean_ranks = np.mean(ranks, axis=0) # Rango promedio de cada modelo
        
        n_folds = matrix_accs.shape[0]
        n_models = len(model_names)
        
        # Error estándar para comparación de rangos promedio en Nemenyi:
        # SE = sqrt( k * (k + 1) / (6 * N) )
        se = np.sqrt(n_models * (n_models + 1) / (6.0 * n_folds))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                m1 = model_names[i]
                m2 = model_names[j]
                
                # Estadístico Z de Nemenyi
                rank_diff = abs(mean_ranks[i] - mean_ranks[j])
                z_stat = rank_diff / se
                
                # p-valor bilateral a partir de la normal estándar
                pval = 2.0 * (1.0 - stats.norm.cdf(z_stat))
                
                # Ajuste de Bonferroni-Dunn (multiplicar por número total de comparaciones: k * (k - 1) / 2)
                num_comparisons = n_models * (n_models - 1) / 2
                pval_adj = min(pval * num_comparisons, 1.0)
                
                key = f"{m1} vs {m2}"
                posthoc_results[key] = {
                    'diff': np.mean(cv_results[m1]['accuracies']) - np.mean(cv_results[m2]['accuracies']),
                    'p_val': pval_adj,
                    'significant': bool(pval_adj < alpha),
                    'rank_diff': rank_diff
                }
                    
    # 3. Prueba T de Wilcoxon (Comparación directa del mejor clásico vs mejor híbrido)
    classics = ['Regresión Logística (Clásico)', 'Random Forest (Clásico)', 'Red Neuronal MLP (Clásico)']
    hybrids = ['Híbrido Votación (RF+MLP)', 'Híbrido Stacking (Meta-GB)']
    
    # Asegurar compatibilidad de nombres si los modelos se llaman diferente
    existing_classics = [n for n in classics if n in cv_results]
    existing_hybrids = [n for n in hybrids if n in cv_results]
    
    if not existing_classics:
        existing_classics = [n for n in cv_results.keys() if 'clásico' in n.lower() or 'clasico' in n.lower()]
    if not existing_hybrids:
        existing_hybrids = [n for n in cv_results.keys() if 'híbrido' in n.lower() or 'hibrido' in n.lower()]
    
    if not existing_classics:
        existing_classics = list(cv_results.keys())[:3]
    if not existing_hybrids:
        existing_hybrids = list(cv_results.keys())[3:]
        
    best_classic = max(existing_classics, key=lambda n: np.mean(cv_results[n]['accuracies']))
    best_hybrid = max(existing_hybrids, key=lambda n: np.mean(cv_results[n]['accuracies']))
    
    # Si no se proveen predicciones de test, buscar del diccionario de resultados
    if y_pred_classic is None and results is not None and best_classic in results:
        y_pred_classic = results[best_classic]['y_pred']
    if y_pred_hybrid is None and results is not None and best_hybrid in results:
        y_pred_hybrid = results[best_hybrid]['y_pred']
        
    # Ejecutar comparación best_classic vs best_hybrid
    if y_pred_classic is not None and y_pred_hybrid is not None:
        _, wilc_pval = stats.wilcoxon(cv_results[best_classic]['accuracies'], cv_results[best_hybrid]['accuracies'])
        wilc_stat, _ = stats.wilcoxon(cv_results[best_classic]['accuracies'], cv_results[best_hybrid]['accuracies'])
        
        # 4. Prueba de McNemar (Consistencia de clasificaciones en el Test set)
        contingency = [[0, 0], [0, 0]]
        for true, pred1, pred2 in zip(y_test, y_pred_classic, y_pred_hybrid):
            c1 = (pred1 == true)
            c2 = (pred2 == true)
            if c1 and c2:
                contingency[0][0] += 1
            elif c1 and not c2:
                contingency[0][1] += 1
            elif not c1 and c2:
                contingency[1][0] += 1
            else:
                contingency[1][1] += 1
                
        b = contingency[0][1]
        c = contingency[1][0]
        mcnemar_stat = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0.0
        mcnemar_pval = 1.0 - stats.chi2.cdf(mcnemar_stat, df=1) if (b + c) > 0 else 1.0
    else:
        wilc_stat, wilc_pval = 0.0, 1.0
        contingency = [[0, 0], [0, 0]]
        mcnemar_stat, mcnemar_pval = 0.0, 1.0
    
    # 5. Bootstrap Confidence Intervals (para todos los modelos si 'results' existe)
    bootstrap_ci = {}
    bootstrap_ci_f1 = {}
    if results is not None:
        for name, res in results.items():
            bootstrap_ci[name] = compute_bootstrap_ci(y_test, res['y_pred'], metric='accuracy')
            bootstrap_ci_f1[name] = compute_bootstrap_ci(y_test, res['y_pred'], metric='f1-score')
    else:
        if y_pred_classic is not None:
            bootstrap_ci[best_classic] = compute_bootstrap_ci(y_test, y_pred_classic, metric='accuracy')
            bootstrap_ci_f1[best_classic] = compute_bootstrap_ci(y_test, y_pred_classic, metric='f1-score')
        if y_pred_hybrid is not None:
            bootstrap_ci[best_hybrid] = compute_bootstrap_ci(y_test, y_pred_hybrid, metric='accuracy')
            bootstrap_ci_f1[best_hybrid] = compute_bootstrap_ci(y_test, y_pred_hybrid, metric='f1-score')
            
    # 6. Comparaciones pareadas dinámicas (T-Student vs Wilcoxon según Shapiro-Wilk)
    best_model_name = max(cv_results.keys(), key=lambda n: np.mean(cv_results[n]['accuracies']))
    pairwise_comparisons = {}
    
    for model_name in model_names:
        if model_name == best_model_name:
            continue
            
        acc_a = cv_results[best_model_name]['accuracies']
        acc_b = cv_results[model_name]['accuracies']
        
        # Verificar normalidad de ambos
        norm_a = shapiro_pvals.get(best_model_name, 1.0) >= alpha
        norm_b = shapiro_pvals.get(model_name, 1.0) >= alpha
        
        if norm_a and norm_b:
            # T-Student pareada
            try:
                t_stat, p_val = stats.ttest_rel(acc_a, acc_b)
                test_name = "T-Student pareada (Paramétrica)"
                reason = "Ambos grupos cumplen supuesto de normalidad"
            except Exception:
                t_stat, p_val = 0.0, 1.0
                test_name = "T-Student pareada (Fallo)"
                reason = "Fallo en la prueba T-Student"
        else:
            # Wilcoxon signed-rank
            try:
                t_stat, p_val = stats.wilcoxon(acc_a, acc_b)
                test_name = "Wilcoxon signed-rank (No Paramétrica)"
                reason = "Al menos un grupo no cumple normalidad"
            except Exception:
                t_stat, p_val = 0.0, 1.0
                test_name = "Wilcoxon signed-rank (Fallo)"
                reason = "Fallo en la prueba de Wilcoxon"
                
        pairwise_comparisons[f"{best_model_name} vs {model_name}"] = {
            'test_name': test_name,
            'stat': float(t_stat),
            'p_val': float(p_val),
            'significant': bool(p_val < alpha),
            'reason': reason
        }
    
    # Graficar heatmap de p-valores post-hoc
    plt.figure(figsize=(8, 6))
    p_matrix = np.ones((len(model_names), len(model_names)))
    for idx_i, m1 in enumerate(model_names):
        for idx_j, m2 in enumerate(model_names):
            if m1 == m2:
                p_matrix[idx_i][idx_j] = 1.0
            else:
                key = f"{m1} vs {m2}" if f"{m1} vs {m2}" in posthoc_results else f"{m2} vs {m1}"
                if key in posthoc_results:
                    p_matrix[idx_i][idx_j] = 0.01 if posthoc_results[key]['significant'] else 0.5
                else:
                    p_matrix[idx_i][idx_j] = 0.5
                
    lang_key = lang if lang in ['es', 'en', 'pt'] else 'es'
    title_lbl = {
        'es': f'Mapa de Significancia Estadística Post-Hoc ({test_type})',
        'en': f'Post-Hoc Statistical Significance Map ({test_type})',
        'pt': f'Mapa de Significância Estatística Post-Hoc ({test_type})'
    }.get(lang_key)
    
    translated_model_names = []
    for name in model_names:
        if name == 'Regresión Logística (Clásico)':
            translated_model_names.append({'es': 'Regresión Logística (Clásico)', 'en': 'Logistic Regression (Classic)', 'pt': 'Regressão Logística (Clássico)'}.get(lang_key))
        elif name == 'Random Forest (Clásico)':
            translated_model_names.append({'es': 'Random Forest (Clásico)', 'en': 'Random Forest (Classic)', 'pt': 'Random Forest (Clássico)'}.get(lang_key))
        elif name == 'Red Neuronal MLP (Clásico)':
            translated_model_names.append({'es': 'Red Neuronal MLP (Clásico)', 'en': 'MLP Neural Network (Classic)', 'pt': 'Rede Neural MLP (Clássico)'}.get(lang_key))
        elif name == 'Híbrido Votación (RF+MLP)':
            translated_model_names.append({'es': 'Híbrido Votación (RF+MLP)', 'en': 'Voting Hybrid (RF+MLP)', 'pt': 'Híbrido Votação (RF+MLP)'}.get(lang_key))
        elif name == 'Híbrido Stacking (Meta-GB)':
            translated_model_names.append({'es': 'Híbrido Stacking (Meta-GB)', 'en': 'Stacking Hybrid (Meta-GB)', 'pt': 'Híbrido Stacking (Meta-GB)'}.get(lang_key))
        else:
            translated_model_names.append(name)
            
    sns.heatmap(p_matrix, annot=True, cmap="PiYG", xticklabels=translated_model_names, yticklabels=translated_model_names, cbar=False)
    plt.title(title_lbl)
    plt.tight_layout()
    stats_chart = os.path.join(save_path, 'stats_significance.png')
    plt.savefig(stats_chart, dpi=150)
    plt.close()
    
    return {
        'test_type': test_type,
        'overall_stat': overall_stat,
        'overall_pval': overall_pval,
        'shapiro_pvals': shapiro_pvals,
        'levene_pval': levene_pval,
        'use_parametric': use_parametric,
        'posthoc_results': posthoc_results,
        'bootstrap_ci': bootstrap_ci,
        'bootstrap_ci_f1': bootstrap_ci_f1,
        'pairwise_comparisons': pairwise_comparisons,
        'wilcoxon': {
            'best_classic': best_classic,
            'best_hybrid': best_hybrid,
            'stat': wilc_stat,
            'p_val': wilc_pval
        },
        'mcnemar': {
            'contingency_table': contingency,
            'stat': mcnemar_stat,
            'p_val': mcnemar_pval
        },
        'stats_chart': stats_chart
    }

def interpret_stats(results, alpha=0.05, lang='es'):
    """
    Genera interpretación automática en lenguaje natural de las pruebas estadísticas.
    """
    from src.interpretation import interpretar_stats
    return interpretar_stats(results, alpha=alpha, lang=lang)

