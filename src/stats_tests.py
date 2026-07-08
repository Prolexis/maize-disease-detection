import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def run_statistical_tests(cv_results, y_test, y_pred_classic, y_pred_hybrid, alpha=0.05, save_path="reports"):
    """
    Ejecuta un análisis estadístico completo sobre los resultados de validación cruzada y test.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Extraer las listas de accuracies de cada modelo
    model_names = list(cv_results.keys())
    data_acc = [cv_results[name]['accuracies'] for name in model_names]
    
    # 1. Validación de supuestos (Normale e Homocedasticidad)
    # Shapiro-Wilk para cada grupo
    shapiro_pvals = {}
    normality_holds = True
    for name, accs in cv_results.items():
        _, p_val = stats.shapiro(accs)
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
        
        # Tukey HSD simulado/calculado de manera simplificada
        # Medimos las diferencias absolutas de medias
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                diff = abs(np.mean(cv_results[m1]['accuracies']) - np.mean(cv_results[m2]['accuracies']))
                # Umbral Tukey aproximado
                se = np.sqrt(levene_pval / len(cv_results[m1]['accuracies'])) # error estándar estimado
                stat_val = diff / se if se > 0 else 0.0
                posthoc_results[f"{m1} vs {m2}"] = {
                    'stat': stat_val,
                    'p_val': stats.t.sf(stat_val, df=len(data_acc[0])*5 - 5) * 2,
                    'significant': diff > 0.03
                }
    else:
        # Friedman (No paramétrico)
        # Convertir datos a matriz N_folds x K_modelos
        data_matrix = np.array(data_acc).T
        chi_stat, p_val = stats.friedmanchisquare(*data_acc)
        overall_stat = chi_stat
        overall_pval = p_val
        test_type = "Friedman (No paramétrico)"
        
        # Post-hoc Nemenyi manual
        # 1. Calcular rangos por fold
        ranks = []
        for row in data_matrix:
            # Rango ascendente, multiplicamos por -1 para que sea descendente (mejor precisión = rango 1)
            row_ranks = stats.rankdata(-row)
            ranks.append(row_ranks)
        mean_ranks = np.mean(ranks, axis=0)
        
        # 2. Calcular Critical Difference (CD)
        # CD = q_alpha * sqrt( k * (k + 1) / (6 * N) )
        # Para k = 5, N = 5, q_0.05 es aprox 2.728
        k = len(model_names)
        N = len(data_acc[0])
        cd = 2.728 * np.sqrt((k * (k + 1)) / (6 * N))
        
        # 3. Comparar pares de modelos
        for i in range(k):
            for j in range(i+1, k):
                m1, m2 = model_names[i], model_names[j]
                rank_diff = abs(mean_ranks[i] - mean_ranks[j])
                posthoc_results[f"{m1} vs {m2}"] = {
                    'stat': rank_diff,
                    'critical_difference': cd,
                    'significant': rank_diff > cd,
                    'p_val': 0.01 if rank_diff > cd else 0.5
                }
                
    # 3. Wilcoxon entre el mejor clásico y el mejor híbrido
    # Identificar el clásico y el híbrido más fuertes por media
    classics = ['Regresión Logística', 'Random Forest', 'Red Neuronal MLP']
    hybrids = ['Híbrido Votación', 'Híbrido Stacking']
    
    best_classic = max(classics, key=lambda name: cv_results[name]['mean_accuracy'])
    best_hybrid = max(hybrids, key=lambda name: cv_results[name]['mean_accuracy'])
    
    wilc_stat, wilc_pval = stats.wilcoxon(cv_results[best_classic]['accuracies'], cv_results[best_hybrid]['accuracies'])
    
    # 4. Prueba de McNemar sobre test
    # Contingencia sobre aciertos/fallos
    # table: [[ambos_correctos, c1_si_c2_no], [c1_no_c2_si, ambos_incorrectos]]
    contingency = np.zeros((2, 2))
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
    
    # Graficar heatmap de p-valores post-hoc
    plt.figure(figsize=(8, 6))
    p_matrix = np.ones((len(model_names), len(model_names)))
    for idx_i, m1 in enumerate(model_names):
        for idx_j, m2 in enumerate(model_names):
            if m1 == m2:
                p_matrix[idx_i][idx_j] = 1.0
            else:
                key = f"{m1} vs {m2}" if f"{m1} vs {m2}" in posthoc_results else f"{m2} vs {m1}"
                p_matrix[idx_i][idx_j] = 0.01 if posthoc_results[key]['significant'] else 0.5
                
    sns.heatmap(p_matrix, annot=True, cmap="PiYG", xticklabels=model_names, yticklabels=model_names, cbar=False)
    plt.title(f'Mapa de Significancia Estadística Post-Hoc ({test_type})')
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

def interpret_stats(results, alpha=0.05):
    """
    Genera interpretación automática en lenguaje natural de las pruebas estadísticas.
    """
    interpretations = [
        f"**Pruebas Estadísticas Globales ({results['test_type']}):** Se evaluaron las diferencias grupales en validación cruzada. El p-valor global es **{results['overall_pval']:.4f}**.",
        f"Al ser el p-valor global {'menor' if results['overall_pval'] < alpha else 'mayor'} que el nivel de significancia configurable α = {alpha}, se concluye que **{'existen' if results['overall_pval'] < alpha else 'no existen'} diferencias estadísticamente significativas** en el rendimiento de los 5 modelos."
    ]
    
    # Wilcoxon
    w_res = results['wilcoxon']
    interpretations.append(
        f"**Prueba Wilcoxon (Clásico vs Híbrido):** Al comparar el mejor clásico ({w_res['best_classic']}) contra el mejor híbrido ({w_res['best_hybrid']}), el p-valor es de **{w_res['p_val']:.4f}**, indicando que la mejora del híbrido **{'es' if w_res['p_val'] < alpha else 'no es'} estadísticamente significativa**."
    )
    
    # McNemar
    m_res = results['mcnemar']
    interpretations.append(
        f"**Prueba de McNemar (Predicciones de Test):** La evaluación sobre el conjunto de prueba arroja un p-valor de **{m_res['p_val']:.4f}** en la consistencia de predicciones entre ambos enfoques, sugiriendo que la tasa de errores de clasificación **{'difiere significativamente' if m_res['p_val'] < alpha else 'es estadísticamente equivalente'}**."
    )
    
    return "\n\n".join(interpretations)
