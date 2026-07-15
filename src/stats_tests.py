import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def run_statistical_tests(cv_results, y_test, y_pred_classic, y_pred_hybrid, alpha=0.05, save_path="reports", lang="es"):
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
            _, p_val = stats.shapiro(accs)
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
        
        # Fallback a Wilcoxon para comparaciones post-hoc apareadas con corrección Bonferroni
        from statsmodels.stats.multitest import multipletests
        p_vals_temp = []
        keys_temp = []
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                m1 = model_names[i]
                m2 = model_names[j]
                _, pval = stats.wilcoxon(cv_results[m1]['accuracies'], cv_results[m2]['accuracies'])
                p_vals_temp.append(pval)
                keys_temp.append(f"{m1} vs {m2}")
        if p_vals_temp:
            rejects, p_adjusted, _, _ = multipletests(p_vals_temp, alpha=alpha, method='bonferroni')
            for key, pval, rej in zip(keys_temp, p_adjusted, rejects):
                m1, m2 = key.split(" vs ")
                posthoc_results[key] = {
                    'diff': np.mean(cv_results[m1]['accuracies']) - np.mean(cv_results[m2]['accuracies']),
                    'p_val': pval,
                    'significant': rej
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
    lang_key = lang if lang in ['es', 'en', 'pt'] else 'es'
    
    if lang_key == 'en':
        interpretations = [
            f"**Global Statistical Tests ({results['test_type']}):** Group differences in cross validation were evaluated. The global p-value is **{results['overall_pval']:.4f}**.",
            f"Since the global p-value is {'less' if results['overall_pval'] < alpha else 'greater'} than the configurable significance level alpha = {alpha}, we conclude that **{'there are' if results['overall_pval'] < alpha else 'there are no'} statistically significant differences** in the performance of the 5 models."
        ]
        w_res = results['wilcoxon']
        interpretations.append(
            f"**Wilcoxon Test (Classic vs Hybrid):** When comparing the best classic ({w_res['best_classic']}) against the best hybrid ({w_res['best_hybrid']}), the p-value is **{w_res['p_val']:.4f}**, indicating that the hybrid improvement **{'is' if w_res['p_val'] < alpha else 'is not'} statistically significant**."
        )
        m_res = results['mcnemar']
        interpretations.append(
            f"**McNemar Test (Test Predictions):** Evaluation on the test set yields a p-value of **{m_res['p_val']:.4f}** in classification consistency between both approaches, suggesting that classification error rates **{'differ significantly' if m_res['p_val'] < alpha else 'are statistically equivalent'}**."
        )
    elif lang_key == 'pt':
        interpretations = [
            f"**Testes Estatísticos Globais ({results['test_type']}):** Avaliaram-se as diferenças grupais em validação cruzada. O p-valor global é **{results['overall_pval']:.4f}**.",
            f"Sendo o p-valor global {'menor' if results['overall_pval'] < alpha else 'maior'} que o nível de significância configurável alfa = {alpha}, conclui-se que **{'existem' if results['overall_pval'] < alpha else 'não existem'} diferenças estatisticamente significativas** no desempenho dos 5 modelos."
        ]
        w_res = results['wilcoxon']
        interpretations.append(
            f"**Teste Wilcoxon (Clássico vs Híbrido):** Ao comparar o melhor clássico ({w_res['best_classic']}) contra o melhor híbrido ({w_res['best_hybrid']}), o p-valor é de **{w_res['p_val']:.4f}**, indicando que a melhoria do híbrido **{'é' if w_res['p_val'] < alpha else 'não é'} estatisticamente significativa**."
        )
        m_res = results['mcnemar']
        interpretations.append(
            f"**Teste de McNemar (Previsões de Teste):** A avaliação sobre o conjunto de teste resulta em um p-valor de **{m_res['p_val']:.4f}** na consistência das previsões entre ambas as abordagens, sugerindo que a taxa de erros de classificação **{'difere significativamente' if m_res['p_val'] < alpha else 'é estatisticamente equivalente'}**."
        )
    else:
        interpretations = [
            f"**Pruebas Estadísticas Globales ({results['test_type']}):** Se evaluaron las diferencias grupales en validación cruzada. El p-valor global es **{results['overall_pval']:.4f}**.",
            f"Al ser el p-valor global {'menor' if results['overall_pval'] < alpha else 'mayor'} que el nivel de significancia configurable alfa = {alpha}, se concluye que **{'existen' if results['overall_pval'] < alpha else 'no existen'} diferencias estadísticamente significativas** en el rendimiento de los 5 modelos."
        ]
        w_res = results['wilcoxon']
        interpretations.append(
            f"**Prueba Wilcoxon (Clásico vs Híbrido):** Al comparar el mejor clásico ({w_res['best_classic']}) contra el mejor híbrido ({w_res['best_hybrid']}), el p-valor es de **{w_res['p_val']:.4f}**, indicando que la mejora del híbrido **{'es' if w_res['p_val'] < alpha else 'no es'} estadísticamente significativa**."
        )
        m_res = results['mcnemar']
        interpretations.append(
            f"**Prueba de McNemar (Predicciones de Test):** La evaluación sobre el conjunto de prueba arroja un p-valor de **{m_res['p_val']:.4f}** en la consistencia de predicciones entre ambos enfoques, sugiriendo que la tasa de errores de clasificación **{'difiere significativamente' if m_res['p_val'] < alpha else 'es estadísticamente equivalente'}**."
        )
    return "\n\n".join(interpretations)
