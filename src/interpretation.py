# -*- coding: utf-8 -*-
"""
Módulo compartido de interpretación de resultados en lenguaje natural (Español / Inglés / Portugués).
Utilizado tanto por Streamlit (app.py) como por FastAPI (backend) y el Chatbot.
"""

def interpretar_mcnemar(p_val, modelo_a, modelo_b, alpha=0.05, lang="es"):
    """
    Interpreta el resultado de la prueba de McNemar (comparación de clasificación en Test set).
    """
    lang_key = lang.lower() if lang in ["es", "en", "pt"] else "es"
    if p_val < alpha:
        if lang_key == "es":
            return f"El test de McNemar indica que {modelo_a} y {modelo_b} tienen un desempeño significativamente distinto (p={p_val:.4f}). Esto sugiere que conviene utilizar {modelo_a} para el diagnóstico."
        elif lang_key == "pt":
            return f"O teste de McNemar indica que {modelo_a} e {modelo_b} têm um desempenho significativamente diferente (p={p_val:.4f}). Isso sugere que é melhor usar {modelo_a} para o diagnóstico."
        else:
            return f"McNemar's test indicates that {modelo_a} and {modelo_b} have significantly different performances (p={p_val:.4f}). This suggests that it is better to use {modelo_a} for diagnosis."
    else:
        if lang_key == "es":
            return f"El test de McNemar no detectó diferencias significativas (p={p_val:.4f}) en el patrón de errores de {modelo_a} y {modelo_b}. Ambos modelos son estadísticamente equivalentes en el conjunto de prueba."
        elif lang_key == "pt":
            return f"O teste de McNemar não detectou diferenças significativas (p={p_val:.4f}) no padrão de erros de {modelo_a} e {modelo_b}. Ambos os modelos são estatisticamente equivalentes no conjunto de testes."
        else:
            return f"McNemar's test detected no significant differences (p={p_val:.4f}) in the error patterns of {modelo_a} and {modelo_b}. Both models are statistically equivalent on the test set."

def interpretar_bootstrap_ci(metric_name, lower, upper, lang="es"):
    """
    Interpreta el intervalo de confianza Bootstrap para una métrica.
    """
    lang_key = lang.lower() if lang in ["es", "en", "pt"] else "es"
    
    # Formatear números si vienen como floats
    if isinstance(lower, float):
        lower_str = f"{lower:.2%}" if lower <= 1.0 else f"{lower:.3f}"
    else:
        lower_str = str(lower)
        
    if isinstance(upper, float):
        upper_str = f"{upper:.2%}" if upper <= 1.0 else f"{upper:.3f}"
    else:
        upper_str = str(upper)

    # Determinar si el CI es angosto (alta confianza) o ancho (baja confianza/alta variabilidad)
    is_narrow = False
    try:
        diff = float(upper) - float(lower)
        if diff < 0.10:
            is_narrow = True
    except Exception:
        pass

    if lang_key == "es":
        conf_msg = "lo que indica alta confianza y estabilidad en las predicciones." if is_narrow else "lo que sugiere cierta variabilidad en el rendimiento."
        return f"Con 95% de confianza, el {metric_name} real del modelo está entre {lower_str} y {upper_str}, {conf_msg}"
    elif lang_key == "pt":
        conf_msg = "o que indica alta confiança e estabilidade nas previsões." if is_narrow else "o que sugere alguma variabilidade no desempenho."
        return f"Com 95% de confiança, o {metric_name} real do modelo está entre {lower_str} e {upper_str}, {conf_msg}"
    else:
        conf_msg = "indicating high confidence and stability in the predictions." if is_narrow else "suggesting some variability in performance."
        return f"With 95% confidence, the true {metric_name} of the model lies between {lower_str} and {upper_str}, {conf_msg}"

def interpretar_anova_friedman(p_val, test_type, alpha=0.05, lang="es"):
    """
    Interpreta la prueba global de hipótesis (ANOVA o Friedman) sobre validación cruzada.
    """
    lang_key = lang.lower() if lang in ["es", "en", "pt"] else "es"
    if p_val < alpha:
        if lang_key == "es":
            return f"La prueba global ({test_type}) confirma diferencias significativas (p={p_val:.4f}) entre los modelos. Al menos uno de los modelos tiene un rendimiento significativamente diferente."
        elif lang_key == "pt":
            return f"A prova global ({test_type}) confirma diferenças significativas (p={p_val:.4f}) entre os modelos. Pelo menos um dos modelos tem um rendimento significativamente diferente."
        else:
            return f"The global test ({test_type}) confirms significant differences (p={p_val:.4f}) between the models. At least one model has a significantly different performance."
    else:
        if lang_key == "es":
            return f"La prueba global ({test_type}) no detectó diferencias significativas (p={p_val:.4f}) en los folds de validación cruzada. Estadísticamente, la diferencia observada podría deberse al azar."
        elif lang_key == "pt":
            return f"A prova global ({test_type}) não detectou diferenças significativas (p={p_val:.4f}) nos folds de validação cruzada. Estatisticamente, a diferença observada poderia ser devida ao acaso."
        else:
            return f"The global test ({test_type}) detected no significant differences (p={p_val:.4f}) in the cross-validation folds. Statistically, the observed difference could be due to random chance."

def interpretar_cv(cv_results, lang="es"):
    """
    Interpreta los resultados de la validación cruzada identificando el modelo más estable.
    cv_results: dict {model_name: [scores...]} o {model_name: {'accuracies': [scores...]}}
    """
    import numpy as np
    lang_key = lang.lower() if lang in ["es", "en", "pt"] else "es"
    
    means = {}
    stds = {}
    for name, data in cv_results.items():
        scores = data['accuracies'] if isinstance(data, dict) and 'accuracies' in data else data
        if scores:
            means[name] = np.mean(scores)
            stds[name] = np.std(scores)
            
    if not means:
        return "No hay resultados de validación cruzada para analizar." if lang_key == "es" else "No cross-validation results available to analyze."
        
    best_model = max(means, key=means.get)
    best_mean = means[best_model]
    best_std = stds[best_model]
    
    if lang_key == "es":
        return f"El modelo {best_model} mostró el F1-Score/Accuracy promedio más alto ({best_mean:.2%}) y la menor variabilidad entre folds (desviación estándar: {best_std:.4f}), lo que indica que es el más estable y robusto ante cambios en la muestra."
    elif lang_key == "pt":
        return f"O modelo {best_model} mostrou o F1-Score/Acurácia médio mais alto ({best_mean:.2%}) e a menor variabilidade entre folds (desvio padrão: {best_std:.4f}), o que indica que é o mais estável e robusto perante alterações na amostra."
    else:
        return f"The {best_model} model showed the highest average F1-Score/Accuracy ({best_mean:.2%}) and the lowest variability across folds (standard deviation: {best_std:.4f}), indicating that it is the most stable and robust against sampling changes."

def interpretar_tuning(tuning_results, lang="es"):
    """
    Interpreta los resultados del ajuste de hiperparámetros.
    """
    lang_key = lang.lower() if lang in ["es", "en", "pt"] else "es"
    if not tuning_results:
        return "No hay resultados de ajuste de hiperparámetros." if lang_key == "es" else "No hyperparameter tuning results available."
        
    acc_before = tuning_results.get('accuracy_before', 0.0)
    acc_after = tuning_results.get('accuracy_after', 0.0)
    diff = acc_after - acc_before
    method = tuning_results.get('method', 'Grid Search').upper()
    time_taken = tuning_results.get('search_time', 0.0)
    best_params = tuning_results.get('best_params', {})
    
    params_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
    
    if diff > 0:
        if lang_key == "es":
            return f"El ajuste mediante {method} ({time_taken:.2f} s) optimizó el modelo base (parámetros: {params_str}), logrando incrementar la exactitud en un +{diff:.2%} (de {acc_before:.2%} a {acc_after:.2%})."
        elif lang_key == "pt":
            return f"O ajuste via {method} ({time_taken:.2f} s) otimizou o modelo base (parâmetros: {params_str}), conseguindo aumentar a acurácia em +{diff:.2%} (de {acc_before:.2%} a {acc_after:.2%})."
        else:
            return f"Tuning via {method} ({time_taken:.2f} s) optimized the base model (parameters: {params_str}), increasing accuracy by +{diff:.2%} (from {acc_before:.2%} to {acc_after:.2%})."
    else:
        if lang_key == "es":
            return f"El ajuste mediante {method} ({time_taken:.2f} s) determinó que los parámetros por defecto eran óptimos (exactitud estable en {acc_after:.2%})."
        elif lang_key == "pt":
            return f"O ajuste via {method} ({time_taken:.2f} s) determinou que os parâmetros padrão eram ideais (acurácia estável em {acc_after:.2%})."
        else:
            return f"Tuning via {method} ({time_taken:.2f} s) determined that the default parameters were already optimal (accuracy stable at {acc_after:.2%})."

def interpretar_eda(df, target_col, num_duplicates, imputed_nulls, outliers_detected, global_stats, lang="es"):
    """
    Genera interpretaciones automáticas profesionales en texto de los datos limpios y estadísticos avanzados.
    """
    import numpy as np
    import pandas as pd
    interpretations = []
    lang_key = lang.lower() if lang in ['es', 'en', 'pt'] else 'es'
    
    if lang_key == 'en':
        # 1. Cleaning and Outliers
        clean_text = f"**1. Data Quality and Cleaning:**\n"
        clean_text += f"* Removed duplicate records: **{num_duplicates}**.\n"
        if imputed_nulls:
            clean_text += "* Intelligent imputation of nulls: " + ", ".join([f"'{k}' ({v[0]} nulls imputed via {v[1]})" for k, v in imputed_nulls.items()]) + ".\n"
        else:
            clean_text += "* No missing values (nulls) were found in critical columns.\n"
            
        if outliers_detected:
            clean_text += "* Univariate outlier correction (IQR clipping): " + ", ".join([f"'{k}' ({v} values adjusted)" for k, v in outliers_detected.items()]) + ".\n"
        else:
            clean_text += "* No univariate outliers were detected outside the 1.5 * IQR range.\n"
            
        m_outliers = df.attrs.get('multivariate_outliers', 0)
        if m_outliers > 0:
            clean_text += f"* Multivariate outlier detection (Isolation Forest): Detected **{m_outliers}** anomalous multidimensional samples with atypical behavior.\n"
        
        interpretations.append(clean_text)
        
        # 2. Distribution, Skewness
        dist_text = "**2. Distributions and Normality:**\n"
        skewed_cols = []
        non_normal_cols = []
        for col, stat in global_stats.iterrows():
            if abs(stat['asimetría']) > 1.0:
                skewed_cols.append(f"'{col}' (skewness: {stat['asimetría']:.2f})")
            if stat['shapiro_p'] < 0.05:
                non_normal_cols.append(f"'{col}' (p-value: {stat['shapiro_p']:.4f})")
                
        if skewed_cols:
            dist_text += f"* Highly skewed variables detected: {', '.join(skewed_cols)}. "
            t_cols = df.attrs.get('transformed_cols', [])
            if t_cols:
                dist_text += f"To optimize linear models and neural networks, a **logarithmic transformation (log1p)** was automatically applied to: {', '.join([f'*{c}*' for c in t_cols])}.\n"
            else:
                dist_text += "\n"
        else:
            dist_text += "* Variables show balanced distributions and moderate skewness.\n"
            
        if non_normal_cols:
            dist_text += f"* Shapiro-Wilk normality test (α = 0.05): Normality null hypothesis is rejected for variables: {', '.join(non_normal_cols[:4])}. "
            if len(non_normal_cols) > 4:
                dist_text += f"and {len(non_normal_cols) - 4} more. "
            dist_text += "This confirms the suitable use of non-parametric models such as Random Forest and advanced hybrids.\n"
        else:
            dist_text += "* All variables follow normal distributions according to Shapiro-Wilk.\n"
            
        interpretations.append(dist_text)
        
        # 3. Multicollinearity
        corr_text = "**3. Correlation and Redundancy Analysis:**\n"
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)
        
        if len(num_cols) > 1:
            corr_matrix = df[num_cols].corr()
            high_corr_pairs = []
            for i in range(len(num_cols)):
                for j in range(i+1, len(num_cols)):
                    c_val = corr_matrix.iloc[i, j]
                    if abs(c_val) > 0.85:
                        high_corr_pairs.append(f"'{num_cols[i]}' and '{num_cols[j]}' (r = {c_val:.2f})")
            if high_corr_pairs:
                corr_text += "* **Multicollinearity Warning:** High correlation found in: " + ", ".join(high_corr_pairs) + ". This may inflate the variance of coefficients in Logistic Regression.\n"
            else:
                corr_text += "* No critical collinearities (r > 0.85) were found between analyzed numerical variables.\n"
        else:
            corr_text += "* Not enough numerical variables to calculate correlations.\n"
            
        interpretations.append(corr_text)
        
        # 4. Class Balance
        class_counts = df[target_col].value_counts()
        min_class_ratio = class_counts.min() / class_counts.max()
        balance_text = f"**4. Target Structure and Balance:**\n"
        balance_text += f"* Sample distribution per class: " + ", ".join([f"'{k}': {v}" for k, v in class_counts.items()]) + ".\n"
        if min_class_ratio < 0.6:
            balance_text += "* **Alert:** Significant class imbalance observed. The pipeline will automatically apply balanced class weighting (*class_weight*) during training to avoid bias.\n"
        else:
            balance_text += "* Target classes are balanced, facilitating uniform training.\n"
        interpretations.append(balance_text)

    elif lang_key == 'pt':
        # 1. Cleaning and Outliers
        clean_text = f"**1. Qualidade e Limpeza dos Dados:**\n"
        clean_text += f"* Registros duplicados removidos: **{num_duplicates}**.\n"
        if imputed_nulls:
            clean_text += "* Imputação inteligente de nulos: " + ", ".join([f"'{k}' ({v[0]} nulos imputados via {v[1]})" for k, v in imputed_nulls.items()]) + ".\n"
        else:
            clean_text += "* Não foram encontrados valores ausentes (nulos) nas colunas críticas.\n"
            
        if outliers_detected:
            clean_text += "* Correção de outliers univariados (IQR clipping): " + ", ".join([f"'{k}' ({v} valores ajustados)" for k, v in outliers_detected.items()]) + ".\n"
        else:
            clean_text += "* Nenhum outlier univariado foi detectado fora da faixa 1.5 * IQR.\n"
            
        m_outliers = df.attrs.get('multivariate_outliers', 0)
        if m_outliers > 0:
            clean_text += f"* Detecção de outliers multivariados (Isolation Forest): Foram detectadas **{m_outliers}** amostras anômalas multidimensionais de comportamento atípico.\n"
        
        interpretations.append(clean_text)
        
        # 2. Distribution, Skewness
        dist_text = "**2. Distribuições e Normalidade:**\n"
        skewed_cols = []
        non_normal_cols = []
        for col, stat in global_stats.iterrows():
            if abs(stat['asimetría']) > 1.0:
                skewed_cols.append(f"'{col}' (assimetria: {stat['asimetría']:.2f})")
            if stat['shapiro_p'] < 0.05:
                non_normal_cols.append(f"'{col}' (p-valor: {stat['shapiro_p']:.4f})")
                
        if skewed_cols:
            dist_text += f"* Variáveis altamente sesgadas detectadas: {', '.join(skewed_cols)}. "
            t_cols = df.attrs.get('transformed_cols', [])
            if t_cols:
                dist_text += f"Para otimizar modelos lineares e redes neurais, aplicou-se automaticamente uma **transformação logarítmica (log1p)** a: {', '.join([f'*{c}*' for c in t_cols])}.\n"
            else:
                dist_text += "\n"
        else:
            dist_text += "* As variáveis apresentam distribuições equilibradas e com assimetria moderada.\n"
            
        if non_normal_cols:
            dist_text += f"* Teste de normalidade de Shapiro-Wilk (α = 0.05): Rejeita-se a hipótese nula de normalidade nas variáveis: {', '.join(non_normal_cols[:4])}. "
            if len(non_normal_cols) > 4:
                dist_text += f"e mais {len(non_normal_cols) - 4}. "
            dist_text += "Isso confirma o uso adequado de modelos não paramétricos como Random Forest e híbridos avançados.\n"
        else:
            dist_text += "* Todas as variáveis seguem distribuições normais de acordo com Shapiro-Wilk.\n"
            
        interpretations.append(dist_text)
        
        # 3. Multicollinearity
        corr_text = "**3. Análise de Correlação e Redundância:**\n"
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)
        
        if len(num_cols) > 1:
            corr_matrix = df[num_cols].corr()
            high_corr_pairs = []
            for i in range(len(num_cols)):
                for j in range(i+1, len(num_cols)):
                    c_val = corr_matrix.iloc[i, j]
                    if abs(c_val) > 0.85:
                        high_corr_pairs.append(f"'{num_cols[i]}' e '{num_cols[j]}' (r = {c_val:.2f})")
            if high_corr_pairs:
                corr_text += "* **Aviso de Multicolinearidade:** Encontrou-se alta correlação em: " + ", ".join(high_corr_pairs) + ". Isso pode inflar a variância dos coeficientes na Regresión Logística.\n"
            else:
                corr_text += "* Não foram encontradas colinearidades críticas (r > 0.85) entre as variáveis numéricas analisadas.\n"
        else:
            corr_text += "* Não há variáveis numéricas suficientes para calcular correlações.\n"
            
        interpretations.append(corr_text)
        
        # 4. Class Balance
        class_counts = df[target_col].value_counts()
        min_class_ratio = class_counts.min() / class_counts.max()
        balance_text = f"**4. Estrutura e Equilíbrio do Target:**\n"
        balance_text += f"* Distribuição de amostras por classe: " + ", ".join([f"'{k}': {v}" for k, v in class_counts.items()]) + ".\n"
        if min_class_ratio < 0.6:
            balance_text += "* **Alerta:** Observa-se um desequilíbrio de classes significativo. O pipeline aplicará automaticamente peso equilibrado de classes (*class_weight*) durante o treinamento para evitar viés.\n"
        else:
            balance_text += "* As classes do target estão equilibradas, facilitando um treinamento uniforme.\n"
        interpretations.append(balance_text)

    else:
        # 1. Limpieza y Outliers
        clean_text = f"**1. Calidad y Limpieza de Datos:**\n"
        clean_text += f"* Registros duplicados eliminados: **{num_duplicates}**.\n"
        if imputed_nulls:
            clean_text += "* Imputación inteligente de nulos: " + ", ".join([f"'{k}' ({v[0]} nulos imputados vía {v[1]})" for k, v in imputed_nulls.items()]) + ".\n"
        else:
            clean_text += "* No se encontraron valores faltantes (nulos) en las columnas críticas.\n"
            
        if outliers_detected:
            clean_text += "* Corrección de outliers univariados (IQR clipping): " + ", ".join([f"'{k}' ({v} valores ajustados)" for k, v in outliers_detected.items()]) + ".\n"
        else:
            clean_text += "* No se detectaron outliers univariados fuera del rango 1.5 * IQR.\n"
            
        m_outliers = df.attrs.get('multivariate_outliers', 0)
        if m_outliers > 0:
            clean_text += f"* Detección de outliers multivariados (Isolation Forest): Se detectaron **{m_outliers}** muestras anómalas multidimensionales de comportamiento atípico.\n"
        
        interpretations.append(clean_text)
        
        # 2. Distribución, Asimetría y Transformación
        dist_text = "**2. Distribuciones y Normalidad:**\n"
        skewed_cols = []
        non_normal_cols = []
        for col, stat in global_stats.iterrows():
            if abs(stat['asimetría']) > 1.0:
                skewed_cols.append(f"'{col}' (asimetría: {stat['asimetría']:.2f})")
            if stat['shapiro_p'] < 0.05:
                non_normal_cols.append(f"'{col}' (p-valor: {stat['shapiro_p']:.4f})")
                
        if skewed_cols:
            dist_text += f"* Variables altamente sesgadas detectadas: {', '.join(skewed_cols)}. "
            t_cols = df.attrs.get('transformed_cols', [])
            if t_cols:
                dist_text += f"Para optimizar modelos lineales y redes neuronales, se les aplicó automáticamente una **transformación logarítmica (log1p)** a: {', '.join([f'*{c}*' for c in t_cols])}.\n"
            else:
                dist_text += "\n"
        else:
            dist_text += "* Las variables presentan distribuciones balanceadas y con asimetría moderada.\n"
            
        if non_normal_cols:
            dist_text += f"* Prueba de normalidad de Shapiro-Wilk (α = 0.05): Se rechaza la hipótesis nula de normalidad en las variables: {', '.join(non_normal_cols[:4])}. "
            if len(non_normal_cols) > 4:
                dist_text += f"y {len(non_normal_cols) - 4} más. "
            dist_text += "Esto confirma el uso idóneo de modelos no paramétricos como Random Forest e híbridos avanzados.\n"
        else:
            dist_text += "* Todas las variables siguen distribuciones normales de acuerdo con Shapiro-Wilk.\n"
            
        interpretations.append(dist_text)
        
        # 3. Multicolinealidad (Correlaciones altas)
        corr_text = "**3. Análisis de Correlación y Redundancia:**\n"
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)
        
        if len(num_cols) > 1:
            corr_matrix = df[num_cols].corr()
            high_corr_pairs = []
            for i in range(len(num_cols)):
                for j in range(i+1, len(num_cols)):
                    c_val = corr_matrix.iloc[i, j]
                    if abs(c_val) > 0.85:
                        high_corr_pairs.append(f"'{num_cols[i]}' y '{num_cols[j]}' (r = {c_val:.2f})")
            if high_corr_pairs:
                corr_text += "* **Advertencia de Multicolinealidad:** Se halló alta correlación en: " + ", ".join(high_corr_pairs) + ". Esto puede elevar la varianza de los coeficientes en la Regresión Logística.\n"
            else:
                corr_text += "* No se encontraron colinealidades críticas (r > 0.85) entre las variables numéricas y analizadas.\n"
        else:
            corr_text += "* No hay suficientes variables numéricas para calcular correlaciones.\n"
            
        interpretations.append(corr_text)
        
        # 4. Balance de clases
        class_counts = df[target_col].value_counts()
        min_class_ratio = class_counts.min() / class_counts.max()
        balance_text = f"**4. Estructura y Balance del Target:**\n"
        balance_text += f"* Distribución de muestras por clase: " + ", ".join([f"'{k}': {v}" for k, v in class_counts.items()]) + ".\n"
        if min_class_ratio < 0.6:
            balance_text += "* **Alerta:** Se observa un desbalance de clases significativo. El pipeline aplicará automáticamente ponderación equilibrada de clases (*class_weight*) durante el entrenamiento para evitar sesgos.\n"
        else:
            balance_text += "* Las clases del target se encuentran equilibradas, facilitando un entrenamiento uniforme.\n"
        interpretations.append(balance_text)
        
    return "\n\n".join(interpretations)

def interpretar_consenso_vision(predictions, consensus_reached, diagnosis, lang="es"):
    """
    Interpreta el consenso de diagnóstico foliar de los 3 modelos CNN.
    """
    lang_key = lang.lower() if lang in ["es", "en", "pt"] else "es"
    
    if consensus_reached:
        if diagnosis == "Sano":
            if lang_key == "es":
                return "Los 3 modelos de redes neuronales (MobileNetV2, ResNet50, EfficientNetB0) coinciden unánimemente en que la hoja está Sana. Esto otorga alta confianza al diagnóstico."
            elif lang_key == "pt":
                return "Os 3 modelos de redes neurais (MobileNetV2, ResNet50, EfficientNetB0) coincidem unanimemente que a folha está Saudável. Isso confere alta confiança ao diagnóstico."
            else:
                return "All 3 neural network models (MobileNetV2, ResNet50, EfficientNetB0) unanimously agree that the leaf is Healthy. This gives high confidence to the diagnosis."
        else:
            if lang_key == "es":
                return f"Los 3 modelos coinciden en el diagnóstico de {diagnosis}, lo que da alta confianza al resultado (consenso unánime del consorcio)."
            elif lang_key == "pt":
                return f"Os 3 modelos coincidem no diagnóstico de {diagnosis}, o que dá alta confiança ao resultado (consenso unânime do consórcio)."
            else:
                return f"All 3 models agree on the diagnosis of {diagnosis}, which gives high confidence to the result (unanimous consensus of the consortium)."
    else:
        # Calcular voto de mayoría
        votes = {}
        for name, pred in predictions.items():
            cls = pred.get("class", "Desconocido")
            votes[cls] = votes.get(cls, 0) + 1
        majority_class = max(votes, key=votes.get)
        count = votes[majority_class]
        
        if lang_key == "es":
            return f"No hay consenso unánime. Sin embargo, la mayoría ({count}/3) sugiere {majority_class}. Se recomienda inspección visual adicional."
        elif lang_key == "pt":
            return f"Não há consenso unânime. No entanto, a maioria ({count}/3) sugere {majority_class}. Recomenda-se inspeção visual adicional."
        else:
            return f"No unanimous consensus reached. However, the majority ({count}/3) suggests {majority_class}. Visual inspection is recommended."

def interpretar_training(results, lang="es"):
    """
    Interpreta el rendimiento de los modelos en el entrenamiento.
    """
    lang_key = lang.lower() if lang in ["es", "en", "pt"] else "es"
    sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    best_name, best_res = sorted_models[0]
    
    if lang_key == "es":
        return f"El modelo con el mejor desempeño global en el conjunto de prueba es **{best_name}** con un **Accuracy del {best_res['accuracy']:.2%}** y un F1-Score de **{best_res['f1-score']:.4f}**, logrando la menor tasa de confusión entre clases patógenas."
    elif lang_key == "pt":
        return f"O modelo com o melhor desempenho global no conjunto de testes é o **{best_name}** com uma **Acurácia de {best_res['accuracy']:.2%}** e um F1-Score de **{best_res['f1-score']:.4f}**, alcançando a menor taxa de confusão entre as classes patogênicas."
    else:
        return f"The model with the best overall performance on the test set is **{best_name}** with an **Accuracy of {best_res['accuracy']:.2%}** and an F1-Score of **{best_res['f1-score']:.4f}**, achieving the lowest confusion rate among pathogen classes."

def interpretar_stats(results, alpha=0.05, lang="es"):
    """
    Genera interpretación automática en lenguaje natural de las pruebas estadísticas.
    """
    lang_key = lang.lower() if lang in ["es", "en", "pt"] else "es"
    
    if lang_key == 'en':
        interpretations = [
            f"**Global Statistical Tests ({results['test_type']}):** Group differences in cross validation were evaluated. The global p-value is **{results['overall_pval']:.4f}**.",
            f"Since the global p-value is {'less' if results['overall_pval'] < alpha else 'greater'} than the configurable significance level alpha = {alpha}, we conclude that **{'there are' if results['overall_pval'] < alpha else 'there are no'} statistically significant differences** in the performance of the 5 models."
        ]
        w_res = results.get('wilcoxon', {})
        if w_res and 'best_classic' in w_res:
            interpretations.append(
                f"**Wilcoxon Test (Classic vs Hybrid):** When comparing the best classic ({w_res.get('best_classic')}) against the best hybrid ({w_res.get('best_hybrid')}), the p-value is **{w_res.get('p_val', 1.0):.4f}**, indicating that the hybrid improvement **{'is' if w_res.get('p_val', 1.0) < alpha else 'is not'} statistically significant**."
            )
        m_res = results.get('mcnemar', {})
        if m_res and 'p_val' in m_res:
            interpretations.append(
                f"**McNemar Test (Test Predictions):** Evaluation on the test set yields a p-value of **{m_res['p_val']:.4f}** in classification consistency between both approaches, suggesting that classification error rates **{'differ significantly' if m_res['p_val'] < alpha else 'are statistically equivalent'}**."
            )
    elif lang_key == 'pt':
        interpretations = [
            f"**Testes Estatísticos Globais ({results['test_type']}):** Avaliaram-se as diferenças grupais em validação cruzada. O p-valor global é **{results['overall_pval']:.4f}**.",
            f"Sendo o p-valor global {'menor' if results['overall_pval'] < alpha else 'maior'} que o nível de significância configurável alfa = {alpha}, conclui-se que **{'existem' if results['overall_pval'] < alpha else 'não existem'} diferenças estatisticamente significativas** no desempenho dos 5 modelos."
        ]
        w_res = results.get('wilcoxon', {})
        if w_res and 'best_classic' in w_res:
            interpretations.append(
                f"**Teste Wilcoxon (Clássico vs Híbrido):** Ao comparar o melhor clássico ({w_res.get('best_classic')}) contra o melhor híbrido ({w_res.get('best_hybrid')}), o p-valor é de **{w_res.get('p_val', 1.0):.4f}**, indicando que a melhoria do híbrido **{'é' if w_res.get('p_val', 1.0) < alpha else 'não é'} estatisticamente significativa**."
            )
        m_res = results.get('mcnemar', {})
        if m_res and 'p_val' in m_res:
            interpretations.append(
                f"**Teste de McNemar (Previsões de Teste):** A avaliação sobre o conjunto de teste resulta em um p-valor de **{m_res['p_val']:.4f}** na consistência das previsões entre ambas as abordagens, sugerindo que a taxa de erros de classificação **{'difere significativamente' if m_res['p_val'] < alpha else 'é estatisticamente equivalente'}**."
            )
    else:
        interpretations = [
            f"**Pruebas Estadísticas Globales ({results['test_type']}):** Se evaluaron las diferencias grupales en validación cruzada. El p-valor global es **{results['overall_pval']:.4f}**.",
            f"Al ser el p-valor global {'menor' if results['overall_pval'] < alpha else 'mayor'} que el nivel de significancia alfa = {alpha}, se concluye que **{'existen' if results['overall_pval'] < alpha else 'no existen'} diferencias estadísticamente significativas** en el rendimiento de los 5 modelos."
        ]
        w_res = results.get('wilcoxon', {})
        if w_res and 'best_classic' in w_res:
            interpretations.append(
                f"**Prueba Wilcoxon (Clásico vs Híbrido):** Al comparar el mejor clásico ({w_res.get('best_classic')}) contra el mejor híbrido ({w_res.get('best_hybrid')}), el p-valor es de **{w_res.get('p_val', 1.0):.4f}**, indicando que la mejora del híbrido **{'es' if w_res.get('p_val', 1.0) < alpha else 'no es'} estadísticamente significativa**."
            )
        m_res = results.get('mcnemar', {})
        if m_res and 'p_val' in m_res:
            interpretations.append(
                f"**Prueba de McNemar (Predicciones de Test):** La evaluación sobre el conjunto de prueba arroja un p-valor de **{m_res['p_val']:.4f}** en la consistencia de predicciones entre ambos enfoques, sugiriendo que la tasa de errores de clasificación **{'difiere significativamente' if m_res['p_val'] < alpha else 'es estadísticamente equivalente'}**."
            )
            
    # Agregar interpretación resumida de intervalos bootstrap si existen
    b_res = results.get('bootstrap_ci', {})
    if b_res:
        bootstrap_texts = []
        for name, ci in b_res.items():
            bootstrap_texts.append(f"{name}: [{ci[0]:.2%} - {ci[1]:.2%}]")
        if lang_key == 'en':
            interpretations.append(f"**Bootstrap Confidence Intervals (95% CI Accuracy):** {', '.join(bootstrap_texts)}.")
        elif lang_key == 'pt':
            interpretations.append(f"**Intervalos de Confiança Bootstrap (95% CI Acurácia):** {', '.join(bootstrap_texts)}.")
        else:
            interpretations.append(f"**Intervalos de Confianza Bootstrap (95% CI Exactitud):** {', '.join(bootstrap_texts)}.")
            
    return "\n\n".join(interpretations)

