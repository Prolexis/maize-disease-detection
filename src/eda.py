import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest

def clean_data(df, target_col):
    """
    Realiza limpieza de datos avanzada: imputacion de nulos inteligente por clase, 
    eliminacion de duplicados, deteccion de outliers multivariados (Isolation Forest),
    tratamiento de outliers univariados (IQR clipping) y normalizacion de variables sesgadas.
    """
    df_cleaned = df.copy()
    
    # 1. Eliminar duplicados
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
    num_duplicates = initial_rows - len(df_cleaned)
    
    # Identificar columnas
    num_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    if target_col in num_cols:
        num_cols.remove(target_col)
        
    # 2. Imputacion de nulos inteligente (usando la mediana/moda del grupo del target si es posible)
    imputed_nulls = {}
    for col in num_cols:
        null_count = df_cleaned[col].isnull().sum()
        if null_count > 0:
            # Imputar usando la mediana por clase del target
            class_medians = df_cleaned.groupby(target_col)[col].transform('median')
            df_cleaned[col] = df_cleaned[col].fillna(class_medians)
            # Si aun quedan nulos (por clases vacias), rellenar con mediana global
            if df_cleaned[col].isnull().sum() > 0:
                global_median = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(global_median)
            imputed_nulls[col] = (null_count, "Mediana agrupada por clase")
            
    for col in cat_cols:
        null_count = df_cleaned[col].isnull().sum()
        if null_count > 0:
            # Imputar usando la moda global
            mode_val = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else "Desconocido"
            df_cleaned[col] = df_cleaned[col].fillna(mode_val)
            imputed_nulls[col] = (null_count, f"Moda ({mode_val})")
            
    # 3. Deteccion y tratamiento de outliers univariados (IQR clipping)
    outliers_detected = {}
    for col in num_cols:
        q1 = df_cleaned[col].quantile(0.25)
        q3 = df_cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
        count_outliers = outlier_mask.sum()
        
        if count_outliers > 0:
            outliers_detected[col] = count_outliers
            df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)

    # 4. Deteccion de outliers multivariados con Isolation Forest (sin eliminar para no perder muestras)
    multivariate_outliers = 0
    if len(num_cols) >= 2 and len(df_cleaned) > 10:
        try:
            iso = IsolationForest(contamination=0.02, random_state=42)
            preds = iso.fit_predict(df_cleaned[num_cols])
            multivariate_outliers = int((preds == -1).sum())
        except:
            pass
            
    # 5. Tratamiento automatico de asimetria (Skewness Normalizacion con log1p)
    transformed_cols = []
    for col in num_cols:
        skew = stats.skew(df_cleaned[col])
        if abs(skew) > 1.0:
            # Si los valores son no-negativos, aplicar log1p
            if (df_cleaned[col] >= 0).all():
                df_cleaned[col] = np.log1p(df_cleaned[col])
                transformed_cols.append(col)

    # Guardar metadatos en df.attrs para no romper la firma de retorno de la funcion
    df_cleaned.attrs['transformed_cols'] = transformed_cols
    df_cleaned.attrs['multivariate_outliers'] = multivariate_outliers
            
    return df_cleaned, num_duplicates, imputed_nulls, outliers_detected

def get_descriptive_stats(df, target_col):
    """
    Genera estadisticas descriptivas avanzadas para variables numericas.
    Retorna llaves en minuscula (compatibles con reporting.py) y en mayuscula (para presentacion en UI).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
        
    global_stats = {}
    for col in num_cols:
        desc = df[col].describe(percentiles=[0.25, 0.50, 0.75])
        skewness = stats.skew(df[col])
        kurtosis = stats.kurtosis(df[col])
        mode_res = df[col].mode()
        mode_val = mode_res[0] if len(mode_res) > 0 else np.nan
        
        # Coeficiente de Variacion (CV) y Error Estandar (SEM)
        mean_val = desc['mean']
        std_val = desc['std']
        cv = (std_val / mean_val) if mean_val != 0 else 0
        sem = stats.sem(df[col]) if len(df[col]) > 1 else 0
        
        # Prueba de normalidad de Shapiro-Wilk
        if len(df[col]) >= 3:
            try:
                shapiro_stat, p_val = stats.shapiro(df[col])
            except:
                p_val = 0.0
        else:
            p_val = 0.0
            
        global_stats[col] = {
            'mean': mean_val,
            'media': mean_val,
            'Media': mean_val,
            'std': std_val,
            'Std': std_val,
            'desviación': std_val,
            'Desv. Estándar': std_val,
            'min': desc['min'],
            'Min': desc['min'],
            'Mín': desc['min'],
            'max': desc['max'],
            'Max': desc['max'],
            'Máx': desc['max'],
            'mediana': desc['50%'],
            'Mediana': desc['50%'],
            'moda': mode_val,
            'Moda': mode_val,
            'varianza': std_val ** 2,
            'Varianza': std_val ** 2,
            'rango': desc['max'] - desc['min'],
            'Rango': desc['max'] - desc['min'],
            'error_sem': sem,
            'Error Est. (SEM)': sem,
            'coef_var': cv,
            'Coef. Variación (CV)': cv,
            'asimetría': skewness,
            'Asimetría': skewness,
            'curtosis': kurtosis,
            'Curtosis': kurtosis,
            'shapiro_p': p_val,
            'Norm. p-valor (Shapiro)': p_val,
            'p25': desc['25%'],
            'p75': desc['75%']
        }
        
    # Agrupado por clase
    class_stats = {}
    classes = df[target_col].unique()
    for cls in classes:
        cls_df = df[df[target_col] == cls]
        cls_stat_dict = {}
        for col in num_cols:
            desc = cls_df[col].describe()
            cls_stat_dict[col] = {
                'media': desc['mean'],
                'mediana': desc['50%'] if '50%' in desc else cls_df[col].median(),
                'desviación': desc['std'] if 'std' in desc else cls_df[col].std()
            }
        class_stats[cls] = cls_stat_dict
        
    return pd.DataFrame(global_stats).T, class_stats

def interpret_eda(df, target_col, num_duplicates, imputed_nulls, outliers_detected, global_stats, lang='es'):
    """
    Genera interpretaciones automáticas profesionales en texto de los datos limpios y estadísticos avanzados.
    """
    from src.interpretation import interpretar_eda
    return interpretar_eda(df, target_col, num_duplicates, imputed_nulls, outliers_detected, global_stats, lang=lang)


def plot_eda_charts(df, target_col, save_path="reports", lang="es"):
    """
    Genera y guarda los gráficos de EDA requeridos: correlación, histogramas, boxplots y balance de clases.
    """
    from src.translation import translate_class_lang
    os.makedirs(save_path, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
        
    lang_key = lang if lang in ['es', 'en', 'pt'] else 'es'
    
    # Translations
    bal_title = {
        'es': 'Distribución y Balance de la Variable Objetivo',
        'en': 'Distribution and Balance of the Target Variable',
        'pt': 'Distribuição e Equilíbrio da Variável Objetivo'
    }.get(lang_key)
    
    bal_xlabel = {
        'es': 'Clase Target',
        'en': 'Target Class',
        'pt': 'Classe Target'
    }.get(lang_key)
    
    bal_ylabel = {
        'es': 'Cantidad de Muestras',
        'en': 'Number of Samples',
        'pt': 'Quantidade de Amostras'
    }.get(lang_key)
    
    corr_title = {
        'es': 'Matriz de Correlación de Variables Numéricas',
        'en': 'Correlation Matrix of Numerical Variables',
        'pt': 'Matriz de Correlação de Variáveis Numéricas'
    }.get(lang_key)
    
    dist_title = {
        'es': 'Distribución de {col} por Clase',
        'en': 'Distribution of {col} by Class',
        'pt': 'Distribuição de {col} por Classe'
    }.get(lang_key)
    
    box_title = {
        'es': 'Boxplot de {col} por Clase',
        'en': 'Boxplot of {col} by Class',
        'pt': 'Boxplot de {col} por Classe'
    }.get(lang_key)

    df_plot = df.copy()
    df_plot[target_col] = df_plot[target_col].apply(lambda c: translate_class_lang(c, lang_key))

    charts = {}
    
    # 1. Gráfico de Balance de Clases
    plt.figure(figsize=(8, 5))
    sns.countplot(x=target_col, data=df_plot, palette="viridis")
    plt.title(bal_title)
    plt.xlabel(bal_xlabel)
    plt.ylabel(bal_ylabel)
    plt.tight_layout()
    balance_fig = os.path.join(save_path, 'eda_balance_clases.png')
    plt.savefig(balance_fig, dpi=150)
    plt.close()
    charts['balance'] = balance_fig

    # 2. Matriz de Correlación (Heatmap)
    if len(num_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df_plot[num_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)
        plt.title(corr_title)
        plt.tight_layout()
        corr_fig = os.path.join(save_path, 'eda_correlation.png')
        plt.savefig(corr_fig, dpi=150)
        plt.close()
        charts['correlation'] = corr_fig
        
    # 3. Histogramas de Distribución por clase
    fig, axes = plt.subplots(int(np.ceil(len(num_cols)/2)), 2, figsize=(14, 3 * len(num_cols)//2 + 2))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        sns.histplot(data=df_plot, x=col, hue=target_col, kde=True, ax=axes[idx], palette="tab10", multiple="stack")
        axes[idx].set_title(dist_title.format(col=col))
    # Ocultar subplots sobrantes
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    dist_fig = os.path.join(save_path, 'eda_distribution_by_class.png')
    plt.savefig(dist_fig, dpi=150)
    plt.close()
    charts['distributions'] = dist_fig

    # 4. Boxplots por variable y clase
    fig, axes = plt.subplots(int(np.ceil(len(num_cols)/2)), 2, figsize=(14, 3 * len(num_cols)//2 + 2))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        sns.boxplot(data=df_plot, x=target_col, y=col, ax=axes[idx], palette="Set2")
        axes[idx].set_title(box_title.format(col=col))
        axes[idx].set_xlabel(bal_xlabel)
    # Ocultar subplots sobrantes
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    boxplot_fig = os.path.join(save_path, 'eda_boxplots_by_class.png')
    plt.savefig(boxplot_fig, dpi=150)
    plt.close()
    charts['boxplots'] = boxplot_fig
    
    return charts

def run_predictor_significance_tests(df, target_col, alpha=0.05):
    """
    Ejecuta pruebas estadísticas de significancia (ANOVA o Kruskal-Wallis)
    para cada variable numérica respecto a las clases del target.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
        
    classes = df[target_col].unique()
    significance_results = []
    
    for col in num_cols:
        # Agrupar datos por clase
        groups = [df[df[target_col] == c][col].dropna().values for c in classes]
        # Filtrar grupos vacíos o con muy pocos datos
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
            
        # 1. Validar normalidad por grupo
        normality_holds = True
        for g in groups:
            try:
                _, p_val = stats.shapiro(g)
                if p_val < alpha:
                    normality_holds = False
                    break
            except:
                normality_holds = False
                break
                
        # 2. Validar homocedasticidad
        variance_holds = True
        try:
            _, p_val_lev = stats.levene(*groups)
            if p_val_lev < alpha:
                variance_holds = False
        except:
            variance_holds = False
            
        # 3. Elegir prueba
        if normality_holds and variance_holds:
            # ANOVA
            try:
                stat_val, p_val = stats.f_oneway(*groups)
                test_name = "ANOVA (Paramétrico)"
            except:
                stat_val, p_val = 0.0, 1.0
                test_name = "ANOVA (Fallo)"
        else:
            # Kruskal-Wallis
            try:
                stat_val, p_val = stats.kruskal(*groups)
                test_name = "Kruskal-Wallis (No Paramétrico)"
            except:
                stat_val, p_val = 0.0, 1.0
                test_name = "Kruskal-Wallis (Fallo)"
                
        significance_results.append({
            'Variable': col,
            'Prueba': test_name,
            'Estadístico': stat_val,
            'p-valor': p_val,
            'Significativo': "SÍ" if p_val < alpha else "NO"
        })
        
    return pd.DataFrame(significance_results)
