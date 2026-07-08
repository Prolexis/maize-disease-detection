import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def clean_data(df, target_col):
    """
    Realiza limpieza de datos: imputación de nulos, eliminación de duplicados,
    tratamiento de outliers por IQR y separación de variables.
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
        
    # 2. Imputación de nulos
    imputed_nulls = {}
    for col in num_cols:
        null_count = df_cleaned[col].isnull().sum()
        if null_count > 0:
            median_val = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
            imputed_nulls[col] = (null_count, f"Mediana ({median_val:.2f})")
            
    for col in cat_cols:
        null_count = df_cleaned[col].isnull().sum()
        if null_count > 0:
            mode_val = df_cleaned[col].mode()[0]
            df_cleaned[col] = df_cleaned[col].fillna(mode_val)
            imputed_nulls[col] = (null_count, f"Moda ({mode_val})")
            
    # 3. Tratamiento de outliers (Límites IQR y clipping para mantener rango válido)
    outliers_detected = {}
    for col in num_cols:
        q1 = df_cleaned[col].quantile(0.25)
        q3 = df_cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Conteo de outliers
        outlier_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
        count_outliers = outlier_mask.sum()
        
        if count_outliers > 0:
            outliers_detected[col] = count_outliers
            # Realizar clipping suave
            df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
            
    return df_cleaned, num_duplicates, imputed_nulls, outliers_detected

def get_descriptive_stats(df, target_col):
    """
    Genera estadísticas descriptivas detalladas para variables numéricas,
    tanto a nivel global como agrupado por la clase del target.
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
        
        global_stats[col] = {
            'media': desc['mean'],
            'mediana': desc['50%'],
            'moda': mode_val,
            'desviación': desc['std'],
            'varianza': desc['std'] ** 2,
            'asimetría': skewness,
            'curtosis': kurtosis,
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

def interpret_eda(df, target_col, num_duplicates, imputed_nulls, outliers_detected, global_stats):
    """
    Genera interpretaciones automáticas profesionales en texto de los datos.
    """
    interpretations = []
    
    # Duplicados y nulos
    clean_text = f"**Limpieza de Datos:** Se identificaron y eliminaron **{num_duplicates}** registros duplicados. "
    if imputed_nulls:
        clean_text += "Se imputaron valores faltantes en: " + ", ".join([f"'{k}' ({v[0]} nulos reemplazados por {v[1]})" for k, v in imputed_nulls.items()]) + ". "
    else:
        clean_text += "No se detectaron valores faltantes en las columnas principales. "
        
    if outliers_detected:
        clean_text += "Se corrigieron outliers mediante truncamiento IQR en: " + ", ".join([f"'{k}' ({v} outliers)" for k, v in outliers_detected.items()]) + "."
    else:
        clean_text += "No se observaron outliers severos fuera del rango de 1.5 * IQR."
    interpretations.append(clean_text)
    
    # Distribución y Asimetría
    skewed_cols = []
    for col, stat in global_stats.iterrows():
        if abs(stat['asimetría']) > 1.0:
            skewed_cols.append(f"'{col}' (asimetría: {stat['asimetría']:.2f})")
    
    dist_text = "**Distribución de Variables:** "
    if skewed_cols:
        dist_text += "Se observa una marcada asimetría en " + ", ".join(skewed_cols) + ", lo cual sugiere que estas variables no siguen una distribución estrictamente normal y podrían requerir escalamientos robustos."
    else:
        dist_text += "Las variables numéricas muestran distribuciones con asimetría moderada, indicando comportamientos mayormente simétricos."
    interpretations.append(dist_text)
    
    # Balance de clases
    class_counts = df[target_col].value_counts()
    min_class_ratio = class_counts.min() / class_counts.max()
    balance_text = f"**Balance de Clases:** La variable objetivo '{target_col}' cuenta con las clases: "
    balance_text += ", ".join([f"'{k}': {v} muestras" for k, v in class_counts.items()]) + ". "
    if min_class_ratio < 0.6:
        balance_text += "Existe un **desbalance de clases significativo** (relación mín/máx < 60%). Se recomienda aplicar balanceo de pesos de clase (*class weights*) o SMOTE para evitar sesgos en el entrenamiento."
    else:
        balance_text += "El dataset se encuentra **bien balanceado** con proporciones equilibradas entre clases, permitiendo el uso de métricas estándar como Accuracy de forma segura."
    interpretations.append(balance_text)
    
    return "\n\n".join(interpretations)

def plot_eda_charts(df, target_col, save_path="reports"):
    """
    Genera y guarda los gráficos de EDA requeridos: correlación, histogramas, boxplots y balance de clases.
    """
    os.makedirs(save_path, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
        
    charts = {}
    
    # 1. Gráfico de Balance de Clases
    plt.figure(figsize=(8, 5))
    sns.countplot(x=target_col, data=df, palette="viridis")
    plt.title('Distribución y Balance de la Variable Objetivo')
    plt.xlabel('Clase Target')
    plt.ylabel('Cantidad de Muestras')
    plt.tight_layout()
    balance_fig = os.path.join(save_path, 'eda_balance_clases.png')
    plt.savefig(balance_fig, dpi=150)
    plt.close()
    charts['balance'] = balance_fig

    # 2. Matriz de Correlación (Heatmap)
    if len(num_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[num_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)
        plt.title('Matriz de Correlación de Variables Numéricas')
        plt.tight_layout()
        corr_fig = os.path.join(save_path, 'eda_correlation.png')
        plt.savefig(corr_fig, dpi=150)
        plt.close()
        charts['correlation'] = corr_fig
        
    # 3. Histogramas de Distribución por clase
    fig, axes = plt.subplots(int(np.ceil(len(num_cols)/2)), 2, figsize=(14, 3 * len(num_cols)//2 + 2))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue=target_col, kde=True, ax=axes[idx], palette="tab10", multiple="stack")
        axes[idx].set_title(f'Distribución de {col} por Clase')
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
        sns.boxplot(data=df, x=target_col, y=col, ax=axes[idx], palette="Set2")
        axes[idx].set_title(f'Boxplot de {col} por Clase')
    # Ocultar subplots sobrantes
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    boxplot_fig = os.path.join(save_path, 'eda_boxplots_by_class.png')
    plt.savefig(boxplot_fig, dpi=150)
    plt.close()
    charts['boxplots'] = boxplot_fig
    
    return charts
