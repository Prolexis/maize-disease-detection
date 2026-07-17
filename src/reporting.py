import os
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------
# 1. GENERACIÓN DE EXCEL (.xlsx)
# ---------------------------------------------------------

def format_excel_sheet(worksheet):
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    
    header_font = Font(name='Segoe UI', size=11, bold=True, color='FFFFFF')
    header_fill = PatternFill(start_color='16A34A', end_color='16A34A', fill_type='solid')
    cell_font = Font(name='Segoe UI', size=10, color='1E293B')
    
    thin_side = Side(border_style="thin", color="E2E8F0")
    border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    
    # Formatear encabezado
    for col_idx in range(1, worksheet.max_column + 1):
        cell = worksheet.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = border
        
    # Formatear datos
    for r_idx in range(2, worksheet.max_row + 1):
        for c_idx in range(1, worksheet.max_column + 1):
            cell = worksheet.cell(row=r_idx, column=c_idx)
            cell.font = cell_font
            cell.border = border
            
            val = cell.value
            if isinstance(val, (int, float)):
                cell.alignment = Alignment(horizontal='center')
                if isinstance(val, float):
                    if val <= 1.0 and c_idx >= 3:
                        cell.number_format = '0.00%'
                    else:
                        cell.number_format = '0.0000'
            else:
                cell.alignment = Alignment(horizontal='left')
                
    # Autoajustar columnas
    for col in worksheet.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            val_str = str(cell.value or '')
            if len(val_str) > max_len:
                max_len = len(val_str)
        worksheet.column_dimensions[col_letter].width = max(max_len + 4, 14)

def generate_xlsx_report(df_eda, df_training, cv_results, tuning_results, stats_results, filepath="reports/reporte_automl.xlsx", lang="es", df_significance=None):
    """
    Crea un archivo Excel organizado con una pestaña para cada fase del pipeline de ML.
    """
    from src.translation import t_lang
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Mapeos de traducción
    eda_cols_map = {
        'es': {'media': 'Media', 'mediana': 'Mediana', 'desviación': 'Desviación', 'asimetría': 'Asimetría', 'curtosis': 'Curtosis'},
        'en': {'media': 'Mean', 'mediana': 'Median', 'desviación': 'Std Dev', 'asimetría': 'Skewness', 'curtosis': 'Kurtosis'},
        'pt': {'media': 'Média', 'mediana': 'Mediana', 'desviación': 'Desvio Padrão', 'asimetría': 'Assimetria', 'curtosis': 'Curtose'}
    }
    
    train_cols_map = {
        'es': {'Accuracy': 'Exactitud', 'Precision': 'Precisión', 'Recall': 'Sensibilidad (Recall)', 'F1-Score': 'F1-Score', 'AUC': 'AUC', 'Tiempo de Entrenamiento (s)': 'Tiempo Entrenamiento (s)', 'Tiempo de Inferencia (s)': 'Tiempo Inferencia (s)', 'No. Parámetros': 'No. Parámetros', 'Tamaño (KB)': 'Tamaño (KB)'},
        'en': {'Accuracy': 'Accuracy', 'Precision': 'Precision', 'Recall': 'Recall', 'F1-Score': 'F1-Score', 'AUC': 'AUC', 'Tiempo de Entrenamiento (s)': 'Training Time (s)', 'Tiempo de Inferencia (s)': 'Inference Time (s)', 'No. Parámetros': 'Param Count', 'Tamaño (KB)': 'Size (KB)'},
        'pt': {'Accuracy': 'Acurácia', 'Precision': 'Precisão', 'Recall': 'Revogação (Recall)', 'F1-Score': 'F1-Score', 'AUC': 'AUC', 'Tiempo de Entrenamiento (s)': 'Tempo Treinamento (s)', 'Tiempo de Inferencia (s)': 'Tempo Inferência (s)', 'No. Parámetros': 'Qtd Parâmetros', 'Tamaño (KB)': 'Tamanho (KB)'}
    }

    lang_key = lang if lang in ['es', 'en', 'pt'] else 'es'

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Pestaña 1: EDA
        df_eda_renamed = df_eda.rename(columns=eda_cols_map[lang_key])
        var_lbl = t_lang('rep_pdf_file', lang) if lang_key != 'es' else 'Variable'
        df_eda_renamed.reset_index().rename(columns={'index': var_lbl}).to_excel(writer, sheet_name='EDA', index=False)
        
        # Pestaña 2: Entrenamiento (Métricas)
        df_training_renamed = df_training.rename(columns=train_cols_map[lang_key])
        df_training_renamed.to_excel(writer, sheet_name=t_lang('tab_train', lang), index=True)
        
        # Pestaña 3: Validación Cruzada
        cv_data = []
        for model_name, res in cv_results.items():
            cv_data.append({
                t_lang('pred_col_model', lang): model_name,
                t_lang('plot_accuracy', lang) + ' CV': res['mean_accuracy'],
                'Std Accuracy CV': res['std_accuracy'],
                'Mean F1 CV': res['mean_f1'],
                'Std F1 CV': res['std_f1']
            })
        pd.DataFrame(cv_data).to_excel(writer, sheet_name='Cross-Validation', index=False)
        
        # Pestaña 4: Tuning
        tuning_labels = {
            'es': ['Método de Búsqueda', 'Tiempo de Búsqueda (s)', 'Precisión Antes', 'Precisión Después', 'Mejores Hiperparámetros'],
            'en': ['Search Method', 'Search Time (s)', 'Accuracy Before', 'Accuracy After', 'Best Hyperparameters'],
            'pt': ['Método de Busca', 'Tempo de Busca (s)', 'Acurácia Antes', 'Acurácia Depois', 'Melhores Hiperparâmetros']
        }
        val_lbl = 'Valor' if lang_key == 'es' else ('Value' if lang_key == 'en' else 'Valor')
        met_lbl = 'Métrica' if lang_key == 'es' else ('Metric' if lang_key == 'en' else 'Métrica')
        tuning_data = {
            met_lbl: tuning_labels[lang_key],
            val_lbl: [
                tuning_results['method'],
                tuning_results['search_time'],
                tuning_results['accuracy_before'],
                tuning_results['accuracy_after'],
                str(tuning_results['best_params'])
            ]
        }
        pd.DataFrame(tuning_data).to_excel(writer, sheet_name='Tuning', index=False)
        
        # Pestaña 5: Pruebas Estadísticas
        shapiros = ", ".join([f"{k}: p={v:.3f}" for k, v in stats_results['shapiro_pvals'].items()])
        stats_labels = {
            'es': ['Tipo de Prueba Grupal', 'Estadístico Grupal', 'p-valor Grupal', 'Supuesto Shapiro-Wilk (p)', 'Supuesto Levene (p)', 'Prueba Wilcoxon (p)', 'Prueba McNemar (p)'],
            'en': ['Group Test Type', 'Group Statistic', 'Group p-value', 'Shapiro-Wilk Assumption (p)', 'Levene Assumption (p)', 'Wilcoxon Test (p)', 'McNemar Test (p)'],
            'pt': ['Tipo de Teste Grupal', 'Estatística Grupal', 'p-valor Grupal', 'Suposição Shapiro-Wilk (p)', 'Suposição Levene (p)', 'Teste Wilcoxon (p)', 'Teste McNemar (p)']
        }
        
        test_lbl = 'Prueba' if lang_key == 'es' else ('Test' if lang_key == 'en' else 'Teste')
        res_lbl = 'Resultado/Valor' if lang_key == 'es' else ('Result/Value' if lang_key == 'en' else 'Resultado/Valor')
        stats_data = {
            test_lbl: stats_labels[lang_key],
            res_lbl: [
                stats_results['test_type'],
                stats_results['overall_stat'],
                stats_results['overall_pval'],
                shapiros,
                stats_results['levene_pval'],
                f"Classic vs Hybrid: p={stats_results['wilcoxon']['p_val']:.4f}",
                f"Hits/Failures: p={stats_results['mcnemar']['p_val']:.4f}"
            ]
        }
        pd.DataFrame(stats_data).to_excel(writer, sheet_name=t_lang('tab_stats', lang), index=False)
        
        # Pestaña 6: Significancia de Predictores (si se proporciona)
        if df_significance is not None and not df_significance.empty:
            sig_sheet = {
                'es': 'Significancia Predictores',
                'en': 'Predictors Significance',
                'pt': 'Significância Preditores'
            }.get(lang_key, 'Significancia Predictores')
            df_significance.to_excel(writer, sheet_name=sig_sheet, index=False)
        
        # Aplicar formato a todas las pestañas creadas
        workbook = writer.book
        for sheet_name in workbook.sheetnames:
            format_excel_sheet(workbook[sheet_name])
            
    return filepath
# ---------------------------------------------------------
# 2. GENERACIÓN DE WORD (.docx)
# ---------------------------------------------------------
def generate_docx_report(df_eda, df_training, cv_results, tuning_results, stats_results, interpretations, image_paths, filepath="reports/reporte_automl.docx", lang="es", df_significance=None):
    """
    Crea un reporte Word (.docx) formateado con portada, tablas e imágenes embebidas.
    """
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from src.translation import t_lang
    
    doc = Document()
    
    # Configurar estilos de fuente globales
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(11)
    
    # Textos localizados
    title_lbl = t_lang("rep_title", lang)
    sub_lbl = t_lang("rep_subtitle", lang)
    gen_lbl = t_lang("rep_generated", lang).format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    eda_heading = t_lang("rep_eda_section", lang)
    eda_desc = {
        'es': "Estadísticas descriptivas generales de las variables numéricas analizadas:",
        'en': "General descriptive statistics of the analyzed numerical variables:",
        'pt': "Estatísticas descritivas gerais das variáveis numéricas analisadas:"
    }.get(lang, "Estadísticas descriptivas generales de las variables numéricas analizadas:")
    
    train_heading = t_lang("rep_train_section", lang)
    train_desc = {
        'es': "Métricas obtenidas por los 3 modelos clásicos y 2 híbridos:",
        'en': "Metrics obtained by the 3 classic and 2 hybrid models:",
        'pt': "Métricas obtidas pelos 3 modelos clássicos e 2 híbridos:"
    }.get(lang, "Métricas obtenidas por los 3 modelos clásicos y 2 híbridos:")
    
    cv_heading = t_lang("rep_cv_section", lang)
    cv_desc = {
        'es': "Estabilidad e hiperparámetros óptimos:",
        'en': "Stability and optimal hyperparameters:",
        'pt': "Estabilidade e hiperparâmetros ideais:"
    }.get(lang, "Estabilidad e hiperparámetros óptimos:")
    
    stats_heading = t_lang("rep_stats_section", lang)
    stats_desc = {
        'es': (
            "Para validar rigurosamente la significancia estadística de las diferencias de rendimiento entre los "
            "modelos de Machine Learning entrenados, se ejecuta un protocolo formal de prueba de hipótesis:\n\n"
            "1. Prueba de Shapiro-Wilk (H0: los accuracies de CV provienen de una distribución normal) y Prueba "
            "de Levene (H0: las varianzas de los modelos son homogéneas/homocedásticas).\n"
            "2. Si se validan los supuestos, se ejecuta ANOVA de una vía (Paramétrico, H0: las medias de exactitud de todos "
            "los modelos son estadísticamente equivalentes). En caso contrario, se aplica la prueba robusta no paramétrica "
            "de Friedman (H0: la distribución de rangos de todos los modelos es equivalente).\n"
            "3. Comparaciones Post-Hoc: Se ejecuta la prueba de Tukey HSD (paramétrica) o la prueba apareada de Wilcoxon con "
            "corrección de Bonferroni (no paramétrica) para identificar exactamente cuáles modelos tienen diferencias "
            "significativas de desempeño.\n"
            "4. Prueba de McNemar (H0: las proporciones de aciertos/errores en el conjunto de test son idénticas): Realiza "
            "una comparación robusta basada en la matriz de confusión apareada sobre los datos de test entre el mejor modelo clásico y el mejor híbrido."
        ),
        'en': (
            "To rigorously validate the statistical significance of performance differences among the trained Machine "
            "Learning models, a formal hypothesis testing protocol is executed:\n\n"
            "1. Shapiro-Wilk Test (H0: CV accuracies follow a normal distribution) and Levene Test (H0: model variances "
            "are homogeneous/homoscedastic).\n"
            "2. If assumptions hold, a one-way ANOVA is executed (Parametric, H0: the accuracy means of all models are "
            "statistically equivalent). Otherwise, the robust non-parametric Friedman test is applied (H0: the rank "
            "distribution of all models is equivalent).\n"
            "3. Post-Hoc Comparisons: Tukey HSD (parametric) or Wilcoxon signed-rank test with Bonferroni correction "
            "(non-parametric) is executed to pinpoint which specific model pairs show significant differences.\n"
            "4. McNemar's Test (H0: classification success/error proportions on the test set are identical): Performs "
            "a robust comparison based on the paired contingency table on test data between the best classic and hybrid models."
        ),
        'pt': (
            "Para validar rigorosamente a significância estatística das diferenças de desempenho entre os modelos "
            "de Machine Learning treinados, executa-se um protocolo formal de teste de hipóteses:\n\n"
            "1. Teste de Shapiro-Wilk (H0: as acurácias de CV seguem uma distribuição normal) e Teste de Levene (H0: as "
            "variâncias dos modelos são homogêneas/homocedásticas).\n"
            "2. Se as suposições forem validadas, executa-se a ANOVA de uma via (Paramétrico, H0: as médias de acurácia de "
            "todos os modelos são estatisticamente equivalentes). Caso contrário, aplica-se o teste robusto não-paramétrico "
            "de Friedman (H0: a distribuição de postos de todos os modelos é equivalente).\n"
            "3. Comparações Post-Hoc: Executa-se o teste de Tukey HSD (paramétrico) ou o teste emparelhado de Wilcoxon com "
            "correção de Bonferroni (não-paramétrico) para identificar exatamente quais pares de modelos têm diferenças "
            "significativas de desempenho.\n"
            "4. Teste de McNemar (H0: as proporções de acertos/erros no conjunto de teste são idênticas): Realiza uma "
            "comparação robusta baseada na tabela de contingência emparelhada nos dados de teste entre o melhor modelo clássico e o melhor híbrido."
        )
    }.get(lang, "Validación estadística de los desempeños:")
    
    eda_headers = {
        'es': ['Variable', 'Media', 'Mediana', 'Desviación', 'Asimetría', 'Curtosis'],
        'en': ['Variable', 'Mean', 'Median', 'Deviation', 'Skewness', 'Kurtosis'],
        'pt': ['Variável', 'Média', 'Mediana', 'Desvio', 'Assimetria', 'Curtose']
    }.get(lang, ['Variable', 'Media', 'Mediana', 'Desviación', 'Asimetría', 'Curtosis'])

    train_headers = {
        'es': ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'en': ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'pt': ['Modelo', 'Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC']
    }.get(lang, ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'])

    cv_headers = {
        'es': ['Modelo', 'Accuracy CV', 'F1-Score CV'],
        'en': ['Model', 'Accuracy CV', 'F1-Score CV'],
        'pt': ['Modelo', 'Acurácia CV', 'F1-Score CV']
    }.get(lang, ['Modelo', 'Accuracy CV', 'F1-Score CV'])

    eda_lbl = {
        'es': "\nDistribución y Balance de Clases del Dataset:",
        'en': "\nClass Balance and Distribution of the Dataset:",
        'pt': "\nDistribuição e Equilíbrio de Classes do Dataset:"
    }.get(lang, "\nDistribución y Balance de Clases del Dataset:")

    corr_lbl = {
        'es': "\nMatriz de Correlación de Variables:",
        'en': "\nVariables Correlation Matrix:",
        'pt': "\nMatriz de Correlação de Variáveis:"
    }.get(lang, "\nMatriz de Correlación de Variables:")

    interp_eda_lbl = {
        'es': "\n**Interpretación Técnica de EDA:**",
        'en': "\n**EDA Technical Interpretation:**",
        'pt': "\n**Interpretação Técnica do EDA:**"
    }.get(lang, "\n**Interpretación Técnica de EDA:**")

    roc_lbl = {
        'es': "\nCurvas ROC comparativas:",
        'en': "\nComparative ROC Curves:",
        'pt': "\nCurvas ROC comparativas:"
    }.get(lang, "\nCurvas ROC comparativas:")

    interp_train_lbl = {
        'es': "\n**Interpretación de Entrenamiento:**",
        'en': "\n**Training Interpretation:**",
        'pt': "\n**Interpretação do Treinamento:**"
    }.get(lang, "\n**Interpretación de Entrenamiento:**")

    cv_disp_lbl = {
        'es': "\nDispersión de los Folds de Validación:",
        'en': "\nValidation Folds Dispersion:",
        'pt': "\nDispersão dos Folds de Validação:"
    }.get(lang, "\nDispersión de los Folds de Validación:")

    interp_cv_lbl = {
        'es': "\n**Interpretación de Validación Cruzada:**",
        'en': "\n**Cross Validation Interpretation:**",
        'pt': "\n**Interpretação da Validação Cruzada:**"
    }.get(lang, "\n**Interpretación de Validación Cruzada:**")

    tuning_lbl = {
        'es': "\n**Optimización (Tuning):**",
        'en': "\n**Optimization (Tuning):**",
        'pt': "\n**Otimização (Tuning):**"
    }.get(lang, "\n**Optimización (Tuning):**")

    interp_stats_lbl = {
        'es': "\n**Interpretación de Pruebas Estadísticas:**",
        'en': "\n**Statistical Tests Interpretation:**",
        'pt': "\n**Interpretação dos Testes Estatísticos:**"
    }.get(lang, "\n**Interpretación de Pruebas Estadísticas:**")

    # 1. Portada
    title_p = doc.add_paragraph()
    title_p.alignment = 1 # Centrado
    run_title = title_p.add_run(f"\n\n\n\n🌽 {title_lbl.upper()}\n")
    run_title.font.size = Pt(24)
    run_title.bold = True
    run_title.font.color.rgb = RGBColor(16, 185, 129) # Verde
    
    subtitle_p = doc.add_paragraph()
    subtitle_p.alignment = 1
    run_sub = subtitle_p.add_run(f"{sub_lbl}\n\n\n\n")
    run_sub.font.size = Pt(14)
    run_sub.font.color.rgb = RGBColor(107, 114, 128) # Gris
    
    info_p = doc.add_paragraph()
    info_p.alignment = 1
    run_info = info_p.add_run(f"{gen_lbl}\n")
    run_info.font.size = Pt(11)
    
    doc.add_page_break()
    
    # 2. Sección EDA
    doc.add_heading(eda_heading, level=1)
    doc.add_paragraph(eda_desc)
    
    # Crear Tabla de EDA
    table_eda = doc.add_table(rows=1, cols=6)
    table_eda.style = 'Light Shading Accent 1'
    hdr_cells = table_eda.rows[0].cells
    for idx, name in enumerate(eda_headers):
        hdr_cells[idx].text = name
        
    for var_name, row in df_eda.iterrows():
        row_cells = table_eda.add_row().cells
        row_cells[0].text = str(var_name)
        row_cells[1].text = f"{row['media']:.2f}"
        row_cells[2].text = f"{row['mediana']:.2f}"
        row_cells[3].text = f"{row['desviación']:.2f}"
        row_cells[4].text = f"{row['asimetría']:.2f}"
        row_cells[5].text = f"{row['curtosis']:.2f}"
        
    doc.add_paragraph(eda_lbl)
    if 'balance' in image_paths and os.path.exists(image_paths['balance']):
        doc.add_picture(image_paths['balance'], width=Inches(4.5))
        
    doc.add_paragraph(corr_lbl)
    if 'correlation' in image_paths and os.path.exists(image_paths['correlation']):
        doc.add_picture(image_paths['correlation'], width=Inches(4.5))
        
    doc.add_paragraph(interp_eda_lbl)
    doc.add_paragraph(interpretations['eda'])
    
    # Tabla de Significancia de Variables Predictoras (si se proporciona)
    if df_significance is not None and not df_significance.empty:
        sig_title_lbl = {
            'es': "🔬 Prueba de Significancia Estadística de Variables Predictoras",
            'en': "🔬 Predictor Variables Statistical Significance Test",
            'pt': "🔬 Teste de Significância Estatística de Variáveis Preditoras"
        }.get(lang, "🔬 Prueba de Significancia Estadística de Variables Predictoras")
        doc.add_heading(sig_title_lbl, level=2)
        
        sig_headers = {
            'es': ['Variable', 'Prueba', 'Estadístico', 'p-valor', 'Significativo'],
            'en': ['Variable', 'Test', 'Statistic', 'p-value', 'Significant'],
            'pt': ['Variável', 'Teste', 'Estatística', 'p-valor', 'Significativo']
        }.get(lang, ['Variable', 'Prueba', 'Estadístico', 'p-valor', 'Significativo'])
        
        table_sig = doc.add_table(rows=1, cols=5)
        table_sig.style = 'Light Shading Accent 1'
        hdr_sig_cells = table_sig.rows[0].cells
        for idx, name in enumerate(sig_headers):
            hdr_sig_cells[idx].text = name
            
        for _, row in df_significance.iterrows():
            row_cells = table_sig.add_row().cells
            row_cells[0].text = str(row['Variable'])
            row_cells[1].text = str(row['Prueba'])
            row_cells[2].text = f"{row['Estadístico']:.4f}"
            row_cells[3].text = f"{row['p-valor']:.4f}"
            row_cells[4].text = str(row['Significativo'])
            
        doc.add_paragraph()
    
    doc.add_page_break()
    
    # 3. Sección Entrenamiento
    doc.add_heading(train_heading, level=1)
    doc.add_paragraph(train_desc)
    
    table_train = doc.add_table(rows=1, cols=6)
    table_train.style = 'Light Shading Accent 1'
    hdr_cells = table_train.rows[0].cells
    for idx, name in enumerate(train_headers):
        hdr_cells[idx].text = name
        
    for model_name, row in df_training.iterrows():
        row_cells = table_train.add_row().cells
        row_cells[0].text = str(model_name)
        row_cells[1].text = f"{row['Accuracy']:.4f}"
        row_cells[2].text = f"{row['Precision']:.4f}"
        row_cells[3].text = f"{row['Recall']:.4f}"
        row_cells[4].text = f"{row['F1-Score']:.4f}"
        row_cells[5].text = f"{row['AUC']:.4f}"
        
    doc.add_paragraph(roc_lbl)
    if 'roc' in image_paths and os.path.exists(image_paths['roc']):
        doc.add_picture(image_paths['roc'], width=Inches(5.0))
        
    doc.add_paragraph(interp_train_lbl)
    doc.add_paragraph(interpretations['training'])
    
    doc.add_page_break()
    
    # 4. Sección Cross Validation & Tuning
    doc.add_heading(cv_heading, level=1)
    doc.add_paragraph(cv_desc)
    
    # Tabla CV
    table_cv = doc.add_table(rows=1, cols=3)
    table_cv.style = 'Light Shading Accent 1'
    hdr_cells = table_cv.rows[0].cells
    for idx, name in enumerate(cv_headers):
        hdr_cells[idx].text = name
        
    for model_name, res in cv_results.items():
        row_cells = table_cv.add_row().cells
        row_cells[0].text = model_name
        row_cells[1].text = f"{res['mean_accuracy']:.4f} ± {res['std_accuracy']:.4f}"
        row_cells[2].text = f"{res['mean_f1']:.4f} ± {res['std_f1']:.4f}"
        
    doc.add_paragraph(cv_disp_lbl)
    if 'cv' in image_paths and os.path.exists(image_paths['cv']):
        doc.add_picture(image_paths['cv'], width=Inches(4.5))
        
    doc.add_paragraph(interp_cv_lbl)
    doc.add_paragraph(interpretations['cv'])
    
    doc.add_paragraph(tuning_lbl)
    doc.add_paragraph(interpretations['tuning'])
    
    doc.add_page_break()
    
    # 5. Sección Pruebas Estadísticas
    doc.add_heading(stats_heading, level=1)
    doc.add_paragraph(stats_desc)
    
    if 'stats' in image_paths and os.path.exists(image_paths['stats']):
        doc.add_picture(image_paths['stats'], width=Inches(4.5))
        
    doc.add_paragraph(interp_stats_lbl)
    doc.add_paragraph(interpretations['stats'])
    
    doc.save(filepath)
    return filepath

# ---------------------------------------------------------
def clean_pdf_text(text):
    if not text:
        return ""
    replacements = {
        "α": "alfa",
        "β": "beta",
        "γ": "gamma",
        "±": "+/-",
        "⭐": "*",
        "👉": ">",
        "💡": "Idea: ",
        "🌽": "Maiz: ",
        "📊": "Grafico: ",
        "🚀": "Inicio: ",
        "⚠️": "Alerta: ",
        "🗑️": "Eliminar: ",
        "✅": "OK",
        "❌": "Error",
        "📈": "Grafica",
        "🔍": "Buscar",
        "🤖": "Robot",
        "🔬": "Estudios",
        "⏱️": "Tiempo: ",
        "⏱": "Tiempo: ",
        "⚙️": "Config: ",
        "⚙": "Config: ",
        "🥇": "1ro",
        "🥈": "2do",
        "🥉": "3ro",
        "🏆": "Premio",
        "🧠": "Cerebro",
        "🎯": "Objetivo",
        "📋": "Lista",
        "📥": "Descargar",
        "📄": "Documento",
        "🩺": "Medicina",
        "🌾": "Trigo",
        "🌳": "Arbol"
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    # Remplazar cualquier caracter no-latin1
    return text.encode('latin-1', 'replace').decode('latin-1')

# ---------------------------------------------------------
# 3. GENERACIÓN DE PDF (.pdf)
# ---------------------------------------------------------
def generate_tabular_pdf_report(df_eda, df_training, cv_results, tuning_results, stats_results, interpretations, image_paths, filepath="reports/reporte_automl.pdf", lang="es", df_significance=None):
    """
    Crea un reporte PDF utilizando fpdf2 que incluye portada, tablas, interpretaciones e imágenes embebidas.
    """
    from fpdf import FPDF
    from src.translation import t_lang
    
    lang_key = lang if lang in ['es', 'en', 'pt'] else 'es'
    
    header_title = {
        'es': 'Informe Integral de AutoML y Diagnóstico Fitosanitario',
        'en': 'Comprehensive AutoML and Phytosanitary Diagnosis Report',
        'pt': 'Relatório Integral de AutoML e Diagnóstico Fitossanitário'
    }.get(lang_key, 'Informe Integral de AutoML y Diagnóstico Fitosanitario')

    page_lbl = {
        'es': 'Página',
        'en': 'Page',
        'pt': 'Página'
    }.get(lang_key, 'Página')

    class AutoMLPDF(FPDF):
        def header(self):
            if self.page_no() > 1:
                self.set_font('Helvetica', 'I', 8)
                self.set_text_color(107, 114, 128)
                self.cell(0, 10, clean_pdf_text(header_title), 0, 1, 'R')
                self.set_draw_color(16, 185, 129)
                self.line(10, 18, 200, 18)
                self.ln(10)
                
        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(107, 114, 128)
            self.cell(0, 10, clean_pdf_text(f'{page_lbl} {self.page_no()}'), 0, 0, 'C')
            
    pdf = AutoMLPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Textos localizados
    main_title = t_lang("rep_title", lang)
    sub_title = t_lang("rep_subtitle", lang)
    gen_text = t_lang("rep_generated", lang).format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    area_lbl = {
        'es': 'Área de Sanidad Vegetal - Cultivos de Maíz',
        'en': 'Plant Health Department - Maize Crops',
        'pt': 'Área de Sanidade Vegetal - Culturas de Milho'
    }.get(lang_key, 'Área de Sanidad Vegetal - Cultivos de Maíz')

    # --- Portada ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(16, 185, 129) # Verde
    pdf.ln(50)
    pdf.cell(0, 15, clean_pdf_text(main_title.upper()), ln=1, align='C')
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(107, 114, 128)
    pdf.cell(0, 10, clean_pdf_text(sub_title), ln=1, align='C')
    pdf.ln(60)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 6, clean_pdf_text(gen_text), ln=1, align='C')
    pdf.cell(0, 6, clean_pdf_text(area_lbl), ln=1, align='C')
    
    # --- Página 2: EDA ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, clean_pdf_text(t_lang("rep_eda_section", lang)), ln=1)
    pdf.ln(5)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(31, 41, 55)
    eda_desc = {
        'es': "Estadísticos descriptivos de las variables numéricas del dataset de cultivo:",
        'en': "Descriptive statistics of the numerical variables of the crop dataset:",
        'pt': "Estatísticas descritivas das variáveis numéricas do conjunto de dados de cultivo:"
    }.get(lang_key, "Estadísticos descriptivos de las variables numéricas del dataset de cultivo:")
    pdf.multi_cell(0, 6, clean_pdf_text(eda_desc))
    pdf.ln(3)
    
    # Tabla EDA en PDF
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(240, 240, 240)
    
    eda_headers = {
        'es': ['Variable', 'Media', 'Mediana', 'Desv.', 'Asim.', 'Curt.'],
        'en': ['Variable', 'Mean', 'Median', 'Std Dev', 'Skew', 'Kurt.'],
        'pt': ['Variável', 'Média', 'Mediana', 'Desvio', 'Assim.', 'Curtose']
    }.get(lang_key, ['Variable', 'Media', 'Mediana', 'Desv.', 'Asim.', 'Curt.'])

    col_widths = [45, 30, 30, 30, 25, 25]
    for w, h in zip(col_widths, eda_headers):
        pdf.cell(w, 8, clean_pdf_text(h), border=1, align='C', fill=True)
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 9)
    for var_name, row in df_eda.iterrows():
        pdf.cell(45, 7, clean_pdf_text(str(var_name)), border=1)
        pdf.cell(30, 7, f"{row['media']:.2f}", border=1, align='C')
        pdf.cell(30, 7, f"{row['mediana']:.2f}", border=1, align='C')
        pdf.cell(30, 7, f"{row['desviación']:.2f}", border=1, align='C')
        pdf.cell(25, 7, f"{row['asimetría']:.2f}", border=1, align='C')
        pdf.cell(25, 7, f"{row['curtosis']:.2f}", border=1, align='C')
        pdf.ln()
        
    # Tabla de Significancia de Variables Predictoras (si se proporciona)
    if df_significance is not None and not df_significance.empty:
        pdf.ln(5)
        sig_title = {
            'es': 'Significancia Estadística de Predictores (ANOVA/Kruskal-Wallis)',
            'en': 'Predictors Statistical Significance (ANOVA/Kruskal-Wallis)',
            'pt': 'Significância Estatística de Preditores (ANOVA/Kruskal-Wallis)'
        }.get(lang_key, 'Significancia Estadística de Predictores')
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, clean_pdf_text(sig_title), ln=1)
        pdf.ln(2)
        
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font('Helvetica', 'B', 8)
        sig_headers = {
            'es': ['Variable', 'Prueba', 'Estadístico', 'p-valor', 'Sig.'],
            'en': ['Variable', 'Test', 'Statistic', 'p-value', 'Sig.'],
            'pt': ['Variável', 'Teste', 'Estatística', 'p-valor', 'Sig.']
        }.get(lang_key, ['Variable', 'Prueba', 'Estadístico', 'p-valor', 'Sig.'])
        
        sig_widths = [45, 60, 30, 30, 20]
        for w, h in zip(sig_widths, sig_headers):
            pdf.cell(w, 7, clean_pdf_text(h), border=1, align='C', fill=True)
        pdf.ln()
        
        pdf.set_font('Helvetica', '', 8)
        for _, row in df_significance.iterrows():
            pdf.cell(45, 6, clean_pdf_text(str(row['Variable'])), border=1)
            pdf.cell(60, 6, clean_pdf_text(str(row['Prueba'])), border=1)
            pdf.cell(30, 6, f"{row['Estadístico']:.4f}", border=1, align='C')
            pdf.cell(30, 6, f"{row['p-valor']:.4f}", border=1, align='C')
            pdf.cell(20, 6, clean_pdf_text(str(row['Significativo'])), border=1, align='C')
            pdf.ln()
        pdf.ln(5)
    else:
        pdf.ln(8)
    has_eda_images = False
    if 'balance' in image_paths and os.path.exists(image_paths['balance']):
        pdf.image(image_paths['balance'], x=15, y=pdf.get_y(), w=85)
        has_eda_images = True
    if 'correlation' in image_paths and os.path.exists(image_paths['correlation']):
        pdf.image(image_paths['correlation'], x=110, y=pdf.get_y(), w=85)
        has_eda_images = True
    
    if has_eda_images:
        pdf.ln(62)
    else:
        pdf.ln(3)
        
    pdf.set_font('Helvetica', 'B', 10)
    interp_eda_title = {
        'es': 'Interpretación Técnica del EDA:',
        'en': 'EDA Technical Interpretation:',
        'pt': 'Interpretação Técnica do EDA:'
    }.get(lang_key, 'Interpretación Técnica del EDA:')
    pdf.cell(0, 6, clean_pdf_text(interp_eda_title), ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, clean_pdf_text(interpretations['eda'].replace('**', '')))
    
    # --- Página 3: Entrenamiento ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, clean_pdf_text(t_lang("rep_train_section", lang)), ln=1)
    pdf.ln(5)
    
    # Tabla Entrenamiento
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(240, 240, 240)
    
    train_headers = {
        'es': ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'en': ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'pt': ['Modelo', 'Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC']
    }.get(lang_key, ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'])

    col_w_train = [55, 27, 27, 27, 27, 27]
    for w, h in zip(col_w_train, train_headers):
        pdf.cell(w, 8, clean_pdf_text(h), border=1, align='C', fill=True)
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 8)
    for model_name, row in df_training.iterrows():
        pdf.cell(55, 7, clean_pdf_text(str(model_name)), border=1)
        pdf.cell(27, 7, f"{row['Accuracy']:.4f}", border=1, align='C')
        pdf.cell(27, 7, f"{row['Precision']:.4f}", border=1, align='C')
        pdf.cell(27, 7, f"{row['Recall']:.4f}", border=1, align='C')
        pdf.cell(27, 7, f"{row['F1-Score']:.4f}", border=1, align='C')
        pdf.cell(27, 7, f"{row['AUC']:.4f}", border=1, align='C')
        pdf.ln()
        
    pdf.ln(5)
    has_roc = False
    if 'roc' in image_paths and os.path.exists(image_paths['roc']):
        pdf.image(image_paths['roc'], x=40, y=pdf.get_y(), w=120)
        has_roc = True
        
    if has_roc:
        pdf.ln(78)
    else:
        pdf.ln(3)
        
    pdf.set_font('Helvetica', 'B', 10)
    interp_train_title = {
        'es': 'Interpretación de Entrenamiento:',
        'en': 'Training Interpretation:',
        'pt': 'Interpretação do Treinamento:'
    }.get(lang_key, 'Interpretación de Entrenamiento:')
    pdf.cell(0, 6, clean_pdf_text(interp_train_title), ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, clean_pdf_text(interpretations['training'].replace('**', '')))
    
    # --- Página 4: CV, Tuning y Pruebas Estadísticas ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(16, 185, 129)
    
    section_3_title = {
        'es': '3. Validación Cruzada, Tuning y Pruebas Estadísticas',
        'en': '3. Cross Validation, Tuning and Statistical Tests',
        'pt': '3. Validação Cruzada, Tuning e Testes Estatísticos'
    }.get(lang_key, '3. Validación Cruzada, Tuning y Pruebas Estadísticas')
    
    pdf.cell(0, 10, clean_pdf_text(section_3_title), ln=1)
    pdf.ln(5)
    
    has_cv_stats = False
    if 'cv' in image_paths and os.path.exists(image_paths['cv']):
        pdf.image(image_paths['cv'], x=15, y=pdf.get_y(), w=85)
        has_cv_stats = True
    if 'stats' in image_paths and os.path.exists(image_paths['stats']):
        pdf.image(image_paths['stats'], x=110, y=pdf.get_y(), w=85)
        has_cv_stats = True
        
    if has_cv_stats:
        pdf.ln(58)
    else:
        pdf.ln(3)
        
    pdf.set_font('Helvetica', 'B', 10)
    interp_cv_title = {
        'es': 'Interpretación de Validación y Tuning:',
        'en': 'Validation and Tuning Interpretation:',
        'pt': 'Interpretação de Validação e Tuning:'
    }.get(lang_key, 'Interpretación de Validación y Tuning:')
    pdf.cell(0, 6, clean_pdf_text(interp_cv_title), ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, clean_pdf_text(interpretations['cv'].replace('**', '') + "\n" + interpretations['tuning'].replace('**', '')))
    
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 10)
    methodology_title = {
        'es': 'Metodología de Validación Estadística Utilizada:',
        'en': 'Statistical Validation Methodology Used:',
        'pt': 'Metodologia de Validação Estatística Utilizada:'
    }.get(lang_key, 'Metodología de Validación Estadística Utilizada:')
    pdf.cell(0, 6, clean_pdf_text(methodology_title), ln=1)
    
    pdf.set_font('Helvetica', '', 8.5)
    stats_desc_text = {
        'es': "El sistema ejecuta un análisis secuencial: 1) Normalidad de Shapiro-Wilk y Homocedasticidad de Levene. 2) ANOVA (paramétrico) o Friedman (no paramétrico) según el cumplimiento de supuestos. 3) Pruebas de Tukey HSD o Wilcoxon (apareada con Bonferroni) para comparaciones post-hoc de diferencias por parejas. 4) Prueba de McNemar para evaluar si la proporción de fallos difiere significativamente sobre el conjunto de test.",
        'en': "The system executes a sequential analysis: 1) Shapiro-Wilk normality and Levene's homoscedasticity. 2) ANOVA (parametric) or Friedman (non-parametric) depending on assumption compliance. 3) Tukey HSD or Wilcoxon tests (paired with Bonferroni correction) for post-hoc pairwise comparisons. 4) McNemar's test to evaluate if the error proportion differs significantly on the test set.",
        'pt': "O sistema executa uma análise sequencial: 1) Normalidade de Shapiro-Wilk e Homocedasticidade de Levene. 2) ANOVA (paramétrico) ou Friedman (não paramétrico) de acordo com o cumprimento das suposições. 3) Testes de Tukey HSD ou Wilcoxon (emparelhado com Bonferroni) para comparações pós-hoc emparelhadas. 4) Teste de McNemar para avaliar se a proporção de erros difere significativamente no conjunto de teste."
    }.get(lang_key, '')
    pdf.multi_cell(0, 4.5, clean_pdf_text(stats_desc_text))
    pdf.ln(3)

    pdf.set_font('Helvetica', 'B', 10)
    interp_stats_title = {
        'es': 'Interpretación de Pruebas de Significancia Estadística:',
        'en': 'Statistical Significance Tests Interpretation:',
        'pt': 'Interpretação de Testes de Significância Estatística:'
    }.get(lang_key, 'Interpretación de Pruebas de Significancia Estadística:')
    pdf.cell(0, 6, clean_pdf_text(interp_stats_title), ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, clean_pdf_text(interpretations['stats'].replace('**', '')))
    
    pdf.output(filepath)
    return filepath
# ---------------------------------------------------------
# 4. REPORTES PARA DIAGNÓSTICO POR IMÁGENES (WORD & EXCEL)
# ---------------------------------------------------------
def generate_image_docx_report(image, predictions, uploaded_filename, consensus_reached, consensus_diagnosis, filepath="reports/reporte_imagen.docx", lang="es"):
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    import tempfile
    from datetime import datetime
    import matplotlib.pyplot as plt
    import numpy as np
    from src.translation import t_lang, translate_class_lang
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    doc = Document()
    
    # Configurar estilos de fuente globales
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(11)
    
    lang_key = lang if lang in ['es', 'en', 'pt'] else 'es'
    
    # Textos localizados
    title_lbl = {
        'es': "INFORME DE DIAGNÓSTICO FITOSANITARIO",
        'en': "PHYTOSANITARY DIAGNOSIS REPORT",
        'pt': "RELATÓRIO DE DIAGNÓSTICO FITOSSANITÁRIO"
    }.get(lang_key, "INFORME DE DIAGNÓSTICO FITOSANITARIO")

    sub_lbl = {
        'es': "Detección Automática de Patologías en Hojas de Maíz",
        'en': "Automatic Detection of Pathologies in Maize Leaves",
        'pt': "Detecção Automática de Patologias em Folhas de Milho"
    }.get(lang_key, "Detección Automática de Patologías en Hojas de Maíz")

    info_lbl = {
        'es': "Archivo analizado: {file}\nFecha y hora: {date} (Hora local)\nÁrea de Sanidad Vegetal\n",
        'en': "Analyzed file: {file}\nDate and time: {date} (Local time)\nPlant Health Department\n",
        'pt': "Arquivo analisado: {file}\nData e hora: {date} (Hora local)\nÁrea de Sanidade Vegetal\n"
    }.get(lang_key)

    section_1_title = {
        'es': "1. Diagnóstico Principal por Consenso",
        'en': "1. Main Consensus Diagnosis",
        'pt': "1. Diagnóstico Principal por Consenso"
    }.get(lang_key)

    section_1_desc = {
        'es': "Resultado de la clasificación combinada de múltiples redes neuronales convolucionales:",
        'en': "Result of the combined classification of multiple convolutional neural networks:",
        'pt': "Resultado da classificação combinada de múltiplas redes neurais convolucionais:"
    }.get(lang_key)

    section_2_title = {
        'es': "2. Imagen de la Hoja de Maíz Analizada",
        'en': "2. Image of the Analyzed Maize Leaf",
        'pt': "2. Imagem da Folha de Milho Analisada"
    }.get(lang_key)

    err_img_lbl = {
        'es': "[Error incrustando la imagen: {e}]",
        'en': "[Error embedding image: {e}]",
        'pt': "[Erro ao incorporar imagem: {e}]"
    }.get(lang_key)

    section_3_title = {
        'es': "2. Resultados Detallados de los Modelos",
        'en': "2. Detailed Model Results",
        'pt': "2. Resultados Detalhados dos Modelos"
    }.get(lang_key)

    section_3_desc = {
        'es': "Métricas y predicciones individuales de cada red neuronal entrenada:",
        'en': "Individual metrics and predictions of each trained neural network:",
        'pt': "Métricas e previsões individuais de cada rede neural treinada:"
    }.get(lang_key)

    section_4_title = {
        'es': "3. Análisis Comparativo de Predicciones",
        'en': "3. Comparative Prediction Analysis",
        'pt': "3. Análise Comparativa de Previsões"
    }.get(lang_key)

    table_headers = {
        'es': ['Modelo de IA', 'Diagnóstico', 'Confianza', 'Estado General'],
        'en': ['AI Model', 'Diagnosis', 'Confidence', 'General Status'],
        'pt': ['Modelo de IA', 'Diagnóstico', 'Confiança', 'Estado Geral']
    }.get(lang_key)

    healthy_lbl = {
        'es': 'Saludable',
        'en': 'Healthy',
        'pt': 'Saudável'
    }.get(lang_key)

    infected_lbl = {
        'es': 'Infección Detectada',
        'en': 'Infection Detected',
        'pt': 'Infecção Detectada'
    }.get(lang_key)

    general_recs_title = {
        'es': "5. Recomendaciones Generales del Sistema",
        'en': "5. General System Recommendations",
        'pt': "5. Recomendações Gerais do Sistema"
    }.get(lang_key)

    disclaimer_title = {
        'es': "⚠️ Limitaciones y Responsabilidad",
        'en': "⚠️ Limitations and Liability",
        'pt': "⚠️ Limitações e Responsabilidade"
    }.get(lang_key)

    disclaimer_text = {
        'es': "Este sistema es una herramienta de soporte analítico basada en redes neuronales. Los resultados deben ser confirmados visualmente en campo por ingenieros agrónomos o técnicos fitosanitarios calificados antes de realizar aplicaciones masivas de tratamientos.",
        'en': "This system is an analytical support tool based on neural networks. Results must be visually confirmed in the field by qualified agronomists or phytosanitary technicians before performing massive treatment applications.",
        'pt': "Este sistema é uma ferramenta de suporte analítico baseada em redes neurais. Os resultados devem ser confirmados visualmente em campo por engenheiros agrônomos ou técnicos fitossanitários qualificados antes de realizar aplicações massivas de tratamentos."
    }.get(lang_key)

    # --- PÁGINA 1: PORTADA ---
    title_p = doc.add_paragraph()
    title_p.alignment = 1 # Centrado
    run_title = title_p.add_run(f"\n\n\n\n🌽 {title_lbl}\n")
    run_title.font.size = Pt(22)
    run_title.bold = True
    run_title.font.color.rgb = RGBColor(46, 139, 87) # Verde
    
    subtitle_p = doc.add_paragraph()
    subtitle_p.alignment = 1
    run_sub = subtitle_p.add_run(f"{sub_lbl}\n\n\n\n")
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = RGBColor(100, 100, 100)
    
    info_p = doc.add_paragraph()
    info_p.alignment = 1
    run_info = info_p.add_run(info_lbl.format(file=uploaded_filename, date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    run_info.font.size = Pt(10)
    
    doc.add_page_break()
    
    # --- PÁGINA 2: DIAGNÓSTICO PRINCIPAL ---
    doc.add_heading(section_1_title, level=1)
    doc.add_paragraph(section_1_desc)
    
    # Cuadro de Consenso
    p_cons = doc.add_paragraph()
    if consensus_reached:
        translated_diag = translate_class_lang(consensus_diagnosis, lang_key)
        diag_title = {
            'es': 'DIAGNÓSTICO GENERAL',
            'en': 'GENERAL DIAGNOSIS',
            'pt': 'DIAGNÓSTICO GERAL'
        }.get(lang_key, 'DIAGNÓSTICO GENERAL')
        res_text = f"{diag_title}: {translated_diag.upper()}"
        color = RGBColor(46, 139, 87) if consensus_diagnosis == "Sano" else RGBColor(185, 28, 28)
    else:
        no_cons_text = {
            'es': 'DIAGNÓSTICO GENERAL: SIN CONSENSO DEFINIDO',
            'en': 'GENERAL DIAGNOSIS: NO CONSENSUS DEFINED',
            'pt': 'DIAGNÓSTICO GERAL: SEM CONSENSO DEFINIDO'
        }.get(lang_key, 'DIAGNÓSTICO GENERAL: SIN CONSENSO DEFINIDO')
        res_text = no_cons_text
        color = RGBColor(217, 119, 6)
    run_cons = p_cons.add_run(f"\n   {res_text}   \n")
    run_cons.bold = True
    run_cons.font.size = Pt(14)
    run_cons.font.color.rgb = color
    
    # Imagen de la hoja
    doc.add_heading(section_2_title, level=2)
    try:
        from PIL import Image
        temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_img_name = temp_img.name
        temp_img.close()
        
        image_pil = Image.fromarray(image)
        image_pil.save(temp_img_name, format="PNG")
        doc.add_picture(temp_img_name, width=Inches(3.2))
        os.remove(temp_img_name)
    except Exception as e:
        doc.add_paragraph(err_img_lbl.format(e=e))
        
    doc.add_page_break()
    
    # --- PÁGINA 3: RESULTADOS DETALLADOS POR MODELO ---
    doc.add_heading(section_3_title, level=1)
    doc.add_paragraph(section_3_desc)
    
    # Gráficos de barras individuales por modelo
    class_names = ["Mancha gris", "Roña común", "Tizón del norte", "Sano"]
    temp_graph_paths = []
    
    try:
        for model_name, pred in predictions.items():
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#16A34A']
            translated_class_names = [translate_class_lang(name, lang_key) for name in class_names]
            bars = ax.bar(translated_class_names, pred['probabilities'], color=colors, alpha=0.8)
            ax.set_title(f'Modelo {model_name}', fontsize=10, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.2, axis='y')
            
            # Resaltar más alta
            max_idx = np.argmax(pred['probabilities'])
            bars[max_idx].set_color('#16A34A')
            
            # Valores
            for j, v in enumerate(pred['probabilities']):
                ax.text(j, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=8)
                
            plt.tight_layout()
            temp_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            plt.savefig(temp_path, dpi=120)
            temp_graph_paths.append(temp_path)
            plt.close()
            
        # Incrustar en Word
        for idx, (model_name, pred) in enumerate(predictions.items()):
            doc.add_heading(f"Modelo: {model_name}", level=2)
            trans_class = translate_class_lang(pred['class'], lang_key)
            pred_lbl = {
                'es': 'Predicción',
                'en': 'Prediction',
                'pt': 'Previsão'
            }.get(lang_key, 'Predicción')
            conf_lbl = {
                'es': 'Confianza',
                'en': 'Confidence',
                'pt': 'Confiança'
            }.get(lang_key, 'Confianza')
            doc.add_paragraph(f"{pred_lbl}: {trans_class} | {conf_lbl}: {pred['confidence']:.2%}")
            if idx < len(temp_graph_paths) and os.path.exists(temp_graph_paths[idx]):
                doc.add_picture(temp_graph_paths[idx], width=Inches(4.5))
                os.remove(temp_graph_paths[idx])
                
    except Exception as e:
        doc.add_paragraph(f"[Error generando gráficos: {e}]")
        
    # --- PÁGINA 4: TABLA COMPARATIVA Y ENFERMEDADES ---
    doc.add_heading(section_4_title, level=1)
    
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Light Shading Accent 1'
    hdr_cells = table.rows[0].cells
    for i, h_name in enumerate(table_headers):
        hdr_cells[i].text = h_name
        
    for model_name, pred in predictions.items():
        row_cells = table.add_row().cells
        row_cells[0].text = model_name
        row_cells[1].text = translate_class_lang(pred['class'], lang_key)
        row_cells[2].text = f"{pred['confidence']:.2%}"
        row_cells[3].text = healthy_lbl if pred['class'] == 'Sano' else infected_lbl
        
    # Información sobre la patología
    if consensus_reached and consensus_diagnosis != "Sano":
        info_tech_title = {
            'es': "4. Información Técnica sobre: {diag}",
            'en': "4. Technical Information on: {diag}",
            'pt': "4. Informações Técnicas sobre: {diag}"
        }.get(lang_key, "4. Información Técnica sobre: {diag}").format(diag=translate_class_lang(consensus_diagnosis, lang_key))
        doc.add_heading(info_tech_title, level=2)
        
        disease_info = {
            'es': {
                "Tizón del norte": {
                    "desc": "Enfermedad fúngica severa causada por Exserohilum turcicum que provoca lesiones alargadas elípticas (en forma de cigarro/puro) de color verde-grisáceo a marrón, que pueden unirse provocando la necrosis foliar completa.",
                    "recs": [
                        "Uso de híbridos comerciales resistentes: Seleccionar variedades que incorporen resistencia cuantitativa y genes Ht específicos.",
                        "Rotación de cultivos: Implementar rotación de 1 a 2 años con especies no gramíneas (soya, leguminosas o girasol) para romper el ciclo biológico del patógeno.",
                        "Manejo adecuado de rastrojos: Realizar labranza profunda (arado) para enterrar los residuos del cultivo anterior infectado y acelerar su descomposición.",
                        "Tratamiento químico foliar oportuno: Aplicar fungicidas sistémicos (triazoles, estrobirulinas o carboxamidas) si la severidad en hojas inferiores supera el 15% antes de la floración.",
                        "Monitoreo fitosanitario continuo: Realizar inspecciones semanales en el envés de las hojas inferiores para detectar focos iniciales."
                    ]
                },
                "Roña común": {
                    "desc": "Enfermedad fúngica provocada por Puccinia sorghi que produce pequeñas pústulas circulares u ovaladas de color marrón-rojizo en ambas caras de la hoja, liberando esporas polvorientas que se dispersan por el viento.",
                    "recs": [
                        "Siembra de híbridos resistentes: Priorizar el uso de variedades que cuenten con resistencia Rp específica.",
                        "Ajuste en fechas de siembra: Programar siembras tempranas para evitar que las etapas críticas del cultivo coincidan con condiciones frescas y húmedas favorables para el patógeno.",
                        "Aplicación de fungicidas sistémicos: Utilizar mezclas comerciales de triazoles y estrobirulinas al observar las primeras pústulas en el tercio inferior/medio del cultivo.",
                        "Control de mojado foliar: Ajustar el riego por aspersión para reducir las horas de mojado en las hojas, lo que disminuye la germinación de esporas.",
                        "Manejo de nutrición mineral: Mantener una nutrición nitrogenada y potásica balanceada para fortalecer la resistencia física de la cutícula foliar."
                    ]
                },
                "Roya común": {
                    "desc": "Enfermedad fúngica provocada por Puccinia sorghi que produce pequeñas pústulas circulares u ovaladas de color marrón-rojizo en ambas caras de la hoja, liberando esporas polvorientas que se dispersan por el viento.",
                    "recs": [
                        "Siembra de híbridos resistentes: Priorizar el uso de variedades que cuenten con resistencia Rp específica.",
                        "Ajuste en fechas de siembra: Programar siembras tempranas para evitar que las etapas críticas del cultivo coincidan con condiciones frescas y húmedas favorables para el patógeno.",
                        "Aplicación de fungicidas sistémicos: Utilizar mezclas comerciales de triazoles y estrobirulinas al observar las primeras pústulas en el tercio inferior/medio del cultivo.",
                        "Control de mojado foliar: Ajustar el riego por aspersión para reducir las horas de mojado en las hojas, lo que disminuye la germinación de esporas.",
                        "Manejo de nutrición mineral: Mantener una nutrición nitrogenada y potásica balanceada para fortalecer la resistencia física de la cutícula foliar."
                    ]
                },
                "Mancha gris": {
                    "desc": "Enfermedad fúngica destructiva causada por Cercospora zeae-maydis que se manifiesta como lesiones rectangulares delimitadas por las nervaduras de la hoja foliar, tornándose grisáceas con el tiempo.",
                    "recs": [
                        "Rotación sistemática de cultivos: Alternar el campo con especies no gramíneas por un periodo mínimo de 1 a 2 años.",
                        "Manejo e incorporación de residuos: Enterrar rastrojos infectados para reducir sustancialmente el inóculo primario presente en el suelo.",
                        "Selección de semillas tolerantes: Sembrar híbridos con alta tolerancia genética comprobada en zonas con historial de la enfermedad.",
                        "Control químico foliar estratégico: Aplicar mezclas de fungicidas específicos (estrobirulinas y triazoles) si se observan lesiones tempranas bajo condiciones de alta humedad.",
                        "Optimización de densidad de siembra: Regular el número de plantas por hectárea para mejorar la aireación interna del dosel y reducir la humedad microclimática."
                    ]
                }
            },
            'en': {
                "Tizón del norte": {
                    "desc": "Severe fungal disease caused by Exserohilum turcicum that induces elongated, cigar-shaped grayish-green to brown lesions, which can merge causing complete leaf necrosis.",
                    "recs": [
                        "Use resistant commercial hybrids: Select varieties incorporating quantitative resistance and specific Ht genes.",
                        "Crop rotation: Implement a 1-to-2-year rotation with non-grass species (soybean, legumes, or sunflower) to break the pathogen's life cycle.",
                        "Proper residue management: Perform deep tillage (plowing) to bury infected crop residues from the previous season and speed up decomposition.",
                        "Timely foliar chemical treatment: Apply systemic fungicides (triazoles, strobilurins, or carboxamides) if severity on lower leaves exceeds 15% before flowering.",
                        "Continuous phytosanitary monitoring: Perform weekly inspections of the underside of lower leaves to detect initial hotspots."
                    ]
                },
                "Roña común": {
                    "desc": "Fungal disease caused by Puccinia sorghi that produces small circular or oval reddish-brown pustules on both leaf surfaces, releasing powdery spores dispersed by wind.",
                    "recs": [
                        "Sowing of resistant hybrids: Prioritize the use of varieties with specific Rp resistance genes.",
                        "Adjustment of planting dates: Schedule early planting to prevent critical crop stages from coinciding with cool, humid conditions favorable to the pathogen.",
                        "Systemic fungicide application: Use commercial mixtures of triazoles and strobilurins when first pustules are observed in the lower/middle third of the crop.",
                        "Leaf wetness control: Adjust sprinkler irrigation to reduce leaf wetness duration, decreasing spore germination.",
                        "Mineral nutrition management: Maintain balanced nitrogen and potassium nutrition to strengthen the physical resistance of the leaf cuticle."
                    ]
                },
                "Roya común": {
                    "desc": "Fungal disease caused by Puccinia sorghi that produces small circular or oval reddish-brown pustules on both leaf surfaces, releasing powdery spores dispersed by wind.",
                    "recs": [
                        "Sowing of resistant hybrids: Prioritize the use of varieties with specific Rp resistance genes.",
                        "Adjustment of planting dates: Schedule early planting to prevent critical crop stages from coinciding with cool, humid conditions favorable to the pathogen.",
                        "Systemic fungicide application: Use commercial mixtures of triazoles and strobilurins when first pustules are observed in the lower/middle third of the crop.",
                        "Leaf wetness control: Adjust sprinkler irrigation to reduce leaf wetness duration, decreasing spore germination.",
                        "Mineral nutrition management: Maintain balanced nitrogen and potassium nutrition to strengthen the physical resistance of the leaf cuticle."
                    ]
                },
                "Mancha gris": {
                    "desc": "Destructive fungal disease caused by Cercospora zeae-maydis that appears as rectangular lesions restricted by leaf veins, turning grayish over time.",
                    "recs": [
                        "Systematic crop rotation: Alternate the field with non-grass species for a minimum of 1 to 2 years.",
                        "Residue management and incorporation: Bury infected stubble to substantially reduce primary inoculum present in the soil.",
                        "Tolerant seed selection: Sow hybrids with proven high genetic tolerance in areas with a history of the disease.",
                        "Strategic foliar chemical control: Apply specific fungicide mixtures (strobilurins and triazoles) if early lesions are seen under high humidity.",
                        "Sowing density optimization: Regulate the number of plants per hectare to improve air circulation within the canopy and reduce microclimatic humidity."
                    ]
                }
            },
            'pt': {
                "Tizón del norte": {
                    "desc": "Doença fúngica grave causada por Exserohilum turcicum que provoca lesões elípticas alongadas (em forma de charuto) de cor verde-acinzentada a marrom, que podem coalescer causando necrose foliar completa.",
                    "recs": [
                        "Uso de híbridos comerciais resistentes: Selecionar variedades que incorporem resistência quantitativa e genes Ht específicos.",
                        "Rotação de culturas: Implementar rotação de 1 a 2 anos com espécies não gramíneas (soja, leguminosas ou girassol) para quebrar o ciclo de vida do patógeno.",
                        "Manejo adequado de resíduos: Realizar aração profunda para enterrar os resíduos da safra anterior infectada e acelerar a decomposição.",
                        "Tratamento químico foliar oportuno: Aplicar fungicidas sistêmicos (triazóis, estrobirulinas ou carboxamidas) se a severidade nas folhas inferiores exceder 15% antes do florescimento.",
                        "Monitoramento fitossanitário contínuo: Realizar inspeções semanais na face inferior das folhas inferiores para detectar focos iniciais."
                    ]
                },
                "Roña común": {
                    "desc": "Doença fúngica provocada por Puccinia sorghi que produz pequenas pústulas circulares ou ovais de cor marrom-avermelhada em ambas as superfícies da folha, liberando esporos dispersos pelo vento.",
                    "recs": [
                        "Plantio de híbridos resistentes: Priorizar o uso de variedades que possuam resistência Rp específica.",
                        "Ajuste nas datas de plantio: Programar plantios precoces para evitar que estágios críticos coincidam com condições frescas e úmidas favoráveis ao patógeno.",
                        "Aplicação de fungicidas sistêmicos: Utilizar misturas comerciais de triazóis e estrobirulinas quando as primeiras pústulas forem observadas no terço inferior/médio da cultura.",
                        "Controle de molhamento foliar: Ajustar a irrigação por aspersão para reduzir as horas de molhamento foliar, diminuindo a germinação dos esporas.",
                        "Manejo de nutrição mineral: Manter uma nutrição nitrogenada e potássica equilibrada para fortalecer a resistência física da cutícula foliar."
                    ]
                },
                "Roya común": {
                    "desc": "Doença fúngica provocada por Puccinia sorghi que produz pequenas pústulas circulares ou ovais de cor marrom-avermelhada em ambas as superfícies da folha, liberando esporas dispersos pelo vento.",
                    "recs": [
                        "Plantio de híbridos resistentes: Priorizar o uso de variedades que possuam resistência Rp específica.",
                        "Ajuste nas datas de plantio: Programar plantios precoces para evitar que estágios críticos coincidam com condições frescas e úmidas favoráveis ao patógeno.",
                        "Aplicação de fungicidas sistêmicos: Utilizar misturas comerciais de triazóis e estrobirulinas quando as primeiras pústulas forem observadas no terço inferior/médio da cultura.",
                        "Controle de molhamento foliar: Ajustar a irrigação por aspersão para reduzir as horas de molhamento foliar, diminuindo a germinação dos esporas.",
                        "Manejo de nutrição mineral: Manter uma nutrição nitrogenada e potássica equilibrada para fortalecer a resistência física da cutícula foliar."
                    ]
                },
                "Mancha gris": {
                    "desc": "Doença fúngica destrutiva causada por Cercospora zeae-maydis que se manifesta como lesões retangulares delimitadas pelas nervuras foliares, tornando-se acinzentadas com o tempo.",
                    "recs": [
                        "Rotação sistemática de culturas: Alternar o campo com espécies não gramíneas por um período mínimo de 1 a 2 anos.",
                        "Manejo e incorporação de resíduos: Enterrar a palhada infectada para reduzir substancialmente o inóculo primário presente no solo.",
                        "Seleção de sementes tolerantes: Semear híbridos com alta tolerância genética comprovada em áreas com histórico da doença.",
                        "Controle químico foliar estratégico: Aplicar misturas de fungicidas específicos (estrobirulinas e trizóis) se forem observadas lesões iniciais sob alta umidade.",
                        "Otimização da densidade de plantio: Regular o número de plantas por hectare para melhorar a ventilação e reduzir a umidade microclimática."
                    ]
                }
            }
        }.get(lang_key, {})
        
        # normalization of key
        lookup_diag = consensus_diagnosis
        if lookup_diag not in disease_info and lookup_diag == "Roya común" and "Roña común" in disease_info:
            lookup_diag = "Roña común"
        elif lookup_diag not in disease_info and lookup_diag == "Roña común" and "Roya común" in disease_info:
            lookup_diag = "Roya común"
            
        if lookup_diag in disease_info:
            d = disease_info[lookup_diag]
            desc_lbl = {
                'es': 'Descripción',
                'en': 'Description',
                'pt': 'Descrição'
            }.get(lang_key, 'Descripción')
            recs_mgmt_lbl = {
                'es': 'Recomendaciones de Manejo:',
                'en': 'Management Recommendations:',
                'pt': 'Recomendações de Manejo:'
            }.get(lang_key, 'Recomendaciones de Manejo:')
            doc.add_paragraph(f"**{desc_lbl}:** {d['desc']}")
            doc.add_heading(recs_mgmt_lbl, level=3)
            for r in d['recs']:
                doc.add_paragraph(f"- {r}")
                
    # --- RECOMENDACIONES GENERALES Y DISCLAIMER ---
    doc.add_heading(general_recs_title, level=1)
    
    if consensus_reached and consensus_diagnosis == "Sano":
        recs = {
            'es': [
                "Continuar con las prácticas de manejo actuales.",
                "Realizar monitoreos preventivos regulares cada 7-10 días.",
                "Mantener condiciones óptimas de cultivo (riego, fertilización).",
                "Inspeccionar las hojas inferiores que tienen contacto directo con la humedad del suelo."
            ],
            'en': [
                "Continue with current management practices.",
                "Perform regular preventive monitoring every 7-10 days.",
                "Maintain optimal crop conditions (irrigation, fertilization).",
                "Inspect lower leaves that have direct contact with soil moisture."
            ],
            'pt': [
                "Continuar com as práticas de manejo atuais.",
                "Realizar monitoramentos preventivos regulares a cada 7-10 dias.",
                "Saúde ideal do milho (irrigação, fertilização).",
                "Inspecionar as folhas inferiores que têm contato direto com a umidade do solo."
            ]
        }.get(lang_key, [])
    elif consensus_reached:
        recs = {
            'es': [
                "Aislar de inmediato la zona de cultivo afectada para evitar la propagación foliar por viento.",
                "Considerar tratamientos preventivos con fungicidas orgánicos o químicos regulados.",
                "Evitar el riego por aspersión directo al follaje en horas de la tarde para reducir humedad retenida.",
                "Documentar la evolución de las hojas con fotografías diarias."
            ],
            'en': [
                "Immediately isolate the affected crop area to prevent foliar spread by wind.",
                "Consider preventive treatments with organic or regulated chemical fungicides.",
                "Avoid overhead irrigation directly onto foliage in the late afternoon to reduce retained moisture.",
                "Document the evolution of leaves with daily photographs."
            ],
            'pt': [
                "Isolar imediatamente a área de cultivo afetada para evitar a propagação foliar pelo vento.",
                "Considerar tratamentos preventivos com fungicidas orgânicos ou químicos regulamentados.",
                "Evitar a irrigação por aspersão direta na folhagem no final da tarde para reduzir a umidade retida.",
                "Documentar a evolução das folhas com fotografias diárias."
            ]
        }.get(lang_key, [])
    else:
        recs = {
            'es': [
                "Tomar una nueva imagen con mejor iluminación y enfoque central en la patología.",
                "Asegurar que la hoja no tenga reflejos de luz solar excesivos al momento de capturar.",
                "Realizar análisis de suelo para descartar deficiencias de nutrientes simulando necrosis foliar."
            ],
            'en': [
                "Take a new image with better lighting and a central focus on the pathology.",
                "Ensure the leaf does not have excessive sunlight reflections when capturing.",
                "Perform soil analysis to rule out nutrient deficiencies simulating leaf necrosis."
            ],
            'pt': [
                "Tirar uma nova imagem com melhor iluminação e foco central na patologia.",
                "Garantir que a folha não tenha reflexos excessivos de luz solar no momento da captura.",
                "Realizar análise de solo para descartar deficiências de nutrientes simulando necrose foliar."
            ]
        }.get(lang_key, [])
        
    for r in recs:
        doc.add_paragraph(f"- {r}")
    stats_head = {
        'es': "Validación Estadística Robusta (Pruebas del Ing. Santos)",
        'en': "Robust Statistical Validation (Eng. Santos Tests)",
        'pt': "Validação Estatística Robusta (Testes do Eng. Santos)"
    }.get(lang_key)
    
    doc.add_heading(stats_head, level=2)
    
    stats_body = {
        'es': [
            "Prueba de McNemar: p-valor = 0.0133 (Diferencia significativa en clasificación, se rechaza H0).",
            "Prueba de Mann-Whitney U (CV): MobileNetV2 vs EfficientNetB0 (p = 0.0089). Confirma la superioridad de EfficientNetB0.",
            "Prueba de Kolmogorov-Smirnov: Confirma que las curvas de confianza de inferencia difieren significativamente entre modelos.",
            "Prueba de Morgan-Pitman: p-valor = 0.3821 (Varianza del error equivalente entre ResNet50 y EfficientNetB0, validando parsimonia).",
            "Robustez (DAVT-Adv): Resiliencia de EfficientNetB0 ante ruido foliar y variaciones de luz (+20% de brillo, 5% de ruido de sal y pimienta)."
        ],
        'en': [
            "McNemar's Test: p-value = 0.0133 (Significant difference in classification, H0 is rejected).",
            "Mann-Whitney U Test (CV): MobileNetV2 vs EfficientNetB0 (p = 0.0089). Confirms EfficientNetB0 superiority.",
            "Kolmogorov-Smirnov Test: Confirms that prediction confidence curves differ significantly between architectures.",
            "Morgan-Pitman Test: p-value = 0.3821 (Equivalent error variance between ResNet50 and EfficientNetB0, validating parsimony).",
            "Robustness (DAVT-Adv): EfficientNetB0 resilience against leaf noise and light changes (+20% brightness, 5% salt & pepper noise)."
        ],
        'pt': [
            "Teste de McNemar: p-valor = 0.0133 (Diferença significativa na classificação, H0 é rejeitada).",
            "Teste Mann-Whitney U (CV): MobileNetV2 vs EfficientNetB0 (p = 0.0089). Confirma a superioridade do EfficientNetB0.",
            "Teste Kolmogorov-Smirnov: Confirma que as curvas de confiança de inferência diferem significativamente entre modelos.",
            "Teste de Morgan-Pitman: p-valor = 0.3821 (Variância do erro equivalente entre ResNet50 e EfficientNetB0, validando parcimônia).",
            "Robustez (DAVT-Adv): Resiliência do EfficientNetB0 sob ruído foliar e variações de luz (+20% de brilho, 5% de ruído de sal e pimenta)."
        ]
    }.get(lang_key, [])
    
    for s_line in stats_body:
        doc.add_paragraph(f"- {s_line}")
        
    doc.add_heading(disclaimer_title, level=2)
    doc.add_paragraph(disclaimer_text)
    
    doc.save(filepath)
    return filepath


def generate_image_xlsx_report(predictions, uploaded_filename, consensus_reached, consensus_diagnosis, filepath="reports/reporte_imagen.xlsx", lang="es"):
    import pandas as pd
    from datetime import datetime
    from src.translation import t_lang, translate_class_lang
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    lang_key = lang if lang in ['es', 'en', 'pt'] else 'es'
    
    # Textos localizados
    var_col = {
        'es': 'Variable',
        'en': 'Variable',
        'pt': 'Variável'
    }.get(lang_key)
    
    val_col = {
        'es': 'Valor',
        'en': 'Value',
        'pt': 'Valor'
    }.get(lang_key)

    variables = {
        'es': ['Archivo Analizado', 'Fecha de Análisis', 'Consenso Alcanzado', 'Diagnóstico Final', 'Estado General'],
        'en': ['Analyzed File', 'Analysis Date', 'Consensus Reached', 'Final Diagnosis', 'General Status'],
        'pt': ['Arquivo Analisado', 'Data da Análise', 'Consenso Atingido', 'Diagnóstico Final', 'Estado Geral']
    }.get(lang_key)

    yes_lbl = {
        'es': 'SÍ',
        'en': 'YES',
        'pt': 'SIM'
    }.get(lang_key)

    no_lbl = {
        'es': 'NO',
        'en': 'NO',
        'pt': 'NÃO'
    }.get(lang_key)

    no_cons_lbl = {
        'es': 'Sin Consenso',
        'en': 'No Consensus',
        'pt': 'Sem Consenso'
    }.get(lang_key)

    healthy_lbl = {
        'es': 'Saludable (Sano)',
        'en': 'Healthy',
        'pt': 'Saudável'
    }.get(lang_key)

    infected_lbl = {
        'es': 'Infectado (Enfermo)',
        'en': 'Infected (Ill)',
        'pt': 'Infectado (Doente)'
    }.get(lang_key)

    not_det_lbl = {
        'es': 'No Determinado',
        'en': 'Not Determined',
        'pt': 'Não Determinado'
    }.get(lang_key)

    # Diagnostico values
    final_diag = translate_class_lang(consensus_diagnosis, lang_key) if consensus_diagnosis else no_cons_lbl
    general_status = healthy_lbl if consensus_diagnosis == 'Sano' else infected_lbl if consensus_reached else not_det_lbl
    
    resumen_sheet = {
        'es': 'Diagnostico',
        'en': 'Diagnosis',
        'pt': 'Diagnóstico'
    }.get(lang_key)

    preds_sheet = {
        'es': 'Predicciones por Modelo',
        'en': 'Predictions by Model',
        'pt': 'Previsões por Modelo'
    }.get(lang_key)

    model_hdr = {
        'es': 'Modelo',
        'en': 'Model',
        'pt': 'Modelo'
    }.get(lang_key)

    pred_class_hdr = {
        'es': 'Clase Predicha',
        'en': 'Predicted Class',
        'pt': 'Classe Prevista'
    }.get(lang_key)

    confidence_hdr = {
        'es': 'Confianza',
        'en': 'Confidence',
        'pt': 'Confiança'
    }.get(lang_key)

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Pestaña 1: Resumen de Diagnóstico
        resumen_data = {
            var_col: variables,
            val_col: [
                uploaded_filename,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                yes_lbl if consensus_reached else no_lbl,
                final_diag,
                general_status
            ]
        }
        pd.DataFrame(resumen_data).to_excel(writer, sheet_name=resumen_sheet, index=False)
        
        # Pestaña 2: Predicciones Detalladas
        preds_data = []
        for model_name, pred in predictions.items():
            preds_data.append({
                model_hdr: model_name,
                pred_class_hdr: translate_class_lang(pred['class'], lang_key),
                confidence_hdr: pred['confidence'],
                f"Prob_{translate_class_lang('Mancha gris', lang_key).replace(' ', '_')}": pred['probabilities'][0],
                f"Prob_{translate_class_lang('Roya común', lang_key).replace(' ', '_')}": pred['probabilities'][1],
                f"Prob_{translate_class_lang('Tizón del norte', lang_key).replace(' ', '_')}": pred['probabilities'][2],
                f"Prob_{translate_class_lang('Sano', lang_key).replace(' ', '_')}": pred['probabilities'][3]
            })
        pd.DataFrame(preds_data).to_excel(writer, sheet_name=preds_sheet, index=False)
        
        # Pestaña 3: Recomendaciones de Control
        recs_sheet = {
            'es': 'Recomendaciones',
            'en': 'Recommendations',
            'pt': 'Recomendações'
        }.get(lang_key)

        recs_disease_info = {
            'es': {
                "Tizón del norte": {
                    "desc": "Enfermedad fúngica severa causada por Exserohilum turcicum que provoca lesiones alargadas elípticas (en forma de cigarro/puro) de color verde-grisáceo a marrón.",
                    "recs": [
                        "Uso de híbridos comerciales resistentes: Seleccionar variedades que incorporen resistencia cuantitativa y genes Ht específicos.",
                        "Rotación de cultivos: Implementar rotación de 1 a 2 años con especies no gramíneas (soya, leguminosas o girasol) para romper el ciclo biológico del patógeno.",
                        "Manejo adecuado de rastrojos: Realizar labranza profunda (arado) para enterrar los residuos del cultivo anterior infectado y acelerar su descomposición.",
                        "Tratamiento químico foliar oportuno: Aplicar fungicidas sistémicos (triazoles, estrobirulinas o carboxamidas) si la severidad en hojas inferiores supera el 15% antes de la floración.",
                        "Monitoreo fitosanitario continuo: Realizar inspecciones semanales en el envés de las hojas inferiores para detectar focos iniciales."
                    ]
                },
                "Roña común": {
                    "desc": "Enfermedad fúngica provocada por Puccinia sorghi que produce pequeñas pústulas circulares u ovaladas de color marrón-rojizo en ambas caras de la hoja, liberando esporas polvorientas que se dispersan por el viento.",
                    "recs": [
                        "Siembra de híbridos resistentes: Priorizar el uso de variedades que cuenten con resistencia Rp específica.",
                        "Ajuste en fechas de siembra: Programar siembras tempranas para evitar que las etapas críticas del cultivo coincidan con condiciones frescas y húmedas favorables para el patógeno.",
                        "Aplicación de fungicidas sistémicos: Utilizar mezclas comerciales de triazoles y estrobirulinas al observar las primeras pústulas en el tercio inferior/medio del cultivo.",
                        "Control de mojado foliar: Ajustar el riego por aspersión para reducir las horas de mojado en las hojas, lo que disminuye la germinación de esporas.",
                        "Manejo de nutrición mineral: Mantener una nutrición nitrogenada y potásica balanceada para fortalecer la resistencia física de la cutícula foliar."
                    ]
                },
                "Roya común": {
                    "desc": "Enfermedad fúngica provocada por Puccinia sorghi que produce pequeñas pústulas circulares u ovaladas de color marrón-rojizo en ambas caras de la hoja, liberando esporas polvorientas que se dispersan por el viento.",
                    "recs": [
                        "Siembra de híbridos resistentes: Priorizar el uso de variedades que cuenten con resistencia Rp específica.",
                        "Ajuste en fechas de siembra: Programar siembras tempranas para evitar que las etapas críticas del cultivo coincidan con condiciones frescas y húmedas favorables para el patógeno.",
                        "Aplicación de fungicidas sistémicos: Utilizar mezclas comerciales de triazoles y estrobirulinas al observar las primeras pústulas en el tercio inferior/medio del cultivo.",
                        "Control de mojado foliar: Ajustar el riego por aspersión para reducir las horas de mojado en las hojas, lo que disminuye la germinación de esporas.",
                        "Manejo de nutrición mineral: Mantener una nutrición nitrogenada y potásica balanceada para fortalecer la resistencia física de la cutícula foliar."
                    ]
                },
                "Mancha gris": {
                    "desc": "Enfermedad fúngica destructiva causada por Cercospora zeae-maydis que se manifiesta como lesiones rectangulares delimitadas por las nervaduras de la hoja foliar, tornándose grisáceas con el tiempo.",
                    "recs": [
                        "Rotación sistemática de cultivos: Alternar el campo con especies no gramíneas por un periodo mínimo de 1 a 2 años.",
                        "Manejo e incorporación de residuos: Enterrar rastrojos infectados para reducir sustancialmente el inóculo primario presente en el suelo.",
                        "Selección de semillas tolerantes: Sembrar híbridos con alta tolerancia genética comprobada en zonas con historial de la enfermedad.",
                        "Control químico foliar estratégico: Aplicar mezclas de fungicidas específicos (estrobirulinas y triazoles) si se observan lesiones tempranas bajo condiciones de alta humedad.",
                        "Optimización de densidad de siembra: Regular el número de plantas por hectárea para mejorar la aireación interna del dosel y reducir la humedad microclimática."
                    ]
                },
                "Sano": {
                    "desc": "La planta de maíz presenta hojas completamente saludables, sin indicios de infección fúngica activa ni deficiencias de nutrientes.",
                    "recs": [
                        "Monitoreo preventivo de rutina: Inspeccionar visualmente el campo una vez por semana en busca de focos infecciosos o anomalías.",
                        "Fertilización balanceada de precisión: Continuar el plan de nutrición basado en el análisis periódico de suelos.",
                        "Manejo integral de malezas y plagas: Mantener el cultivo libre de malezas hospedantes y plagas (como gusano cogollero) para evitar heridas de entrada.",
                        "Garantizar buen drenaje en el lote: Evitar encharcamientos prolongados que estimulen el desarrollo de patógenos del suelo.",
                        "Uso de agua de riego limpia: Evitar fuentes de agua estancada que puedan transportar esporas de hongos fitopatógenos."
                    ]
                }
            },
            'en': {
                "Tizón del norte": {
                    "desc": "Severe fungal disease caused by Exserohilum turcicum that induces elongated, cigar-shaped grayish-green to brown lesions, which can merge causing complete leaf necrosis.",
                    "recs": [
                        "Use resistant commercial hybrids: Select varieties incorporating quantitative resistance and specific Ht genes.",
                        "Crop rotation: Implement a 1-to-2-year rotation with non-grass species (soybean, legumes, or sunflower) to break the pathogen's life cycle.",
                        "Proper residue management: Perform deep tillage (plowing) to bury infected crop residues from the previous season and speed up decomposition.",
                        "Timely foliar chemical treatment: Apply systemic fungicides (triazoles, strobilurins, or carboxamides) if severity on lower leaves exceeds 15% before flowering.",
                        "Continuous phytosanitary monitoring: Perform weekly inspections of the underside of lower leaves to detect initial hotspots."
                    ]
                },
                "Roña común": {
                    "desc": "Fungal disease caused by Puccinia sorghi that produces small circular or oval reddish-brown pustules on both leaf surfaces, releasing powdery spores dispersed by wind.",
                    "recs": [
                        "Sowing of resistant hybrids: Prioritize the use of varieties with specific Rp resistance genes.",
                        "Adjustment of planting dates: Schedule early planting to prevent critical crop stages from coinciding with cool, humid conditions favorable to the pathogen.",
                        "Systemic fungicide application: Use commercial mixtures of triazoles and strobilurins when first pustules are observed in the lower/middle third of the crop.",
                        "Leaf wetness control: Adjust sprinkler irrigation to reduce leaf wetness duration, decreasing spore germination.",
                        "Mineral nutrition management: Maintain balanced nitrogen and potassium nutrition to strengthen the physical resistance of the leaf cuticle."
                    ]
                },
                "Roya común": {
                    "desc": "Fungal disease caused by Puccinia sorghi that produces small circular or oval reddish-brown pustules on both leaf surfaces, releasing powdery spores dispersed by wind.",
                    "recs": [
                        "Sowing of resistant hybrids: Prioritize the use of varieties with specific Rp resistance genes.",
                        "Adjustment of planting dates: Schedule early planting to prevent critical crop stages from coinciding with cool, humid conditions favorable to the pathogen.",
                        "Systemic fungicide application: Use commercial mixtures of triazoles and strobilurins when first pustules are observed in the lower/middle third of the crop.",
                        "Leaf wetness control: Adjust sprinkler irrigation to reduce leaf wetness duration, decreasing spore germination.",
                        "Mineral nutrition management: Maintain balanced nitrogen and potassium nutrition to strengthen the physical resistance of the leaf cuticle."
                    ]
                },
                "Mancha gris": {
                    "desc": "Destructive fungal disease caused by Cercospora zeae-maydis that appears as rectangular lesions restricted by leaf veins, turning grayish over time.",
                    "recs": [
                        "Systematic crop rotation: Alternate the field with non-grass species for a minimum of 1 to 2 years.",
                        "Residue management and incorporation: Bury infected stubble to substantially reduce primary inoculum present in the soil.",
                        "Tolerant seed selection: Sow hybrids with proven high genetic tolerance in areas with a history of the disease.",
                        "Strategic foliar chemical control: Apply specific fungicide mixtures (strobilurins and triazoles) if early lesions are seen under high humidity.",
                        "Sowing density optimization: Regulate the number of plants per hectare to improve air circulation within the canopy and reduce microclimatic humidity."
                    ]
                },
                "Sano": {
                    "desc": "The maize plant shows completely healthy leaves, with no signs of active fungal infection or nutritional deficiencies.",
                    "recs": [
                        "Routine preventive monitoring: Visually inspect the field once a week for initial infection focus or anomalies.",
                        "Precision balanced fertilization: Continue the nutrition plan based on periodic soil analysis.",
                        "Integrated weed and pest management: Keep the crop free of host weeds and pests (such as fall armyworm) to avoid entry wounds.",
                        "Ensure good field drainage: Avoid prolonged waterlogging that stimulates the development of soil-borne pathogens.",
                        "Use clean irrigation water: Avoid stagnant water sources that may transport phytopathogenic fungal spores."
                    ]
                }
            },
            'pt': {
                "Tizón del norte": {
                    "desc": "Doença fúngica grave causada por Exserohilum turcicum que provoca lesões elípticas alongadas (em forma de charuto) de cor verde-acinzentada a marrom.",
                    "recs": [
                        "Uso de híbridos comerciais resistentes: Selecionar variedades que incorporem resistência quantitativa e genes Ht específicos.",
                        "Rotação de culturas: Implementar rotação de 1 a 2 anos com espécies não gramíneas (soja, leguminosas ou girassol) para quebrar o ciclo de vida do patógeno.",
                        "Manejo adequado de resíduos: Realizar aração profunda para enterrar os resíduos da safra anterior infectada e acelerar a decomposição.",
                        "Tratamento químico foliar oportuno: Aplicar fungicidas sistêmicos (triazóis, estrobirulinas ou carboxamidas) se a severidade nas folhas inferiores exceder 15% antes do florescimento.",
                        "Monitoramento fitossanitário contínuo: Realizar inspeções semanais na face inferior das folhas inferiores para detectar focos iniciais."
                    ]
                },
                "Roña común": {
                    "desc": "Doença fúngica provocada por Puccinia sorghi que produz pequenas pústulas circulares ou ovais de cor marrom-avermelhada em ambas as superfícies da folha, liberando esporos dispersos pelo vento.",
                    "recs": [
                        "Plantio de híbridos resistentes: Priorizar o uso de variedades que possuam resistência Rp específica.",
                        "Ajuste nas datas de plantio: Programar plantios precoces para evitar que estágios críticos coincidam com condições frescas e úmidas favoráveis ao patógeno.",
                        "Aplicação de fungicidas sistêmicos: Utilizar misturas comerciais de trizóis e estrobirulinas quando as primeiras pústulas forem observadas no terço inferior/médio da cultura.",
                        "Controle de molhamento foliar: Ajustar a irrigação por aspersão para reduzir as horas de molhamento foliar, diminuindo a germinação dos esporas.",
                        "Manejo de nutrição mineral: Manter uma nutrição nitrogenada e potássica equilibrada para fortalecer a resistência física da cutícula foliar."
                    ]
                },
                "Roya común": {
                    "desc": "Doença fúngica provocada por Puccinia sorghi que produz pequenas pústulas circulares ou ovais de cor marrom-avermelhada em ambas as superfícies da folha, liberando esporos dispersos pelo vento.",
                    "recs": [
                        "Plantio de híbridos resistentes: Priorizar o uso de variedades que possuam resistência Rp específica.",
                        "Ajuste nas datas de plantio: Programar plantios precoces para evitar que estágios críticos coincidam com condições frescas e úmidas favoráveis ao patógeno.",
                        "Aplicação de fungicidas sistêmicos: Utilizar misturas comerciais de trizóis e estrobirulinas quando as primeiras pústulas forem observadas no terço inferior/médio da cultura.",
                        "Controle de molhamento foliar: Ajustar a irrigação por aspersão para reduzir as horas de molhamento foliar, diminuindo a germinação dos esporas.",
                        "Manejo de nutrição mineral: Manter uma nutrição nitrogenada e potássica equilibrada para fortalecer a resistência física da cutícula foliar."
                    ]
                },
                "Mancha gris": {
                    "desc": "Doença fúngica destrutiva causada por Cercospora zeae-maydis que se manifesta como lesões retangulares delimitadas pelas nervuras foliares, tornando-se acinzentadas com o tempo.",
                    "recs": [
                        "Rotação sistemática de culturas: Alternar o campo com espécies não gramíneas por um período mínimo de 1 a 2 anos.",
                        "Manejo e incorporação de resíduos: Enterrar a palhada infectada para reduzir substancialmente o inóculo primário presente no solo.",
                        "Seleção de sementes tolerantes: Semear híbridos com alta tolerância genética comprovada em áreas com histórico da doença.",
                        "Controle químico foliar estratégico: Aplicar misturas de fungicidas específicos (estrobirulinas e trizóis) se forem observadas lesões iniciais sob alta umidade.",
                        "Otimização da densidade de plantio: Regular o número de plantas por hectare para melhorar a ventilação e reduzir a umidade microclimática."
                    ]
                },
                "Sano": {
                    "desc": "A planta de milho apresenta folhas completamente saudáveis, sem sinais de infecção fúngica ativa ou deficiências nutricionais.",
                    "recs": [
                        "Monitoramento preventivo de rotina: Inspecionar visualmente o campo uma vez por semana em busca de focos infecciosos ou anomalias.",
                        "Adubação equilibrada de precisão: Continuar o plano de nutrição baseado na análise periódica de solos.",
                        "Manejo integrado de plantas daninhas e pragas: Manter a cultura livre de plantas daninhas hospedeiras e pragas (como lagarta-do-cartucho) para evitar ferimentos de entrada.",
                        "Garantir boa drenagem no lote: Evitar encharcamentos prolongados que estimulem o desenvolvimento de patógenos do solo.",
                        "Uso de água de irrigação limpa: Evitar fontes de água estancada que possam transportar esporos de fungos fitopatogênicos."
                    ]
                }
            }
        }.get(lang_key, {})

        lookup_key = consensus_diagnosis if consensus_diagnosis else "Sano"
        if lookup_key not in recs_disease_info and lookup_key == "Roya común" and "Roña común" in recs_disease_info:
            lookup_key = "Roña común"
        elif lookup_key not in recs_disease_info and lookup_key == "Roña común" and "Roya común" in recs_disease_info:
            lookup_key = "Roya común"

        if lookup_key in recs_disease_info:
            diag_info = recs_disease_info[lookup_key]
            diag_name = translate_class_lang(consensus_diagnosis, lang_key) if consensus_diagnosis else healthy_lbl
            desc_val = diag_info['desc']
            
            recs_headers = {
                'es': ['Diagnóstico', 'Descripción Fitosanitaria', 'Pautas y Recomendaciones de Control'],
                'en': ['Diagnosis', 'Phytosanitary Description', 'Control Guidelines and Recommendations'],
                'pt': ['Diagnóstico', 'Descrição Fitossanitária', 'Diretrizes e Recomendações de Controle']
            }.get(lang_key)

            rows_recs = []
            for r in diag_info['recs']:
                rows_recs.append({
                    recs_headers[0]: diag_name,
                    recs_headers[1]: desc_val,
                    recs_headers[2]: r
                })
            pd.DataFrame(rows_recs).to_excel(writer, sheet_name=recs_sheet, index=False)
            
        # Aplicar formato y autoajustes
        workbook = writer.book
        for sheet_name in workbook.sheetnames:
            format_excel_sheet(workbook[sheet_name])
            
    return filepath

# ---------------------------------------------------------
# 5. GENERACIÓN DE PDF PARA DIAGNÓSTICO POR IMÁGENES
# ---------------------------------------------------------
def generate_image_pdf_report(image, predictions, uploaded_filename, consensus_reached, consensus_diagnosis, filepath="reports/reporte_imagen.pdf", lang="es", img_size=128):
    """
    Crea un reporte PDF para el diagnóstico por imágenes de la hoja de maíz.
    """
    from fpdf import FPDF
    from PIL import Image
    import matplotlib.pyplot as plt
    from src.translation import translate_class_lang
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    peru_time = datetime.now()
    
    pdf_text_dict = {
        'es': {
            'title': 'DIAGNÓSTICO FITOSANITARIO - MAÍZ',
            'subtitle': 'Sistema de Detección Automática de Enfermedades',
            'header': 'RELATORIO DE DIAGNÓSTICO FITOSANITARIO (IMÁGENES)',
            'page': 'Página',
            'generated': 'Generado el',
            'peru_time': '(Hora Local)',
            'info_title': 'INFORMACIÓN DEL ANÁLISIS',
            'file': 'Archivo',
            'datetime': 'Fecha y hora',
            'models_used': 'Modelos utilizados',
            'resolution': 'Resolución de procesamiento',
            'diag_title': 'DIAGNÓSTICO PRINCIPAL',
            'img_title': 'IMAGEN ANALIZADA',
            'img_details': 'Detalles de la imagen:',
            'orig_size': 'Tamaño original',
            'format': 'Formato',
            'channels': 'Canales de color',
            'det_title': 'RESULTADOS DETALLADOS',
            'chart_title': 'Predicciones del Modelo',
            'prob_ylabel': 'Probabilidad',
            'model_card_title': 'Resultado del Modelo',
            'pred_label': 'Predicción',
            'conf_label': 'Confianza',
            'state_label': 'Estado',
            'state_ok': '[OK] Saludable',
            'state_warn': '[!] Enfermedad detectada',
            'prob_per_class': 'Probabilidades por clase:',
            'comp_title': 'ANÁLISIS COMPARATIVO',
            'comp_summary': 'Resumen de las predicciones:',
            'comp_table_header': 'Modelo                Predicción           Confianza    Estado',
            'state_healthy_lbl': '[OK] Saludable',
            'state_diseased_lbl': '[!] Enfermo',
            'consensus_reached_title': '[OK] Consenso Alcanzado',
            'consensus_reached_body': 'Los tres modelos coinciden en el diagnóstico: {diagnosis}\nEsto indica alta confiabilidad en el resultado.\nNivel de acuerdo: 100% (3/3 modelos)',
            'no_consensus_title': '[!] Sin Consenso',
            'no_consensus_body_start': 'Los modelos presentan diferentes diagnósticos:\n',
            'no_consensus_body_end': 'Se recomienda análisis adicional para confirmar.',
            'rec_title': 'RECOMENDACIONES',
            'rec_healthy': [
                '- Continuar con las prácticas de manejo actuales',
                '- Realizar monitoreos preventivos regulares cada 7-10 días',
                '- Mantener condiciones óptimas de cultivo (riego, fertilización)',
                '- Implementar rotación de cultivos para prevenir enfermedades',
                '- Vigilar plantas vecinas por posibles síntomas'
            ],
            'rec_diseased': [
                '- Consultar inmediatamente con un especialista en fitopatología',
                '- Aislar las plantas afectadas si es posible',
                '- Implementar medidas de control específicas para la enfermedad',
                '- Monitorear la extensión de la enfermedad en el cultivo',
                '- Considerar tratamientos preventivos en plantas cercanas',
                '- Documentar la evolución con fotografías regulares',
                '- Revisar condiciones ambientales que favorecen la enfermedad'
            ],
            'rec_no_consensus': [
                '- Tomar una nueva imagen con mejor calidad e iluminación',
                '- Asegurar que la hoja esté bien centrada y enfocada',
                '- Consultar con un especialista para confirmación visual',
                '- Realizar análisis de laboratorio si persisten síntomas',
                '- Considerar múltiples muestras de diferentes partes de la planta'
            ],
            'disease_info_title': 'INFORMACIÓN ESPECÍFICA',
            'disease_header': 'Enfermedad',
            'symptoms_header': 'Síntomas característicos:',
            'conditions_header': 'Condiciones favorables:',
            'treatments_header': 'Estrategias de manejo:',
            'tech_title': 'INFORMACIÓN TÉCNICA',
            'tech_spec_header': 'Especificaciones del sistema:',
            'tech_spec_list': [
                '- Modelos basados en transfer learning con redes neuronales convolucionales',
                '- Dataset de entrenamiento: PlantVillage Corn Leaf Disease',
                '- Arquitecturas: MobileNetV2, ResNet50, EfficientNetB0',
                '- Precisión promedio en validación: >95%',
                '- Resolución de procesamiento: {size}x{size} pixeles',
                '- Preprocesamiento específico por modelo aplicado',
                '- Análisis basado en características visuales de la hoja'
            ],
            'disclaimer_title': '[!] IMPORTANTE - LIMITACIONES Y DISCLAIMER',
            'disclaimer_text': '- Este análisis automatizado debe ser validado por un profesional\n- La precisión del diagnóstico depende de la calidad de la imagen\n- Se recomienda tomar múltiples muestras para mayor certeza\n- Este sistema es una herramienta de apoyo, no un sustituto del diagnóstico profesional\n- En caso de dudas, consulte con un fitopatólogo certificado\n- Los resultados pueden variar según condiciones de iluminación y enfoque',
            'system_info_title': 'Información del sistema:',
            'system_info_name': 'Sistema de Detección Automática de Enfermedades en Maíz',
            'system_info_version': 'Versión: 2.0 | Fecha de generación: {date}',
            'system_info_tech': 'Desarrollado con tecnología de Deep Learning',
            'diseases': {
                'Tizón del norte': {
                    'descripcion': 'Enfermedad fúngica causada por Exserohilum turcicum que afecta principalmente las hojas del maíz.',
                    'sintomas': [
                        '- Lesiones alargadas en forma de cigarro',
                        '- Color marrón grisáceo con bordes definidos',
                        '- Pueden alcanzar varios centímetros de longitud',
                        '- Amarillamiento prematuro de hojas',
                        '- En casos severos, marchitez de la planta'
                    ],
                    'condiciones': 'Favorecido por alta humedad (>90%) y temperaturas de 18-27C',
                    'tratamiento': [
                        '- Aplicación de fungicidas específicos (azoles, estrobilurinas)',
                        '- Uso de variedades resistentes',
                        '- Rotación de cultivos con especies no susceptibles',
                        '- Manejo de residuos de cosecha',
                        '- Espaciamiento adecuado para mejorar ventilación'
                    ]
                },
                'Roña común': {
                    'descripcion': 'Enfermedad fúngica causada por Puccinia sorghi que produce pústulas características en las hojas.',
                    'sintomas': [
                        '- Pústulas pequeñas y circulares de color marrón-rojizo',
                        '- Aparecen en ambas caras de la hoja',
                        '- Pueden coalescer formando áreas grandes',
                        '- Amarillamiento prematuro del follaje',
                        '- Reducción en el vigor de la planta'
                    ],
                    'condiciones': 'Temperaturas moderadas (16-25C) y presencia de rocío matutino',
                    'tratamiento': [
                        '- Fungicidas preventivos antes de la aparición de síntomas',
                        '- Variedades con genes de resistencia',
                        '- Eliminación de hospederos alternativos',
                        '- Monitoreo temprano y control oportuno',
                        '- Aplicación foliar de productos cúpricos'
                    ]
                },
                'Roya común': {
                    'descripcion': 'Enfermedad fúngica causada por Puccinia sorghi que produce pústulas características en las hojas.',
                    'sintomas': [
                        '- Pústulas pequeñas y circulares de color marrón-rojizo',
                        '- Aparecen en ambas caras de la hoja',
                        '- Pueden coalescer formando áreas grandes',
                        '- Amarillamiento prematuro del follaje',
                        '- Reducción en el vigor de la planta'
                    ],
                    'condiciones': 'Temperaturas moderadas (16-25C) y presencia de rocío matutino',
                    'tratamiento': [
                        '- Fungicidas preventivos antes de la aparición de síntomas',
                        '- Variedades con genes de resistencia',
                        '- Eliminación de hospederos alternativos',
                        '- Monitoreo temprano y control oportuno',
                        '- Aplicación foliar de productos cúpricos'
                    ]
                },
                'Mancha gris': {
                    'descripcion': 'Enfermedad fúngica causada por Cercospora zeae-maydis que produce manchas características en las hojas.',
                    'sintomas': [
                        '- Manchas rectangulares de color gris a marrón',
                        '- Delimitadas por las venas de las hojas',
                        '- Pueden desarrollar un halo amarillento',
                        '- Coalescencia causa muerte de tejido foliar',
                        '- Afecta principalmente hojas inferiores'
                    ],
                    'condiciones': 'Alta humedad relativa y temperaturas cálidas (25-30C)',
                    'tratamiento': [
                        '- Rotación con cultivos no gramíneas',
                        '- Aplicación de fungicidas sistémicos',
                        '- Manejo de densidad de siembra',
                        '- Eliminación de residuos infectados',
                        '- Mejoramiento de drenaje del suelo'
                    ]
                }
            }
        },
        'en': {
            'title': 'MAIZE PHYTOSANITARY DIAGNOSIS',
            'subtitle': 'Automatic Disease Detection System',
            'header': 'PHYTOSANITARY DIAGNOSIS REPORT (IMAGES)',
            'page': 'Page',
            'generated': 'Generated on',
            'peru_time': '(Local Time)',
            'info_title': 'ANALYSIS INFORMATION',
            'file': 'File',
            'datetime': 'Date and time',
            'models_used': 'Models used',
            'resolution': 'Processing resolution',
            'diag_title': 'PRIMARY DIAGNOSIS',
            'img_title': 'ANALYZED IMAGE',
            'img_details': 'Image details:',
            'orig_size': 'Original size',
            'format': 'Format',
            'channels': 'Color channels',
            'det_title': 'DETAILED RESULTS',
            'chart_title': 'Model Predictions',
            'prob_ylabel': 'Probability',
            'model_card_title': 'Model Result',
            'pred_label': 'Prediction',
            'conf_label': 'Confidence',
            'state_label': 'State',
            'state_ok': '[OK] Healthy',
            'state_warn': '[!] Disease detected',
            'prob_per_class': 'Probabilities per class:',
            'comp_title': 'COMPARATIVE ANALYSIS',
            'comp_summary': 'Predictions summary:',
            'comp_table_header': 'Model                Prediction           Confidence    State',
            'state_healthy_lbl': '[OK] Healthy',
            'state_diseased_lbl': '[!] Diseased',
            'consensus_reached_title': '[OK] Consensus Reached',
            'consensus_reached_body': 'All three models agree on the diagnosis: {diagnosis}\nThis indicates high reliability in the result.\nAgreement level: 100% (3/3 models)',
            'no_consensus_title': '[!] No Consensus',
            'no_consensus_body_start': 'Models present different diagnoses:\n',
            'no_consensus_body_end': 'Additional analysis is recommended for confirmation.',
            'rec_title': 'RECOMMENDATIONS',
            'rec_healthy': [
                '- Continue with current management practices',
                '- Perform regular preventive monitoring every 7-10 days',
                '- Maintain optimal crop conditions (irrigation, fertilization)',
                '- Implement crop rotation to prevent diseases',
                '- Monitor surrounding plants for potential symptoms'
            ],
            'rec_diseased': [
                '- Consult immediately with a phytopathology specialist',
                '- Isolate affected plants if possible',
                '- Implement specific disease control measures',
                '- Monitor the extension of the disease in the crop',
                '- Consider preventive treatments in nearby plants',
                '- Document the evolution with regular photographs',
                '- Review environmental conditions that favor the disease'
            ],
            'rec_no_consensus': [
                '- Take a new image with better quality and lighting',
                '- Ensure the leaf is well centered and focused',
                '- Consult with a specialist for visual confirmation',
                '- Perform laboratory analysis if symptoms persist',
                '- Consider multiple samples from different parts of the plant'
            ],
            'disease_info_title': 'SPECIFIC INFORMATION',
            'disease_header': 'Disease',
            'symptoms_header': 'Characteristic symptoms:',
            'conditions_header': 'Favorable conditions:',
            'treatments_header': 'Management strategies:',
            'tech_title': 'TECHNICAL INFORMATION',
            'tech_spec_header': 'System specifications:',
            'tech_spec_list': [
                '- Models based on transfer learning with convolutional neural networks',
                '- Training dataset: PlantVillage Corn Leaf Disease',
                '- Architectures: MobileNetV2, ResNet50, EfficientNetB0',
                '- Average validation accuracy: >95%',
                '- Processing resolution: {size}x{size} pixels',
                '- Specific preprocessing applied per model',
                '- Analysis based on foliar visual features'
            ],
            'disclaimer_title': '[!] IMPORTANT - LIMITATIONS AND DISCLAIMER',
            'disclaimer_text': '- This automated analysis should be validated by a professional\n- The diagnostic accuracy depends on the quality of the image\n- It is recommended to take multiple samples for higher certainty\n- This system is a support tool, not a substitute for professional diagnosis\n- In case of doubt, consult a certified phytopathologist\n- Results may vary depending on lighting and focus conditions',
            'system_info_title': 'System information:',
            'system_info_name': 'Automatic Disease Detection System in Maize',
            'system_info_version': 'Version: 2.0 | Generation date: {date}',
            'system_info_tech': 'Developed with Deep Learning technology',
            'diseases': {
                'Tizón del norte': {
                    'descripcion': 'Fungal disease caused by Exserohilum turcicum that mainly affects maize leaves.',
                    'sintomas': [
                        '- Elongated cigar-shaped lesions',
                        '- Grayish-brown color with defined borders',
                        '- Can reach several centimeters in length',
                        '- Premature yellowing of leaves',
                        '- In severe cases, wilting of the plant'
                    ],
                    'condiciones': 'Favored by high humidity (>90%) and temperatures of 18-27C',
                    'tratamiento': [
                        '- Application of specific fungicides (azoles, strobilurins)',
                        '- Use of resistant varieties',
                        '- Crop rotation with non-susceptible species',
                        '- Crop residue management',
                        '- Proper spacing to improve ventilation'
                    ]
                },
                'Roña común': {
                    'descripcion': 'Fungal disease caused by Puccinia sorghi that produces characteristic pustules on leaves.',
                    'sintomas': [
                        '- Small and circular reddish-brown pustules',
                        '- Appear on both sides of the leaf',
                        '- Can coalesce forming large areas',
                        '- Premature yellowing of foliage',
                        '- Reduction in plant vigor'
                    ],
                    'condiciones': 'Moderate temperatures (16-25C) and presence of morning dew',
                    'tratamiento': [
                        '- Preventive fungicides before the appearance of symptoms',
                        '- Varieties with resistance genes',
                        '- Elimination of alternative hosts',
                        '- Early monitoring and timely control',
                        '- Foliar application of copper products'
                    ]
                },
                'Roya común': {
                    'descripcion': 'Fungal disease caused by Puccinia sorghi that produces characteristic pustules on leaves.',
                    'sintomas': [
                        '- Small and circular reddish-brown pustules',
                        '- Appear on both sides of the leaf',
                        '- Can coalesce forming large areas',
                        '- Premature yellowing of foliage',
                        '- Reduction in plant vigor'
                    ],
                    'condiciones': 'Moderate temperatures (16-25C) and presence of morning dew',
                    'tratamiento': [
                        '- Preventive fungicides before the appearance of symptoms',
                        '- Varieties with resistance genes',
                        '- Elimination of alternative hosts',
                        '- Early monitoring and timely control',
                        '- Foliar application of copper products'
                    ]
                },
                'Mancha gris': {
                    'descripcion': 'Fungal disease caused by Cercospora zeae-maydis that produces characteristic spots on leaves.',
                    'sintomas': [
                        '- Rectangular gray to brown spots',
                        '- Delimited by leaf veins',
                        '- Can develop a yellowish halo',
                        '- Coalescence causes death of foliar tissue',
                        '- Mainly affects lower leaves'
                    ],
                    'condiciones': 'High relative humidity and warm temperatures (25-30C)',
                    'tratamiento': [
                        '- Rotation with non-grass crops',
                        '- Application of systemic fungicides',
                        '- Planting density management',
                        '- Elimination of infected residues',
                        '- Soil drainage improvement'
                    ]
                }
            }
        },
        'pt': {
            'title': 'DIAGNÓSTICO FITOSSANITÁRIO - MILHO',
            'subtitle': 'Sistema de Detecção Automática de Doenças',
            'header': 'RELATÓRIO DE DIAGNÓSTICO FITOSSANITÁRIO (IMAGENS)',
            'page': 'Página',
            'generated': 'Gerado em',
            'peru_time': '(Hora Local)',
            'info_title': 'INFORMAÇÃO DO ANÁLISE',
            'file': 'Arquivo',
            'datetime': 'Data e hora',
            'models_used': 'Modelos utilizados',
            'resolution': 'Resolução de processamento',
            'diag_title': 'DIAGNÓSTICO PRINCIPAL',
            'img_title': 'IMAGEM ANALISADA',
            'img_details': 'Detalhes da imagem:',
            'orig_size': 'Tamanho original',
            'format': 'Formato',
            'channels': 'Canais de cor',
            'det_title': 'RESULTADOS DETALHADOS',
            'chart_title': 'Predições do Modelo',
            'prob_ylabel': 'Probabilidade',
            'model_card_title': 'Resultado do Modelo',
            'pred_label': 'Predição',
            'conf_label': 'Confiança',
            'state_label': 'Estado',
            'state_ok': '[OK] Saudável',
            'state_warn': '[!] Doença detectada',
            'prob_per_class': 'Probabilidades por classe:',
            'comp_title': 'ANÁLISE COMPARATIVA',
            'comp_summary': 'Resumo das predições:',
            'comp_table_header': 'Modelo                Predição           Confiança    Estado',
            'state_healthy_lbl': '[OK] Saudável',
            'state_diseased_lbl': '[!] Doente',
            'consensus_reached_title': '[OK] Consenso Alcançado',
            'consensus_reached_body': 'Os três modelos coincidem no diagnóstico: {diagnosis}\nIsso indica alta confiabilidade no resultado.\nNível de acordo: 100% (3/3 modelos)',
            'no_consensus_title': '[!] Sem Consenso',
            'no_consensus_body_start': 'Os modelos apresentam diferentes diagnósticos:\n',
            'no_consensus_body_end': 'Se recomienda análise adicional para confirmar.',
            'rec_title': 'RECOMENDAÇÕES',
            'rec_healthy': [
                '- Continuar com as práticas de manejo atuais',
                '- Realizar monitoramentos preventivos regulares cada 7-10 dias',
                '- Manter condições ótimas de cultivo (irrigação, fertilização)',
                '- Implementar rotação de cultivos para prevenir doenças',
                '- Vigilar plantas vizinhas por possíveis sintomas'
            ],
            'rec_diseased': [
                '- Consultar imediatamente com um especialista em fitopatologia',
                '- Isolar as plantas afetadas se possível',
                '- Implementar medidas de controle específicas para a doença',
                '- Monitorar a extensão da doença no cultivo',
                '- Considerar tratamentos preventivos em plantas próximas',
                '- Documentar a evolução com fotografias regulares',
                '- Revisar condições ambientais que favorecem a doença'
            ],
            'rec_no_consensus': [
                '- Tomar uma nova imagem com melhor qualidade e iluminação',
                '- Garantir que a folha esteja bem centrada e focada',
                '- Consultar com um especialista para confirmação visual',
                '- Realizar análise de laboratório se persistirem sintomas',
                '- Considerar múltiplas amostras de diferentes partes da planta'
            ],
            'disease_info_title': 'INFORMAÇÃO ESPECÍFICA',
            'disease_header': 'Doença',
            'symptoms_header': 'Sintomas característicos:',
            'conditions_header': 'Condições favoráveis:',
            'treatments_header': 'Estratégias de manejo:',
            'tech_title': 'INFORMAÇÃO TÉCNICA',
            'tech_spec_header': 'Especificações do sistema:',
            'tech_spec_list': [
                '- Modelos baseados em transfer learning com redes neurais convolucionais',
                '- Dataset de treinamento: PlantVillage Corn Leaf Disease',
                '- Arquiteturas: MobileNetV2, ResNet50, EfficientNetB0',
                '- Precisão média em validação: >95%',
                '- Resolução de processamento: {size}x{size} pixels',
                '- Pré-processamento específico por modelo aplicado',
                '- Análise baseada em características visuais da folha'
            ],
            'disclaimer_title': '[!] IMPORTANTE - LIMITAÇÕES E DISCLAIMER',
            'disclaimer_text': '- Este análise automatizado deve ser validado por um profissional\n- A precisão do diagnóstico depende da qualidade da imagem\n- Se recomienda tomar múltiplas amostras para maior certeza\n- Este sistema é uma ferramenta de apoio, não um substituto do diagnóstico profissional\n- Em caso de dúvidas, consulte com um fitopatólogo certificado\n- Os resultados podem variar conforme condições de iluminação e enfoque',
            'system_info_title': 'Informação do sistema:',
            'system_info_name': 'Sistema de Detecção Automática de Doenças em Milho',
            'system_info_version': 'Versão: 2.0 | Data de geração: {date}',
            'system_info_tech': 'Desenvolvido com tecnologia de Deep Learning',
            'diseases': {
                'Tizón del norte': {
                    'descripcion': 'Doença fúngica grave causada por Exserohilum turcicum que provoca lesões elípticas alongadas (em forma de charuto) de cor verde-acinzentada a marrom.',
                    'sintomas': [
                        '- Lesões elípticas alongadas em forma de charuto',
                        '- Cor verde-acinzentada a marrom com bordas definidas',
                        '- Podem coalescer formando áreas grandes',
                        '- Amarelecimento prematuro da folhagem',
                        '- Redução no vigor da planta'
                    ],
                    'condiciones': 'Favorecido por alta umidade (>90%) e temperaturas de 18-27C',
                    'tratamiento': [
                        '- Uso de híbridos comerciais resistentes',
                        '- Rotação de cultivos com espécies não gramíneas',
                        '- Manejo adequado de resíduos',
                        '- Tratamento químico foliar oportuno',
                        '- Monitoramento fitosanitário contínuo'
                    ]
                },
                'Roña común': {
                    'descripcion': 'Doença fúngica provocada por Puccinia sorghi que produce pústulas pequenas circulares ou ovais de cor marrom-avermelhada em ambas as superfícies da folha.',
                    'sintomas': [
                        '- Pústulas pequenas circulares ou ovais de cor marrom-avermelhada',
                        '- Aparecem em ambas as superfícies da folha',
                        '- Podem coalescer formando áreas grandes',
                        '- Amarelecimento prematuro da folhagem',
                        '- Redução no vigor da planta'
                    ],
                    'condiciones': 'Temperaturas moderadas (16-25C) e presença de rocío matutino',
                    'tratamiento': [
                        '- Sementes de híbridos resistentes',
                        '- Ajuste em datas de plantio',
                        '- Aplicação de fungicidas sistêmicos',
                        '- Controle de molhamento foliar',
                        '- Manejo de nutrição mineral'
                    ]
                },
                'Roya común': {
                    'descripcion': 'Doença fúngica provocada por Puccinia sorghi que produce pústulas pequenas circulares ou ovais de cor marrom-avermelhada em ambas as superfícies da folha.',
                    'sintomas': [
                        '- Pústulas pequenas circulares ou ovais de cor marrom-avermelhada',
                        '- Aparecem em ambas as superfícies da folha',
                        '- Podem coalescer formando áreas grandes',
                        '- Amarelecimento prematuro da folhagem',
                        '- Redução no vigor da planta'
                    ],
                    'condiciones': 'Temperaturas moderadas (16-25C) e presença de rocío matutino',
                    'tratamiento': [
                        '- Sementes de híbridos resistentes',
                        '- Ajuste em datas de plantio',
                        '- Aplicação de fungicidas sistêmicos',
                        '- Controle de molhamento foliar',
                        '- Manejo de nutrição mineral'
                    ]
                },
                'Mancha gris': {
                    'descripcion': 'Doença fúngica destrutiva causada por Cercospora zeae-maydis que se manifesta como lesões retangulares delimitadas pelas nervuras foliares.',
                    'sintomas': [
                        '- Lesões retangulares delimitadas pelas nervuras foliares',
                        '- Cor acinzentada a marrom com bordas definidas',
                        '- Podem desenvolver um halo amarelo',
                        '- Coalescência causa morte de tecido foliar',
                        '- Afecta principalmente folhas inferiores'
                    ],
                    'condiciones': 'Alta umidade relativa e temperaturas cálidas (25-30C)',
                    'tratamiento': [
                        '- Rotação sistemática de cultivos',
                        '- Manejo e incorporação de resíduos',
                        '- Seleção de sementes tolerantes',
                        '- Controle químico foliar estratégico',
                        '- Otimização da densidade de plantio'
                    ]
                },
                'Sano': {
                    'descripcion': 'A planta de milho apresenta folhas completamente saudáveis, sem sinais de infecção fúngica ativa ou deficiências nutricionais.',
                    'sintomas': [
                        '- Folhas completamente verdes e saudáveis',
                        '- Ausência de lesões ou manchas',
                        '- Vigor normal da planta'
                    ],
                    'condiciones': 'Condições ideais de cultivo',
                    'tratamiento': [
                        '- Continuar com as práticas de manejo atuais',
                        '- Monitoreo preventivo de rotina',
                        '- Fertilização balanceada de precisão'
                    ]
                }
            }
        }
    }
    
    lang_key = lang if lang in ["es", "en", "pt"] else "es"
    tx = pdf_text_dict[lang_key]
    
    def clean_text_for_pdf(text):
        # Eliminar caracteres problemáticos para PDF
        import re
        return re.sub(r'[^\x20-\x7EáéíóúñÁÉÍÓÚÑ]', '', text)
    
    consensus_diagnosis_cleaned = translate_class_lang(consensus_diagnosis, lang_key) if consensus_diagnosis else "Sin consenso"
    
    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=15)

        def header(self):
            if self.page_no() == 1:
                # Portada/Primera pagina header grande
                self.set_font('Arial', 'B', 18)
                self.set_text_color(46, 139, 87)
                self.cell(0, 15, clean_text_for_pdf(tx["title"]), 0, 1, 'C')
                self.set_font('Arial', 'I', 11)
                self.set_text_color(100, 100, 100)
                self.cell(0, 8, clean_text_for_pdf(tx["subtitle"]), 0, 1, 'C')
                self.set_draw_color(46, 139, 87)
                self.line(10, 35, 200, 35)
                self.ln(10)
            else:
                # Paginas siguientes header compacto para ahorrar espacio
                self.set_font('Arial', 'B', 9)
                self.set_text_color(46, 139, 87)
                self.cell(0, 6, clean_text_for_pdf(tx["header"]), 0, 0, 'L')
                self.set_font('Arial', 'I', 8)
                self.set_text_color(128, 128, 128)
                self.cell(0, 6, f'{clean_text_for_pdf(tx["file"])}: {uploaded_filename}', 0, 1, 'R')
                self.set_draw_color(200, 200, 200)
                self.line(10, 17, 200, 17)
                self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            date_str = tx["generated"] + f" {peru_time.strftime('%Y-%m-%d %H:%M:%S')} " + tx["peru_time"]
            self.cell(0, 10, clean_text_for_pdf(f'{tx["page"]} {self.page_no()} | {date_str}'), 0, 0, 'C')

        def check_and_add_page(self, needed_height):
            # Agregar pagina si el elemento excede el limite
            if self.get_y() + needed_height > 265:
                self.add_page()

        def chapter_title(self, title, icon=""):
            self.check_and_add_page(25)
            self.ln(3)
            self.set_font('Arial', 'B', 14)
            self.set_text_color(46, 139, 87)
            self.cell(0, 10, f'{icon} {title}', 0, 1, 'L')
            self.set_draw_color(46, 139, 87)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def section_title(self, title, icon=""):
            self.check_and_add_page(15)
            self.ln(2)
            self.set_font('Arial', 'B', 11)
            self.set_text_color(70, 70, 70)
            self.cell(0, 7, f'{icon} {title}', 0, 1, 'L')
            self.ln(1)

        def normal_text(self, text, bold=False):
            self.check_and_add_page(8)
            self.set_font('Arial', 'B' if bold else '', 9.5)
            self.set_text_color(0, 0, 0)
            self.cell(0, 5.5, text, 0, 1, 'L')

        def info_box(self, title, content, bg_color=(240, 248, 255)):
            lines = content.split('\n')
            needed = len(lines) * 5 + 15
            self.check_and_add_page(needed)
            x, y = self.get_x(), self.get_y()
            self.set_fill_color(*bg_color)
            self.rect(x, y, 190, needed, 'F')

            self.set_font('Arial', 'B', 10.5)
            self.set_text_color(25, 25, 112)
            self.cell(0, 7, title, 0, 1, 'L')

            self.set_font('Arial', '', 9)
            self.set_text_color(0, 0, 0)
            for line in lines:
                if line.strip():
                    self.cell(0, 4.5, f"  {line.strip()}", 0, 1, 'L')
            self.ln(3)

        def add_consensus_result(self, consensus_reached, consensus_diagnosis):
            self.check_and_add_page(20)
            if consensus_reached:
                if consensus_diagnosis == "Sano":
                    bg_color = (212, 237, 218)
                    title = f"[OK] {tx['diag_title']}: {tx['state_healthy_lbl'].upper()}"
                else:
                    bg_color = (248, 215, 218)
                    title = f"[!] {tx['diag_title']}: {tx['state_diseased_lbl'].upper()} ({consensus_diagnosis_cleaned.upper()})"
            else:
                bg_color = (255, 243, 205)
                title = f"[?] {tx['diag_title']}: SIN CONSENSO"

            self.set_fill_color(*bg_color)
            self.rect(10, self.get_y(), 190, 12, 'F')

            self.set_font('Arial', 'B', 12)
            self.set_text_color(0, 0, 0)
            self.cell(0, 12, title, 0, 1, 'C')
            self.ln(4)

    pdf = PDF()
    pdf.add_page()

    # 1. INFORMACIÓN GENERAL
    pdf.chapter_title(clean_text_for_pdf(tx["info_title"]), "[INFO]")
    pdf.normal_text(clean_text_for_pdf(f"{tx['file']}: {uploaded_filename}"), bold=True)
    pdf.normal_text(clean_text_for_pdf(f"{tx['datetime']}: {peru_time.strftime('%Y-%m-%d %H:%M:%S')} {tx['peru_time']}"))
    pdf.normal_text(clean_text_for_pdf(f"{tx['models_used']}: MobileNetV2, ResNet50, EfficientNetB0"))
    pdf.normal_text(clean_text_for_pdf(f"{tx['resolution']}: {img_size}x{img_size} px"))

    # 2. DIAGNÓSTICO PRINCIPAL
    pdf.chapter_title(clean_text_for_pdf(tx["diag_title"]), "[DIAG]")
    pdf.add_consensus_result(consensus_reached, consensus_diagnosis)

    # 3. IMAGEN ANALIZADA
    pdf.chapter_title(clean_text_for_pdf(tx["img_title"]), "[IMG]")
    try:
        image_pil = Image.fromarray(image)
        temp_img_path = f"temp_analysis_img_{int(peru_time.timestamp())}.png"
        image_pil.save(temp_img_path, format='PNG')

        img_width = 80
        page_width = 190
        x_position = (page_width - img_width) / 2 + 10

        # Calcular altura real de la imagen según relación de aspecto
        img_w, img_h = image_pil.size
        aspect = img_h / img_w
        pdf_img_height = img_width * aspect

        pdf.check_and_add_page(pdf_img_height + 25)
        pdf.image(temp_img_path, x=x_position, w=img_width)
        pdf.ln(pdf_img_height + 3)

        pdf.section_title(clean_text_for_pdf(tx["img_details"]), "[i]")
        pdf.normal_text(clean_text_for_pdf(f"- {tx['orig_size']}: {image_pil.size[0]}x{image_pil.size[1]} px"))
        pdf.normal_text(clean_text_for_pdf(f"- {tx['format']}: PNG"))
        pdf.normal_text(clean_text_for_pdf(f"- {tx['channels']}: RGB"))

        try:
            os.remove(temp_img_path)
        except:
            pass
    except Exception as e:
        pdf.normal_text(clean_text_for_pdf(f"[Error al procesar la imagen: {e}]"))
        pdf.ln(5)

    # 4. RESULTADOS DETALLADOS POR MODELO
    pdf.chapter_title(clean_text_for_pdf(tx["det_title"]), "[MODELS]")

    temp_graph_paths = []
    translated_class_names = [translate_class_lang(c, lang_key) for c in ["Mancha gris", "Roña común", "Tizón del norte", "Sano"]]
    try:
        for model_name, pred in predictions.items():
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#2E8B57']
            bars = ax.bar(translated_class_names, pred['probabilities'], color=colors, alpha=0.8)
            ax.set_title(clean_text_for_pdf(f"{tx['chart_title']} {model_name}"), fontsize=11, fontweight='bold', pad=10)
            ax.set_ylabel(clean_text_for_pdf(tx["prob_ylabel"]), fontsize=9)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.2, axis='y')

            max_idx = np.argmax(pred['probabilities'])
            bars[max_idx].set_color('#2E8B57')
            bars[max_idx].set_alpha(1.0)

            for j, v in enumerate(pred['probabilities']):
                ax.text(j, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')

            plt.xticks(rotation=30, ha='right', fontsize=8)
            plt.tight_layout()

            temp_graph_path = f"temp_graph_{model_name}_{int(peru_time.timestamp())}.png"
            plt.savefig(temp_graph_path, dpi=120, bbox_inches='tight')
            temp_graph_paths.append(temp_graph_path)
            plt.close()

        # Añadir las gráficas y tablas
        for i, (model_name, pred) in enumerate(predictions.items()):
            pdf.section_title(clean_text_for_pdf(f"{tx['model_card_title']} {model_name}"), "[M]")
            confidence_level = "ALTA" if pred['confidence'] > 0.8 else "MEDIA" if pred['confidence'] > 0.6 else "BAJA"

            translated_pred_class = translate_class_lang(pred['class'], lang_key)
            status_text = tx["state_ok"] if pred['class'] == 'Sano' else tx["state_warn"]
            pdf.info_box(
                clean_text_for_pdf(f"{tx['model_card_title']}"),
                f"{tx['pred_label']}: {clean_text_for_pdf(translated_pred_class)}\n{tx['conf_label']}: {pred['confidence']:.2%} ({confidence_level})\n{tx['state_label']}: {clean_text_for_pdf(status_text)}"
            )

            # Insertar gráfico con altura dinámica calculada
            chart_width = 130
            chart_height = chart_width * 0.58
            
            if i < len(temp_graph_paths) and os.path.exists(temp_graph_paths[i]):
                pdf.check_and_add_page(chart_height + 5)
                pdf.image(temp_graph_paths[i], x=40, w=chart_width)
                pdf.ln(chart_height + 2)

            pdf.section_title(clean_text_for_pdf(tx["prob_per_class"]), "[DATA]")
            for j, class_name in enumerate(["Mancha gris", "Roña común", "Tizón del norte", "Sano"]):
                prob = pred['probabilities'][j]
                marker = "=>" if j == np.argmax(pred['probabilities']) else "  "
                translated_cn = translate_class_lang(class_name, lang_key)
                pdf.normal_text(clean_text_for_pdf(f"{marker} {translated_cn}: {prob:.2%}"))
            pdf.ln(4)

    except Exception as e:
        pdf.normal_text(clean_text_for_pdf(f"Error generando gráficos: {e}"))
    finally:
        for temp_path in temp_graph_paths:
            try:
                os.remove(temp_path)
            except:
                pass

    # 5. ANÁLISIS COMPARATIVO
    pdf.chapter_title(clean_text_for_pdf(tx["comp_title"]), "[COMP]")
    pdf.section_title(clean_text_for_pdf(tx["comp_summary"]), "[SUM]")
    pdf.normal_text(clean_text_for_pdf(tx["comp_table_header"]))
    pdf.normal_text("-" * 65)

    for model_name, pred in predictions.items():
        status = tx["state_healthy_lbl"] if pred['class'] == 'Sano' else tx["state_diseased_lbl"]
        clean_class = clean_text_for_pdf(translate_class_lang(pred['class'], lang_key))
        line = f"{model_name:<15} {clean_class:<15} {pred['confidence']:>8.1%}    {status}"
        pdf.normal_text(clean_text_for_pdf(line))
    pdf.ln(4)

    if consensus_reached:
        consensus_reached_text = tx["consensus_reached_body"].format(diagnosis=consensus_diagnosis_cleaned)
        pdf.info_box(
            clean_text_for_pdf(tx["consensus_reached_title"]),
            clean_text_for_pdf(consensus_reached_text)
        )
    else:
        predictions_list = [translate_class_lang(pred['class'], lang_key) for pred in predictions.values()]
        unique_predictions = list(set(predictions_list))
        consensus_text = clean_text_for_pdf(tx["no_consensus_body_start"])
        for pred in unique_predictions:
            count = predictions_list.count(pred)
            clean_pred = clean_text_for_pdf(pred)
            consensus_text += f"- {clean_pred}: {count} modelo(s)\n"
        consensus_text += clean_text_for_pdf(tx["no_consensus_body_end"])
        pdf.info_box(clean_text_for_pdf(tx["no_consensus_title"]), consensus_text)

    # 6. RECOMENDACIONES
    pdf.chapter_title(clean_text_for_pdf(tx["rec_title"]), "[REC]")
    if consensus_reached:
        if consensus_diagnosis == "Sano":
            recommendations = tx["rec_healthy"]
        else:
            recommendations = tx["rec_diseased"]
    else:
        recommendations = tx["rec_no_consensus"]
        
    for rec in recommendations:
        pdf.normal_text(clean_text_for_pdf(rec))

    # 7. INFORMACIÓN SOBRE ENFERMEDADES
    if consensus_reached and consensus_diagnosis != "Sano":
        pdf.chapter_title(clean_text_for_pdf(tx["disease_info_title"]), "[DISEASE]")
        details_lang = tx["diseases"]
        if consensus_diagnosis in details_lang:
            details = details_lang[consensus_diagnosis]
        elif consensus_diagnosis == "Roya común" and "Roña común" in details_lang:
            details = details_lang["Roña común"]
        elif consensus_diagnosis == "Roña común" and "Roya común" in details_lang:
            details = details_lang["Roya común"]
        else:
            details = None
            
        if details:
            pdf.section_title(clean_text_for_pdf(f"{tx['disease_header']}: {consensus_diagnosis_cleaned}"), "[PATHOGEN]")
            pdf.normal_text(clean_text_for_pdf(details['descripcion']))
            pdf.ln(2)

            pdf.section_title(clean_text_for_pdf(tx["symptoms_header"]), "[SYMPT]")
            for sintoma in details['sintomas']:
                pdf.normal_text(clean_text_for_pdf(sintoma))
            pdf.ln(2)

            pdf.section_title(clean_text_for_pdf(tx["conditions_header"]), "[ENV]")
            pdf.normal_text(clean_text_for_pdf(details['condiciones']))
            pdf.ln(2)

            pdf.section_title(clean_text_for_pdf(tx["treatments_header"]), "[TREAT]")
            for tratamiento in details['tratamiento']:
                pdf.normal_text(clean_text_for_pdf(tratamiento))

    # 7.5. VALIDACIÓN ESTADÍSTICA ROBUSTA (ING. SANTOS)
    pdf.chapter_title(clean_text_for_pdf({
        'es': "VALIDACIÓN ESTADÍSTICA ROBUSTA (ING. SANTOS)",
        'en': "ROBUST STATISTICAL VALIDATION (ENG. SANTOS)",
        'pt': "VALIDAÇÃO ESTATÍSTICA ROBUSTA (ENG. SANTOS)"
    }.get(lang_key, "VALIDACIÓN ESTADÍSTICA ROBUSTA")), "[STATS]")
    
    stats_lines_dict = {
        'es': [
            "- Prueba de McNemar: p-valor = 0.0133 (Diferencia significativa en clasificación, se rechaza H0)",
            "- Prueba de Mann-Whitney U (CV): MobileNetV2 vs EfficientNetB0 (p = 0.0089). Confirma la superioridad de EfficientNetB0",
            "- Prueba de Kolmogorov-Smirnov: Confirma que las curvas de confianza de inferencia difieren significativamente entre modelos",
            "- Prueba de Morgan-Pitman: p-valor = 0.3821 (Varianza del error equivalente entre ResNet50 y EfficientNetB0, validando parsimonia)",
            "- Robustez (DAVT-Adv): Resiliencia de EfficientNetB0 ante ruido foliar y variaciones de luz (+20% de brillo, 5% de ruido de sal y pimienta)"
        ],
        'en': [
            "- McNemar's Test: p-value = 0.0133 (Significant difference in classification, H0 is rejected)",
            "- Mann-Whitney U Test (CV): MobileNetV2 vs EfficientNetB0 (p = 0.0089). Confirms EfficientNetB0 superiority",
            "- Kolmogorov-Smirnov Test: Confirms that prediction confidence curves differ significantly between architectures",
            "- Morgan-Pitman Test: p-value = 0.3821 (Equivalent error variance between ResNet50 and EfficientNetB0, validating parsimony)",
            "- Robustness (DAVT-Adv): EfficientNetB0 resilience against leaf noise and light changes (+20% brightness, 5% salt & pepper noise)"
        ],
        'pt': [
            "- Teste de McNemar: p-valor = 0.0133 (Diferença significativa na classificação, H0 é rejeitado)",
            "- Teste Mann-Whitney U (CV): MobileNetV2 vs EfficientNetB0 (p = 0.0089). Confirma a superioridade de EfficientNetB0",
            "- Teste Kolmogorov-Smirnov: Confirma que as curvas de confiança de inferência diferem significativamente entre modelos",
            "- Teste de Morgan-Pitman: p-valor = 0.3821 (Variância do erro equivalente entre ResNet50 e EfficientNetB0, validando parcimônia)",
            "- Robustez (DAVT-Adv): Resiliência de EfficientNetB0 sob ruído foliar e variaciones de luz (+20% de brilho, 5% de ruido de sal e pimenta)"
        ]
    }
    stats_lines = stats_lines_dict.get(lang_key, stats_lines_dict['es'])
    
    for s_line in stats_lines:
        pdf.normal_text(clean_text_for_pdf(s_line))

    # 8. INFORMACIÓN TÉCNICA DEL SISTEMA
    pdf.chapter_title(clean_text_for_pdf(tx["tech_title"]), "[TECH]")
    pdf.section_title(clean_text_for_pdf(tx["tech_spec_header"]), "[SPEC]")
    for line in tx["tech_spec_list"]:
        pdf.normal_text(clean_text_for_pdf(line.format(size=img_size)))

    # 9. DISCLAIMER
    pdf.chapter_title(clean_text_for_pdf(tx["disclaimer_title"]), "[!]")
    pdf.info_box(
        clean_text_for_pdf(tx["disclaimer_title"]),
        clean_text_for_pdf(tx["disclaimer_text"]),
        bg_color=(255, 248, 220)
    )

    # Guardar el PDF
    pdf.output(filepath)
    return filepath
