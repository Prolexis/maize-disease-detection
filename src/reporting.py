import os
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------
# 1. GENERACIÓN DE EXCEL (.xlsx)
# ---------------------------------------------------------
def generate_xlsx_report(df_eda, df_training, cv_results, tuning_results, stats_results, filepath="reports/reporte_automl.xlsx"):
    """
    Crea un archivo Excel organizado con una pestaña para cada fase del pipeline de ML.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Pestaña 1: EDA
        df_eda.reset_index().rename(columns={'index': 'Variable'}).to_excel(writer, sheet_name='EDA', index=False)
        
        # Pestaña 2: Entrenamiento (Métricas)
        df_training.to_excel(writer, sheet_name='Entrenamiento', index=True)
        
        # Pestaña 3: Validación Cruzada
        cv_data = []
        for model_name, res in cv_results.items():
            cv_data.append({
                'Modelo': model_name,
                'Mean Accuracy': res['mean_accuracy'],
                'Std Accuracy': res['std_accuracy'],
                'Mean F1': res['mean_f1'],
                'Std F1': res['std_f1']
            })
        pd.DataFrame(cv_data).to_excel(writer, sheet_name='Cross-Validation', index=False)
        
        # Pestaña 4: Tuning
        tuning_data = {
            'Métrica': ['Método de Búsqueda', 'Tiempo de Búsqueda (s)', 'Precisión Antes', 'Precisión Después', 'Mejores Hiperparámetros'],
            'Valor': [
                tuning_results['method'],
                f"{tuning_results['search_time']:.2f}",
                f"{tuning_results['accuracy_before']:.4f}",
                f"{tuning_results['accuracy_after']:.4f}",
                str(tuning_results['best_params'])
            ]
        }
        pd.DataFrame(tuning_data).to_excel(writer, sheet_name='Tuning', index=False)
        
        # Pestaña 5: Pruebas Estadísticas
        shapiros = ", ".join([f"{k}: p={v:.3f}" for k, v in stats_results['shapiro_pvals'].items()])
        stats_data = {
            'Prueba': [
                'Tipo de Prueba Grupal', 'Estadístico Grupal', 'p-valor Grupal',
                'Supuesto Shapiro-Wilk (p)', 'Supuesto Levene (p)',
                'Prueba Wilcoxon (p)', 'Prueba McNemar (p)'
            ],
            'Resultado/Valor': [
                stats_results['test_type'],
                f"{stats_results['overall_stat']:.4f}",
                f"{stats_results['overall_pval']:.4f}",
                shapiros,
                f"{stats_results['levene_pval']:.4f}",
                f"Classic vs Hybrid: p={stats_results['wilcoxon']['p_val']:.4f}",
                f"Aciertos/Fallos: p={stats_results['mcnemar']['p_val']:.4f}"
            ]
        }
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='Pruebas Estadisticas', index=False)
        
    return filepath

# ---------------------------------------------------------
# 2. GENERACIÓN DE WORD (.docx)
# ---------------------------------------------------------
def generate_docx_report(df_eda, df_training, cv_results, tuning_results, stats_results, interpretations, image_paths, filepath="reports/reporte_automl.docx"):
    """
    Crea un reporte Word (.docx) formateado con portada, tablas e imágenes embebidas.
    """
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    
    doc = Document()
    
    # Configurar estilos de fuente globales
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(11)
    
    # 1. Portada
    title_p = doc.add_paragraph()
    title_p.alignment = 1 # Centrado
    run_title = title_p.add_run("\n\n\n\n🌽 INFORME INTEGRAL DE AUTOML FITOSANITARIO\n")
    run_title.font.size = Pt(24)
    run_title.bold = True
    run_title.font.color.rgb = RGBColor(16, 185, 129) # Verde
    
    subtitle_p = doc.add_paragraph()
    subtitle_p.alignment = 1
    run_sub = subtitle_p.add_run("Pipeline de Machine Learning Tabular y Auditoría Fitosanitaria\n\n\n\n")
    run_sub.font.size = Pt(14)
    run_sub.font.color.rgb = RGBColor(107, 114, 128) # Gris
    
    info_p = doc.add_paragraph()
    info_p.alignment = 1
    run_info = info_p.add_run(f"Generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nHora local (Perú)\n")
    run_info.font.size = Pt(11)
    
    doc.add_page_break()
    
    # 2. Sección EDA
    doc.add_heading("1. Análisis Exploratorio de Datos (EDA)", level=1)
    doc.add_paragraph("Estadísticas descriptivas generales de las variables numéricas analizadas:")
    
    # Crear Tabla de EDA
    table_eda = doc.add_table(rows=1, cols=6)
    table_eda.style = 'Light Shading Accent 1'
    hdr_cells = table_eda.rows[0].cells
    headers = ['Variable', 'Media', 'Mediana', 'Desviación', 'Asimetría', 'Curtosis']
    for idx, name in enumerate(headers):
        hdr_cells[idx].text = name
        
    for var_name, row in df_eda.iterrows():
        row_cells = table_eda.add_row().cells
        row_cells[0].text = str(var_name)
        row_cells[1].text = f"{row['media']:.2f}"
        row_cells[2].text = f"{row['mediana']:.2f}"
        row_cells[3].text = f"{row['desviación']:.2f}"
        row_cells[4].text = f"{row['asimetría']:.2f}"
        row_cells[5].text = f"{row['curtosis']:.2f}"
        
    doc.add_paragraph("\nDistribución y Balance de Clases del Dataset:")
    if 'balance' in image_paths and os.path.exists(image_paths['balance']):
        doc.add_picture(image_paths['balance'], width=Inches(4.5))
        
    doc.add_paragraph("\nMatriz de Correlación de Variables:")
    if 'correlation' in image_paths and os.path.exists(image_paths['correlation']):
        doc.add_picture(image_paths['correlation'], width=Inches(4.5))
        
    doc.add_paragraph("\n**Interpretación Técnica de EDA:**")
    doc.add_paragraph(interpretations['eda'])
    
    doc.add_page_break()
    
    # 3. Sección Entrenamiento
    doc.add_heading("2. Fase de Entrenamiento de Modelos", level=1)
    doc.add_paragraph("Métricas obtenidas por los 3 modelos clásicos y 2 híbridos:")
    
    table_train = doc.add_table(rows=1, cols=6)
    table_train.style = 'Light Shading Accent 1'
    hdr_cells = table_train.rows[0].cells
    hdr_names = ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    for idx, name in enumerate(hdr_names):
        hdr_cells[idx].text = name
        
    for model_name, row in df_training.iterrows():
        row_cells = table_train.add_row().cells
        row_cells[0].text = str(model_name)
        row_cells[1].text = f"{row['Accuracy']:.4f}"
        row_cells[2].text = f"{row['Precision']:.4f}"
        row_cells[3].text = f"{row['Recall']:.4f}"
        row_cells[4].text = f"{row['F1-Score']:.4f}"
        row_cells[5].text = f"{row['AUC']:.4f}"
        
    doc.add_paragraph("\nCurvas ROC comparativas:")
    if 'roc' in image_paths and os.path.exists(image_paths['roc']):
        doc.add_picture(image_paths['roc'], width=Inches(5.0))
        
    doc.add_paragraph("\n**Interpretación de Entrenamiento:**")
    doc.add_paragraph(interpretations['training'])
    
    doc.add_page_break()
    
    # 4. Sección Cross Validation & Tuning
    doc.add_heading("3. Validación Cruzada y Optimización de Hiperparámetros", level=1)
    doc.add_paragraph("Estabilidad e hiperparámetros óptimos:")
    
    # Tabla CV
    table_cv = doc.add_table(rows=1, cols=3)
    table_cv.style = 'Light Shading Accent 1'
    hdr_cells = table_cv.rows[0].cells
    hdr_cells[0].text = 'Modelo'
    hdr_cells[1].text = 'Accuracy CV'
    hdr_cells[2].text = 'F1-Score CV'
    
    for model_name, res in cv_results.items():
        row_cells = table_cv.add_row().cells
        row_cells[0].text = model_name
        row_cells[1].text = f"{res['mean_accuracy']:.4f} ± {res['std_accuracy']:.4f}"
        row_cells[2].text = f"{res['mean_f1']:.4f} ± {res['std_f1']:.4f}"
        
    doc.add_paragraph("\nDispersión de los Folds de Validación:")
    if 'cv' in image_paths and os.path.exists(image_paths['cv']):
        doc.add_picture(image_paths['cv'], width=Inches(4.5))
        
    doc.add_paragraph("\n**Interpretación de Validación Cruzada:**")
    doc.add_paragraph(interpretations['cv'])
    
    doc.add_paragraph("\n**Optimización (Tuning):**")
    doc.add_paragraph(interpretations['tuning'])
    
    doc.add_page_break()
    
    # 5. Sección Pruebas Estadísticas
    doc.add_heading("4. Pruebas de Significancia Estadística", level=1)
    doc.add_paragraph("Validación estadística de los desempeños:")
    
    if 'stats' in image_paths and os.path.exists(image_paths['stats']):
        doc.add_picture(image_paths['stats'], width=Inches(4.5))
        
    doc.add_paragraph("\n**Interpretación de Pruebas Estadísticas:**")
    doc.add_paragraph(interpretations['stats'])
    
    doc.save(filepath)
    return filepath

# ---------------------------------------------------------
# 3. GENERACIÓN DE PDF (.pdf)
# ---------------------------------------------------------
def generate_pdf_report(df_eda, df_training, cv_results, tuning_results, stats_results, interpretations, image_paths, filepath="reports/reporte_automl.pdf"):
    """
    Crea un reporte PDF utilizando fpdf2 que incluye portada, tablas, interpretaciones e imágenes embebidas.
    """
    from fpdf import FPDF
    
    class AutoMLPDF(FPDF):
        def header(self):
            if self.page_no() > 1:
                self.set_font('Helvetica', 'I', 8)
                self.set_text_color(107, 114, 128)
                self.cell(0, 10, 'Informe Integral de AutoML y Diagnóstico Fitosanitario', 0, 1, 'R')
                self.set_draw_color(16, 185, 129)
                self.line(10, 18, 200, 18)
                self.ln(10)
                
        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(107, 114, 128)
            self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
            
    pdf = AutoMLPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- Portada ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(16, 185, 129) # Verde
    pdf.ln(50)
    pdf.cell(0, 15, 'INFORME INTEGRAL DE AUTOML', ln=1, align='C')
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(107, 114, 128)
    pdf.cell(0, 10, 'Detección Fitosanitaria y Pipeline Tabular de Maíz', ln=1, align='C')
    pdf.ln(60)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 6, f'Generado automáticamente: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=1, align='C')
    pdf.cell(0, 6, 'Área de Sanidad Vegetal - Cultivos de Maíz', ln=1, align='C')
    
    # --- Página 2: EDA ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, '1. Análisis Exploratorio de Datos (EDA)', ln=1)
    pdf.ln(5)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(31, 41, 55)
    pdf.multi_cell(0, 6, 'Estadísticos descriptivos de las variables numéricas del dataset de cultivo:')
    pdf.ln(3)
    
    # Tabla EDA en PDF
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(240, 240, 240)
    headers = ['Variable', 'Media', 'Mediana', 'Desv.', 'Asim.', 'Curt.']
    col_widths = [45, 30, 30, 30, 25, 25]
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 8, h, border=1, align='C', fill=True)
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 9)
    for var_name, row in df_eda.iterrows():
        pdf.cell(45, 7, str(var_name), border=1)
        pdf.cell(30, 7, f"{row['media']:.2f}", border=1, align='C')
        pdf.cell(30, 7, f"{row['mediana']:.2f}", border=1, align='C')
        pdf.cell(30, 7, f"{row['desviación']:.2f}", border=1, align='C')
        pdf.cell(25, 7, f"{row['asimetría']:.2f}", border=1, align='C')
        pdf.cell(25, 7, f"{row['curtosis']:.2f}", border=1, align='C')
        pdf.ln()
        
    pdf.ln(10)
    if 'balance' in image_paths and os.path.exists(image_paths['balance']):
        pdf.image(image_paths['balance'], x=15, y=pdf.get_y(), w=85)
    if 'correlation' in image_paths and os.path.exists(image_paths['correlation']):
        pdf.image(image_paths['correlation'], x=110, y=pdf.get_y(), w=85)
    
    pdf.ln(65)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Interpretación Técnica del EDA:', ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, interpretations['eda'].replace('**', ''))
    
    # --- Página 3: Entrenamiento ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, '2. Fase de Entrenamiento y Comparación', ln=1)
    pdf.ln(5)
    
    # Tabla Entrenamiento
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(240, 240, 240)
    hdr_train = ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    col_w_train = [55, 27, 27, 27, 27, 27]
    for w, h in zip(col_w_train, hdr_train):
        pdf.cell(w, 8, h, border=1, align='C', fill=True)
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 8)
    for model_name, row in df_training.iterrows():
        pdf.cell(55, 7, str(model_name), border=1)
        pdf.cell(27, 7, f"{row['Accuracy']:.4f}", border=1, align='C')
        pdf.cell(27, 7, f"{row['Precision']:.4f}", border=1, align='C')
        pdf.cell(27, 7, f"{row['Recall']:.4f}", border=1, align='C')
        pdf.cell(27, 7, f"{row['F1-Score']:.4f}", border=1, align='C')
        pdf.cell(27, 7, f"{row['AUC']:.4f}", border=1, align='C')
        pdf.ln()
        
    pdf.ln(5)
    if 'roc' in image_paths and os.path.exists(image_paths['roc']):
        pdf.image(image_paths['roc'], x=40, y=pdf.get_y(), w=120)
        
    pdf.ln(90)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Interpretación de Entrenamiento:', ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, interpretations['training'].replace('**', ''))
    
    # --- Página 4: CV, Tuning y Pruebas Estadísticas ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, '3. Validación Cruzada, Tuning y Pruebas Estadísticas', ln=1)
    pdf.ln(5)
    
    if 'cv' in image_paths and os.path.exists(image_paths['cv']):
        pdf.image(image_paths['cv'], x=15, y=pdf.get_y(), w=85)
    if 'stats' in image_paths and os.path.exists(image_paths['stats']):
        pdf.image(image_paths['stats'], x=110, y=pdf.get_y(), w=85)
        
    pdf.ln(70)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Interpretación de Validación y Tuning:', ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, interpretations['cv'].replace('**', '') + "\n" + interpretations['tuning'].replace('**', ''))
    
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Interpretación de Pruebas de Significancia Estadística:', ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, interpretations['stats'].replace('**', ''))
    
    pdf.output(filepath)
    return filepath
