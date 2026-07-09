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
                tuning_results['search_time'],
                tuning_results['accuracy_before'],
                tuning_results['accuracy_after'],
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
                stats_results['overall_stat'],
                stats_results['overall_pval'],
                shapiros,
                stats_results['levene_pval'],
                f"Classic vs Hybrid: p={stats_results['wilcoxon']['p_val']:.4f}",
                f"Aciertos/Fallos: p={stats_results['mcnemar']['p_val']:.4f}"
            ]
        }
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='Pruebas Estadisticas', index=False)
        
        # Aplicar formato a todas las pestañas creadas
        workbook = writer.book
        for sheet_name in workbook.sheetnames:
            format_excel_sheet(workbook[sheet_name])
            
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
def generate_tabular_pdf_report(df_eda, df_training, cv_results, tuning_results, stats_results, interpretations, image_paths, filepath="reports/reporte_automl.pdf"):
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
        pdf.cell(45, 7, clean_pdf_text(str(var_name)), border=1)
        pdf.cell(30, 7, f"{row['media']:.2f}", border=1, align='C')
        pdf.cell(30, 7, f"{row['mediana']:.2f}", border=1, align='C')
        pdf.cell(30, 7, f"{row['desviación']:.2f}", border=1, align='C')
        pdf.cell(25, 7, f"{row['asimetría']:.2f}", border=1, align='C')
        pdf.cell(25, 7, f"{row['curtosis']:.2f}", border=1, align='C')
        pdf.ln()
        
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
    pdf.cell(0, 6, 'Interpretación Técnica del EDA:', ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, clean_pdf_text(interpretations['eda'].replace('**', '')))
    
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
    pdf.cell(0, 6, 'Interpretación de Entrenamiento:', ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, clean_pdf_text(interpretations['training'].replace('**', '')))
    
    # --- Página 4: CV, Tuning y Pruebas Estadísticas ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, '3. Validación Cruzada, Tuning y Pruebas Estadísticas', ln=1)
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
    pdf.cell(0, 6, 'Interpretación de Validación y Tuning:', ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, clean_pdf_text(interpretations['cv'].replace('**', '') + "\n" + interpretations['tuning'].replace('**', '')))
    
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Interpretación de Pruebas de Significancia Estadística:', ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, clean_pdf_text(interpretations['stats'].replace('**', '')))
    
    pdf.output(filepath)
    return filepath
# ---------------------------------------------------------
# 4. REPORTES PARA DIAGNÓSTICO POR IMÁGENES (WORD & EXCEL)
# ---------------------------------------------------------
def generate_image_docx_report(image, predictions, uploaded_filename, consensus_reached, consensus_diagnosis, filepath="reports/reporte_imagen.docx"):
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    import tempfile
    from datetime import datetime
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    doc = Document()
    
    # Configurar estilos de fuente globales
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(11)
    
    # --- PÁGINA 1: PORTADA ---
    title_p = doc.add_paragraph()
    title_p.alignment = 1 # Centrado
    run_title = title_p.add_run("\n\n\n\n🌽 INFORME DE DIAGNÓSTICO FITOSANITARIO\n")
    run_title.font.size = Pt(22)
    run_title.bold = True
    run_title.font.color.rgb = RGBColor(46, 139, 87) # Verde
    
    subtitle_p = doc.add_paragraph()
    subtitle_p.alignment = 1
    run_sub = subtitle_p.add_run("Detección Automática de Patologías en Hojas de Maíz\n\n\n\n")
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = RGBColor(100, 100, 100)
    
    info_p = doc.add_paragraph()
    info_p.alignment = 1
    run_info = info_p.add_run(f"Archivo analizado: {uploaded_filename}\nFecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Hora local)\nÁrea de Sanidad Vegetal\n")
    run_info.font.size = Pt(10)
    
    doc.add_page_break()
    
    # --- PÁGINA 2: DIAGNÓSTICO PRINCIPAL ---
    doc.add_heading("1. Diagnóstico Principal por Consenso", level=1)
    doc.add_paragraph("Resultado de la clasificación combinada de múltiples redes neuronales convolucionales:")
    
    # Cuadro de Consenso
    p_cons = doc.add_paragraph()
    if consensus_reached:
        res_text = f"DIAGNÓSTICO GENERAL: {consensus_diagnosis.upper()}"
        color = RGBColor(46, 139, 87) if consensus_diagnosis == "Sano" else RGBColor(185, 28, 28)
    else:
        res_text = "DIAGNÓSTICO GENERAL: SIN CONSENSO DEFINIDO"
        color = RGBColor(217, 119, 6)
    run_cons = p_cons.add_run(f"\n   {res_text}   \n")
    run_cons.bold = True
    run_cons.font.size = Pt(14)
    run_cons.font.color.rgb = color
    
    # Imagen de la hoja
    doc.add_heading("2. Imagen de la Hoja de Maíz Analizada", level=2)
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
        doc.add_paragraph(f"[Error incrustando la imagen: {e}]")
        
    doc.add_page_break()
    
    # --- PÁGINA 3: RESULTADOS DETALLADOS POR MODELO ---
    doc.add_heading("2. Resultados Detallados de los Modelos", level=1)
    doc.add_paragraph("Métricas y predicciones individuales de cada red neuronal entrenada:")
    
    # Gráficos de barras individuales por modelo
    class_names = ["Mancha gris", "Roña común", "Tizón del norte", "Sano"]
    temp_graph_paths = []
    
    try:
        for model_name, pred in predictions.items():
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#16A34A']
            bars = ax.bar(class_names, pred['probabilities'], color=colors, alpha=0.8)
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
            doc.add_paragraph(f"Predicción: {pred['class']} | Confianza: {pred['confidence']:.2%}")
            if idx < len(temp_graph_paths) and os.path.exists(temp_graph_paths[idx]):
                doc.add_picture(temp_graph_paths[idx], width=Inches(4.5))
                os.remove(temp_graph_paths[idx])
                
    except Exception as e:
        doc.add_paragraph(f"[Error generando gráficos: {e}]")
        
    # Eliminamos el salto de página forzado aquí para flujo continuo
    
    # --- PÁGINA 4: TABLA COMPARATIVA Y ENFERMEDADES ---
    doc.add_heading("3. Análisis Comparativo de Predicciones", level=1)
    
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Light Shading Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Modelo de IA'
    hdr_cells[1].text = 'Diagnóstico'
    hdr_cells[2].text = 'Confianza'
    hdr_cells[3].text = 'Estado General'
    
    for model_name, pred in predictions.items():
        row_cells = table.add_row().cells
        row_cells[0].text = model_name
        row_cells[1].text = pred['class']
        row_cells[2].text = f"{pred['confidence']:.2%}"
        row_cells[3].text = 'Saludable' if pred['class'] == 'Sano' else 'Infección Detectada'
        
    # Información sobre la patología
    if consensus_reached and consensus_diagnosis != "Sano":
        doc.add_heading(f"4. Información Técnica sobre: {consensus_diagnosis}", level=2)
        
        disease_info = {
            "Tizón del norte": {
                "desc": "Causado por Exserohilum turcicum. Provoca lesiones alargadas en forma de cigarro de color marrón-grisáceo.",
                "recs": ["Consultar con un fitopatólogo.", "Aplicación foliar preventiva de fungicidas.", "Uso de semillas híbridas con tolerancia genética."]
            },
            "Roña común": {
                "desc": "Causado por Puccinia sorghi. Produce pústulas circulares de color rojizo-marrón en ambas caras de las hojas.",
                "recs": ["Eliminar malezas hospederas.", "Monitorear roció matutino.", "Aplicación temprana de compuestos cúpricos."]
            },
            "Mancha gris": {
                "desc": "Causado por Cercospora zeae-maydis. Provoca lesiones rectangulares delimitadas por las venas foliares.",
                "recs": ["Rotación de cultivos por 2 temporadas.", "Mejora del drenaje.", "Fungicidas sistémicos en etapas vegetativas."]
            }
        }
        
        if consensus_diagnosis in disease_info:
            d = disease_info[consensus_diagnosis]
            doc.add_paragraph(f"**Descripción:** {d['desc']}")
            doc.add_heading("Recomendaciones de Manejo:", level=3)
            for r in d['recs']:
                doc.add_paragraph(f"- {r}")
                
    # Eliminamos salto de página para flujo continuo
    
    # --- RECOMENDACIONES GENERALES Y DISCLAIMER ---
    doc.add_heading("5. Recomendaciones Generales del Sistema", level=1)
    
    if consensus_reached and consensus_diagnosis == "Sano":
        recs = [
            "Continuar con las prácticas de manejo actuales.",
            "Realizar monitoreos preventivos regulares cada 7-10 días.",
            "Mantener condiciones óptimas de cultivo (riego, fertilización).",
            "Inspeccionar las hojas inferiores que tienen contacto directo con la humedad del suelo."
        ]
    elif consensus_reached:
        recs = [
            "Aislar de inmediato la zona de cultivo afectada para evitar la propagación foliar por viento.",
            "Considerar tratamientos preventivos con fungicidas orgánicos o químicos regulados.",
            "Evitar el riego por aspersión directo al follaje en horas de la tarde para reducir humedad retenida.",
            "Documentar la evolución de las hojas con fotografías diarias."
        ]
    else:
        recs = [
            "Tomar una nueva imagen con mejor iluminación y enfoque central en la patología.",
            "Asegurar que la hoja no tenga reflejos de luz solar excesivos al momento de capturar.",
            "Realizar análisis de suelo para descartar deficiencias de nutrientes simulando necrosis foliar."
        ]
        
    for r in recs:
        doc.add_paragraph(f"- {r}")
        
    doc.add_heading("⚠️ Limitaciones y Responsabilidad", level=2)
    doc.add_paragraph("Este sistema es una herramienta de soporte analítico basada en redes neuronales. Los resultados deben ser confirmados visualmente en campo por ingenieros agrónomos o técnicos fitosanitarios calificados antes de realizar aplicaciones masivas de tratamientos.")
    
    doc.save(filepath)
    return filepath


def generate_image_xlsx_report(predictions, uploaded_filename, consensus_reached, consensus_diagnosis, filepath="reports/reporte_imagen.xlsx"):
    import pandas as pd
    from datetime import datetime
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Pestaña 1: Resumen de Diagnóstico
        resumen_data = {
            'Variable': ['Archivo Analizado', 'Fecha de Análisis', 'Consenso Alcanzado', 'Diagnóstico Final', 'Estado General'],
            'Valor': [
                uploaded_filename,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'SÍ' if consensus_reached else 'NO',
                consensus_diagnosis if consensus_diagnosis else 'Sin Consenso',
                'Saludable (Sano)' if consensus_diagnosis == 'Sano' else 'Infectado (Enfermo)' if consensus_reached else 'No Determinado'
            ]
        }
        pd.DataFrame(resumen_data).to_excel(writer, sheet_name='Diagnostico', index=False)
        
        # Pestaña 2: Predicciones Detalladas
        preds_data = []
        for model_name, pred in predictions.items():
            preds_data.append({
                'Modelo': model_name,
                'Clase Predicha': pred['class'],
                'Confianza': pred['confidence'],
                'Prob_Mancha_Gris': pred['probabilities'][0],
                'Prob_Rona_Comun': pred['probabilities'][1],
                'Prob_Tizon_Norte': pred['probabilities'][2],
                'Prob_Sano': pred['probabilities'][3]
            })
        pd.DataFrame(preds_data).to_excel(writer, sheet_name='Predicciones por Modelo', index=False)
        
        # Aplicar formato y autoajustes
        workbook = writer.book
        for sheet_name in workbook.sheetnames:
            format_excel_sheet(workbook[sheet_name])
            
    return filepath
