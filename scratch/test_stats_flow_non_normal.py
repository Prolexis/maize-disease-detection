import sys
import os
import numpy as np

# Asegurar que la raíz del proyecto está en el PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stats_tests import run_statistical_tests

def run_verification_non_normal():
    print("--- INICIANDO VERIFICACIÓN DE CÁLCULOS ESTADÍSTICOS (NO PARAMÉTRICOS) ---")
    
    # 1. Crear datos de validación cruzada simulados (5 pliegues, 5 modelos)
    # Hacemos que uno de los modelos tenga valores constantes para forzar que falle Shapiro-Wilk (p-valor < 0.05 o NaN)
    # y así forzar el uso de Friedman y Wilcoxon
    cv_results = {
        'Regresión Logística (Clásico)': {'accuracies': [0.90, 0.89, 0.91, 0.20, 0.90]}, # p-valor SW será muy bajo (< 0.05) obligando a test no paramétrico
        'Random Forest (Clásico)': {'accuracies': [0.88, 0.89, 0.87, 0.90, 0.88]},
        'Red Neuronal MLP (Clásico)': {'accuracies': [0.89, 0.91, 0.90, 0.92, 0.89]},
        'Híbrido Votación (RF+MLP)': {'accuracies': [0.91, 0.92, 0.93, 0.90, 0.92]},
        'Híbrido Stacking (Meta-GB)': {'accuracies': [0.93, 0.94, 0.95, 0.92, 0.94]}
    }
    
    np.random.seed(42)
    y_test = np.random.randint(0, 2, size=100)
    
    results = {}
    for name in cv_results.keys():
        acc = np.mean(cv_results[name]['accuracies'])
        y_pred = np.copy(y_test)
        mask = np.random.rand(100) > acc
        y_pred[mask] = 1 - y_pred[mask]
        results[name] = {'y_pred': y_pred}
        
    print("Ejecutando run_statistical_tests con datos no normales y alpha = 0.05...")
    stats_results = run_statistical_tests(
        cv_results=cv_results,
        y_test=y_test,
        alpha=0.05,
        save_path="reports",
        lang="es",
        results=results
    )
    
    print("\nResultados obtenidos:")
    print(f"Tipo de Prueba Global Utilizada: {stats_results['test_type']}")
    print(f"Estadístico Global: {stats_results['overall_stat']:.4f}")
    print(f"p-valor Global: {stats_results['overall_pval']:.5e}")
    print(f"¿Usa Paramétrica (ANOVA)?: {stats_results['use_parametric']}")
    
    print("\np-valores de Shapiro-Wilk (Normalidad):")
    for name, pval in stats_results['shapiro_pvals'].items():
        print(f"  - {name}: p = {pval:.5f} (Normal: {pval >= 0.05})")
        
    print(f"p-valor de Levene: {stats_results['levene_pval']:.5f}")
    
    print("\nComparaciones por Pares (vs. Mejor Modelo):")
    for pair, info in stats_results['pairwise_comparisons'].items():
        print(f"  - {pair}:")
        print(f"    * Prueba: {info['test_name']}")
        print(f"    * Estadístico: {info['stat']:.4f}")
        print(f"    * p-valor: {info['p_val']:.5f}")
        print(f"    * Significativo: {info['significant']} ({info['reason']})")
        
    print("\nResultados Post-Hoc (deben incluir Nemenyi):")
    for pair, info in list(stats_results['posthoc_results'].items())[:5]:
        print(f"  - {pair}: p-adj = {info['p_val']:.5f}, Significativo: {info['significant']}")
        
    # Validaciones clave
    assert not stats_results['use_parametric'], "Debería haber seleccionado Friedman por falta de normalidad"
    assert "Friedman" in stats_results['test_type'], "El tipo de test debería ser Friedman"
    print("\n✅ ¡La validación de la prueba no paramétrica pasó con éxito!")

if __name__ == "__main__":
    run_verification_non_normal()
