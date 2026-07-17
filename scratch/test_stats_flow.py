import sys
import os
import numpy as np

# Asegurar que la raíz del proyecto está en el PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stats_tests import run_statistical_tests

def run_verification():
    print("--- INICIANDO VERIFICACIÓN DE CÁLCULOS ESTADÍSTICOS ---")
    
    # 1. Crear datos de validación cruzada simulados (5 pliegues, 5 modelos)
    np.random.seed(42)
    
    cv_results = {
        'Regresión Logística (Clásico)': {'accuracies': [0.81, 0.82, 0.83, 0.80, 0.82]},
        'Random Forest (Clásico)': {'accuracies': [0.88, 0.89, 0.87, 0.90, 0.88]},
        'Red Neuronal MLP (Clásico)': {'accuracies': [0.89, 0.91, 0.90, 0.92, 0.89]},
        'Híbrido Votación (RF+MLP)': {'accuracies': [0.91, 0.92, 0.93, 0.90, 0.92]},
        'Híbrido Stacking (Meta-GB)': {'accuracies': [0.93, 0.94, 0.95, 0.92, 0.94]}
    }
    
    # Datos de prueba simulados (N = 100 muestras)
    y_test = np.random.randint(0, 2, size=100)
    
    # Resultados del test simulados
    results = {}
    for name in cv_results.keys():
        acc = np.mean(cv_results[name]['accuracies'])
        y_pred = np.copy(y_test)
        # Errores basados en la precisión
        mask = np.random.rand(100) > acc
        y_pred[mask] = 1 - y_pred[mask]
        results[name] = {'y_pred': y_pred}
        
    print("Ejecutando run_statistical_tests con datos normales y alpha = 0.05...")
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
        
    print("\nIntervalos de Confianza por Bootstrap (Accuracy):")
    for name, ci in stats_results['bootstrap_ci'].items():
        print(f"  - {name}: [{ci[0]:.2%} - {ci[1]:.2%}]")
        
    print("\nIntervalos de Confianza por Bootstrap (F1-Score):")
    for name, ci in stats_results['bootstrap_ci_f1'].items():
        print(f"  - {name}: [{ci[0]:.4f} - {ci[1]:.4f}]")
        
    # Validar campos clave
    assert 'bootstrap_ci_f1' in stats_results, "Falta el campo bootstrap_ci_f1"
    assert 'pairwise_comparisons' in stats_results, "Falta el campo pairwise_comparisons"
    assert len(stats_results['bootstrap_ci_f1']) == 5, "Faltan intervalos de bootstrap F1 para los 5 modelos"
    print("\n✅ ¡Todos los checks de la prueba pasaron con éxito!")

if __name__ == "__main__":
    run_verification()
