# -*- coding: utf-8 -*-
"""
Chatbot fitosanitario con motor semántico RAG (TF-IDF + similitud de coseno).
Si la similitud del query supera el umbral, retorna el fragmento más relevante.
En caso contrario, cae al fallback de keywords exactas.
"""

import re
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  BASE DE CONOCIMIENTO TRILINGÜE
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = {
    'es': {
        'rust': {
            'keywords': ['roya', 'roña', 'puccinia', 'sorghi'],
            'corpus': (
                "roya común puccinia sorghi síntomas pústulas circulares alargadas marrón rojizo "
                "hoja tratamiento fungicida triazoles estrobirulinas manejo cultural híbridos "
                "resistentes humedad malezas hospederas prevención diagnóstico"
            ),
            'response': (
                "**Roya Común (Puccinia sorghi):**\n"
                "* **Síntomas:** Pústulas circulares o alargadas de color marrón rojizo en ambas caras de la hoja.\n"
                "* **Manejo Cultural:** Eliminar malezas hospederas, evitar exceso de humedad y sembrar híbridos resistentes.\n"
                "* **Manejo Químico:** Aplicación foliar preventiva de fungicidas a base de triazoles o estrobirulinas al observar las primeras pústulas."
            )
        },
        'blight': {
            'keywords': ['tizon', 'tizón', 'exserohilum', 'turcicum'],
            'corpus': (
                "tizón norte exserohilum turcicum lesiones alargadas cigarro grisáceo marrón "
                "rotación cultivos arado rastrojo fungicidas umbral daño económico floración "
                "resistente diagnóstico síntomas prevención"
            ),
            'response': (
                "**Tizón del Norte (Exserohilum turcicum):**\n"
                "* **Síntomas:** Grandes lesiones alargadas (en forma de cigarro) de color marrón grisáceo.\n"
                "* **Manejo Cultural:** Rotación de cultivos por al menos un año, arado profundo para enterrar rastrojos infectados y uso de híbridos resistentes.\n"
                "* **Manejo Químico:** Fungicidas foliares si se alcanza el umbral de daño económico antes de la floración."
            )
        },
        'gray': {
            'keywords': ['mancha', 'gris', 'cercospora'],
            'corpus': (
                "mancha gris cercospora zeae maydis lesiones rectangulares venas grises "
                "labranza mínima rotación gramíneas fungicidas sistémicos vegetativo "
                "síntomas diagnóstico tratamiento prevención"
            ),
            'response': (
                "**Mancha Gris (Cercospora zeae-maydis):**\n"
                "* **Síntomas:** Lesiones rectangulares delimitadas por las venas de las hojas, de color grisáceo.\n"
                "* **Manejo Cultural:** Reducir la labranza mínima si hay antecedentes de la enfermedad, rotación de cultivos con especies no gramíneas.\n"
                "* **Manejo Químico:** Aplicación oportuna de fungicidas sistémicos en etapas vegetativas críticas."
            )
        },
        'healthy': {
            'keywords': ['sano', 'saludable', 'prevencion', 'prevenir'],
            'corpus': (
                "hoja sana saludable prevención monitoreo fitosanitario fertilización nitrógeno "
                "fósforo potasio densidad siembra humedad suelo riego aspersión maíz cultivo"
            ),
            'response': (
                "**Prevención y Hojas Sanas:**\n"
                "Para mantener las hojas de maíz saludables y libres de patógenos:\n"
                "1. Realizar monitoreo fitosanitario preventivo cada 7-10 días.\n"
                "2. Asegurar una fertilización balanceada (nitrógeno, fósforo y potasio) y una densidad de siembra adecuada.\n"
                "3. Mantener un control adecuado de la humedad del suelo y evitar riegos por aspersión tardíos."
            )
        },
        'automl': {
            'keywords': ['automl', 'tabular', 'datos', 'clasificacion', 'stacking', 'voting', 'random forest', 'mlp', 'regresion logistica'],
            'corpus': (
                "automl tabular pipeline datos clasificación modelos regresión logística random forest "
                "red neuronal mlp perceptrón votación stacking gradient boosting validación cruzada "
                "hiperparámetros tuning"
            ),
            'response': (
                "**Módulo AutoML Tabular:**\n"
                "Este módulo entrena y evalúa 5 modelos sobre variables del cultivo:\n"
                "1. **Regresión Logística:** Modelo lineal base.\n"
                "2. **Random Forest:** Conjunto de árboles de decisión robusto.\n"
                "3. **Red Neuronal MLP:** Perceptrón multicapa para patrones complejos.\n"
                "4. **Híbrido Votación (Voting):** Promedia las decisiones de Random Forest y MLP.\n"
                "5. **Híbrido Stacking:** Combina las predicciones usando un meta-aprendiz de Gradient Boosting.\n"
                "El sistema realiza validación cruzada de 5 pliegues y pruebas estadísticas automáticas (Shapiro-Wilk → ANOVA o Friedman, T-Student o Wilcoxon, Bootstrap CI)."
            )
        },
        'stats': {
            'keywords': ['estadistica', 'estadísticas', 'shapiro', 'anova', 'friedman', 'wilcoxon', 'bootstrap', 'prueba', 'significancia', 'normalidad'],
            'corpus': (
                "pruebas estadísticas shapiro wilk normalidad anova friedman comparación modelos "
                "t-student wilcoxon pareado bootstrap intervalo confianza significancia paramétrico "
                "no paramétrico post-hoc nemenyi tukey p-valor alpha hipótesis"
            ),
            'response': (
                "**Pruebas Estadísticas del Panel Tabular:**\n"
                "El sistema aplica un protocolo estadístico automático de 4 pasos:\n"
                "1. **Shapiro-Wilk:** Evalúa si las métricas de cada modelo siguen una distribución normal.\n"
                "2. **ANOVA o Friedman:** Si todos los modelos son normales → ANOVA de una vía; si alguno no lo es → Friedman (no paramétrico) con post-hoc de Nemenyi.\n"
                "3. **T-Student o Wilcoxon:** Comparación por pares contra el mejor modelo, con selección automática según normalidad.\n"
                "4. **Bootstrap CI (n=1000):** Calcula intervalos de confianza del 95% para Accuracy y F1-Score de cada modelo."
            )
        },
        'tinyml': {
            'keywords': ['tinyml', 'c++', 'c', 'microcontrolador', 'arduino', 'esp32', 'stm32', 'edge'],
            'corpus': (
                "tinyml exportar C c header microcontrolador arduino esp32 stm32 edge "
                "sin conexión campo bajo recursos tflite cuantización modelo"
            ),
            'response': (
                "**Integración con TinyML / C++:**\n"
                "El sistema permite exportar los árboles de decisión o modelos de Random Forest entrenados directamente a código fuente C/C++.\n"
                "Esto facilita su compilación e implementación en microcontroladores de bajos recursos (como Arduino, ESP32 o STM32) para realizar diagnósticos fitosanitarios directamente en campo sin conexión a internet."
            )
        },
        'credentials': {
            'keywords': ['credenciales', 'usuario', 'contraseña', 'login', 'acceso', 'admin'],
            'corpus': (
                "credenciales usuario contraseña login acceso admin autenticación sistema panel"
            ),
            'response': (
                "**Credenciales del Sistema:**\n"
                "El acceso al panel está protegido con autenticación segura (bcrypt + JWT).\n"
                "Las credenciales de demostración por defecto son:\n"
                "* **Usuario:** `admin`\n"
                "* **Contraseña:** `admin123`"
            )
        },
        'consensus': {
            'keywords': ['consenso', 'mobilenet', 'resnet', 'efficientnet', 'modelos', 'sin consenso'],
            'corpus': (
                "consenso modelos CNN visión mobilenet resnet efficientnet clasificación hojas "
                "mayoría sin consenso diagnóstico foliar imagen convolucional"
            ),
            'response': (
                "**Consenso de Modelos CNN (Diagnóstico Foliar):**\n"
                "Para la clasificación de imágenes de hojas se utilizan 3 redes convolucionales avanzadas (MobileNetV2, ResNet50 y EfficientNetB0).\n"
                "El diagnóstico se valida mediante consenso mayoritario (al menos 2 modelos deben coincidir). Si los 3 modelos predicen clases diferentes, se muestra una alerta de 'Sin consenso'."
            )
        }
    },
    'en': {
        'rust': {
            'keywords': ['rust', 'puccinia', 'sorghi'],
            'corpus': (
                "common rust puccinia sorghi symptoms pustules brown reddish leaf treatment "
                "fungicide triazoles strobilurins cultural management resistant hybrids humidity weeds prevention"
            ),
            'response': (
                "**Common Rust (Puccinia sorghi):**\n"
                "* **Symptoms:** Circular or elongated reddish-brown pustules on both leaf surfaces.\n"
                "* **Cultural Control:** Eliminate host weeds, avoid excess humidity, and plant resistant hybrids.\n"
                "* **Chemical Control:** Preventive foliar fungicide application (triazoles or strobilurins) when the first pustules are observed."
            )
        },
        'blight': {
            'keywords': ['blight', 'exserohilum', 'turcicum'],
            'corpus': (
                "northern leaf blight exserohilum turcicum lesions elongated cigar grayish brown "
                "crop rotation plowing residue fungicides economic threshold flowering resistant"
            ),
            'response': (
                "**Northern Leaf Blight (Exserohilum turcicum):**\n"
                "* **Symptoms:** Large, elongated (cigar-shaped) grayish-brown lesions.\n"
                "* **Cultural Control:** Crop rotation for at least one year, deep plowing to bury infected crop residues, and use of resistant hybrids.\n"
                "* **Chemical Control:** Foliar fungicides if the economic threshold is reached before flowering."
            )
        },
        'gray': {
            'keywords': ['gray', 'grey', 'spot', 'cercospora'],
            'corpus': (
                "gray leaf spot cercospora zeae maydis rectangular lesions veins grayish "
                "minimum tillage rotation grasses systemic fungicides vegetative treatment"
            ),
            'response': (
                "**Gray Leaf Spot (Cercospora zeae-maydis):**\n"
                "* **Symptoms:** Rectangular grayish-brown lesions restricted by leaf veins.\n"
                "* **Cultural Control:** Reduce minimum tillage if the disease history exists, rotate crops with non-grass species.\n"
                "* **Chemical Control:** Timely application of systemic fungicides in critical vegetative stages."
            )
        },
        'healthy': {
            'keywords': ['healthy', 'sane', 'prevention', 'prevent'],
            'corpus': (
                "healthy leaf prevention phytosanitary monitoring fertilization nitrogen phosphorus "
                "potassium planting density soil moisture irrigation maize crop"
            ),
            'response': (
                "**Prevention and Healthy Leaves:**\n"
                "To keep maize leaves healthy and pathogen-free:\n"
                "1. Perform preventive phytosanitary monitoring every 7-10 days.\n"
                "2. Ensure balanced fertilization (N, P, K) and adequate planting density.\n"
                "3. Maintain proper soil moisture control and avoid late overhead sprinkler irrigation."
            )
        },
        'automl': {
            'keywords': ['automl', 'tabular', 'data', 'stacking', 'voting', 'random forest', 'mlp', 'logistic regression'],
            'corpus': (
                "automl tabular pipeline data classification models logistic regression random forest "
                "neural network mlp voting stacking gradient boosting cross-validation hyperparameter tuning"
            ),
            'response': (
                "**AutoML Tabular Module:**\n"
                "This module trains and evaluates 5 models on crop features:\n"
                "1. **Logistic Regression:** Base linear model.\n"
                "2. **Random Forest:** Robust decision tree ensemble.\n"
                "3. **MLP Neural Network:** Multi-layer perceptron for complex patterns.\n"
                "4. **Voting Hybrid:** Averages predictions from Random Forest and MLP.\n"
                "5. **Stacking Hybrid:** Combines predictions using a Gradient Boosting meta-learner.\n"
                "The system performs 5-fold cross-validation and automatic statistical tests (Shapiro-Wilk → ANOVA or Friedman, T-Student or Wilcoxon, Bootstrap CI)."
            )
        },
        'stats': {
            'keywords': ['statistics', 'shapiro', 'anova', 'friedman', 'wilcoxon', 'bootstrap', 'test', 'significance', 'normality'],
            'corpus': (
                "statistical tests shapiro wilk normality anova friedman model comparison "
                "t-student wilcoxon paired bootstrap confidence interval significance parametric "
                "non-parametric post-hoc nemenyi tukey p-value alpha hypothesis"
            ),
            'response': (
                "**Statistical Tests (Tabular Panel):**\n"
                "The system applies an automatic 4-step statistical protocol:\n"
                "1. **Shapiro-Wilk:** Checks if each model's metrics follow a normal distribution.\n"
                "2. **ANOVA or Friedman:** If all models are normal → One-way ANOVA; otherwise → Friedman (non-parametric) with Nemenyi post-hoc.\n"
                "3. **T-Student or Wilcoxon:** Pairwise comparison against the best model, selected automatically per pair.\n"
                "4. **Bootstrap CI (n=1000):** Computes 95% confidence intervals for Accuracy and F1-Score of each model."
            )
        },
        'tinyml': {
            'keywords': ['tinyml', 'c++', 'c', 'microcontroller', 'arduino', 'esp32', 'stm32', 'edge'],
            'corpus': (
                "tinyml export C c header microcontroller arduino esp32 stm32 edge "
                "offline field low resources tflite quantization model"
            ),
            'response': (
                "**TinyML / C++ Integration:**\n"
                "The system allows exporting trained decision trees or Random Forest models directly into C/C++ source code.\n"
                "This enables compilation and deployment on resource-constrained microcontrollers (like Arduino, ESP32, or STM32) for offline field diagnostics."
            )
        },
        'credentials': {
            'keywords': ['credentials', 'username', 'password', 'login', 'access', 'admin'],
            'corpus': (
                "credentials username password login access admin authentication system panel"
            ),
            'response': (
                "**System Credentials:**\n"
                "Access is protected with secure authentication (bcrypt + JWT).\n"
                "Default demonstration credentials are:\n"
                "* **Username:** `admin`\n"
                "* **Password:** `admin123`"
            )
        },
        'consensus': {
            'keywords': ['consensus', 'mobilenet', 'resnet', 'efficientnet', 'models', 'no consensus'],
            'corpus': (
                "consensus CNN models vision mobilenet resnet efficientnet classification leaves "
                "majority no consensus foliar diagnosis image convolutional"
            ),
            'response': (
                "**CNN Model Consensus (Foliar Diagnosis):**\n"
                "Image classification uses 3 advanced convolutional networks (MobileNetV2, ResNet50, and EfficientNetB0).\n"
                "The diagnosis is validated by majority consensus (at least 2 models must agree). If all 3 models predict different classes, a 'No consensus' warning is triggered."
            )
        }
    },
    'pt': {
        'rust': {
            'keywords': ['ferrugem', 'roia', 'roña', 'puccinia', 'sorghi'],
            'corpus': (
                "ferrugem comum puccinia sorghi sintomas pústulas marrom avermelhado folha "
                "fungicidas triazóis estrobirulinas manejo cultural híbridos resistentes humidade"
            ),
            'response': (
                "**Ferrugem Comum (Puccinia sorghi):**\n"
                "* **Sintomas:** Pústulas circulares ou alongadas de cor marrom-avermelhada em ambas as faces da folha.\n"
                "* **Manejo Cultural:** Eliminar ervas daninhas hospedeiras, evitar excesso de umidade e plantar híbridos resistentes.\n"
                "* **Manejo Químico:** Aplicação foliar preventiva de fungicidas à base de triazóis ou estrobirulinas ao observar as primeiras pústulas."
            )
        },
        'blight': {
            'keywords': ['helmintosporiose', 'tizon', 'tizón', 'exserohilum', 'turcicum'],
            'corpus': (
                "helmintosporiose tizón norte exserohilum turcicum lesões alongadas charuto acinzentado "
                "rotação culturas aração resíduos fungicidas limiar dano florescimento resistente"
            ),
            'response': (
                "**Helmintosporiose / Tizón do Norte (Exserohilum turcicum):**\n"
                "* **Sintomas:** Grandes lesões alongadas (em forma de charuto) de cor marrom-acinzentada.\n"
                "* **Manejo Cultural:** Rotação de culturas por pelo menos um ano, aração profunda para enterrar restos de culturas infectados e uso de híbridos resistentes.\n"
                "* **Manejo Químico:** Fungicidas foliares se o limiar de dano econômico for atingido antes do florescimento."
            )
        },
        'gray': {
            'keywords': ['cercospora', 'mancha', 'gris', 'cinzenta'],
            'corpus': (
                "mancha cercospora zeae maydis lesões retangulares nervuras acinzentadas "
                "plantio direto rotação gramíneas fungicidas sistêmicos vegetativo"
            ),
            'response': (
                "**Mancha de Cercospora (Cercospora zeae-maydis):**\n"
                "* **Sintomas:** Lesões retangulares acinzentadas delimitadas pelas nervuras das folhas.\n"
                "* **Manejo Cultural:** Reduzir o plantio direto se houver histórico da doença, rotação de culturas com espécies não gramíneas.\n"
                "* **Manejo Químico:** Aplicação oportuna de fungicidas sistêmicos em estágios vegetativos críticos."
            )
        },
        'healthy': {
            'keywords': ['saudavel', 'saudável', 'sano', 'prevencao', 'prevenção'],
            'corpus': (
                "folha saudável prevenção monitoramento fitossanitário fertilização nitrogênio "
                "fósforo potássio densidade solo umidade irrigação milho"
            ),
            'response': (
                "**Prevenção e Folhas Saudáveis:**\n"
                "Para manter as folhas de milho saudáveis e livres de patógenos:\n"
                "1. Realizar monitoramento fitossanitário preventivo a cada 7-10 dias.\n"
                "2. Garantir fertilização equilibrada (N, P, K) e densidade de plantio adequada.\n"
                "3. Manter o controle adequado da umidade do solo e evitar irrigações por aspersão tardias."
            )
        },
        'automl': {
            'keywords': ['automl', 'tabular', 'dados', 'stacking', 'voting', 'random forest', 'mlp', 'regressao logistica'],
            'corpus': (
                "automl tabular pipeline dados classificação modelos regressão logística random forest "
                "rede neural mlp votação stacking gradient boosting validação cruzada hiperparâmetros"
            ),
            'response': (
                "**Módulo AutoML Tabular:**\n"
                "Este módulo treina e avalia 5 modelos sobre características de cultivo:\n"
                "1. **Regressão Logística:** Modelo linear básico.\n"
                "2. **Random Forest:** Conjunto robusto de árvores de decisão.\n"
                "3. **Rede Neural MLP:** Perceptron multicamada para padrões complexos.\n"
                "4. **Híbrido Votação (Voting):** Média das previsões do Random Forest e MLP.\n"
                "5. **Híbrido Stacking:** Combina as previsões usando um meta-aprendiz de Gradient Boosting.\n"
                "O sistema realiza validação cruzada de 5 dobras e testes estatísticos automáticos (Shapiro-Wilk → ANOVA ou Friedman, T-Student ou Wilcoxon, Bootstrap CI)."
            )
        },
        'stats': {
            'keywords': ['estatistica', 'estatísticas', 'shapiro', 'anova', 'friedman', 'wilcoxon', 'bootstrap', 'teste', 'significância'],
            'corpus': (
                "testes estatísticos shapiro wilk normalidade anova friedman comparação modelos "
                "t-student wilcoxon pareado bootstrap intervalo confiança significância paramétrico "
                "não paramétrico post-hoc nemenyi tukey p-valor alpha hipótese"
            ),
            'response': (
                "**Testes Estatísticos (Painel Tabular):**\n"
                "O sistema aplica um protocolo estatístico automático de 4 etapas:\n"
                "1. **Shapiro-Wilk:** Verifica se as métricas de cada modelo seguem distribuição normal.\n"
                "2. **ANOVA ou Friedman:** Se todos normais → ANOVA; caso contrário → Friedman com post-hoc Nemenyi.\n"
                "3. **T-Student ou Wilcoxon:** Comparação par-a-par com o melhor modelo, selecionada automaticamente.\n"
                "4. **Bootstrap CI (n=1000):** Intervalos de confiança de 95% para Acurácia e F1-Score."
            )
        },
        'tinyml': {
            'keywords': ['tinyml', 'c++', 'c', 'microcontrolador', 'arduino', 'esp32', 'stm32', 'edge'],
            'corpus': (
                "tinyml exportar C c header microcontrolador arduino esp32 stm32 edge "
                "sem conexão campo baixo custo tflite quantização modelo"
            ),
            'response': (
                "**Integração con TinyML / C++:**\n"
                "O sistema permite exportar as árvores de decisão ou modelos de Random Forest treinados diretamente para código C/C++.\n"
                "Isso facilita sua compilação e implementação em microcontroladores de baixo custo (como Arduino, ESP32 ou STM32) para diagnósticos fitossanitários diretamente em campo sem conexão à internet."
            )
        },
        'credentials': {
            'keywords': ['credenciais', 'usuario', 'senha', 'login', 'acesso', 'admin'],
            'corpus': (
                "credenciais usuário senha login acesso admin autenticação sistema painel"
            ),
            'response': (
                "**Credenciais do Sistema:**\n"
                "O acesso ao painel é protegido com autenticação segura (bcrypt + JWT).\n"
                "As credenciais de demonstração padrão são:\n"
                "* **Usuário:** `admin`\n"
                "* **Senha:** `admin123`"
            )
        },
        'consensus': {
            'keywords': ['consenso', 'mobilenet', 'resnet', 'efficientnet', 'modelos', 'sem consenso'],
            'corpus': (
                "consenso modelos CNN visão mobilenet resnet efficientnet classificação folhas "
                "maioria sem consenso diagnóstico foliar imagem convolucional"
            ),
            'response': (
                "**Consenso de Modelos CNN (Diagnóstico Foliar):**\n"
                "Para classificação de imagens de folhas, são usadas 3 redes convolucionais avançadas (MobileNetV2, ResNet50 e EfficientNetB0).\n"
                "O diagnóstico é validado por consenso maioritario (pelo menos 2 modelos devem concordar). Se todos os 3 preverem classes diferentes, um alerta de 'Sem consenso' é disparado."
            )
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
#  MOTOR RAG  (TF-IDF + cosine similarity — inicializado por idioma)
# ─────────────────────────────────────────────────────────────────────────────
_rag_engines: dict = {}

def _get_rag_engine(lang: str):
    """Construye o recupera el vectorizador TF-IDF para el idioma dado."""
    if not _SKLEARN_AVAILABLE:
        return None
    if lang not in _rag_engines:
        kb = KNOWLEDGE_BASE.get(lang, KNOWLEDGE_BASE['es'])
        topics = list(kb.keys())
        corpus = [kb[t]['corpus'] for t in topics]
        vectorizer = TfidfVectorizer(
            analyzer='word', ngram_range=(1, 2),
            max_df=1.0, min_df=1
        )
        vectors = vectorizer.fit_transform(corpus)
        _rag_engines[lang] = {
            'topics': topics,
            'vectorizer': vectorizer,
            'vectors': vectors,
        }
    return _rag_engines[lang]


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def get_chatbot_response(user_query: str, lang: str = 'es') -> str:
    """
    Retorna la respuesta más relevante para ``user_query`` en el idioma ``lang``.

    Estrategia híbrida:
      1. TF-IDF cosine similarity  →  si score ≥ RAG_THRESHOLD → respuesta RAG.
      2. Keyword matching exacto   →  fallback.
      3. Respuesta genérica        →  si nada coincide.
    """
    RAG_THRESHOLD = 0.18
    lang_key = lang if lang in ('es', 'en', 'pt') else 'es'
    kb = KNOWLEDGE_BASE[lang_key]
    query_clean = re.sub(r'[^\w\s]', '', user_query.lower())

    # ── 1. RAG semántico ────────────────────────────────────────────────────
    engine = _get_rag_engine(lang_key)
    if engine is not None:
        try:
            query_vec = engine['vectorizer'].transform([query_clean])
            scores = cosine_similarity(query_vec, engine['vectors'])[0]
            best_idx = int(np.argmax(scores))
            if float(scores[best_idx]) >= RAG_THRESHOLD:
                return kb[engine['topics'][best_idx]]['response']
        except Exception:
            pass  # Si falla el RAG, continúa al fallback

    # ── 2. Fallback de keywords ─────────────────────────────────────────────
    for area, data in kb.items():
        for kw in data['keywords']:
            if kw in query_clean:
                return data['response']

    # ── 3. Respuesta genérica ───────────────────────────────────────────────
    defaults = {
        'es': (
            "Lo siento, no he podido identificar el tema de tu pregunta. "
            "Puedo ayudarte con información sobre:\n"
            "* **Patologías del maíz:** Roya común, Tizón del norte, Mancha gris o prevención.\n"
            "* **AutoML Tabular:** Los 5 modelos y pruebas estadísticas.\n"
            "* **TinyML:** Código C/C++ para microcontroladores.\n"
            "* **Credenciales** del sistema o el **consenso** de modelos CNN."
        ),
        'en': (
            "Sorry, I couldn't identify the topic of your question. "
            "I can help you with information about:\n"
            "* **Maize diseases:** Common rust, Northern leaf blight, Gray leaf spot, or prevention.\n"
            "* **AutoML Tabular:** The 5 models and statistical tests.\n"
            "* **TinyML:** C/C++ code for microcontrollers.\n"
            "* **Credentials** of the system or **consensus** of CNN models."
        ),
        'pt': (
            "Desculpe, não consegui identificar o assunto da sua pergunta. "
            "Posso te ajudar com informações sobre:\n"
            "* **Doenças do milho:** Ferrugem comum, Helmintosporiose, Mancha de cercospora ou prevenção.\n"
            "* **AutoML Tabular:** Os 5 modelos e testes estatísticos.\n"
            "* **TinyML:** Código C/C++ para microcontroladores.\n"
            "* **Credenciais** do sistema ou o **consenso** dos modelos CNN."
        )
    }
    return defaults[lang_key]
