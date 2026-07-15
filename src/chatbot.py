# -*- coding: utf-8 -*-
import re

# Base de conocimiento estructurada para el Chatbot
KNOWLEDGE_BASE = {
    'es': {
        'rust': {
            'keywords': ['roya', 'roña', 'puccinia', 'sorghi'],
            'response': (
                "**Roya Común (Puccinia sorghi):**\n"
                "* **Síntomas:** Pústulas circulares o alargadas de color marrón rojizo en ambas caras de la hoja.\n"
                "* **Manejo Cultural:** Eliminar malezas hospederas, evitar exceso de humedad y sembrar híbridos resistentes.\n"
                "* **Manejo Químico:** Aplicación foliar preventiva de fungicidas a base de triazoles o estrobirulinas al observar las primeras pústulas."
            )
        },
        'blight': {
            'keywords': ['tizon', 'tizón', 'exserohilum', 'turcicum'],
            'response': (
                "**Tizón del Norte (Exserohilum turcicum):**\n"
                "* **Síntomas:** Grandes lesiones alargadas (en forma de cigarro) de color marrón grisáceo.\n"
                "* **Manejo Cultural:** Rotación de cultivos por al menos un año, arado profundo para enterrar rastrojos infectados y uso de híbridos resistentes.\n"
                "* **Manejo Químico:** Fungicidas foliares si se alcanza el umbral de daño económico antes de la floración."
            )
        },
        'gray': {
            'keywords': ['mancha', 'gris', 'cercospora'],
            'response': (
                "**Mancha Gris (Cercospora zeae-maydis):**\n"
                "* **Síntomas:** Lesiones rectangulares delimitadas por las venas de las hojas, de color grisáceo.\n"
                "* **Manejo Cultural:** Reducir la labranza mínima si hay antecedentes de la enfermedad, rotación de cultivos con especies no gramíneas.\n"
                "* **Manejo Químico:** Aplicación oportuna de fungicidas sistémicos en etapas vegetativas críticas."
            )
        },
        'healthy': {
            'keywords': ['sano', 'saludable', 'prevencion', 'prevenir'],
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
            'response': (
                "**Módulo AutoML Tabular:**\n"
                "Este módulo entrena y evalúa 5 modelos sobre variables del cultivo:\n"
                "1. **Regresión Logística:** Modelo lineal base.\n"
                "2. **Random Forest:** Conjunto de árboles de decisión robusto.\n"
                "3. **Red Neuronal MLP:** Perceptrón multicapa para patrones complejos.\n"
                "4. **Híbrido Votación (Voting):** Promedia las decisiones de Random Forest y MLP.\n"
                "5. **Híbrido Stacking:** Combina las predicciones usando un meta-aprendiz de Gradient Boosting.\n"
                "El sistema realiza validación cruzada de 5 pliegues y pruebas de significancia estadística (Wilcoxon y McNemar)."
            )
        },
        'tinyml': {
            'keywords': ['tinyml', 'c++', 'c', 'microcontrolador', 'arduino', 'esp32', 'stm32', 'edge'],
            'response': (
                "**Integración con TinyML / C++:**\n"
                "El sistema permite exportar los árboles de decisión o modelos de Random Forest entrenados directamente a código fuente C/C++.\n"
                "Esto facilita su compilación e implementación en microcontroladores de bajos recursos (como Arduino, ESP32 o STM32) para realizar diagnósticos fitosanitarios directamente en campo sin conexión a internet."
            )
        },
        'credentials': {
            'keywords': ['credenciales', 'usuario', 'contraseña', 'login', 'acceso', 'admin'],
            'response': (
                "**Credenciales del Sistema:**\n"
                "El acceso al panel está protegido. Las credenciales de demostración por defecto son:\n"
                "* **Usuario:** `admin`\n"
                "* **Contraseña:** `admin123`"
            )
        },
        'consensus': {
            'keywords': ['consenso', 'mobilenet', 'resnet', 'efficientnet', 'modelos', 'sin consenso'],
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
            'response': (
                "**Common Rust (Puccinia sorghi):**\n"
                "* **Symptoms:** Circular or elongated reddish-brown pustules on both leaf surfaces.\n"
                "* **Cultural Control:** Eliminate host weeds, avoid excess humidity, and plant resistant hybrids.\n"
                "* **Chemical Control:** Preventive foliar fungicide application (triazoles or strobilurins) when the first pustules are observed."
            )
        },
        'blight': {
            'keywords': ['blight', 'exserohilum', 'turcicum'],
            'response': (
                "**Northern Leaf Blight (Exserohilum turcicum):**\n"
                "* **Symptoms:** Large, elongated (cigar-shaped) grayish-brown lesions.\n"
                "* **Cultural Control:** Crop rotation for at least one year, deep plowing to bury infected crop residues, and use of resistant hybrids.\n"
                "* **Chemical Control:** Foliar fungicides if the economic threshold is reached before flowering."
            )
        },
        'gray': {
            'keywords': ['gray', 'grey', 'spot', 'cercospora'],
            'response': (
                "**Gray Leaf Spot (Cercospora zeae-maydis):**\n"
                "* **Symptoms:** Rectangular grayish-brown lesions restricted by leaf veins.\n"
                "* **Cultural Control:** Reduce minimum tillage if the disease history exists, rotate crops with non-grass species.\n"
                "* **Chemical Control:** Timely application of systemic fungicides in critical vegetative stages."
            )
        },
        'healthy': {
            'keywords': ['healthy', 'sane', 'prevention', 'prevent'],
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
            'response': (
                "**AutoML Tabular Module:**\n"
                "This module trains and evaluates 5 models on crop features:\n"
                "1. **Logistic Regression:** Base linear model.\n"
                "2. **Random Forest:** Robust decision tree ensemble.\n"
                "3. **MLP Neural Network:** Multi-layer perceptron for complex patterns.\n"
                "4. **Voting Hybrid:** Averages predictions from Random Forest and MLP.\n"
                "5. **Stacking Hybrid:** Combines predictions using a Gradient Boosting meta-learner.\n"
                "The system performs 5-fold cross-validation and statistical significance tests (Wilcoxon and McNemar)."
            )
        },
        'tinyml': {
            'keywords': ['tinyml', 'c++', 'c', 'microcontroller', 'arduino', 'esp32', 'stm32', 'edge'],
            'response': (
                "**TinyML / C++ Integration:**\n"
                "The system allows exporting trained decision trees or Random Forest models directly into C/C++ source code.\n"
                "This enables compilation and deployment on resource-constrained microcontrollers (like Arduino, ESP32, or STM32) for offline field diagnostics."
            )
        },
        'credentials': {
            'keywords': ['credentials', 'username', 'password', 'login', 'access', 'admin'],
            'response': (
                "**System Credentials:**\n"
                "Access is protected. Default demonstration credentials are:\n"
                "* **Username:** `admin`\n"
                "* **Password:** `admin123`"
            )
        },
        'consensus': {
            'keywords': ['consensus', 'mobilenet', 'resnet', 'efficientnet', 'models', 'no consensus'],
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
            'response': (
                "**Ferrugem Comum (Puccinia sorghi):**\n"
                "* **Sintomas:** Pústulas circulares ou alongadas de cor marrom-avermelhada em ambas as faces da folha.\n"
                "* **Manejo Cultural:** Eliminar ervas daninhas hospedeiras, evitar excesso de umidade e plantar híbridos resistentes.\n"
                "* **Manejo Químico:** Aplicação foliar preventiva de fungicidas à base de triazóis ou estrobirulinas ao observar as primeiras pústulas."
            )
        },
        'blight': {
            'keywords': ['helmintosporiose', 'tizon', 'tizón', 'exserohilum', 'turcicum'],
            'response': (
                "**Helmintosporiose / Tizón do Norte (Exserohilum turcicum):**\n"
                "* **Sintomas:** Grandes lesões alongadas (em forma de charuto) de cor marrom-acinzentada.\n"
                "* **Manejo Cultural:** Rotação de culturas por pelo menos um ano, aração profunda para enterrar restos de culturas infectados e uso de híbridos resistentes.\n"
                "* **Manejo Químico:** Fungicidas foliares se o limiar de dano econômico for atingido antes do florescimento."
            )
        },
        'gray': {
            'keywords': ['cercospora', 'mancha', 'gris', 'cinzenta'],
            'response': (
                "**Mancha de Cercospora (Cercospora zeae-maydis):**\n"
                "* **Sintomas:** Lesões retangulares acinzentadas delimitadas pelas nervuras das folhas.\n"
                "* **Manejo Cultural:** Reduzir o plantio direto se houver histórico da doença, rotação de culturas com espécies não gramíneas.\n"
                "* **Manejo Químico:** Aplicação oportuna de fungicidas sistêmicos em estágios vegetativos críticos."
            )
        },
        'healthy': {
            'keywords': ['saudavel', 'saudável', 'sano', 'prevencao', 'prevenção'],
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
            'response': (
                "**Módulo AutoML Tabular:**\n"
                "Este módulo treina e avalia 5 modelos sobre características de cultivo:\n"
                "1. **Regressão Logística:** Modelo linear básico.\n"
                "2. **Random Forest:** Conjunto robusto de árvores de decisão.\n"
                "3. **Rede Neural MLP:** Perceptron multicamada para padrões complexos.\n"
                "4. **Híbrido Votação (Voting):** Média das previsões do Random Forest e MLP.\n"
                "5. **Híbrido Stacking:** Combina as previsões usando um meta-aprendiz de Gradient Boosting.\n"
                "O sistema realiza validação cruzada de 5 dobras e testes de significância estatística (Wilcoxon e McNemar)."
            )
        },
        'tinyml': {
            'keywords': ['tinyml', 'c++', 'c', 'microcontrolador', 'arduino', 'esp32', 'stm32', 'edge'],
            'response': (
                "**Integração con TinyML / C++:**\n"
                "O sistema permite exportar as árvores de decisão ou modelos de Random Forest treinados diretamente para código C/C++.\n"
                "Isso facilita sua compilação e implementação em microcontroladores de baixo custo (como Arduino, ESP32 ou STM32) para diagnósticos fitossanitários diretamente em campo sem conexão à internet."
            )
        },
        'credentials': {
            'keywords': ['credenciais', 'usuario', 'senha', 'login', 'acesso', 'admin'],
            'response': (
                "**Credenciais do Sistema:**\n"
                "O acesso ao panel é protegido. As credenciais de demonstração padrão são:\n"
                "* **Usuário:** `admin`\n"
                "* **Senha:** `admin123`"
            )
        },
        'consensus': {
            'keywords': ['consenso', 'mobilenet', 'resnet', 'efficientnet', 'modelos', 'sem consenso'],
            'response': (
                "**Consenso de Modelos CNN (Diagnóstico Foliar):**\n"
                "Para classificação de imagens de folhas, são usadas 3 redes convolucionais avançadas (MobileNetV2, ResNet50 e EfficientNetB0).\n"
                "O diagnóstico é validado por consenso maioritario (pelo menos 2 modelos devem concordar). Se todos os 3 preverem classes diferentes, um alerta de 'Sem consenso' é disparado."
            )
        }
    }
}

def get_chatbot_response(user_query, lang='es'):
    """
    Analiza la consulta del usuario y retorna una respuesta basada en la base de conocimiento.
    """
    lang_key = lang if lang in ['es', 'en', 'pt'] else 'es'
    query_clean = re.sub(r'[^\w\s]', '', user_query.lower())
    
    # Buscar coincidencia de palabras clave
    for area, data in KNOWLEDGE_BASE[lang_key].items():
        for kw in data['keywords']:
            if kw in query_clean:
                return data['response']
                
    # Respuesta por defecto si no se encuentra coincidencia
    default_responses = {
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
    
    return default_responses[lang_key]
