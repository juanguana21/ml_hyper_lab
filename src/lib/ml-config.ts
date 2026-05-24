/**
 * ML HyperLab - Configuración de algoritmos e hiperparámetros
 * Cada algoritmo tiene su configuración con explicaciones didácticas en español.
 */

export interface HyperparamConfig {
  key: string;
  label: string;
  pythonParam: string;
  pythonLib: string;
  type: "slider" | "select" | "number";
  min?: number;
  max?: number;
  step?: number;
  defaultValue: number | string;
  options?: { value: string | number; label: string }[];
  explanation: string;
  tip: string;
  impact: "alto" | "medio" | "bajo";
}

export interface AlgorithmConfig {
  id: string;
  name: string;
  shortName: string;
  emoji: string;
  color: string;
  description: string;
  concept: string;
  whenToUse: string;
  pros: string[];
  cons: string[];
  hyperparams: HyperparamConfig[];
}

export const ALGORITHMS: AlgorithmConfig[] = [
  {
    id: "xgboost",
    name: "XGBoost",
    shortName: "XGB",
    emoji: "🚀",
    color: "#10b981",
    description:
      "Extreme Gradient Boosting. Uno de los algoritmos más potentes para competencias de ML. Construye árboles secuencialmente donde cada árbol corrige los errores del anterior.",
    concept:
      "XGBoost funciona como un equipo donde cada miembro aprende de los errores del anterior. El primer árbol hace predicciones, el segundo intenta corregir los errores del primero, el tercero corrige los del segundo, y así sucesivamente. Cada nuevo árbol se enfoca en los ejemplos que son más difíciles de clasificar.",
    whenToUse:
      "Cuando necesitas alto rendimiento predictivo, tienes datos tabulares, y puedes permitirte un modelo más complejo. Ganador frecuente en competencias de Kaggle.",
    pros: [
      "Alto rendimiento predictivo",
      "Maneja datos faltantes automáticamente",
      "Regularización integrada contra overfitting",
      "Rápido y eficiente",
    ],
    cons: [
      "Puede hacer overfitting si no se ajusta bien",
      "Muchos hiperparámetros para afinar",
      "Menos interpretable que un árbol simple",
    ],
    hyperparams: [
      {
        key: "n_estimators",
        label: "Número de árboles (n_estimators)",
        pythonParam: "n_estimators",
        pythonLib: "xgboost.XGBClassifier",
        type: "slider",
        min: 10,
        max: 500,
        step: 10,
        defaultValue: 100,
        explanation:
          "Define cuántos árboles secuenciales se construyen. Cada árbol nuevo intenta corregir los errores residuales del conjunto anterior. Más árboles = más capacidad de aprender patrones complejos, pero también más riesgo de sobreajuste y mayor tiempo de entrenamiento.",
        tip: "Empieza con 100 y aumenta gradualmente. Si el accuracy de entrenamiento es mucho mayor que el de test, reduce este valor.",
        impact: "alto",
      },
      {
        key: "max_depth",
        label: "Profundidad máxima (max_depth)",
        pythonParam: "max_depth",
        pythonLib: "xgboost.XGBClassifier",
        type: "slider",
        min: 1,
        max: 20,
        step: 1,
        defaultValue: 6,
        explanation:
          "Controla la profundidad de cada árbol individual. Árboles más profundos capturan relaciones más complejas pero también memorizan ruido. Una profundidad de 3-6 suele ser un buen punto de partida para la mayoría de problemas.",
        tip: "Valores bajos (3-5) producen modelos más robustos. Valores altos (>10) suelen causar sobreajuste.",
        impact: "alto",
      },
      {
        key: "learning_rate",
        label: "Tasa de aprendizaje (learning_rate)",
        pythonParam: "learning_rate",
        pythonLib: "xgboost.XGBClassifier",
        type: "slider",
        min: 0.01,
        max: 1.0,
        step: 0.01,
        defaultValue: 0.3,
        explanation:
          "Determina cuánto contribuye cada árbol nuevo a la predicción final. Una tasa baja significa que cada árbol aporta una corrección pequeña, requiriendo más árboles pero logrando mejor generalización. Una tasa alta hace que cada árbol tenga más influencia, converge más rápido pero puede ser inestable.",
        tip: "Usualmente se usa entre 0.01 y 0.3. Si usas learning_rate bajo, necesitas más n_estimators.",
        impact: "alto",
      },
      {
        key: "min_child_weight",
        label: "Peso mínimo del hijo (min_child_weight)",
        pythonParam: "min_child_weight",
        pythonLib: "xgboost.XGBClassifier",
        type: "slider",
        min: 1,
        max: 20,
        step: 1,
        defaultValue: 1,
        explanation:
          "Peso mínimo que debe tener un nodo hijo para crear una nueva división. Funciona como regularización: valores altos impiden que el modelo aprenda patrones muy específicos que solo aplican a pocas muestras. En esencia, controla la creación de particiones en regiones con pocos datos.",
        tip: "Aumentar este valor ayuda a prevenir sobreajuste en datasets pequeños o ruidosos.",
        impact: "medio",
      },
      {
        key: "subsample",
        label: "Submuestreo (subsample)",
        pythonParam: "subsample",
        pythonLib: "xgboost.XGBClassifier",
        type: "slider",
        min: 0.1,
        max: 1.0,
        step: 0.1,
        defaultValue: 1.0,
        explanation:
          "Fracción de muestras usadas para entrenar cada árbol. Con 0.8, cada árbol usa el 80% de los datos aleatoriamente. Esto agrega aleatoriedad que ayuda a prevenir sobreajuste, similar al concepto de bagging en Random Forest.",
        tip: "Valores entre 0.6 y 0.9 suelen mejorar la generalización. 1.0 usa todos los datos en cada árbol.",
        impact: "medio",
      },
      {
        key: "colsample_bytree",
        label: "Submuestreo de features (colsample_bytree)",
        pythonParam: "colsample_bytree",
        pythonLib: "xgboost.XGBClassifier",
        type: "slider",
        min: 0.1,
        max: 1.0,
        step: 0.1,
        defaultValue: 1.0,
        explanation:
          "Fracción de características (features) usadas para construir cada árbol. Si tienes 10 features y usas 0.8, cada árbol selecciona aleatoriamente 8 features. Esto reduce la correlación entre árboles y mejora la generalización, especialmente cuando hay features irrelevantes.",
        tip: "Útil cuando hay muchas features. Valores entre 0.5 y 0.8 suelen funcionar bien.",
        impact: "medio",
      },
      {
        key: "gamma",
        label: "Gamma (reducción mínima de pérdida)",
        pythonParam: "gamma",
        pythonLib: "xgboost.XGBClassifier",
        type: "slider",
        min: 0,
        max: 5,
        step: 0.1,
        defaultValue: 0,
        explanation:
          "Reducción mínima de pérdida requerida para hacer una nueva división en un nodo. Actúa como regularización: valores altos hacen que el algoritmo sea más conservador, requiriendo que cada división aporte una mejora significativa. Con gamma=0, cualquier mejora positiva genera una división.",
        tip: "Aumentar gamma simplifica el modelo. Útil para controlar sobreajuste.",
        impact: "medio",
      },
      {
        key: "reg_alpha",
        label: "Regularización L1 (reg_alpha)",
        pythonParam: "reg_alpha",
        pythonLib: "xgboost.XGBClassifier",
        type: "slider",
        min: 0,
        max: 10,
        step: 0.1,
        defaultValue: 0,
        explanation:
          "Penalización L1 (Lasso) aplicada a los pesos de las hojas. Fuerza a que algunos pesos sean exactamente cero, realizando selección de features automáticamente. Es útil cuando sospechas que muchas features son irrelevantes.",
        tip: "Empieza con 0 y aumenta gradualmente si ves sobreajuste. Valores típicos: 0, 0.1, 1, 10.",
        impact: "bajo",
      },
      {
        key: "reg_lambda",
        label: "Regularización L2 (reg_lambda)",
        pythonParam: "reg_lambda",
        pythonLib: "xgboost.XGBClassifier",
        type: "slider",
        min: 0,
        max: 10,
        step: 0.1,
        defaultValue: 1,
        explanation:
          "Penalización L2 (Ridge) aplicada a los pesos de las hojas. Reduce los pesos grandes sin hacerlos cero, suavizando el modelo. A diferencia de L1, L2 no elimina features pero reduce su influencia. El valor por defecto de 1 ya proporciona regularización moderada.",
        tip: "El valor por defecto de 1 suele funcionar bien. Aumentar si el modelo sobreajusta.",
        impact: "bajo",
      },
    ],
  },
  {
    id: "random_forest",
    name: "Random Forest",
    shortName: "RF",
    emoji: "🌲",
    color: "#22c55e",
    description:
      "Bosque Aleatorio. Construye múltiples árboles de decisión independientes y combina sus predicciones por votación. Cada árbol se entrena con una muestra aleatoria de datos y features.",
    concept:
      "Imagina que pides la opinión a 100 expertos para tomar una decisión, donde cada experto solo ve una parte aleatoria de la información. Al combinar todas las opiniones por votación mayoritaria, el resultado final es más robusto que cualquier experto individual. Eso es Random Forest: la sabiduría de la multitud aplicada a árboles de decisión.",
    whenToUse:
      "Como primer modelo para cualquier problema tabular. Excelente baseline, robusto, requiere poca afinación de hiperparámetros.",
    pros: [
      "Robusto y difícil de sobreajustar",
      "Poco sensible a hiperparámetros",
      "Paralelizable (árboles independientes)",
      "Proporciona importancia de features",
    ],
    cons: [
      "Menos potente que XGBoost en competencias",
      "Muchos árboles pueden ser lentos para predecir",
      "No extrapoliza fuera del rango de entrenamiento",
    ],
    hyperparams: [
      {
        key: "n_estimators",
        label: "Número de árboles (n_estimators)",
        pythonParam: "n_estimators",
        pythonLib: "sklearn.ensemble.RandomForestClassifier",
        type: "slider",
        min: 10,
        max: 500,
        step: 10,
        defaultValue: 100,
        explanation:
          "Cantidad de árboles independientes en el bosque. Cada árbol vota y la clase mayoritaria es la predicción final. Más árboles generalmente mejoran el rendimiento hasta un punto donde se estabiliza, sin causar sobreajuste (a diferencia de XGBoost). El costo es mayor tiempo de entrenamiento y predicción.",
        tip: "Entre 100-200 suele ser suficiente. Más allá de 300 raramente mejora significativamente.",
        impact: "alto",
      },
      {
        key: "max_depth",
        label: "Profundidad máxima (max_depth)",
        pythonParam: "max_depth",
        pythonLib: "sklearn.ensemble.RandomForestClassifier",
        type: "slider",
        min: 1,
        max: 30,
        step: 1,
        defaultValue: 0,
        explanation:
          "Profundidad máxima de cada árbol. Valor 0 significa sin límite (los árboles crecen hasta que todas las hojas sean puras). Limitar la profundidad produce árboles más simples que generalizan mejor, aunque cada árbol individual es menos potente, el ensemble compensa.",
        tip: "0 (sin límite) suele funcionar bien en RF ya que el promedio de muchos árboles controla el sobreajuste naturalmente.",
        impact: "alto",
      },
      {
        key: "min_samples_split",
        label: "Mínimo muestras para dividir (min_samples_split)",
        pythonParam: "min_samples_split",
        pythonLib: "sklearn.ensemble.RandomForestClassifier",
        type: "slider",
        min: 2,
        max: 20,
        step: 1,
        defaultValue: 2,
        explanation:
          "Número mínimo de muestras requeridas para dividir un nodo interno. Un valor de 2 (por defecto) permite divisiones que crean nodos con una sola muestra, capturando ruido. Aumentar este valor fuerza a que cada división se base en más datos, produciendo árboles más conservadores.",
        tip: "Aumentar a 5-10 en datasets pequeños o ruidosos para prevenir sobreajuste.",
        impact: "medio",
      },
      {
        key: "min_samples_leaf",
        label: "Mínimo muestras en hoja (min_samples_leaf)",
        pythonParam: "min_samples_leaf",
        pythonLib: "sklearn.ensemble.RandomForestClassifier",
        type: "slider",
        min: 1,
        max: 20,
        step: 1,
        defaultValue: 1,
        explanation:
          "Número mínimo de muestras que debe tener una hoja (nodo final). Similar a min_samples_split pero actúa en las hojas. Un valor de 1 permite hojas con una sola muestra (sobreajuste), mientras que valores más altos aseguran que cada predicción se base en múltiples muestras, suavizando el modelo.",
        tip: "Valores de 1-5 son comunes. Para datasets ruidosos, probar 5-10.",
        impact: "medio",
      },
      {
        key: "max_features",
        label: "Máximo features por división (max_features)",
        pythonParam: "max_features",
        pythonLib: "sklearn.ensemble.RandomForestClassifier",
        type: "select",
        defaultValue: "sqrt",
        options: [
          { value: "sqrt", label: "√n_features (sqrt)" },
          { value: "log2", label: "log2(n_features)" },
          { value: "0.5", label: "50% de features" },
          { value: "0.8", label: "80% de features" },
          { value: "None", label: "Todas las features" },
        ],
        explanation:
          "Número de features consideradas en cada división. Usar solo una fracción de features en cada división descorrelaciona los árboles, lo cual es clave para que el ensemble funcione bien. Si todos los árboles usan las mismas features dominantes, el bosque se comportaría como un solo árbol grande.",
        tip: "sqrt es el valor por defecto y suele funcionar excelente. log2 es más conservador.",
        impact: "alto",
      },
      {
        key: "bootstrap",
        label: "Bootstrap",
        pythonParam: "bootstrap",
        pythonLib: "sklearn.ensemble.RandomForestClassifier",
        type: "select",
        defaultValue: "True",
        options: [
          { value: "True", label: "Sí (con reemplazo)" },
          { value: "False", label: "No (sin reemplazo)" },
        ],
        explanation:
          "Si usar muestreo con reemplazo (bootstrap) para construir cada árbol. Con bootstrap=True, cada árbol usa una muestra aleatoria con reemplazo del dataset completo, creando variabilidad. Con bootstrap=False, cada árbol ve todos los datos de entrenamiento, reduciendo la diversidad del ensemble.",
        tip: "Mantener en True para obtener el comportamiento clásico de Random Forest.",
        impact: "medio",
      },
    ],
  },
  {
    id: "knn",
    name: "K-Nearest Neighbors",
    shortName: "KNN",
    emoji: "👥",
    color: "#f59e0b",
    description:
      "K-Vecinos más Cercanos. Clasifica cada punto nuevo basándose en la clase mayoritaria de sus K vecinos más cercanos en el espacio de features. Es un modelo 'perezoso' que no aprende parámetros, sino que memoriza los datos.",
    concept:
      "KNN es como pedir consejos a tus vecinos más cercanos: si la mayoría de tus K vecinos más cercanos pertenecen a una clase, tú probablemente también. No hay fase de entrenamiento como tal — el modelo simplemente almacena los datos y hace cálculos en el momento de la predicción. Es intuitivo pero costoso computacionalmente con datos grandes.",
    whenToUse:
      "Cuando tienes datasets pequeños, necesidad de un modelo interpretable, o como baseline. No recomendado para datos de alta dimensionalidad.",
    pros: [
      "Simple e intuitivo",
      "Sin fase de entrenamiento",
      "Naturalmente multi-clase",
      "No asume forma funcional",
    ],
    cons: [
      "Lento en predicción con datos grandes",
      "Sensible a la escala de features",
      "Maldición de la dimensionalidad",
      "Sensible a features irrelevantes",
    ],
    hyperparams: [
      {
        key: "n_neighbors",
        label: "Número de vecinos (k)",
        pythonParam: "n_neighbors",
        pythonLib: "sklearn.neighbors.KNeighborsClassifier",
        type: "slider",
        min: 1,
        max: 50,
        step: 1,
        defaultValue: 5,
        explanation:
          "Cantidad de vecinos a consultar para clasificar un punto nuevo. K=1 usa solo el vecino más cercano (muy sensible al ruido), mientras que K grande suaviza las fronteras de decisión pero puede ignorar patrones locales. La elección de K es el hiperparámetro más importante de KNN.",
        tip: "K=1 sobreajusta, K muy grande subajusta. Prueba valores impares para evitar empates. Regla común: K ≈ √(n_samples).",
        impact: "alto",
      },
      {
        key: "weights",
        label: "Pesos de los vecinos (weights)",
        pythonParam: "weights",
        pythonLib: "sklearn.neighbors.KNeighborsClassifier",
        type: "select",
        defaultValue: "uniform",
        options: [
          { value: "uniform", label: "Uniforme (todos igual)" },
          { value: "distance", label: "Distancia (más cercanos = más peso)" },
        ],
        explanation:
          "Cómo ponderar el voto de cada vecino. 'Uniform' da el mismo peso a todos los K vecinos. 'Distance' asigna más peso a los vecinos más cercanos, haciendo que las muestras que están más cerca del punto a clasificar tengan mayor influencia. Esto suele mejorar los resultados especialmente con K grande.",
        tip: "'distance' suele funcionar mejor ya que vecinos más cercanos son más relevantes para la clasificación.",
        impact: "alto",
      },
      {
        key: "p",
        label: "Parámetro de distancia (p)",
        pythonParam: "p",
        pythonLib: "sklearn.neighbors.KNeighborsClassifier",
        type: "slider",
        min: 1,
        max: 5,
        step: 1,
        defaultValue: 2,
        explanation:
          "Define la métrica de distancia usada. p=1 es la distancia Manhattan (suma de diferencias absolutas), p=2 es la distancia Euclidiana (la más común). Valores mayores que 2 dan más peso a las diferencias grandes en alguna dimensión. La elección afecta cómo se miden las 'cercanías' entre puntos.",
        tip: "p=2 (Euclidiana) es el estándar. Probar p=1 si hay muchas features con diferentes escalas.",
        impact: "medio",
      },
      {
        key: "metric",
        label: "Métrica de distancia (metric)",
        pythonParam: "metric",
        pythonLib: "sklearn.neighbors.KNeighborsClassifier",
        type: "select",
        defaultValue: "minkowski",
        options: [
          { value: "minkowski", label: "Minkowski (generalización)" },
          { value: "euclidean", label: "Euclidiana" },
          { value: "manhattan", label: "Manhattan (L1)" },
          { value: "chebyshev", label: "Chebyshev (máxima diferencia)" },
        ],
        explanation:
          "La fórmula matemática para calcular la distancia entre puntos. Minkowski es la más general (el parámetro p la define). Euclidiana es la distancia 'en línea recta'. Manhattan suma las diferencias absolutas por dimensión. Chebyshev usa la mayor diferencia en cualquier dimensión. La métrica define el concepto de 'cercanía'.",
        tip: "Minkowski con p=2 equivale a Euclidiana. Manhattan puede funcionar mejor con datos de alta dimensionalidad.",
        impact: "medio",
      },
    ],
  },
  {
    id: "neural_network",
    name: "Red Neuronal (MLP)",
    shortName: "MLP",
    emoji: "🧠",
    color: "#8b5cf6",
    description:
      "Multi-Layer Perceptron. Red neuronal con capas ocultas de neuronas completamente conectadas. Cada neurona aplica una transformación lineal seguida de una función de activación no lineal, permitiendo aprender representaciones complejas de los datos.",
    concept:
      "Imagina una fábrica con múltiples estaciones de trabajo en cadena. Cada estación (capa) recibe información, la transforma de diferentes maneras (funciones de activación), y la pasa a la siguiente. Las primeras capas detectan patrones simples, las intermedias los combinan en patrones más complejos, y la última capa produce la predicción. El aprendizaje consiste en ajustar los pesos de cada conexión para minimizar el error.",
    whenToUse:
      "Cuando hay relaciones complejas y no lineales, datasets grandes, y features numéricos. Evitar con datasets muy pequeños o sin preprocesamiento.",
    pros: [
      "Aprende representaciones complejas",
      "Universalmente aproximador",
      "Flexible en arquitectura",
    ],
    cons: [
      "Requiere mucho preprocesamiento",
      "Difícil de interpretar (caja negra)",
      "Lento de entrenar",
      "Sensible a la escala de features",
    ],
    hyperparams: [
      {
        key: "hidden_layer_sizes",
        label: "Capas ocultas (hidden_layer_sizes)",
        pythonParam: "hidden_layer_sizes",
        pythonLib: "sklearn.neural_network.MLPClassifier",
        type: "select",
        defaultValue: "100",
        options: [
          { value: "50", label: "1 capa: [50]" },
          { value: "100", label: "1 capa: [100]" },
          { value: "200", label: "1 capa: [200]" },
          { value: "50,50", label: "2 capas: [50, 50]" },
          { value: "100,50", label: "2 capas: [100, 50]" },
          { value: "100,100", label: "2 capas: [100, 100]" },
          { value: "100,50,25", label: "3 capas: [100, 50, 25]" },
          { value: "200,100,50", label: "3 capas: [200, 100, 50]" },
        ],
        explanation:
          "Define el número de neuronas en cada capa oculta. Más capas y neuronas = más capacidad de aprender patrones complejos, pero también más riesgo de sobreajuste y más tiempo de entrenamiento. La última capa suele ser más pequeña (como un embudo), forzando al modelo a comprimir la información.",
        tip: "Empieza con [100] y aumenta complejidad si es necesario. Más capas ayuda con relaciones muy no lineales.",
        impact: "alto",
      },
      {
        key: "activation",
        label: "Función de activación",
        pythonParam: "activation",
        pythonLib: "sklearn.neural_network.MLPClassifier",
        type: "select",
        defaultValue: "relu",
        options: [
          { value: "relu", label: "ReLU (Rectified Linear Unit)" },
          { value: "tanh", label: "Tangente hiperbólica" },
          { value: "logistic", label: "Sigmoide (logistic)" },
          { value: "identity", label: "Identidad (sin activación)" },
        ],
        explanation:
          "Función matemática que se aplica después de cada transformación lineal. Es lo que introduce la no linealidad en la red. ReLU f(x)=max(0,x) es la más usada por ser simple y evitar el problema del gradiente desvaneciente. Tanh y Sigmoide acotan la salida a rangos específicos. Identity no transforma (lineal puro).",
        tip: "ReLU es el estándar moderno. Tanh puede funcionar mejor en problemas específicos. Identity solo si el problema es lineal.",
        impact: "alto",
      },
      {
        key: "solver",
        label: "Optimizador (solver)",
        pythonParam: "solver",
        pythonLib: "sklearn.neural_network.MLPClassifier",
        type: "select",
        defaultValue: "adam",
        options: [
          { value: "adam", label: "Adam (estocástico adaptativo)" },
          { value: "sgd", label: "SGD (descenso de gradiente)" },
          { value: "lbfgs", label: "L-BFGS (cuasi-Newton)" },
        ],
        explanation:
          "Algoritmo que ajusta los pesos de la red para minimizar el error. Adam adapta la tasa de aprendizaje automáticamente para cada parámetro (recomendado para la mayoría de casos). SGD es el clásico descenso de gradiente (más simple, requiere más ajuste). L-BFGS es un método de optimización de segundo orden (mejor para datasets pequeños).",
        tip: "Adam funciona bien en la mayoría de casos. L-BFGS puede ser mejor para datasets pequeños (<1000 muestras).",
        impact: "alto",
      },
      {
        key: "alpha",
        label: "Regularización L2 (alpha)",
        pythonParam: "alpha",
        pythonLib: "sklearn.neural_network.MLPClassifier",
        type: "slider",
        min: 0.0001,
        max: 1,
        step: 0.001,
        defaultValue: 0.0001,
        explanation:
          "Penalización L2 aplicada a los pesos de la red. Valores grandes fuerzan pesos pequeños, simplificando el modelo y previniendo sobreajuste. Valores muy pequeños permiten que los pesos crezcan libremente, lo que puede causar sobreajuste. Es el control principal de complejidad en redes neuronales.",
        tip: "0.0001 es el valor por defecto. Aumentar a 0.001-0.01 si ves sobreajuste. Probar 0.1 para datasets pequeños.",
        impact: "alto",
      },
      {
        key: "learning_rate_init",
        label: "Tasa de aprendizaje inicial (learning_rate_init)",
        pythonParam: "learning_rate_init",
        pythonLib: "sklearn.neural_network.MLPClassifier",
        type: "slider",
        min: 0.0001,
        max: 0.1,
        step: 0.001,
        defaultValue: 0.001,
        explanation:
          "Qué tan grandes son los pasos que da el optimizador al ajustar los pesos. Tasa muy alta puede causar que el optimizador 'salte' sobre el mínimo y diverja. Tasa muy baja hace que el entrenamiento sea extremadamente lento y puede quedarse atascado en mínimos locales.",
        tip: "0.001 es un buen punto de partida. Si el loss oscila mucho, reduce la tasa. Si converge muy lento, auméntala.",
        impact: "alto",
      },
      {
        key: "max_iter",
        label: "Iteraciones máximas (max_iter)",
        pythonParam: "max_iter",
        pythonLib: "sklearn.neural_network.MLPClassifier",
        type: "slider",
        min: 50,
        max: 1000,
        step: 50,
        defaultValue: 200,
        explanation:
          "Número máximo de pasadas completas sobre los datos de entrenamiento (epochs). Si el modelo converge antes, se detiene automáticamente. Si alcanza el máximo sin converger, aparece un warning. Más iteraciones permiten que el modelo se ajuste más, pero también pueden llevar al sobreajuste.",
        tip: "Si ves el warning de convergencia, aumenta este valor. Para redes grandes, 500-1000 puede ser necesario.",
        impact: "medio",
      },
    ],
  },
  {
    id: "decision_tree",
    name: "Árbol de Decisión",
    shortName: "DT",
    emoji: "🌳",
    color: "#06b6d4",
    description:
      "Modelo que crea una serie de reglas de decisión binarias (preguntas sí/no) organizadas en forma de árbol. Cada nodo interno representa una condición sobre una feature, cada rama es el resultado de la condición, y cada hoja es una predicción.",
    concept:
      "Imagina un juego de 20 preguntas: el árbol empieza con la pregunta más informativa ('¿el pétalo mide más de 2.5cm?'), y según la respuesta, hace la siguiente pregunta más útil. Eventualmente llega a una conclusión (hoja). La belleza del árbol es que puedes seguir el camino desde la raíz hasta la hoja y entender exactamente por qué tomó cada decisión.",
    whenToUse:
      "Cuando la interpretabilidad es prioritaria, para entender qué features importan, o como componente de ensembles (RF, XGBoost).",
    pros: [
      "Altamente interpretable (caja blanca)",
      "No requiere escalado de features",
      "Maneja datos numéricos y categóricos",
      "Selección automática de features",
    ],
    cons: [
      "Muy propenso al sobreajuste",
      "Inestable (pequeños cambios en datos = árbol diferente)",
      "Fronteras de decisión escalonadas",
      "Pobre para relaciones lineales",
    ],
    hyperparams: [
      {
        key: "criterion",
        label: "Criterio de división",
        pythonParam: "criterion",
        pythonLib: "sklearn.tree.DecisionTreeClassifier",
        type: "select",
        defaultValue: "gini",
        options: [
          { value: "gini", label: "Gini (impureza Gini)" },
          { value: "entropy", label: "Entropía (ganancia de información)" },
        ],
        explanation:
          "Medida matemática para evaluar la calidad de una división. Gini mide qué tan 'impura' es una partición (0 = pura, 0.5 = máxima impureza binaria). Entropía mide la 'desorden' usando conceptos de teoría de la información. Ambos suelen dar resultados similares, pero Gini es computacionalmente más rápido.",
        tip: "Gini es el estándar y ligeramente más rápido. Entropía puede producir árboles más balanceados en algunos casos.",
        impact: "medio",
      },
      {
        key: "max_depth",
        label: "Profundidad máxima (max_depth)",
        pythonParam: "max_depth",
        pythonLib: "sklearn.tree.DecisionTreeClassifier",
        type: "slider",
        min: 1,
        max: 20,
        step: 1,
        defaultValue: 0,
        explanation:
          "Límite de profundidad del árbol. Es el hiperparámetro MÁS IMPORTANTE para controlar el sobreajuste. Profundidad 1 = solo una división (underfitting extremo). Sin límite = el árbol crece hasta que todas las hojas sean puras (overfitting extremo). El equilibrio está en el medio.",
        tip: "Para datasets pequeños, 3-5 suele ser suficiente. Visualiza el árbol para ver si es demasiado complejo.",
        impact: "alto",
      },
      {
        key: "min_samples_split",
        label: "Mínimo muestras para dividir (min_samples_split)",
        pythonParam: "min_samples_split",
        pythonLib: "sklearn.tree.DecisionTreeClassifier",
        type: "slider",
        min: 2,
        max: 20,
        step: 1,
        defaultValue: 2,
        explanation:
          "Muestras mínimas en un nodo para poder dividirlo. Con 2 (por defecto), un nodo con 2 muestras se puede dividir en dos hojas de 1 muestra cada una (sobreajuste). Aumentar este valor fuerza a que cada división se base en más datos, produciendo un árbol más robusto y generalizable.",
        tip: "Aumentar a 5-10 es una forma efectiva de controlar sobreajuste en árboles de decisión.",
        impact: "medio",
      },
      {
        key: "min_samples_leaf",
        label: "Mínimo muestras en hoja (min_samples_leaf)",
        pythonParam: "min_samples_leaf",
        pythonLib: "sklearn.tree.DecisionTreeClassifier",
        type: "slider",
        min: 1,
        max: 20,
        step: 1,
        defaultValue: 1,
        explanation:
          "Muestras mínimas que debe tener una hoja final. A diferencia de min_samples_split, este parámetro garantiza que cada predicción final se base en al menos N muestras. Tiene un efecto de suavizado: evita que el árbol cree reglas muy específicas que solo aplican a casos aislados.",
        tip: "Valores de 1-5 son comunes. Para evitar reglas triviales, usar mínimo 5.",
        impact: "medio",
      },
      {
        key: "splitter",
        label: "Estrategia de división (splitter)",
        pythonParam: "splitter",
        pythonLib: "sklearn.tree.DecisionTreeClassifier",
        type: "select",
        defaultValue: "best",
        options: [
          { value: "best", label: "Mejor división (best)" },
          { value: "random", label: "División aleatoria (random)" },
        ],
        explanation:
          "Cómo elegir la división en cada nodo. 'Best' evalúa todas las divisiones posibles y elige la mejor — produce el árbol más puro pero es más propenso al sobreajuste. 'Random' evalúa un subconjunto aleatorio de divisiones — introduce variabilidad que puede mejorar la generalización, especialmente en ensembles.",
        tip: "Usar 'best' para un solo árbol. 'Random' es más útil cuando el árbol es parte de un ensemble.",
        impact: "bajo",
      },
    ],
  },
  {
    id: "logistic_regression",
    name: "Regresión Logística",
    shortName: "LR",
    emoji: "📈",
    color: "#ef4444",
    description:
      "Modelo lineal para clasificación que estima la probabilidad de pertenencia a cada clase usando la función sigmoide. A pesar de su nombre, es un clasificador, no un regresor.",
    concept:
      "La regresión logística es como trazar una línea (o hiperplano) que separa las clases, pero en lugar de dar una clasificación dura, estima probabilidades. La función sigmoide transforma cualquier valor numérico a un rango [0,1], interpretándose como probabilidad. Es el modelo lineal más fundamental para clasificación.",
    whenToUse:
      "Como baseline para cualquier problema de clasificación. Cuando la interpretabilidad es clave. Cuando la relación es aproximadamente lineal.",
    pros: [
      "Simple y rápido",
      "Probabilidades calibradas",
      "Altamente interpretable",
      "Poco propenso al sobreajuste",
    ],
    cons: [
      "Solo captura relaciones lineales",
      "Fronteras de decisión lineales",
      "Sensible a features irrelevantes",
      "Requiere escalado de features",
    ],
    hyperparams: [
      {
        key: "C",
        label: "Inversa de regularización (C)",
        pythonParam: "C",
        pythonLib: "sklearn.linear_model.LogisticRegression",
        type: "slider",
        min: 0.01,
        max: 100,
        step: 0.1,
        defaultValue: 1.0,
        explanation:
          "Controla la fuerza de la regularización, pero de forma inversa: C pequeño = regularización fuerte (modelo simple), C grande = regularización débil (modelo complejo). Es el hiperparámetro más importante de la regresión logística. C=1 es el punto de equilibrio estándar.",
        tip: "C pequeño (0.01-0.1) para modelos simples, C grande (10-100) para modelos más flexibles. Usar escala logarítmica para buscar.",
        impact: "alto",
      },
      {
        key: "penalty",
        label: "Tipo de regularización (penalty)",
        pythonParam: "penalty",
        pythonLib: "sklearn.linear_model.LogisticRegression",
        type: "select",
        defaultValue: "l2",
        options: [
          { value: "l1", label: "L1 (Lasso) - selección de features" },
          { value: "l2", label: "L2 (Ridge) - pesos pequeños" },
          { value: "elasticnet", label: "ElasticNet - combinación L1+L2" },
          { value: "None", label: "Sin regularización" },
        ],
        explanation:
          "Tipo de penalización aplicada a los coeficientes. L1 (Lasso) fuerza algunos coeficientes a cero, seleccionando features automáticamente. L2 (Ridge) reduce los coeficientes sin hacerlos cero, distribuyendo el peso entre features correlacionadas. ElasticNet combina ambos. Sin regularización, el modelo puede sobreajustar fácilmente.",
        tip: "L2 es el estándar. Probar L1 si hay muchas features irrelevantes. ElasticNet es un buen compromiso.",
        impact: "alto",
      },
      {
        key: "solver",
        label: "Optimizador (solver)",
        pythonParam: "solver",
        pythonLib: "sklearn.linear_model.LogisticRegression",
        type: "select",
        defaultValue: "lbfgs",
        options: [
          { value: "lbfgs", label: "L-BFGS (cuasi-Newton)" },
          { value: "liblinear", label: "LibLinear (coordenada descendente)" },
          { value: "saga", label: "SAGA (gradiente estocástico)" },
          { value: "newton-cg", label: "Newton-CG" },
        ],
        explanation:
          "Algoritmo para encontrar los coeficientes óptimos. L-BFGS es el más versátil y rápido para la mayoría de casos. LibLinear es bueno para datasets pequeños y soporta L1. SAGA es eficiente para datasets grandes y soporta todas las penalizaciones. Newton-CG es similar a L-BFGS pero con implementación diferente.",
        tip: "L-BFGS es el default y funciona bien. Cambiar a liblinear si usas penalty=l1 con datasets pequeños.",
        impact: "medio",
      },
      {
        key: "max_iter",
        label: "Iteraciones máximas (max_iter)",
        pythonParam: "max_iter",
        pythonLib: "sklearn.linear_model.LogisticRegression",
        type: "slider",
        min: 50,
        max: 1000,
        step: 50,
        defaultValue: 100,
        explanation:
          "Número máximo de iteraciones del optimizador. Si el modelo no converge en este número, scikit-learn muestra un warning. Más iteraciones permiten que el optimizador encuentre una mejor solución, pero generalmente si no converge en 100-200 iteraciones, hay otros problemas (escala de features, learning rate, etc.).",
        tip: "Si ves warning de convergencia, aumenta este valor. Pero primero verifica que las features estén escaladas.",
        impact: "bajo",
      },
    ],
  },
];

export const DATASET_PRESETS = [
  {
    id: "moons",
    name: "Two Moons",
    emoji: "🌙",
    description: "Dos lunas intercaladas — perfecto para ver fronteras de decisión",
    bestFor2D: true,
  },
  {
    id: "circles",
    name: "Circles",
    emoji: "⭕",
    description: "Círculos concéntricos — desafío para modelos lineales",
    bestFor2D: true,
  },
  {
    id: "classification",
    name: "Synthetic",
    emoji: "📊",
    description: "Clasificación binaria simple — buen punto de partida",
    bestFor2D: true,
  },
  {
    id: "iris",
    name: "Iris",
    emoji: "🌸",
    description: "El dataset clásico de flores — multi-clase",
    bestFor2D: false,
  },
  {
    id: "wine",
    name: "Wine",
    emoji: "🍷",
    description: "Clasificación de vinos — muchas features",
    bestFor2D: false,
  },
  {
    id: "breast_cancer",
    name: "Breast Cancer",
    emoji: "🔬",
    description: "Diagnóstico médico — clasificación binaria real",
    bestFor2D: false,
  },
];

// ============================================================
// REGRESIÓN: Configuración de algoritmos
// ============================================================
export const REGRESSION_ALGORITHMS: AlgorithmConfig[] = [
  {
    id: "xgboost_regressor",
    name: "XGBoost Regresor",
    shortName: "XGB-R",
    emoji: "🚀",
    color: "#10b981",
    description:
      "Extreme Gradient Boosting para regresión. Construye árboles secuencialmente donde cada árbol corrige los errores residuales del anterior, prediciendo valores continuos.",
    concept:
      "XGBoost para regresión funciona como un equipo donde cada miembro aprende de los errores del anterior. El primer árbol hace predicciones, el segundo intenta corregir los residuos (diferencia entre valor real y predicho), el tercero corrige los nuevos residuos, y así sucesivamente. La predicción final es la suma de todas las contribuciones, cada una multiplicada por la tasa de aprendizaje.",
    whenToUse:
      "Cuando necesitas alto rendimiento predictivo en problemas de regresión, tienes datos tabulares, y puedes permitirte un modelo más complejo. Excelente para predecir precios, cantidades, o cualquier valor continuo.",
    pros: [
      "Alto rendimiento predictivo",
      "Maneja relaciones no lineales",
      "Regularización integrada contra overfitting",
      "Rápido y eficiente",
    ],
    cons: [
      "Puede hacer overfitting si no se ajusta bien",
      "Muchos hiperparámetros para afinar",
      "Menos interpretable que un árbol simple",
    ],
    hyperparams: [
      {
        key: "n_estimators",
        label: "Número de árboles (n_estimators)",
        pythonParam: "n_estimators",
        pythonLib: "xgboost.XGBRegressor",
        type: "slider",
        min: 10,
        max: 500,
        step: 10,
        defaultValue: 100,
        explanation:
          "Define cuántos árboles secuenciales se construyen. Cada árbol nuevo intenta corregir los residuos del conjunto anterior. Más árboles = más capacidad de aprender patrones complejos, pero también más riesgo de sobreajuste.",
        tip: "Empieza con 100 y aumenta gradualmente. Si el error de entrenamiento es mucho menor que el de test, reduce este valor.",
        impact: "alto",
      },
      {
        key: "max_depth",
        label: "Profundidad máxima (max_depth)",
        pythonParam: "max_depth",
        pythonLib: "xgboost.XGBRegressor",
        type: "slider",
        min: 1,
        max: 20,
        step: 1,
        defaultValue: 6,
        explanation:
          "Controla la profundidad de cada árbol individual. Árboles más profundos capturan relaciones más complejas pero también memorizan ruido. Para regresión, profundidades de 3-6 suelen dar buenos resultados.",
        tip: "Valores bajos (3-5) producen modelos más robustos. Valores altos (>10) suelen causar sobreajuste.",
        impact: "alto",
      },
      {
        key: "learning_rate",
        label: "Tasa de aprendizaje (learning_rate)",
        pythonParam: "learning_rate",
        pythonLib: "xgboost.XGBRegressor",
        type: "slider",
        min: 0.01,
        max: 1.0,
        step: 0.01,
        defaultValue: 0.3,
        explanation:
          "Determina cuánto contribuye cada árbol nuevo a la predicción final. Una tasa baja significa correcciones pequeñas, requiriendo más árboles pero logrando mejor generalización. Una tasa alta converge más rápido pero puede ser inestable.",
        tip: "Usualmente se usa entre 0.01 y 0.3. Si usas learning_rate bajo, necesitas más n_estimators.",
        impact: "alto",
      },
      {
        key: "min_child_weight",
        label: "Peso mínimo del hijo (min_child_weight)",
        pythonParam: "min_child_weight",
        pythonLib: "xgboost.XGBRegressor",
        type: "slider",
        min: 1,
        max: 20,
        step: 1,
        defaultValue: 1,
        explanation:
          "Peso mínimo que debe tener un nodo hijo para crear una nueva división. Valores altos impiden que el modelo aprenda patrones muy específicos. En regresión, controla que las divisiones se basen en suficientes muestras.",
        tip: "Aumentar este valor ayuda a prevenir sobreajuste en datasets pequeños o ruidosos.",
        impact: "medio",
      },
      {
        key: "subsample",
        label: "Submuestreo (subsample)",
        pythonParam: "subsample",
        pythonLib: "xgboost.XGBRegressor",
        type: "slider",
        min: 0.1,
        max: 1.0,
        step: 0.1,
        defaultValue: 1.0,
        explanation:
          "Fracción de muestras usadas para entrenar cada árbol. Con 0.8, cada árbol usa el 80% de los datos aleatoriamente. Agrega aleatoriedad que ayuda a prevenir sobreajuste.",
        tip: "Valores entre 0.6 y 0.9 suelen mejorar la generalización. 1.0 usa todos los datos.",
        impact: "medio",
      },
      {
        key: "colsample_bytree",
        label: "Submuestreo de features (colsample_bytree)",
        pythonParam: "colsample_bytree",
        pythonLib: "xgboost.XGBRegressor",
        type: "slider",
        min: 0.1,
        max: 1.0,
        step: 0.1,
        defaultValue: 1.0,
        explanation:
          "Fracción de características usadas para construir cada árbol. Reduce la correlación entre árboles y mejora la generalización, especialmente cuando hay features irrelevantes.",
        tip: "Útil cuando hay muchas features. Valores entre 0.5 y 0.8 suelen funcionar bien.",
        impact: "medio",
      },
      {
        key: "gamma",
        label: "Gamma (reducción mínima de pérdida)",
        pythonParam: "gamma",
        pythonLib: "xgboost.XGBRegressor",
        type: "slider",
        min: 0,
        max: 5,
        step: 0.1,
        defaultValue: 0,
        explanation:
          "Reducción mínima de pérdida requerida para hacer una nueva división. Valores altos hacen que el algoritmo sea más conservador, requiriendo mejoras significativas para dividir.",
        tip: "Aumentar gamma simplifica el modelo. Útil para controlar sobreajuste.",
        impact: "medio",
      },
      {
        key: "reg_alpha",
        label: "Regularización L1 (reg_alpha)",
        pythonParam: "reg_alpha",
        pythonLib: "xgboost.XGBRegressor",
        type: "slider",
        min: 0,
        max: 10,
        step: 0.1,
        defaultValue: 0,
        explanation:
          "Penalización L1 (Lasso) aplicada a los pesos de las hojas. Fuerza a que algunos pesos sean cero, realizando selección de features automáticamente.",
        tip: "Empieza con 0 y aumenta gradualmente si ves sobreajuste.",
        impact: "bajo",
      },
      {
        key: "reg_lambda",
        label: "Regularización L2 (reg_lambda)",
        pythonParam: "reg_lambda",
        pythonLib: "xgboost.XGBRegressor",
        type: "slider",
        min: 0,
        max: 10,
        step: 0.1,
        defaultValue: 1,
        explanation:
          "Penalización L2 (Ridge) aplicada a los pesos. Reduce los pesos grandes sin hacerlos cero, suavizando el modelo. El valor por defecto de 1 ya proporciona regularización moderada.",
        tip: "El valor por defecto de 1 suele funcionar bien. Aumentar si el modelo sobreajusta.",
        impact: "bajo",
      },
    ],
  },
  {
    id: "random_forest_regressor",
    name: "Random Forest Regresor",
    shortName: "RF-R",
    emoji: "🌲",
    color: "#22c55e",
    description:
      "Bosque Aleatorio para regresión. Construye múltiples árboles de decisión independientes y promedia sus predicciones continuas. Cada árbol se entrena con una muestra aleatoria de datos y features.",
    concept:
      "Imagina que pides a 100 expertos que estimen el precio de una casa, donde cada experto solo ve una parte aleatoria de la información. Al promediar todas las estimaciones, el resultado final es más robusto que cualquier estimación individual. Eso es Random Forest para regresión: la sabiduría de la multitud aplicada a la predicción de valores continuos.",
    whenToUse:
      "Como primer modelo para cualquier problema de regresión tabular. Excelente baseline, robusto, requiere poca afinación de hiperparámetros.",
    pros: [
      "Robusto y difícil de sobreajustar",
      "Poco sensible a hiperparámetros",
      "Proporciona importancia de features",
      "No requiere escalado estricto",
    ],
    cons: [
      "Menos potente que XGBoost en competencias",
      "No extrapoliza fuera del rango de entrenamiento",
      "Muchos árboles pueden ser lentos",
    ],
    hyperparams: [
      {
        key: "n_estimators",
        label: "Número de árboles (n_estimators)",
        pythonParam: "n_estimators",
        pythonLib: "sklearn.ensemble.RandomForestRegressor",
        type: "slider",
        min: 10,
        max: 500,
        step: 10,
        defaultValue: 100,
        explanation:
          "Cantidad de árboles independientes en el bosque. Más árboles generalmente mejoran el rendimiento hasta un punto donde se estabiliza, sin causar sobreajuste.",
        tip: "Entre 100-200 suele ser suficiente. Más allá de 300 raramente mejora significativamente.",
        impact: "alto",
      },
      {
        key: "max_depth",
        label: "Profundidad máxima (max_depth)",
        pythonParam: "max_depth",
        pythonLib: "sklearn.ensemble.RandomForestRegressor",
        type: "slider",
        min: 1,
        max: 30,
        step: 1,
        defaultValue: 0,
        explanation:
          "Profundidad máxima de cada árbol. Valor 0 significa sin límite. Limitar la profundidad produce árboles más simples que generalizan mejor.",
        tip: "0 (sin límite) suele funcionar bien en RF ya que el promedio controla el sobreajuste naturalmente.",
        impact: "alto",
      },
      {
        key: "min_samples_split",
        label: "Mínimo muestras para dividir (min_samples_split)",
        pythonParam: "min_samples_split",
        pythonLib: "sklearn.ensemble.RandomForestRegressor",
        type: "slider",
        min: 2,
        max: 20,
        step: 1,
        defaultValue: 2,
        explanation:
          "Número mínimo de muestras requeridas para dividir un nodo interno. Aumentar este valor fuerza a que cada división se base en más datos.",
        tip: "Aumentar a 5-10 en datasets pequeños o ruidosos para prevenir sobreajuste.",
        impact: "medio",
      },
      {
        key: "min_samples_leaf",
        label: "Mínimo muestras en hoja (min_samples_leaf)",
        pythonParam: "min_samples_leaf",
        pythonLib: "sklearn.ensemble.RandomForestRegressor",
        type: "slider",
        min: 1,
        max: 20,
        step: 1,
        defaultValue: 1,
        explanation:
          "Número mínimo de muestras que debe tener una hoja. Valores más altos aseguran que cada predicción se base en múltiples muestras, suavizando el modelo.",
        tip: "Valores de 1-5 son comunes. Para datasets ruidosos, probar 5-10.",
        impact: "medio",
      },
      {
        key: "max_features",
        label: "Máximo features por división (max_features)",
        pythonParam: "max_features",
        pythonLib: "sklearn.ensemble.RandomForestRegressor",
        type: "select",
        defaultValue: "sqrt",
        options: [
          { value: "sqrt", label: "√n_features (sqrt)" },
          { value: "log2", label: "log2(n_features)" },
          { value: "0.5", label: "50% de features" },
          { value: "0.8", label: "80% de features" },
          { value: "None", label: "Todas las features" },
        ],
        explanation:
          "Número de features consideradas en cada división. Usar solo una fracción descorrelaciona los árboles, lo cual es clave para que el ensemble funcione bien.",
        tip: "sqrt es el valor por defecto y suele funcionar excelente.",
        impact: "alto",
      },
      {
        key: "bootstrap",
        label: "Bootstrap",
        pythonParam: "bootstrap",
        pythonLib: "sklearn.ensemble.RandomForestRegressor",
        type: "select",
        defaultValue: "True",
        options: [
          { value: "True", label: "Sí (con reemplazo)" },
          { value: "False", label: "No (sin reemplazo)" },
        ],
        explanation:
          "Si usar muestreo con reemplazo para construir cada árbol. Con bootstrap=True, cada árbol usa una muestra aleatoria con reemplazo.",
        tip: "Mantener en True para obtener el comportamiento clásico de Random Forest.",
        impact: "medio",
      },
    ],
  },
  {
    id: "knn_regressor",
    name: "KNN Regresor",
    shortName: "KNN-R",
    emoji: "👥",
    color: "#f59e0b",
    description:
      "K-Vecinos más Cercanos para regresión. Predice el valor de cada punto nuevo promediando los valores de sus K vecinos más cercanos. Es un modelo 'perezoso' que no aprende parámetros.",
    concept:
      "KNN para regresión es como pedir a tus K vecinos más cercanos que te digan cuánto vale tu casa y luego promediar sus respuestas. Los vecinos más cercanos tienen mayor influencia si usas pesos por distancia. Es intuitivo pero costoso computacionalmente con datos grandes.",
    whenToUse:
      "Cuando tienes datasets pequeños o medianos, relaciones locales suaves, o como baseline de regresión. No recomendado para datos de alta dimensionalidad.",
    pros: [
      "Simple e intuitivo",
      "Sin fase de entrenamiento",
      "Captura relaciones locales",
      "No asume forma funcional",
    ],
    cons: [
      "Lento en predicción con datos grandes",
      "Sensible a la escala de features",
      "Maldición de la dimensionalidad",
      "Sensible a features irrelevantes",
    ],
    hyperparams: [
      {
        key: "n_neighbors",
        label: "Número de vecinos (k)",
        pythonParam: "n_neighbors",
        pythonLib: "sklearn.neighbors.KNeighborsRegressor",
        type: "slider",
        min: 1,
        max: 50,
        step: 1,
        defaultValue: 5,
        explanation:
          "Cantidad de vecinos a consultar. K=1 usa solo el vecino más cercano (muy sensible al ruido), mientras que K grande suaviza las predicciones pero puede ignorar patrones locales.",
        tip: "K=1 sobreajusta, K muy grande subajusta. Regla común: K ≈ √(n_samples).",
        impact: "alto",
      },
      {
        key: "weights",
        label: "Pesos de los vecinos (weights)",
        pythonParam: "weights",
        pythonLib: "sklearn.neighbors.KNeighborsRegressor",
        type: "select",
        defaultValue: "uniform",
        options: [
          { value: "uniform", label: "Uniforme (todos igual)" },
          { value: "distance", label: "Distancia (más cercanos = más peso)" },
        ],
        explanation:
          "Cómo ponderar el voto de cada vecino. 'Uniform' da el mismo peso a todos. 'Distance' asigna más peso a los vecinos más cercanos, mejorando resultados especialmente con K grande.",
        tip: "'distance' suele funcionar mejor ya que vecinos más cercanos son más relevantes.",
        impact: "alto",
      },
      {
        key: "p",
        label: "Parámetro de distancia (p)",
        pythonParam: "p",
        pythonLib: "sklearn.neighbors.KNeighborsRegressor",
        type: "slider",
        min: 1,
        max: 5,
        step: 1,
        defaultValue: 2,
        explanation:
          "Define la métrica de distancia. p=1 es Manhattan, p=2 es Euclidiana. La elección afecta cómo se miden las 'cercanías' entre puntos.",
        tip: "p=2 (Euclidiana) es el estándar. Probar p=1 si hay muchas features con diferentes escalas.",
        impact: "medio",
      },
    ],
  },
  {
    id: "mlp_regressor",
    name: "Red Neuronal Regresora (MLP)",
    shortName: "MLP-R",
    emoji: "🧠",
    color: "#8b5cf6",
    description:
      "Multi-Layer Perceptron para regresión. Red neuronal con capas ocultas que aprende representaciones complejas y produce una salida continua. Usa MSE como función de pérdida.",
    concept:
      "Igual que la versión de clasificación, imagina una fábrica con múltiples estaciones de trabajo. La diferencia es que la última estación produce un valor numérico continuo en lugar de una clase. El aprendizaje ajusta los pesos para minimizar el error cuadrático medio entre las predicciones y los valores reales.",
    whenToUse:
      "Cuando hay relaciones complejas y no lineales en datos de regresión, datasets grandes, y features numéricos. Evitar con datasets muy pequeños.",
    pros: [
      "Aprende representaciones complejas",
      "Universalmente aproximador",
      "Flexible en arquitectura",
    ],
    cons: [
      "Requiere mucho preprocesamiento",
      "Difícil de interpretar (caja negra)",
      "Lento de entrenar",
      "Sensible a la escala de features",
    ],
    hyperparams: [
      {
        key: "hidden_layer_sizes",
        label: "Capas ocultas (hidden_layer_sizes)",
        pythonParam: "hidden_layer_sizes",
        pythonLib: "sklearn.neural_network.MLPRegressor",
        type: "select",
        defaultValue: "100",
        options: [
          { value: "50", label: "1 capa: [50]" },
          { value: "100", label: "1 capa: [100]" },
          { value: "200", label: "1 capa: [200]" },
          { value: "50,50", label: "2 capas: [50, 50]" },
          { value: "100,50", label: "2 capas: [100, 50]" },
          { value: "100,100", label: "2 capas: [100, 100]" },
          { value: "100,50,25", label: "3 capas: [100, 50, 25]" },
          { value: "200,100,50", label: "3 capas: [200, 100, 50]" },
        ],
        explanation:
          "Define el número de neuronas en cada capa oculta. Más capas y neuronas = más capacidad de aprender patrones complejos, pero también más riesgo de sobreajuste.",
        tip: "Empieza con [100] y aumenta complejidad si es necesario.",
        impact: "alto",
      },
      {
        key: "activation",
        label: "Función de activación",
        pythonParam: "activation",
        pythonLib: "sklearn.neural_network.MLPRegressor",
        type: "select",
        defaultValue: "relu",
        options: [
          { value: "relu", label: "ReLU (Rectified Linear Unit)" },
          { value: "tanh", label: "Tangente hiperbólica" },
          { value: "logistic", label: "Sigmoide (logistic)" },
          { value: "identity", label: "Identidad (sin activación)" },
        ],
        explanation:
          "Función matemática que introduce no linealidad en la red. ReLU es la más usada por ser simple y eficiente. Tanh y Sigmoide acotan la salida.",
        tip: "ReLU es el estándar moderno. Identity solo si el problema es lineal.",
        impact: "alto",
      },
      {
        key: "alpha",
        label: "Regularización L2 (alpha)",
        pythonParam: "alpha",
        pythonLib: "sklearn.neural_network.MLPRegressor",
        type: "slider",
        min: 0.0001,
        max: 1,
        step: 0.001,
        defaultValue: 0.0001,
        explanation:
          "Penalización L2 aplicada a los pesos de la red. Valores grandes fuerzan pesos pequeños, simplificando el modelo y previniendo sobreajuste.",
        tip: "0.0001 es el valor por defecto. Aumentar a 0.001-0.01 si ves sobreajuste.",
        impact: "alto",
      },
      {
        key: "learning_rate_init",
        label: "Tasa de aprendizaje inicial (learning_rate_init)",
        pythonParam: "learning_rate_init",
        pythonLib: "sklearn.neural_network.MLPRegressor",
        type: "slider",
        min: 0.0001,
        max: 0.1,
        step: 0.001,
        defaultValue: 0.001,
        explanation:
          "Qué tan grandes son los pasos que da el optimizador al ajustar los pesos. Tasa muy alta puede causar divergencia. Tasa muy baja hace el entrenamiento lento.",
        tip: "0.001 es un buen punto de partida. Si el loss oscila mucho, reduce la tasa.",
        impact: "alto",
      },
      {
        key: "max_iter",
        label: "Iteraciones máximas (max_iter)",
        pythonParam: "max_iter",
        pythonLib: "sklearn.neural_network.MLPRegressor",
        type: "slider",
        min: 50,
        max: 1000,
        step: 50,
        defaultValue: 200,
        explanation:
          "Número máximo de pasadas completas sobre los datos (epochs). Más iteraciones permiten mejor ajuste, pero pueden llevar al sobreajuste.",
        tip: "Si ves warning de convergencia, aumenta este valor.",
        impact: "medio",
      },
    ],
  },
  {
    id: "decision_tree_regressor",
    name: "Árbol de Decisión Regresor",
    shortName: "DT-R",
    emoji: "🌳",
    color: "#06b6d4",
    description:
      "Árbol de decisión para regresión. Crea reglas binarias organizadas en forma de árbol, donde cada hoja contiene el valor promedio de las muestras que llegan a ella. Usa reducción de varianza (MSE) como criterio de división.",
    concept:
      "Igual que el árbol de clasificación, pero en lugar de predecir una clase, cada hoja predice un valor numérico (el promedio de las muestras que llegan a ella). El árbol busca las divisiones que más reduzcan la varianza de los valores objetivo en cada partición.",
    whenToUse:
      "Cuando la interpretabilidad es prioritaria en problemas de regresión, para entender qué features importan, o como componente de ensembles.",
    pros: [
      "Altamente interpretable (caja blanca)",
      "No requiere escalado de features",
      "Maneja datos numéricos y categóricos",
      "Selección automática de features",
    ],
    cons: [
      "Muy propenso al sobreajuste",
      "Inestable (pequeños cambios = árbol diferente)",
      "Predicciones escalonadas (no suaves)",
      "Pobre para relaciones lineales",
    ],
    hyperparams: [
      {
        key: "criterion",
        label: "Criterio de división",
        pythonParam: "criterion",
        pythonLib: "sklearn.tree.DecisionTreeRegressor",
        type: "select",
        defaultValue: "mse",
        options: [
          { value: "mse", label: "MSE (Error Cuadrático Medio)" },
          { value: "mae", label: "MAE (Error Absoluto Medio)" },
        ],
        explanation:
          "Medida para evaluar la calidad de una división. MSE minimiza el error cuadrático (penaliza errores grandes). MAE minimiza el error absoluto (más robusto a outliers).",
        tip: "MSE es el estándar y produce predicciones más suaves. MAE es más robusto a valores atípicos.",
        impact: "medio",
      },
      {
        key: "max_depth",
        label: "Profundidad máxima (max_depth)",
        pythonParam: "max_depth",
        pythonLib: "sklearn.tree.DecisionTreeRegressor",
        type: "slider",
        min: 1,
        max: 20,
        step: 1,
        defaultValue: 0,
        explanation:
          "Límite de profundidad del árbol. Es el hiperparámetro MÁS IMPORTANTE para controlar el sobreajuste. Sin límite = el árbol crece hasta que todas las hojas tengan varianza cero (overfitting).",
        tip: "Para datasets pequeños, 3-5 suele ser suficiente. Visualiza el árbol para ver si es demasiado complejo.",
        impact: "alto",
      },
      {
        key: "min_samples_split",
        label: "Mínimo muestras para dividir (min_samples_split)",
        pythonParam: "min_samples_split",
        pythonLib: "sklearn.tree.DecisionTreeRegressor",
        type: "slider",
        min: 2,
        max: 20,
        step: 1,
        defaultValue: 2,
        explanation:
          "Muestras mínimas en un nodo para poder dividirlo. Aumentar este valor fuerza a que cada división se base en más datos, produciendo un árbol más robusto.",
        tip: "Aumentar a 5-10 es efectivo para controlar sobreajuste.",
        impact: "medio",
      },
      {
        key: "min_samples_leaf",
        label: "Mínimo muestras en hoja (min_samples_leaf)",
        pythonParam: "min_samples_leaf",
        pythonLib: "sklearn.tree.DecisionTreeRegressor",
        type: "slider",
        min: 1,
        max: 20,
        step: 1,
        defaultValue: 1,
        explanation:
          "Muestras mínimas que debe tener una hoja final. Garantiza que cada predicción se base en al menos N muestras.",
        tip: "Valores de 1-5 son comunes. Para evitar predicciones extremas, usar mínimo 5.",
        impact: "medio",
      },
    ],
  },
  {
    id: "linear_regression",
    name: "Regresión Lineal",
    shortName: "LinReg",
    emoji: "📈",
    color: "#ef4444",
    description:
      "Modelo lineal que predice un valor continuo como combinación lineal de las features. Usa descenso de gradiente estocástico con regularización L2 (Ridge). Es el modelo más fundamental de regresión.",
    concept:
      "La regresión lineal encuentra la mejor línea (o hiperplano) que se ajusta a los datos minimizando el error cuadrático medio. Cada feature tiene un coeficiente que indica cuánto cambia la predicción por cada unidad de cambio en esa feature. La regularización L2 evita que los coeficientes crezcan demasiado.",
    whenToUse:
      "Como baseline para cualquier problema de regresión. Cuando la interpretabilidad es clave. Cuando la relación es aproximadamente lineal.",
    pros: [
      "Simple y rápido",
      "Altamente interpretable",
      "Poco propenso al sobreajuste",
      "Coeficientes explicables",
    ],
    cons: [
      "Solo captura relaciones lineales",
      "Sensible a features irrelevantes",
      "Requiere escalado de features",
      "No captura interacciones automáticamente",
    ],
    hyperparams: [
      {
        key: "alpha",
        label: "Regularización L2 (alpha)",
        pythonParam: "alpha",
        pythonLib: "sklearn.linear_model.Ridge",
        type: "slider",
        min: 0.0001,
        max: 10,
        step: 0.001,
        defaultValue: 0.0001,
        explanation:
          "Penalización L2 (Ridge) aplicada a los coeficientes. Valores grandes fuerzan coeficientes pequeños, simplificando el modelo. alpha=0 es regresión lineal sin regularización.",
        tip: "0.0001 es el valor por defecto (casi sin regularización). Aumentar a 0.01-1 si hay sobreajuste.",
        impact: "alto",
      },
      {
        key: "max_iter",
        label: "Iteraciones máximas (max_iter)",
        pythonParam: "max_iter",
        pythonLib: "sklearn.linear_model.Ridge",
        type: "slider",
        min: 50,
        max: 1000,
        step: 50,
        defaultValue: 200,
        explanation:
          "Número máximo de pasadas sobre los datos para el descenso de gradiente. Más iteraciones permiten mejor convergencia.",
        tip: "Si el modelo no converge, aumenta este valor. Pero primero verifica que las features estén escaladas.",
        impact: "medio",
      },
      {
        key: "learning_rate",
        label: "Tasa de aprendizaje (learning_rate)",
        pythonParam: "learning_rate",
        pythonLib: "sklearn.linear_model.SGDRegressor",
        type: "slider",
        min: 0.001,
        max: 0.1,
        step: 0.001,
        defaultValue: 0.01,
        explanation:
          "Tamaño de los pasos que da el descenso de gradiente. Tasa muy alta puede causar divergencia. Tasa muy baja hace el entrenamiento lento.",
        tip: "0.01 es un buen punto de partida. Si el error oscila, reduce la tasa.",
        impact: "alto",
      },
    ],
  },
];

export const REGRESSION_DATASET_PRESETS = [
  {
    id: "make_regression",
    name: "Regresión Sintética",
    emoji: "📉",
    description: "Datos lineales con ruido — perfecto para experimentar con regresión",
    bestFor2D: true,
  },
  {
    id: "boston",
    name: "Boston Housing",
    emoji: "🏠",
    description: "Predecir valor de viviendas — dataset clásico de regresión",
    bestFor2D: false,
  },
];
