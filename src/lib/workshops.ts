/**
 * ML HyperLab — Talleres guiados en español
 * Configuración de 6 workshops interactivos para experimentar con hiperparámetros de ML.
 */

// ============================================================
// Tipos
// ============================================================

export type Difficulty = "principiante" | "intermedio" | "avanzado";

export type AlgorithmId =
  | "xgboost"
  | "random_forest"
  | "knn"
  | "neural_network"
  | "decision_tree"
  | "logistic_regression";

export type DatasetId =
  | "moons"
  | "circles"
  | "classification"
  | "iris"
  | "wine"
  | "breast_cancer";

export interface WorkshopStep {
  instruction: string;
  algorithm: AlgorithmId;
  dataset: DatasetId;
  hyperparams: Record<string, number | string>;
  question: string;
  hint: string;
  expectedObservation: string;
}

export interface Workshop {
  id: string;
  title: string;
  description: string;
  difficulty: Difficulty;
  emoji: string;
  duration: string;
  learningObjectives: string[];
  steps: WorkshopStep[];
  finalChallenge: string;
}

// ============================================================
// Talleres
// ============================================================

export const WORKSHOPS: Workshop[] = [
  // ──────────────────────────────────────────────────────────────
  // Taller 1: Tu Primer Modelo
  // ──────────────────────────────────────────────────────────────
  {
    id: "taller-1-primer-modelo",
    title: "Tu Primer Modelo",
    description:
      "Aprende los conceptos fundamentales del Machine Learning entrenando tu primer modelo. Descubrirás qué es el sobreajuste (overfitting) y el subajuste (underfitting) experimentando con la profundidad de un Árbol de Decisión y comparándolo con un Random Forest.",
    difficulty: "principiante",
    emoji: "🌱",
    duration: "15 min",
    learningObjectives: [
      "Entender qué es entrenar un modelo de clasificación",
      "Observar cómo la profundidad del árbol afecta las fronteras de decisión",
      "Identificar visualmente el sobreajuste y el subajuste",
      "Comprender por qué un ensemble (Random Forest) mejora los resultados",
    ],
    steps: [
      {
        instruction:
          "Selecciona el algoritmo Árbol de Decisión y el dataset 'Two Moons'. Deja todos los hiperparámetros con sus valores por defecto y entrena el modelo. Observa la frontera de decisión y las métricas.",
        algorithm: "decision_tree",
        dataset: "moons",
        hyperparams: {
          criterion: "gini",
          max_depth: 0,
          min_samples_split: 2,
          min_samples_leaf: 1,
          splitter: "best",
        },
        question:
          "¿Qué forma tiene la frontera de decisión? ¿Son líneas rectas o curvas?",
        hint:
          "Los árboles de decisión solo pueden hacer divisiones horizontales y verticales, creando regiones rectangulares.",
        expectedObservation:
          "La frontera de decisión es muy compleja, con muchas líneas rectas que forman escalones intentando seguir la forma curva de las lunas. El accuracy de entrenamiento es muy alto (cercano a 1.0) pero el de test puede ser menor.",
      },
      {
        instruction:
          "Ahora cambia max_depth a 1. Entrena de nuevo y observa cómo cambia la frontera de decisión y las métricas.",
        algorithm: "decision_tree",
        dataset: "moons",
        hyperparams: {
          criterion: "gini",
          max_depth: 1,
          min_samples_split: 2,
          min_samples_leaf: 1,
          splitter: "best",
        },
        question:
          "¿Cuántas regiones de decisión se crean con max_depth=1? ¿El modelo clasifica bien?",
        hint:
          "Con profundidad 1, el árbol hace una sola pregunta, creando exactamente 2 regiones.",
        expectedObservation:
          "Solo hay 2 regiones separadas por una línea vertical u horizontal. El accuracy es bajo (alrededor de 0.70-0.80) porque una sola división no puede capturar la forma de las lunas. Esto es underfitting extremo.",
      },
      {
        instruction:
          "Aumenta max_depth a 3 y entrena. Luego prueba con max_depth=5 y max_depth=10. Compara los resultados.",
        algorithm: "decision_tree",
        dataset: "moons",
        hyperparams: {
          criterion: "gini",
          max_depth: 5,
          min_samples_split: 2,
          min_samples_leaf: 1,
          splitter: "best",
        },
        question:
          "¿Con qué profundidad obtienes el mejor balance entre accuracy de entrenamiento y de test? ¿Qué pasa con depth=10?",
        hint:
          "Con depth=3-5 el modelo captura bien la forma. Con depth=10, el árbol memoriza el ruido de los datos.",
        expectedObservation:
          "Con max_depth=3-5, la frontera sigue razonablemente la forma de las lunas y el accuracy de test es bueno. Con max_depth=10, la frontera es excesivamente compleja con muchas subdivisiones pequeñas, y aunque el accuracy de entrenamiento es casi perfecto, el de test puede ser menor: esto es overfitting.",
      },
      {
        instruction:
          "Cambia el algoritmo a Random Forest con n_estimators=100 y max_depth=5. Entrena con el mismo dataset Moons y compara con el Árbol de Decisión individual.",
        algorithm: "random_forest",
        dataset: "moons",
        hyperparams: {
          n_estimators: 100,
          max_depth: 5,
          min_samples_split: 2,
          min_samples_leaf: 1,
          max_features: "sqrt",
          bootstrap: "True",
        },
        question:
          "¿Por qué el Random Forest produce una frontera de decisión más suave que un solo Árbol de Decisión con la misma profundidad?",
        hint:
          "Random Forest promedia las predicciones de muchos árboles independientes, cada uno entrenado con una muestra diferente de los datos.",
        expectedObservation:
          "La frontera de decisión es más suave y se adapta mejor a la forma de las lunas. El accuracy de test mejora respecto al árbol individual porque el promedio de muchos árboles reduce la varianza y el sobreajuste.",
      },
      {
        instruction:
          "Experimenta libremente: prueba el Random Forest sin límite de profundidad (max_depth=0) y observa si sobreajusta tanto como el Árbol de Decisión individual sin límite.",
        algorithm: "random_forest",
        dataset: "moons",
        hyperparams: {
          n_estimators: 100,
          max_depth: 0,
          min_samples_split: 2,
          min_samples_leaf: 1,
          max_features: "sqrt",
          bootstrap: "True",
        },
        question:
          "¿Random Forest sobreajusta tanto como un solo árbol cuando no hay límite de profundidad? ¿Por qué?",
        hint:
          "El promedio de muchos árboles sobreajustados aún produce una predicción razonable porque cada árbol sobreajusta de forma diferente.",
        expectedObservation:
          "Aunque cada árbol individual sobreajusta, la votación mayoritaria del ensemble suaviza el resultado. El accuracy de test se mantiene razonablemente alto incluso sin límite de profundidad, demostrando la robustez natural de Random Forest contra el sobreajuste.",
      },
    ],
    finalChallenge:
      "Usa el dataset 'Classification' y encuentra la configuración de Árbol de Decisión que mejor equilibre accuracy de entrenamiento y test. Anota la profundidad óptima y explica por qué ni muy poca ni demasiada profundidad funcionan bien.",
  },

  // ──────────────────────────────────────────────────────────────
  // Taller 2: El Efecto de K en KNN
  // ──────────────────────────────────────────────────────────────
  {
    id: "taller-2-knn-k",
    title: "El Efecto de K en KNN",
    description:
      "Explora cómo el número de vecinos (K) transforma el comportamiento del algoritmo KNN. Verás cómo K=1 memoriza cada punto de entrenamiento, K muy grande ignora los detalles, y existe un valor óptimo intermedio.",
    difficulty: "principiante",
    emoji: "👥",
    duration: "20 min",
    learningObjectives: [
      "Comprender el impacto del hiperparámetro K en KNN",
      "Identificar sobreajuste con K=1 y subajuste con K grande",
      "Entender la diferencia entre pesos uniformes y por distancia",
      "Experimentar con diferentes métricas de distancia",
    ],
    steps: [
      {
        instruction:
          "Selecciona el algoritmo KNN con el dataset 'Two Moons'. Configura K=1 (n_neighbors=1) y entrena. Observa la frontera de decisión.",
        algorithm: "knn",
        dataset: "moons",
        hyperparams: {
          n_neighbors: 1,
          weights: "uniform",
          p: 2,
          metric: "minkowski",
        },
        question:
          "¿Qué forma tiene la frontera de decisión con K=1? ¿Hay alguna isla o región aislada?",
        hint:
          "Con K=1, cada punto de entrenamiento crea su propia 'zona de influencia' y los puntos ruidosos generan regiones pequeñas aisladas.",
        expectedObservation:
          "La frontera de decisión es muy irregular, con muchas regiones pequeñas e islas alrededor de puntos aislados. Cada punto de entrenamiento clasifica su vecindario inmediato en su propia clase. El accuracy de entrenamiento es 100% pero el de test muestra errores por el ruido.",
      },
      {
        instruction:
          "Ahora cambia K a 50 (n_neighbors=50). Entrena de nuevo y observa los cambios en la frontera de decisión y las métricas.",
        algorithm: "knn",
        dataset: "moons",
        hyperparams: {
          n_neighbors: 50,
          weights: "uniform",
          p: 2,
          metric: "minkowski",
        },
        question:
          "¿Por qué la frontera de decisión con K=50 es tan suave? ¿Qué información se pierde?",
        hint:
          "Con K=50, cada predicción consulta a 50 vecinos, lo que suaviza las variaciones locales pero también ignora patrones pequeños pero reales.",
        expectedObservation:
          "La frontera de decisión es excesivamente suave, casi una línea recta, incapaz de seguir la curva de las lunas. El accuracy de test es bajo porque el modelo no captura la estructura no lineal del dataset. Esto es underfitting.",
      },
      {
        instruction:
          "Busca el K óptimo: prueba con K=5, K=7, K=11 y K=15. Compara los resultados para encontrar el mejor valor.",
        algorithm: "knn",
        dataset: "moons",
        hyperparams: {
          n_neighbors: 7,
          weights: "uniform",
          p: 2,
          metric: "minkowski",
        },
        question:
          "¿Cuál es el valor de K que da el mejor accuracy de test? ¿Por qué los valores impares son preferibles?",
        hint:
          "Valores impares evitan empates en la votación. K≈√(n_samples) es una regla común: con 500 muestras, √500 ≈ 22, pero el óptimo suele ser menor.",
        expectedObservation:
          "Con K=5 a K=11, la frontera sigue bien la forma de las lunas sin ser demasiado irregular ni demasiado suave. El accuracy de test es el más alto en este rango. K impares evitan empates en clasificación binaria.",
      },
      {
        instruction:
          "Con el K óptimo que encontraste, cambia weights de 'uniform' a 'distance'. Entrena y compara.",
        algorithm: "knn",
        dataset: "moons",
        hyperparams: {
          n_neighbors: 7,
          weights: "distance",
          p: 2,
          metric: "minkowski",
        },
        question:
          "¿Qué cambia al usar pesos por distancia? ¿Mejora el accuracy de test?",
        hint:
          "Con pesos por distancia, los vecinos más cercanos al punto a clasificar tienen más influencia en la decisión que los vecinos lejanos.",
        expectedObservation:
          "La frontera de decisión se adapta mejor a las variaciones locales porque los vecinos más cercanos tienen más peso. El accuracy de test puede mejorar ligeramente, especialmente en las zonas donde las clases se mezclan cerca de la frontera.",
      },
      {
        instruction:
          "Ahora experimenta con la métrica de distancia: prueba metric='manhattan' (p=1) y luego metric='chebyshev'. Compara con la métrica Euclidiana por defecto.",
        algorithm: "knn",
        dataset: "moons",
        hyperparams: {
          n_neighbors: 7,
          weights: "distance",
          p: 1,
          metric: "manhattan",
        },
        question:
          "¿Cómo cambia la forma de la frontera de decisión con diferentes métricas? ¿Cuál funciona mejor para las lunas?",
        hint:
          "Manhattan mide distancia en forma de 'L', Chebyshev usa la mayor diferencia en cualquier dimensión. Euclidiana es la distancia 'en línea recta'.",
        expectedObservation:
          "Con Manhattan, las fronteras tienden a ser más angulares (forma de diamante). Con Chebyshev, las regiones son más cuadradas. La Euclidiana suele funcionar mejor para datos con estructura curva como las lunas, porque mide la distancia más natural.",
      },
      {
        instruction:
          "Cambia al dataset 'Circles' y prueba KNN con K=5 y weights='distance'. Observa cómo KNN maneja círculos concéntricos.",
        algorithm: "knn",
        dataset: "circles",
        hyperparams: {
          n_neighbors: 5,
          weights: "distance",
          p: 2,
          metric: "minkowski",
        },
        question:
          "¿Por qué KNN funciona bien con el dataset de círculos mientras que un modelo lineal fallaría?",
        hint:
          "KNN es un modelo no paramétrico que no asume ninguna forma funcional. Puede capturar cualquier patrón geométrico si hay suficientes datos.",
        expectedObservation:
          "KNN logra una frontera de decisión aproximadamente circular que separa bien los dos anillos concéntricos. El accuracy es alto porque la estructura local (vecinos cercanos pertenecen a la misma clase) es consistente con la forma del dataset.",
      },
    ],
    finalChallenge:
      "Usa el dataset 'Circles' y encuentra la combinación de K, weights y métrica que maximice el F1-score. Registra tus resultados y explica por qué esa configuración funciona mejor. ¿Qué pasaría si el dataset tuviera más ruido?",
  },

  // ──────────────────────────────────────────────────────────────
  // Taller 3: Regularización en Redes Neuronales
  // ──────────────────────────────────────────────────────────────
  {
    id: "taller-3-regularizacion-nn",
    title: "Regularización en Redes Neuronales",
    description:
      "Explora cómo la arquitectura y la regularización controlan el comportamiento de una Red Neuronal. Verás cómo una red grande memoriza los datos, y cómo el parámetro alpha de regularización la obliga a generalizar.",
    difficulty: "intermedio",
    emoji: "🧠",
    duration: "25 min",
    learningObjectives: [
      "Entender cómo la arquitectura (capas y neuronas) afecta la capacidad del modelo",
      "Observar sobreajuste en redes neuronales con arquitecturas grandes",
      "Comprender el efecto de la regularización L2 (alpha)",
      "Experimentar con diferentes funciones de activación",
    ],
    steps: [
      {
        instruction:
          "Selecciona Red Neuronal (MLP) con el dataset 'Two Moons'. Usa la arquitectura más simple: hidden_layer_sizes=[50], activación ReLU, alpha=0.0001. Entrena y observa.",
        algorithm: "neural_network",
        dataset: "moons",
        hyperparams: {
          hidden_layer_sizes: "50",
          activation: "relu",
          solver: "adam",
          alpha: 0.0001,
          learning_rate_init: 0.001,
          max_iter: 200,
        },
        question:
          "¿Qué forma tiene la frontera de decisión? ¿Es suave o escalonada?",
        hint:
          "Una red con 50 neuronas en una capa tiene suficiente capacidad para crear fronteras curvas suaves.",
        expectedObservation:
          "La frontera de decisión es una curva suave que sigue razonablemente la forma de las lunas. El accuracy es decente pero puede no ser perfecto porque la arquitectura es relativamente simple.",
      },
      {
        instruction:
          "Aumenta la complejidad de la red: usa hidden_layer_sizes=[200,100,50] (3 capas). Mantén alpha=0.0001. Entrena y observa los cambios.",
        algorithm: "neural_network",
        dataset: "moons",
        hyperparams: {
          hidden_layer_sizes: "200,100,50",
          activation: "relu",
          solver: "adam",
          alpha: 0.0001,
          learning_rate_init: 0.001,
          max_iter: 200,
        },
        question:
          "¿El accuracy de entrenamiento mejoró? ¿Y el de test? ¿Hay signos de sobreajuste?",
        hint:
          "Una red muy grande con poca regularización puede memorizar los datos de entrenamiento, incluyendo el ruido.",
        expectedObservation:
          "El accuracy de entrenamiento es casi perfecto, pero la frontera de decisión puede mostrar irregularidades o 'islas' pequeñas donde el modelo memorizó puntos ruidosos. La brecha entre accuracy de entrenamiento y test es un signo de sobreajuste.",
      },
      {
        instruction:
          "Con la misma arquitectura [200,100,50], aumenta alpha a 0.01 y luego a 0.1. Observa cómo cambia la frontera de decisión.",
        algorithm: "neural_network",
        dataset: "moons",
        hyperparams: {
          hidden_layer_sizes: "200,100,50",
          activation: "relu",
          solver: "adam",
          alpha: 0.1,
          learning_rate_init: 0.001,
          max_iter: 200,
        },
        question:
          "¿Cómo afecta el aumento de alpha a la forma de la frontera de decisión? ¿El modelo se vuelve más simple o más complejo?",
        hint:
          "Alpha penaliza los pesos grandes. Pesos más pequeños = transformaciones más suaves = modelo más simple.",
        expectedObservation:
          "Con alpha=0.01, la frontera se suaviza y las irregularidades desaparecen. Con alpha=0.1, la frontera es excesivamente simple, casi lineal, y el accuracy de test disminuye porque el modelo ya no tiene suficiente flexibilidad. La regularización encontró el equilibrio con alpha alrededor de 0.01.",
      },
      {
        instruction:
          "Vuelve a alpha=0.001 y experimenta con la función de activación: prueba 'tanh' y luego 'logistic' (sigmoide). Compara con ReLU.",
        algorithm: "neural_network",
        dataset: "moons",
        hyperparams: {
          hidden_layer_sizes: "100,50",
          activation: "tanh",
          solver: "adam",
          alpha: 0.001,
          learning_rate_init: 0.001,
          max_iter: 200,
        },
        question:
          "¿Qué función de activación produce la mejor frontera de decisión para las lunas? ¿Por qué?",
        hint:
          "Tanh produce valores entre -1 y 1 (centrados en 0), lo cual puede ayudar a que el gradiente fluya mejor en algunos casos. ReLU es más simple pero puede 'apagar' neuronas.",
        expectedObservation:
          "Tanh suele producir fronteras ligeramente más suaves y puede converger mejor para datos con estructura simétrica como las lunas. La sigmoide (logistic) puede ser más lenta en converger. ReLU es la más común y funciona bien en la mayoría de casos.",
      },
      {
        instruction:
          "Prueba el optimizador 'lbfgs' con el dataset 'Iris' y hidden_layer_sizes=[100]. L-BFGS suele funcionar mejor con datasets pequeños.",
        algorithm: "neural_network",
        dataset: "iris",
        hyperparams: {
          hidden_layer_sizes: "100",
          activation: "relu",
          solver: "lbfgs",
          alpha: 0.001,
          learning_rate_init: 0.001,
          max_iter: 300,
        },
        question:
          "¿El optimizador L-BFGS converge mejor que Adam con el dataset Iris? ¿Por qué?",
        hint:
          "L-BFGS es un método de segundo orden que estima la curvatura de la función de pérdida, siendo más eficiente con datasets pequeños.",
        expectedObservation:
          "Con el dataset Iris (solo 150 muestras), L-BFGS converge más rápido y con mayor estabilidad que Adam. El accuracy es alto y consistente entre ejecuciones. Para datasets pequeños, L-BFGS es la mejor opción.",
      },
    ],
    finalChallenge:
      "Usa el dataset 'Circles' y diseña una red neuronal que logre más de 0.90 de F1-score. Experimenta con la arquitectura, activación, alpha y learning_rate. Documenta cada intento y explica qué cambio produjo la mayor mejora.",
  },

  // ──────────────────────────────────────────────────────────────
  // Taller 4: XGBoost vs Random Forest
  // ──────────────────────────────────────────────────────────────
  {
    id: "taller-4-xgboost-vs-rf",
    title: "XGBoost vs Random Forest",
    description:
      "Compara los dos algoritmos de ensemble más populares. Entrenarás ambos con los mismos datos y descubrirás sus fortalezas y debilidades ajustando sus hiperparámetros clave.",
    difficulty: "intermedio",
    emoji: "🚀",
    duration: "30 min",
    learningObjectives: [
      "Entender la diferencia fundamental entre Bagging (RF) y Boosting (XGBoost)",
      "Comparar el rendimiento de ambos algoritmos en el mismo dataset",
      "Ajustar el learning_rate de XGBoost y observar su impacto",
      "Entender la relación entre n_estimators y learning_rate en XGBoost",
    ],
    steps: [
      {
        instruction:
          "Selecciona Random Forest con el dataset 'Iris'. Usa n_estimators=100, max_depth=0 (sin límite). Entrena y anota las métricas (accuracy, precision, recall, F1).",
        algorithm: "random_forest",
        dataset: "iris",
        hyperparams: {
          n_estimators: 100,
          max_depth: 0,
          min_samples_split: 2,
          min_samples_leaf: 1,
          max_features: "sqrt",
          bootstrap: "True",
        },
        question:
          "¿Qué accuracy obtiene Random Forest en Iris? ¿Es un dataset fácil o difícil para este algoritmo?",
        hint:
          "Iris es un dataset relativamente fácil con 3 clases bien separadas. La mayoría de algoritmos logran más del 95% de accuracy.",
        expectedObservation:
          "Random Forest logra un accuracy alto (alrededor de 0.95-1.0) en Iris. Las 3 clases están bien separadas, especialmente setosa vs las demás. La importancia de features suele mostrar que petal length y petal width son las más relevantes.",
      },
      {
        instruction:
          "Ahora selecciona XGBoost con el mismo dataset 'Iris'. Usa los valores por defecto: n_estimators=100, max_depth=6, learning_rate=0.3. Compara las métricas con Random Forest.",
        algorithm: "xgboost",
        dataset: "iris",
        hyperparams: {
          n_estimators: 100,
          max_depth: 6,
          learning_rate: 0.3,
          min_child_weight: 1,
          subsample: 1.0,
          colsample_bytree: 1.0,
          gamma: 0,
          reg_alpha: 0,
          reg_lambda: 1,
        },
        question:
          "¿XGBoost supera a Random Forest en Iris? ¿La diferencia es grande?",
        hint:
          "En datasets limpios y pequeños como Iris, ambos algoritmos suelen tener un rendimiento similar. La diferencia se nota más en datasets grandes y complejos.",
        expectedObservation:
          "XGBoost obtiene métricas similares o ligeramente superiores a Random Forest en Iris. La diferencia es pequeña porque Iris es un dataset limpio y bien estructurado. Ambos algoritmos alcanzan accuracy cercano a 1.0.",
      },
      {
        instruction:
          "Con XGBoost, reduce learning_rate a 0.01 y aumenta n_estimators a 300. Entrena y compara con el paso anterior.",
        algorithm: "xgboost",
        dataset: "iris",
        hyperparams: {
          n_estimators: 300,
          max_depth: 6,
          learning_rate: 0.01,
          min_child_weight: 1,
          subsample: 1.0,
          colsample_bytree: 1.0,
          gamma: 0,
          reg_alpha: 0,
          reg_lambda: 1,
        },
        question:
          "¿Por qué necesitamos más árboles cuando el learning_rate es bajo? ¿Qué pasa si usas learning_rate bajo con pocos árboles?",
        hint:
          "Cada árbol contribuye learning_rate × predicción. Si learning_rate es pequeño, cada árbol aporta muy poco, necesitando más árboles para alcanzar el mismo nivel de corrección.",
        expectedObservation:
          "Con learning_rate=0.01 y n_estimators=300, el rendimiento es comparable o ligeramente mejor que con learning_rate=0.3 y n_estimators=100. Si usaras learning_rate=0.01 con solo 100 árboles, el modelo estaría subentrenado (underfitting) porque no habría suficientes correcciones acumuladas.",
      },
      {
        instruction:
          "Cambia al dataset 'Wine' (más features, más complejo). Entrena ambos algoritmos: primero Random Forest (n_estimators=200, max_depth=0), luego XGBoost (n_estimators=200, max_depth=4, learning_rate=0.1). Compara los F1-scores.",
        algorithm: "xgboost",
        dataset: "wine",
        hyperparams: {
          n_estimators: 200,
          max_depth: 4,
          learning_rate: 0.1,
          min_child_weight: 1,
          subsample: 0.8,
          colsample_bytree: 0.8,
          gamma: 0,
          reg_alpha: 0,
          reg_lambda: 1,
        },
        question:
          "¿Qué algoritmo maneja mejor el dataset Wine con sus 13 features? ¿La diferencia es más notable que en Iris?",
        hint:
          "Wine tiene más features y relaciones más sutiles. XGBoost con boosting puede capturar interacciones entre features que RF podría necesitar más árboles para lograr.",
        expectedObservation:
          "Con el dataset Wine, XGBoost puede tener una ventaja más notoria sobre Random Forest gracias a su capacidad de capturar interacciones complejas entre las 13 features. La importancia de features puede diferir entre ambos algoritmos.",
      },
      {
        instruction:
          "Usa XGBoost con el dataset 'Breast Cancer'. Experimenta con subsample=0.7 y colsample_bytree=0.7. Compara con subsample=1.0 y colsample_bytree=1.0.",
        algorithm: "xgboost",
        dataset: "breast_cancer",
        hyperparams: {
          n_estimators: 200,
          max_depth: 4,
          learning_rate: 0.1,
          min_child_weight: 1,
          subsample: 0.7,
          colsample_bytree: 0.7,
          gamma: 0,
          reg_alpha: 0,
          reg_lambda: 1,
        },
        question:
          "¿Por qué usar solo una fracción de los datos y features en cada árbol puede mejorar el rendimiento?",
        hint:
          "El submuestreo introduce diversidad: cada árbol ve datos y features diferentes, reduciendo la correlación entre árboles y mejorando la generalización del ensemble.",
        expectedObservation:
          "Con subsample=0.7 y colsample_bytree=0.7, el modelo puede generalizar mejor (mejor accuracy de test) que con los valores completos, especialmente en el dataset Breast Cancer que tiene 30 features. El submuestreo de features es particularmente útil cuando hay features irrelevantes o redundantes.",
      },
    ],
    finalChallenge:
      "Usa el dataset 'Breast Cancer' y logra el mayor F1-score posible con XGBoost. Ajusta sistemáticamente: 1) learning_rate y n_estimators, 2) max_depth y min_child_weight, 3) subsample y colsample_bytree, 4) gamma y regularización. Compara tu mejor resultado con el mejor Random Forest que puedas lograr. ¿Cuál gana y por qué?",
  },

  // ──────────────────────────────────────────────────────────────
  // Taller 5: Fronteras de Decisión
  // ──────────────────────────────────────────────────────────────
  {
    id: "taller-5-fronteras-decision",
    title: "Fronteras de Decisión",
    description:
      "Visualiza cómo diferentes algoritmos trazan las fronteras entre clases. Verás desde líneas rectas (Regresión Logística) hasta curvas complejas (Redes Neuronales), entendiendo qué hace que cada algoritmo produzca fronteras diferentes.",
    difficulty: "intermedio",
    emoji: "🗺️",
    duration: "20 min",
    learningObjectives: [
      "Comprender qué es una frontera de decisión y por qué importa su forma",
      "Observar fronteras lineales vs no lineales",
      "Relacionar la complejidad del algoritmo con la flexibilidad de la frontera",
      "Entender por qué ciertos algoritmos fallan con datos no lineales",
    ],
    steps: [
      {
        instruction:
          "Selecciona Regresión Logística con el dataset 'Circles'. Usa C=1.0. Entrena y observa la frontera de decisión.",
        algorithm: "logistic_regression",
        dataset: "circles",
        hyperparams: {
          C: 1.0,
          penalty: "l2",
          solver: "lbfgs",
          max_iter: 100,
        },
        question:
          "¿Qué forma tiene la frontera de decisión? ¿Puede una línea recta separar dos círculos concéntricos?",
        hint:
          "La Regresión Logística solo puede producir fronteras lineales (líneas rectas en 2D, planos en 3D). Los círculos concéntricos requieren una frontera circular.",
        expectedObservation:
          "La frontera de decisión es una línea recta que corta los círculos, clasificando aproximadamente la mitad de cada clase correctamente. El accuracy es bajo (alrededor de 0.50-0.60) porque es imposible separar círculos concéntricos con una línea recta. Este es un ejemplo clásico donde un modelo lineal falla.",
      },
      {
        instruction:
          "Sin cambiar el dataset, cambia el algoritmo a KNN con K=5 y weights='distance'. Entrena y compara la frontera.",
        algorithm: "knn",
        dataset: "circles",
        hyperparams: {
          n_neighbors: 5,
          weights: "distance",
          p: 2,
          metric: "minkowski",
        },
        question:
          "¿Cómo cambia la frontera? ¿KNN logra seguir la forma circular?",
        hint:
          "KNN es un modelo basado en distancia. Los puntos dentro del círculo interior tienen vecinos que también están dentro, formando una región circular.",
        expectedObservation:
          "KNN produce una frontera aproximadamente circular que sigue la forma del círculo interior. El accuracy es alto porque la estructura local del dataset (puntos cercanos pertenecen a la misma clase) es perfecta para KNN.",
      },
      {
        instruction:
          "Cambia a Árbol de Decisión con max_depth=3. Observa la frontera escalonada.",
        algorithm: "decision_tree",
        dataset: "circles",
        hyperparams: {
          criterion: "gini",
          max_depth: 3,
          min_samples_split: 2,
          min_samples_leaf: 1,
          splitter: "best",
        },
        question:
          "¿Por qué la frontera del Árbol de Decisión tiene forma de escalera y no de círculo?",
        hint:
          "Cada división del árbol corta en una sola feature (horizontal o vertical), creando regiones rectangulares anidadas.",
        expectedObservation:
          "La frontera tiene forma de escalera o píxeles, con regiones cuadradas que aproximan el círculo. Con max_depth=3, solo hay 8 regiones máximo, insuficientes para una aproximación suave. Aumentar la profundidad mejora la aproximación pero con más escalones.",
      },
      {
        instruction:
          "Cambia a Red Neuronal con hidden_layer_sizes=[100,50], activación ReLU, alpha=0.001. Entrena y observa.",
        algorithm: "neural_network",
        dataset: "circles",
        hyperparams: {
          hidden_layer_sizes: "100,50",
          activation: "relu",
          solver: "adam",
          alpha: 0.001,
          learning_rate_init: 0.001,
          max_iter: 300,
        },
        question:
          "¿La Red Neuronal produce una frontera más circular que el Árbol de Decisión? ¿Por qué?",
        hint:
          "Las redes neuronales combinan múltiples transformaciones no lineales, permitiendo aproximar cualquier forma continua. Con suficientes neuronas, puede crear una frontera suave y circular.",
        expectedObservation:
          "La frontera de decisión es una curva suave y aproximadamente circular que se adapta bien a la forma del dataset. La red neuronal con 2 capas ocultas tiene suficiente capacidad para aprender la transformación que convierte las coordenadas cartesianas en una frontera circular.",
      },
      {
        instruction:
          "Ahora cambia al dataset 'Two Moons' y prueba con Regresión Logística (C=10). Observa cómo intenta separar las lunas con una línea.",
        algorithm: "logistic_regression",
        dataset: "moons",
        hyperparams: {
          C: 10,
          penalty: "l2",
          solver: "lbfgs",
          max_iter: 100,
        },
        question:
          "¿Puede la Regresión Logística separar las lunas mejor que los círculos? ¿Por qué?",
        hint:
          "Las lunas tienen una forma más 'alargada' que puede ser parcialmente separada por una línea diagonal, aunque no perfectamente.",
        expectedObservation:
          "La Regresión Logística traza una línea diagonal que separa parcialmente las lunas. El accuracy es mejor que con los círculos (alrededor de 0.80-0.85) pero todavía lejos de ser óptimo porque las lunas requieren una frontera curva que un modelo lineal no puede proporcionar.",
      },
    ],
    finalChallenge:
      "Usa el dataset 'Moons' y prueba TODOS los algoritmos disponibles (Regresión Logística, Árbol de Decisión, KNN, Random Forest, Red Neuronal, XGBoost) con configuraciones razonables. Ordena los algoritmos de mejor a peor según su F1-score y explica la relación entre la forma de la frontera y el rendimiento.",
  },

  // ──────────────────────────────────────────────────────────────
  // Taller 6: Optimización de Hiperparámetros
  // ──────────────────────────────────────────────────────────────
  {
    id: "taller-6-optimizacion-hp",
    title: "Optimización de Hiperparámetros",
    description:
      "Aprende a optimizar hiperparámetros de forma sistemática. En lugar de probar valores al azar, descubrirás cómo ajustar un hiperparámetro a la vez, leer la importancia de features, y encontrar la mejor configuración paso a paso.",
    difficulty: "avanzado",
    emoji: "🎯",
    duration: "40 min",
    learningObjectives: [
      "Aprender una metodología sistemática para ajuste de hiperparámetros",
      "Entender la importancia de features y cómo usarla para guiar el ajuste",
      "Observar cómo cada hiperparámetro afecta las métricas de forma diferente",
      "Desarrollar intuición para seleccionar rangos de búsqueda efectivos",
    ],
    steps: [
      {
        instruction:
          "Selecciona XGBoost con el dataset 'Breast Cancer'. Usa todos los valores por defecto (n_estimators=100, max_depth=6, learning_rate=0.3, alpha=0, lambda=1). Entrena y anota el F1-score como baseline.",
        algorithm: "xgboost",
        dataset: "breast_cancer",
        hyperparams: {
          n_estimators: 100,
          max_depth: 6,
          learning_rate: 0.3,
          min_child_weight: 1,
          subsample: 1.0,
          colsample_bytree: 1.0,
          gamma: 0,
          reg_alpha: 0,
          reg_lambda: 1,
        },
        question:
          "¿Cuál es el F1-score baseline? ¿Qué features son las más importantes según el modelo?",
        hint:
          "En Breast Cancer, las features 'worst' (peor valor) suelen ser las más predictivas, especialmente worst radius, worst perimeter y worst concave points.",
        expectedObservation:
          "El F1-score baseline está alrededor de 0.90-0.95. Las features más importantes suelen ser worst concave points, worst radius y worst perimeter, que miden los valores más extremos de las características celulares.",
      },
      {
        instruction:
          "Ajusta max_depth: prueba con 2, 4, 6, 8 y 10. Mantén los demás hiperparámetros fijos. Anota el F1-score para cada valor.",
        algorithm: "xgboost",
        dataset: "breast_cancer",
        hyperparams: {
          n_estimators: 100,
          max_depth: 4,
          learning_rate: 0.3,
          min_child_weight: 1,
          subsample: 1.0,
          colsample_bytree: 1.0,
          gamma: 0,
          reg_alpha: 0,
          reg_lambda: 1,
        },
        question:
          "¿Cuál es la profundidad óptima? ¿El F1-score mejora mucho respecto al baseline?",
        hint:
          "Profundidades muy bajas (2) producen underfitting. Profundidades muy altas (10) producen overfitting. El óptimo suele estar entre 3 y 6 para datasets pequeños.",
        expectedObservation:
          "El F1-score es más alto con max_depth entre 3 y 5. Con max_depth=2, el modelo es demasiado simple (underfitting). Con max_depth=10, el accuracy de entrenamiento es alto pero el de test puede disminuir (overfitting). La mejora respecto al baseline puede ser modesta pero consistente.",
      },
      {
        instruction:
          "Con la mejor max_depth encontrada, ahora ajusta learning_rate y n_estimators juntos. Prueba: (lr=0.01, n=500), (lr=0.05, n=300), (lr=0.1, n=200), (lr=0.3, n=100).",
        algorithm: "xgboost",
        dataset: "breast_cancer",
        hyperparams: {
          n_estimators: 200,
          max_depth: 4,
          learning_rate: 0.1,
          min_child_weight: 1,
          subsample: 1.0,
          colsample_bytree: 1.0,
          gamma: 0,
          reg_alpha: 0,
          reg_lambda: 1,
        },
        question:
          "¿Qué combinación de learning_rate y n_estimators da el mejor resultado? ¿Hay una relación entre ambos?",
        hint:
          "Learning_rate bajo con más árboles suele dar la mejor generalización, pero con rendimientos decrecientes. Learning_rate alto converge rápido pero puede ser inestable.",
        expectedObservation:
          "La combinación learning_rate=0.1 con n_estimators=200 suele dar un buen equilibrio. Learning_rate=0.01 con 500 árboles puede ser ligeramente mejor pero con mucho más tiempo de entrenamiento. Learning_rate=0.3 con 100 árboles es más rápido pero puede no converger tan bien.",
      },
      {
        instruction:
          "Con los mejores valores hasta ahora, ajusta la regularización: prueba gamma=0, 0.5, 1.0; luego reg_alpha=0, 0.1, 1.0; y reg_lambda=1, 5, 10. Cambia UN parámetro a la vez.",
        algorithm: "xgboost",
        dataset: "breast_cancer",
        hyperparams: {
          n_estimators: 200,
          max_depth: 4,
          learning_rate: 0.1,
          min_child_weight: 1,
          subsample: 0.8,
          colsample_bytree: 0.8,
          gamma: 0.5,
          reg_alpha: 0.1,
          reg_lambda: 1,
        },
        question:
          "¿Qué parámetro de regularización tuvo el mayor impacto en el F1-score?",
        hint:
          "Gamma controla la creación de nuevas divisiones (más conservador = más simple). reg_alpha (L1) elimina features. reg_lambda (L2) reduce la magnitud de los pesos.",
        expectedObservation:
          "Gamma suele tener el impacto más notable: aumentar a 0.5 simplifica el modelo y puede mejorar la generalización. reg_alpha y reg_lambda tienen efecto más sutil en este dataset. La regularización es más importante cuando hay overfitting, que se nota si el accuracy de entrenamiento es muy superior al de test.",
      },
      {
        instruction:
          "Cambia al dataset 'Wine' y aplica la misma metodología: empieza con valores por defecto, ajusta max_depth, luego learning_rate/n_estimators, luego regularización.",
        algorithm: "xgboost",
        dataset: "wine",
        hyperparams: {
          n_estimators: 200,
          max_depth: 4,
          learning_rate: 0.1,
          min_child_weight: 3,
          subsample: 0.8,
          colsample_bytree: 0.7,
          gamma: 0.5,
          reg_alpha: 0.1,
          reg_lambda: 1,
        },
        question:
          "¿Son los mismos hiperparámetros óptimos para Wine que para Breast Cancer? ¿Por qué?",
        hint:
          "Cada dataset tiene diferente número de features, muestras, clases y estructura. Los hiperparámetros óptimos dependen de la complejidad inherente de los datos.",
        expectedObservation:
          "Los hiperparámetros óptimos difieren entre datasets. Wine tiene 13 features y 3 clases (vs 30 features y 2 clases en Breast Cancer), así que puede necesitar menos regularización pero más cuidado con el submuestreo de features. colsample_bytree=0.7 puede ser más importante en Wine para evitar overfitting en features irrelevantes.",
      },
      {
        instruction:
          "Prueba Random Forest en el dataset 'Wine' con n_estimators=300, max_depth=0, max_features='sqrt'. Compara con tu mejor XGBoost.",
        algorithm: "random_forest",
        dataset: "wine",
        hyperparams: {
          n_estimators: 300,
          max_depth: 0,
          min_samples_split: 2,
          min_samples_leaf: 1,
          max_features: "sqrt",
          bootstrap: "True",
        },
        question:
          "¿Random Forest con ajuste mínimo supera a XGBoost con ajuste cuidadoso en este dataset? ¿Cuándo preferirías cada uno?",
        hint:
          "Random Forest es más robusto con menos ajuste. XGBoost tiene más potencial pero requiere más cuidado en la configuración.",
        expectedObservation:
          "Random Forest puede obtener resultados competitivos con XGBoost en Wine con mucha menos configuración. Sin embargo, con el ajuste adecuado, XGBoost puede alcanzar un F1-score ligeramente superior. La elección depende del contexto: RF para rapidez y simplicidad, XGBoost para máximo rendimiento cuando se puede invertir tiempo en ajuste.",
      },
    ],
    finalChallenge:
      "Elige el dataset que prefieras y el algoritmo que quieras. Tu objetivo: obtener el mayor F1-score posible. Registra cada experimento en una tabla (hiperparámetros → F1-score) y describe tu estrategia de optimización. ¿Qué hiperparámetro tuvo el mayor impacto? ¿Encontraste alguna interacción sorprendente entre hiperparámetros?",
  },
];
