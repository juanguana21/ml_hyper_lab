#!/bin/bash
# ============================================================
# ML HyperLab - Instalador Automático
# Laboratorio Interactivo de Machine Learning
# ============================================================
# Uso: curl -sL <URL> | bash
#   o: chmod +x install.sh && ./install.sh
# ============================================================

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "============================================================"
echo "  🧠 ML HyperLab — Instalador Automático"
echo "  Laboratorio Interactivo de Machine Learning"
echo "============================================================"
echo ""

# ----------------------------------------------------------
# PASO 0: Verificar prerrequisitos
# ----------------------------------------------------------
echo -e "${BLUE}[PASO 1/6]${NC} Verificando prerrequisitos..."
echo ""

ERRORS=0

# Verificar Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "  ${GREEN}✓${NC} Node.js encontrado: $NODE_VERSION"
else
    echo -e "  ${RED}✗${NC} Node.js NO encontrado"
    echo -e "  ${YELLOW}→ Instalar desde: https://nodejs.org/${NC} (versión LTS recomendada)"
    ERRORS=$((ERRORS + 1))
fi

# Verificar Python3
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "  ${GREEN}✓${NC} Python3 encontrado: $PYTHON_VERSION"
else
    echo -e "  ${RED}✗${NC} Python3 NO encontrado"
    echo -e "  ${YELLOW}→ Instalar desde: https://www.python.org/downloads/${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Verificar pip3
if command -v pip3 &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} pip3 encontrado"
else
    echo -e "  ${RED}✗${NC} pip3 NO encontrado"
    echo -e "  ${YELLOW}→ Instalar con: python3 -m ensurepip${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Verificar git
if command -v git &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Git encontrado"
else
    echo -e "  ${YELLOW}⚠${NC} Git NO encontrado (opcional, solo si clonas desde repo)"
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ Faltan dependencias. Instala las marcadas con ✗ y vuelve a ejecutar este script.${NC}"
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Todos los prerrequisitos están instalados${NC}"
echo ""

# ----------------------------------------------------------
# PASO 1: Crear proyecto Next.js
# ----------------------------------------------------------
echo -e "${BLUE}[PASO 2/6]${NC} Creando proyecto Next.js..."
echo ""

PROJECT_NAME="ml-hyperlab"

if [ -d "$PROJECT_NAME" ]; then
    echo -e "  ${YELLOW}⚠${NC} El directorio '$PROJECT_NAME' ya existe."
    read -p "  ¿Deseas sobreescribirlo? (s/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        rm -rf "$PROJECT_NAME"
    else
        echo -e "  ${YELLOW}Usando directorio existente...${NC}"
    fi
fi

if [ ! -d "$PROJECT_NAME" ]; then
    echo "  Creando proyecto Next.js con TypeScript y Tailwind..."
    npx create-next-app@latest "$PROJECT_NAME" --typescript --tailwind --eslint --app --src-dir --no-turbopack --use-npm > /dev/null 2>&1
    echo -e "  ${GREEN}✓${NC} Proyecto Next.js creado"
fi

cd "$PROJECT_NAME"

# ----------------------------------------------------------
# PASO 2: Instalar shadcn/ui y componentes
# ----------------------------------------------------------
echo ""
echo -e "${BLUE}[PASO 3/6]${NC} Instalando componentes de UI..."
echo ""

# Inicializar shadcn/ui
echo "  Inicializando shadcn/ui..."
npx shadcn@latest init -d > /dev/null 2>&1
echo -e "  ${GREEN}✓${NC} shadcn/ui inicializado"

# Instalar componentes necesarios
echo "  Instalando componentes..."
npx shadcn@latest add card button badge slider select tooltip tabs separator scroll-area -y > /dev/null 2>&1
echo -e "  ${GREEN}✓${NC} Componentes instalados"

# Instalar lucide-react para iconos
echo "  Instalando lucide-react..."
npm install lucide-react > /dev/null 2>&1
echo -e "  ${GREEN}✓${NC} lucide-react instalado"

# ----------------------------------------------------------
# PASO 3: Instalar dependencias Python
# ----------------------------------------------------------
echo ""
echo -e "${BLUE}[PASO 4/6]${NC} Instalando librerías de Python para ML..."
echo ""

pip3 install scikit-learn xgboost numpy pandas 2>&1 | tail -1
echo -e "  ${GREEN}✓${NC} Librerías ML instaladas (scikit-learn, xgboost, numpy, pandas)"

# ----------------------------------------------------------
# PASO 4: Copiar archivos fuente
# ----------------------------------------------------------
echo ""
echo -e "${BLUE}[PASO 5/6]${NC} Copiando archivos fuente del laboratorio..."
echo ""

# Crear directorios
mkdir -p ml-scripts
mkdir -p src/lib

# --- train.py ---
cat > ml-scripts/train.py << 'PYTHON_TRAIN_EOF'
"""
ML HyperLab - Train endpoint script
Receives JSON via stdin, outputs JSON via stdout
"""

import sys
import json
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def get_dataset(name):
    if name == "iris":
        data = load_iris()
        return data.data, data.target, list(data.feature_names), list(data.target_names)
    elif name == "wine":
        data = load_wine()
        return data.data, data.target, list(data.feature_names), list(data.target_names)
    elif name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target, list(data.feature_names), list(data.target_names)
    elif name == "moons":
        X, y = make_moons(n_samples=500, noise=0.15, random_state=42)
        return X, y, ["Feature X1", "Feature X2"], ["Clase 0", "Clase 1"]
    elif name == "circles":
        X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
        return X, y, ["Feature X1", "Feature X2"], ["Clase 0", "Clase 1"]
    elif name == "classification":
        X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
        return X, y, ["Feature X1", "Feature X2"], ["Clase 0", "Clase 1"]
    else:
        raise ValueError(f"Dataset desconocido: {name}")


def create_model(algorithm, params):
    if algorithm == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.3)),
            min_child_weight=int(params.get("min_child_weight", 1)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            gamma=float(params.get("gamma", 0)),
            reg_alpha=float(params.get("reg_alpha", 0)),
            reg_lambda=float(params.get("reg_lambda", 1)),
            use_label_encoder=False, eval_metric='logloss', random_state=42
        )
    elif algorithm == "random_forest":
        max_depth = params.get("max_depth")
        if max_depth is not None and str(max_depth).lower() not in ('none', 'null', ''):
            max_depth = int(max_depth)
        else:
            max_depth = None
        max_features = params.get("max_features", "sqrt")
        if str(max_features) in ("None", "null"):
            max_features = None
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=max_depth,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=max_features,
            bootstrap=str(params.get("bootstrap", "True")) == "True",
            random_state=42
        )
    elif algorithm == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 5)),
            weights=params.get("weights", "uniform"),
            p=int(params.get("p", 2)),
            metric=params.get("metric", "minkowski")
        )
    elif algorithm == "neural_network":
        hidden = tuple(int(x) for x in params.get("hidden_layer_sizes", "100").split(","))
        lr_type = params.get("learning_rate_init_type", "constant")
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=params.get("activation", "relu"),
            solver=params.get("solver", "adam"),
            alpha=float(params.get("alpha", 0.0001)),
            learning_rate=lr_type,
            learning_rate_init=float(params.get("learning_rate_init", 0.001)),
            max_iter=int(params.get("max_iter", 200)),
            random_state=42
        )
    elif algorithm == "decision_tree":
        max_depth = params.get("max_depth")
        if max_depth is not None and str(max_depth).lower() not in ('none', 'null', ''):
            max_depth = int(max_depth)
        else:
            max_depth = None
        max_features = params.get("max_features")
        if max_features and str(max_features) in ("None", "null"):
            max_features = None
        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            criterion=params.get("criterion", "gini"),
            splitter=params.get("splitter", "best"),
            max_features=max_features,
            random_state=42
        )
    elif algorithm == "logistic_regression":
        penalty = params.get("penalty", "l2")
        if str(penalty) == "None":
            penalty = None
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            penalty=penalty,
            solver=params.get("solver", "lbfgs"),
            max_iter=int(params.get("max_iter", 100)),
            random_state=42
        )
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")


def main():
    try:
        input_data = json.loads(sys.stdin.read())
        algorithm = input_data["algorithm"]
        dataset_name = input_data["dataset"]
        params = input_data.get("hyperparameters", {})
        test_size = input_data.get("test_size", 0.3)

        X, y, feature_names, target_names = get_dataset(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = create_model(algorithm, params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        }

        cm = confusion_matrix(y_test, y_pred).tolist()

        feature_importance = []
        if hasattr(model, 'feature_importances_'):
            for i, (name, imp) in enumerate(zip(feature_names, model.feature_importances_)):
                feature_importance.append({"feature": name, "importance": round(float(imp), 4), "index": i})
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        elif algorithm == "logistic_regression":
            coefs = np.mean(np.abs(model.coef_), axis=0) if len(model.coef_.shape) > 1 else model.coef_
            for i, (name, imp) in enumerate(zip(feature_names, coefs)):
                feature_importance.append({"feature": name, "importance": round(float(abs(imp)), 4), "index": i})
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        tree_text = None
        if algorithm == "decision_tree":
            try:
                tree_text = export_text(model, feature_names=feature_names, max_depth=4)
            except:
                pass

        result = {
            "success": True,
            "metrics": metrics,
            "confusion_matrix": cm,
            "feature_importance": feature_importance,
            "tree_text": tree_text,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "target_names": target_names,
            "feature_names": feature_names
        }
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))


if __name__ == "__main__":
    main()
PYTHON_TRAIN_EOF

echo -e "  ${GREEN}✓${NC} ml-scripts/train.py"

# --- boundary.py ---
cat > ml-scripts/boundary.py << 'PYTHON_BOUNDARY_EOF'
"""
ML HyperLab - Decision Boundary script
"""

import sys
import json
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def get_dataset(name):
    if name == "iris":
        data = load_iris()
        return data.data, data.target, list(data.feature_names), list(data.target_names)
    elif name == "wine":
        data = load_wine()
        return data.data, data.target, list(data.feature_names), list(data.target_names)
    elif name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target, list(data.feature_names), list(data.target_names)
    elif name == "moons":
        X, y = make_moons(n_samples=500, noise=0.15, random_state=42)
        return X, y, ["Feature X1", "Feature X2"], ["Clase 0", "Clase 1"]
    elif name == "circles":
        X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
        return X, y, ["Feature X1", "Feature X2"], ["Clase 0", "Clase 1"]
    elif name == "classification":
        X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
        return X, y, ["Feature X1", "Feature X2"], ["Clase 0", "Clase 1"]
    else:
        raise ValueError(f"Dataset desconocido: {name}")


def create_model(algorithm, params):
    if algorithm == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.3)),
            min_child_weight=int(params.get("min_child_weight", 1)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            gamma=float(params.get("gamma", 0)),
            reg_alpha=float(params.get("reg_alpha", 0)),
            reg_lambda=float(params.get("reg_lambda", 1)),
            use_label_encoder=False, eval_metric='logloss', random_state=42
        )
    elif algorithm == "random_forest":
        max_depth = params.get("max_depth")
        if max_depth is not None and str(max_depth).lower() not in ('none', 'null', ''):
            max_depth = int(max_depth)
        else:
            max_depth = None
        max_features = params.get("max_features", "sqrt")
        if str(max_features) in ("None", "null"):
            max_features = None
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=max_depth,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=max_features,
            bootstrap=str(params.get("bootstrap", "True")) == "True",
            random_state=42
        )
    elif algorithm == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 5)),
            weights=params.get("weights", "uniform"),
            p=int(params.get("p", 2)),
            metric=params.get("metric", "minkowski")
        )
    elif algorithm == "neural_network":
        hidden = tuple(int(x) for x in params.get("hidden_layer_sizes", "100").split(","))
        lr_type = params.get("learning_rate_init_type", "constant")
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=params.get("activation", "relu"),
            solver=params.get("solver", "adam"),
            alpha=float(params.get("alpha", 0.0001)),
            learning_rate=lr_type,
            learning_rate_init=float(params.get("learning_rate_init", 0.001)),
            max_iter=int(params.get("max_iter", 200)),
            random_state=42
        )
    elif algorithm == "decision_tree":
        max_depth = params.get("max_depth")
        if max_depth is not None and str(max_depth).lower() not in ('none', 'null', ''):
            max_depth = int(max_depth)
        else:
            max_depth = None
        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            criterion=params.get("criterion", "gini"),
            splitter=params.get("splitter", "best"),
            random_state=42
        )
    elif algorithm == "logistic_regression":
        penalty = params.get("penalty", "l2")
        if str(penalty) == "None":
            penalty = None
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            penalty=penalty,
            solver=params.get("solver", "lbfgs"),
            max_iter=int(params.get("max_iter", 100)),
            random_state=42
        )
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")


def main():
    try:
        input_data = json.loads(sys.stdin.read())
        algorithm = input_data["algorithm"]
        dataset_name = input_data["dataset"]
        params = input_data.get("hyperparameters", {})
        feature_indices = input_data.get("feature_indices", [0, 1])
        resolution = input_data.get("grid_resolution", 50)

        X, y, feature_names, target_names = get_dataset(dataset_name)
        fi = feature_indices[:2]
        X_2d = X[:, fi]
        feature_names_2d = [feature_names[i] for i in fi]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_2d)

        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        model = create_model(algorithm, params)
        model.fit(X_scaled, y)
        Z = model.predict(grid).reshape(xx.shape)

        step = max(1, resolution // 50)
        grid_x = xx[::step, ::step].tolist()
        grid_y = yy[::step, ::step].tolist()
        grid_z = Z[::step, ::step].tolist()

        result = {
            "success": True,
            "boundary": {
                "grid_x": grid_x,
                "grid_y": grid_y,
                "grid_z": grid_z,
                "points_x": X_scaled[:, 0].tolist(),
                "points_y": X_scaled[:, 1].tolist(),
                "points_labels": y.tolist(),
                "feature_names": feature_names_2d,
                "n_classes": len(np.unique(y)),
                "target_names": target_names
            }
        }
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))


if __name__ == "__main__":
    main()
PYTHON_BOUNDARY_EOF

echo -e "  ${GREEN}✓${NC} ml-scripts/boundary.py"

echo ""
echo "  Descargando archivos fuente del frontend..."
echo "  (Estos archivos se deben copiar manualmente desde el proyecto original)"
echo ""

# ----------------------------------------------------------
# PASO 5: Crear script de inicio
# ----------------------------------------------------------
echo -e "${BLUE}[PASO 6/6]${NC} Creando script de inicio..."
echo ""

cat > start.sh << 'START_EOF'
#!/bin/bash
# ML HyperLab - Script de inicio
echo "🧠 Iniciando ML HyperLab..."
echo ""
cd "$(dirname "$0")"
npm run dev
START_EOF

chmod +x start.sh

# Script para Windows
cat > start.bat << 'START_WIN_EOF'
@echo off
echo 🧠 Iniciando ML HyperLab...
echo.
cd /d "%~dp0"
npm run dev
START_WIN_EOF

echo -e "  ${GREEN}✓${NC} start.sh (Mac/Linux)"
echo -e "  ${GREEN}✓${NC} start.bat (Windows)"

# ----------------------------------------------------------
# FIN
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo -e "  ${GREEN}✅ Instalación completada exitosamente!${NC}"
echo "============================================================"
echo ""
echo "  📂 Proyecto creado en: ./$PROJECT_NAME/"
echo ""
echo "  ⚠️  IMPORTANTE: Falta copiar los archivos del frontend"
echo "  Necesitas copiar estos archivos desde el proyecto original:"
echo ""
echo "    src/app/page.tsx          → Interfaz principal"
echo "    src/app/layout.tsx        → Layout de la app"
echo "    src/lib/ml-config.ts      → Configuración de algoritmos"
echo "    src/app/api/ml/train/route.ts"
echo "    src/app/api/ml/boundary/route.ts"
echo "    src/app/api/ml/datasets/route.ts"
echo ""
echo "  🚀 Para ejecutar la app:"
echo "    cd $PROJECT_NAME"
echo "    ./start.sh        # Mac/Linux"
echo "    start.bat         # Windows"
echo ""
echo "  Luego abre: http://localhost:3000"
echo ""
