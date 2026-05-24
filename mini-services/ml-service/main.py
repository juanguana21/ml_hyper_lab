"""
ML HyperLab - Python FastAPI Service
Servicio de Machine Learning para el laboratorio interactivo de hiperparámetros.
"""

import json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import uvicorn

app = FastAPI(title="ML HyperLab Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATASETS
# ============================================================

def get_dataset(name: str):
    """Load a dataset by name and return standardized data."""
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    elif name == "breast_cancer":
        data = load_breast_cancer()
    elif name == "moons":
        X, y = make_moons(n_samples=500, noise=0.15, random_state=42)
        return {
            "X": X, "y": y,
            "feature_names": ["Feature X1", "Feature X2"],
            "target_names": ["Clase 0", "Clase 1"],
            "description": "Dataset sintético con dos lunas intercaladas. Ideal para visualizar fronteras de decisión no lineales.",
            "n_samples": 500, "n_features": 2, "n_classes": 2
        }
    elif name == "circles":
        X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
        return {
            "X": X, "y": y,
            "feature_names": ["Feature X1", "Feature X2"],
            "target_names": ["Clase 0", "Clase 1"],
            "description": "Dataset sintético con círculos concéntricos. Desafío para clasificadores lineales.",
            "n_samples": 500, "n_features": 2, "n_classes": 2
        }
    elif name == "classification":
        X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
        return {
            "X": X, "y": y,
            "feature_names": ["Feature X1", "Feature X2"],
            "target_names": ["Clase 0", "Clase 1"],
            "description": "Dataset sintético de clasificación binaria con 2 características informativas.",
            "n_samples": 500, "n_features": 2, "n_classes": 2
        }
    else:
        raise ValueError(f"Dataset desconocido: {name}")

    return {
        "X": data.data, "y": data.target,
        "feature_names": list(data.feature_names),
        "target_names": list(data.target_names) if hasattr(data, 'target_names') else [f"Clase {i}" for i in range(len(np.unique(data.target)))],
        "description": data.DESCR.split('\n')[0] if data.DESCR else "",
        "n_samples": data.data.shape[0],
        "n_features": data.data.shape[1],
        "n_classes": len(np.unique(data.target))
    }


DATASET_INFO = {
    "iris": {
        "name": "Iris",
        "emoji": "🌸",
        "description": "Clasificación de 3 especies de flores iris según medidas de pétalos y sépalos.",
        "n_samples": 150, "n_features": 4, "n_classes": 3,
        "best_for": "Introducción a clasificación multi-clase"
    },
    "wine": {
        "name": "Wine",
        "emoji": "🍷",
        "description": "Clasificación de vinos según análisis químico. Dataset con características correlacionadas.",
        "n_samples": 178, "n_features": 13, "n_classes": 3,
        "best_for": "Features de alta dimensionalidad"
    },
    "breast_cancer": {
        "name": "Breast Cancer",
        "emoji": "🔬",
        "description": "Diagnóstico de tumores benignos/malignos basado en características de imágenes digitales.",
        "n_samples": 569, "n_features": 30, "n_classes": 2,
        "best_for": "Clasificación binaria con muchas features"
    },
    "moons": {
        "name": "Two Moons",
        "emoji": "🌙",
        "description": "Dos lunas intercaladas. Ideal para visualizar fronteras de decisión no lineales.",
        "n_samples": 500, "n_features": 2, "n_classes": 2,
        "best_for": "Visualizar fronteras de decisión 2D"
    },
    "circles": {
        "name": "Circles",
        "emoji": "⭕",
        "description": "Círculos concéntricos. Desafío para clasificadores lineales.",
        "n_samples": 500, "n_features": 2, "n_classes": 2,
        "best_for": "Probar clasificadores no lineales"
    },
    "classification": {
        "name": "Synthetic Classification",
        "emoji": "📊",
        "description": "Dataset sintético de clasificación binaria con 2 features informativos.",
        "n_samples": 500, "n_features": 2, "n_classes": 2,
        "best_for": "Visualización 2D simple"
    }
}

# ============================================================
# MODELS
# ============================================================

def create_model(algorithm: str, params: dict):
    """Create a model instance with the given hyperparameters."""
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
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
    elif algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", None)) if params.get("max_depth") is not None else None,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            bootstrap=params.get("bootstrap", True),
            random_state=42
        )
    elif algorithm == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 5)),
            weights=params.get("weights", "uniform"),
            algorithm=params.get("algorithm", "auto"),
            p=int(params.get("p", 2)),
            metric=params.get("metric", "minkowski")
        )
    elif algorithm == "neural_network":
        hidden_layer_sizes = tuple(int(x) for x in params.get("hidden_layer_sizes", "100").split(","))
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=params.get("activation", "relu"),
            solver=params.get("solver", "adam"),
            alpha=float(params.get("alpha", 0.0001)),
            learning_rate=params.get("learning_rate_init_type", "constant"),
            learning_rate_init=float(params.get("learning_rate_init", 0.001)),
            max_iter=int(params.get("max_iter", 200)),
            random_state=42
        )
    elif algorithm == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=int(params.get("max_depth", None)) if params.get("max_depth") is not None else None,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            criterion=params.get("criterion", "gini"),
            splitter=params.get("splitter", "best"),
            max_features=params.get("max_features", None),
            random_state=42
        )
    elif algorithm == "logistic_regression":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            penalty=params.get("penalty", "l2"),
            solver=params.get("solver", "lbfgs"),
            max_iter=int(params.get("max_iter", 100)),
            multi_class=params.get("multi_class", "auto"),
            random_state=42
        )
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")

# ============================================================
# API MODELS
# ============================================================

class TrainRequest(BaseModel):
    algorithm: str
    dataset: str
    hyperparameters: dict
    test_size: float = 0.3
    features_for_boundary: Optional[list] = None

class DecisionBoundaryRequest(BaseModel):
    algorithm: str
    dataset: str
    hyperparameters: dict
    feature_indices: list = [0, 1]
    grid_resolution: int = 50

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/api/datasets")
def list_datasets():
    """List all available datasets."""
    return {"datasets": DATASET_INFO}

@app.get("/api/datasets/{name}")
def get_dataset_info(name: str):
    """Get details about a specific dataset."""
    if name not in DATASET_INFO:
        return {"error": f"Dataset '{name}' no encontrado"}
    ds = get_dataset(name)
    info = DATASET_INFO[name].copy()
    info["feature_names"] = ds["feature_names"]
    info["target_names"] = ds["target_names"]
    # Preview first 5 rows
    X_df = pd.DataFrame(ds["X"][:5], columns=ds["feature_names"])
    info["preview"] = X_df.round(3).to_dict(orient="records")
    info["target_preview"] = ds["y"][:5].tolist()
    return info

@app.post("/api/train")
def train_model(request: TrainRequest):
    """Train a model and return metrics, confusion matrix, and feature importance."""
    try:
        ds = get_dataset(request.dataset)
        X, y = ds["X"], ds["y"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create and train model
        model = create_model(request.algorithm, request.hyperparameters)
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Metrics
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_list = cm.tolist()

        # Feature importance
        feature_importance = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, (name, imp) in enumerate(zip(ds["feature_names"], importances)):
                feature_importance.append({
                    "feature": name,
                    "importance": round(float(imp), 4),
                    "index": i
                })
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        elif request.algorithm == "logistic_regression":
            if len(model.coef_.shape) == 1:
                coefs = model.coef_
            else:
                coefs = np.mean(np.abs(model.coef_), axis=0)
            for i, (name, imp) in enumerate(zip(ds["feature_names"], coefs)):
                feature_importance.append({
                    "feature": name,
                    "importance": round(float(abs(imp)), 4),
                    "index": i
                })
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        elif request.algorithm == "knn":
            feature_importance = []

        # Decision tree text (if applicable)
        tree_text = None
        if request.algorithm == "decision_tree":
            try:
                tree_text = export_text(model, feature_names=ds["feature_names"], max_depth=4)
            except:
                tree_text = None

        return {
            "success": True,
            "metrics": metrics,
            "confusion_matrix": cm_list,
            "feature_importance": feature_importance,
            "tree_text": tree_text,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "target_names": ds["target_names"],
            "feature_names": ds["feature_names"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/decision-boundary")
def get_decision_boundary(request: DecisionBoundaryRequest):
    """Compute decision boundary data for 2D visualization."""
    try:
        ds = get_dataset(request.dataset)
        X, y = ds["X"], ds["y"]

        # Use only 2 features
        fi = request.feature_indices[:2]
        X_2d = X[:, fi]
        feature_names_2d = [ds["feature_names"][i] for i in fi]

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_2d)

        # Create mesh grid
        resolution = request.grid_resolution
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Train model on 2D data
        model = create_model(request.algorithm, request.hyperparameters)
        model.fit(X_scaled, y)

        # Predict on grid
        Z = model.predict(grid)
        Z = Z.reshape(xx.shape)

        # Prepare data for visualization
        boundary_data = {
            "grid_x": xx.tolist(),
            "grid_y": yy.tolist(),
            "grid_z": Z.tolist(),
            "points_x": X_scaled[:, 0].tolist(),
            "points_y": X_scaled[:, 1].tolist(),
            "points_labels": y.tolist(),
            "feature_names": feature_names_2d,
            "n_classes": len(np.unique(y)),
            "target_names": ds["target_names"]
        }

        return {"success": True, "boundary": boundary_data}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "ML HyperLab", "version": "1.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3030)
