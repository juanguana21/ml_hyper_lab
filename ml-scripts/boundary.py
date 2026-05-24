"""
ML HyperLab - Decision Boundary script
Receives JSON via stdin, outputs JSON via stdout
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

        # Convert to lists for JSON serialization
        # For efficiency, only send every other point if resolution is high
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
