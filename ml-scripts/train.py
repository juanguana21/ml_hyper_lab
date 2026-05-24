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
