"""
Task-aware AutoML: trains a set of candidate models for whichever task type
(classification or regression) was detected, evaluates each on a held-out
split, and returns the best one plus every model's scores (for the
experiment log / leaderboard).
"""
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

CLASSIFICATION_MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
}

REGRESSION_MODELS = {
    "linear_regression": lambda: LinearRegression(),
    "random_forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=42),
}


def select_models(task_type: str, n_rows: int):
    if task_type == "classification":
        names = list(CLASSIFICATION_MODELS.keys())
        registry = CLASSIFICATION_MODELS
    else:
        names = list(REGRESSION_MODELS.keys())
        registry = REGRESSION_MODELS

    if n_rows > 20000:
        # skip the slowest-to-converge simple model on very large datasets
        simple = "logistic_regression" if task_type == "classification" else "linear_regression"
        names = [n for n in names if n != simple]

    return names, registry


def evaluate_classification(y_true, y_pred, y_proba):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
    return metrics


def evaluate_regression(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def primary_metric(task_type: str, metrics: dict):
    if task_type == "classification":
        if "roc_auc" in metrics:
            return "roc_auc", metrics["roc_auc"]
        return "f1_weighted", metrics["f1_weighted"]
    return "r2", metrics["r2"]


def run_automl(X, y, task_type: str):
    model_names, registry = select_models(task_type, len(X))
    stratify = y if task_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    results = []
    best = {"name": None, "model": None, "score": -np.inf, "metrics": None, "metric_name": None}

    for name in model_names:
        try:
            model = registry[name]()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if task_type == "classification":
                proba = None
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(X_test)
                    if p.shape[1] == 2:
                        proba = p[:, 1]
                metrics = evaluate_classification(y_test, preds, proba)
            else:
                metrics = evaluate_regression(y_test, preds)

            metric_name, score = primary_metric(task_type, metrics)
            results.append({"name": name, "metrics": metrics, "metric_name": metric_name, "score": score})

            if score is not None and score > best["score"]:
                best = {"name": name, "model": model, "score": score, "metrics": metrics, "metric_name": metric_name}
        except Exception as e:
            results.append({"name": name, "metrics": {}, "metric_name": None, "score": None, "error": str(e)})

    return best, results, X_train, X_test, y_train, y_test
