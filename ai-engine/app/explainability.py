"""
Feature importance for whichever model won AutoML.
Tries SHAP first (best explanations for tree models & linear coefficients),
and falls back to permutation importance (model-agnostic, always works) if
SHAP errors out for any reason. Never raises -- worst case returns [].
"""
import numpy as np


def get_top_features(model, model_name, X_test, task_type, top_k=5):
    feature_names = list(X_test.columns)

    try:
        import shap

        if model_name in ("random_forest", "gradient_boosting"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_test)
            sv = np.array(sv)
            if sv.ndim == 3:
                # (n_samples, n_features, n_classes) -> use the positive/last class
                sv = sv[:, :, -1]
            importance = np.mean(np.abs(sv), axis=0)
        elif model_name == "logistic_regression":
            importance = np.abs(model.coef_[0])
        elif model_name == "linear_regression":
            importance = np.abs(model.coef_)
        else:
            raise ValueError(f"no SHAP path defined for '{model_name}'")

        importance = np.asarray(importance).reshape(-1)
        if importance.shape[0] != len(feature_names):
            raise ValueError("SHAP importance shape mismatch")

        pairs = sorted(zip(feature_names, importance.tolist()), key=lambda t: -t[1])
        return pairs[:top_k], "shap"

    except Exception as e:
        print("⚠️ SHAP failed, falling back to permutation importance:", e)

    try:
        from sklearn.inspection import permutation_importance

        scoring = "accuracy" if task_type == "classification" else "r2"
        r = permutation_importance(
            model, X_test, model.predict(X_test), n_repeats=5, random_state=42, scoring=scoring
        )
        pairs = sorted(zip(feature_names, r.importances_mean.tolist()), key=lambda t: -t[1])
        return pairs[:top_k], "permutation_importance"
    except Exception as e2:
        print("⚠️ Permutation importance also failed:", e2)
        return [], "unavailable"
