import shap


def explain_prediction(model, feature_row, feature_names, top_n: int = 3):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_row)

    if shap_values.ndim > 1:
        shap_values = shap_values[0]

    contributions = list(zip(feature_names, shap_values, strict=True))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {"feature": name, "shap_value": float(value)}
        for name, value in contributions[:top_n]
    ]
