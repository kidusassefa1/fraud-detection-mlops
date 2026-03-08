from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test, amounts=None):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    false_positive_rate = fp / (fp + tn)

    fraud_detection_rate = tp / (tp + fn)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "false_positive_rate": false_positive_rate,
        "fraud_detection_rate": fraud_detection_rate,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn
    }

    # Optional business metrics
    if amounts is not None:

        fraud_amount_detected = amounts[(predictions == 1) & (y_test == 1)].sum()
        fraud_amount_missed = amounts[(predictions == 0) & (y_test == 1)].sum()

        metrics["fraud_amount_detected"] = fraud_amount_detected
        metrics["fraud_amount_missed"] = fraud_amount_missed

    return metrics