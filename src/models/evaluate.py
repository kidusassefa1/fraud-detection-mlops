from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    proba_predictions = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, proba_predictions)

    report = classification_report(y_test, predictions)

    return {
        "roc_auc": roc_auc,
        "report": report
    }