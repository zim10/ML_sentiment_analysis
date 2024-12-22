from sklearn.metrics import accuracy_score, f1_score

def evaluate_experiment(y_true, y_pred):
    """evaluete expermient"""
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1 score" : f1_score(y_true, y_pred, average= "weighted")
    }
    return results