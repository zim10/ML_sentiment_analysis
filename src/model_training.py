from sklearn.ensemble import RandomForestClassifier

def create_model(n_estimators=100):
    """Create Randomforest model"""
    return RandomForestClassifier(n_estimators=n_estimators)
def create_model(n_estimators =300):
    """create random forest model"""
    return RandomForestClassifier(n_estimators=n_estimators)

def train_model(model, X_train, y_train):
    """Train the model"""
    return model.fit(X_train, y_train)

def create_model(n_estimators=400): # Changed default value
    """Create improved RandomForest model"""
    return RandomForestClassifier(n_estimators= n_estimators,max_depth=10)