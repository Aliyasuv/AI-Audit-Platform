from sklearn.metrics import accuracy_score, classification_report

def evaluate_detector(detector, X_test, y_test):
    y_pred = detector.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report

# Add more evaluation functions as you expand your backend