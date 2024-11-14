import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

TEST_DATA_PATH = "data/val.csv" # will replace this with  "data/test.csv"

def evaluate_submission():


    # Load the test data
    test = pd.read_csv(TEST_DATA_PATH)
    X_test= test.drop(columns=['label'])
    y_test = test['label']

    # Load model from student's directory (you can modify this to train from scratch)
    model_path = os.path.join("lg_model.pkl")
    if not os.path.exists(model_path):
        print(f"Model not found")
        return

    vectorizer_path = os.path.join("vectorizer.pkl")
    if not os.path.exists(vectorizer_path):
        print(f"Vectorizer not found")
        return

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Predict on test data
    X_test = vectorizer.transform(X_test['text'])
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Log Loss: {loss:.4f}")

evaluate_submission()
