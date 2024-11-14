# name: 

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse


def load_data(train_file, val_file):
    """
    Don't touch the code below
    Loading the training and validation data splits
    """

    train = pd.read_csv(train_file)
    X_train = train.drop(columns=['label'])
    y_train = train['label']

    val = pd.read_csv(val_file)
    X_val = val.drop(columns=['label'])
    y_val = val['label']

    return X_train, y_train, X_val, y_val


def create_features(X_train):
    """
    Create the feature set by applying various feature engineering techniques such as:

    a. **n_grams** with **TF/IDF** weights: Extract unigrams and bigrams and use "TfidfVectorizer" of sklearn
    b. **Feature Scaling/Normalization**: Standardize or normalize features to ensure they are on the same scale
    c. **Logarithmic/Exponential Transformations**: Apply logarithmic or exponential transformations to handle skewed distributions or non-linearities.
    d. **Polynomial Features**: Although these features  capture non-linear relationships between the original features, they may not be effective/efficient for text classification, why?
    4. **Strategic Feature Selection**: Be mindful of the size of the feature space, as expanding it leads to increased computational cost, sparsity, overfitting, or longer training times.

    The goal is to enhance the predictive power of the model without excessively increasing the dimensionality of the feature space.

    Returns:
    X_extended: The extended feature matrix, including the original features in X and newly added features.
    y: labels for all examples
    """

    """
    Add your code below
    """
    X_extended = X_train

    # creating vectorizer for unigrams and bigrams
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=25000, sublinear_tf=True, max_df=0.95, min_df=2)

    # fit vectorizer on training data and transform
    X_extended = vectorizer.fit_transform(X_train['text'])

    # scaling
    scaler = StandardScaler(with_mean=False)
    X_extended = scaler.fit_transform(X_extended)

    return X_extended, vectorizer


def train_logistic_regression(X_train, y_train, C=1.0, max_iter=1000):
    """
    Train a logistic regression model with the given data.
    X_train (array-like): Feature matrix of the training data.
    y_train (array-like): Labels or target values for the training data.
    C (float, default=1.0): Inverse of regularization strength. Smaller values mean stronger regularization.
    max_iter (int, default=1000): Maximum number of iterations for the solver to converge.

    Returns:
    model: The trained logistic regression model.
    """

    """
    Add your code below
    - create the LogisticRegression model
    - fit the model with the training data
    """

    model = LogisticRegression(C=C, max_iter=max_iter, solver ='sag')

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_val, y_val):
    """
    Don't touch the code below
    Evaluates the model on validation data and return accuracy and log-loss.
    """
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    loss = log_loss(y_val, y_pred_proba)

    return accuracy, loss


if __name__ == "__main__":
    """
    Don't touch the code below
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Path to training examples", type=str, default="data/train.csv")
    parser.add_argument("--val", help="Path to validation examples", type=str, default="data/val.csv")
    parser.add_argument("--C", help="Inverse regularization strength", type=float, default=1.0)
    args = parser.parse_args()

    # Load and preprocess training and validation data
    X_train, y_train, X_val, y_val = load_data(args.train, args.val)

    # Add/Expand features, return the resulting vectorizer to apply to validation data
    X_train, vectorizer = create_features(X_train)

    # Train logistic regression model
    model = train_logistic_regression(X_train, y_train, C=args.C)

    # Transform validation data using the vectorizer
    X_val = vectorizer.transform(X_val['text'])

    # Evaluate the model
    accuracy, loss = evaluate_model(model, X_val, y_val)

    print(f"Accuracy (higher better): {accuracy:.4f}")
    print(f"Log Loss (lower better): {loss:.4f}")

    # Save the trained model
    joblib.dump(model, 'lg_model.pkl')
    print("Model saved as 'lg_model.pkl'")

    # Save the vectorizer for later use
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("vectorizer saved as 'vectorizer.pkl'")
