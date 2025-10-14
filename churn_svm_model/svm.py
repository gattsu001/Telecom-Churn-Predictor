#Importing the required libraries
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_and_preprocess(csv_path, label_col):
    """
    Load a CSV, return (X, y).
    - Uses only numeric feature columns (simple and robust).
    - Drops rows with missing label.
    """
    #Load Dataset
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Columns: {list(df.columns)}")

    # Drop the label and ID columns
    df = df.dropna(subset=[label_col]).copy()
    y = df[label_col].astype(str).values

    # Drop irrelevant identifier columns
    drop_cols = ["state", "area code", "phone number"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Use only numeric columns
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
    if X.size == 0:
        raise ValueError("No numeric feature columns found.")
    return X, y


def split_dataset(X, y, test_size: float = 0.2, seed: int = 42):
    """
    Split dataset into train and test using scikit learn train_test_split function

    Args:
        X : Feature matrix.
        y : Label vector.
        test_size (float): default 0.2.
        seed (int): Random seed (default 42).

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_pipeline():
    """
    Build a svm pipeline
    StandardScaler -> SVC(RBF)
    - class_weight='balanced' helps when classes are imbalanced.
    - probability=True lets us add ROC-AUC later if we want.
    """
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", clf),
    ])
    return pipe


def train_and_evaluate(csv_path: str, label_col: str, test_size: float = 0.2):
    """
    Train an SVM pipeline on a CSV and print Accuracy, Precision, Recall, F1.

    Args:
        csv_path (str): Path to the dataset CSV.
        label_col (str): Target column name in the CSV (e.g., 'Churn').
        test_size (float): Test split fraction (default 0.2).

    Returns:
        None
    """
    X, y = load_and_preprocess(csv_path, label_col)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=test_size, seed=42)

    #Build the pipeline
    model = build_pipeline()

    #Train the model
    model.fit(X_train, y_train)

    #Predict
    y_pred = model.predict(X_test)

    #Evaluate
    print(classification_report(y_test, y_pred))


def parse_args():
    p = argparse.ArgumentParser(description="SVM for Telecom Churn")
    p.add_argument("--csv", required=True, help="Path to dataset CSV")
    p.add_argument("--label", required=True, help="Label column name (e.g., Churn)")
    p.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default 0.2)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args.csv, args.label, test_size=args.test_size)