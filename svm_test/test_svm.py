import os, sys, tempfile
import pandas as pd

# Ensure the repo root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from churn_svm_model.svm import train_and_evaluate

def test_svm_runs_on_tiny_csv():
    """
    Minimal test to ensure train_and_evaluate() runs successfully on a small synthetic dataset.
    """
    # Create a small dataset with numeric features + binary label
    df = pd.DataFrame({
        "f1": [0, 1, 1, 0, 2, 1],
        "f2": [1, 0, 1, 0, 1, 2],
        "churn": ["False", "True", "True", "False", "True", "False"],
    })

    # Save to a temporary CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        csv_path = tmp.name
    df.to_csv(csv_path, index=False)

    try:
        # Function should execute without throwing errors
        train_and_evaluate(csv_path, label_col="churn", test_size=0.5)
    finally:
        os.remove(csv_path)
