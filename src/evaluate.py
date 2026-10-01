import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def run():
    print("[INFO] Evaluating Baseline Model (Lab 2)...")

    # Load the original Admission dataset

    raw = pd.read_csv("data/raw/admission.csv")

    # Remove unwanted spaces from column names
    raw.columns = raw.columns.str.strip()

    # Target column
    target = "Chance of Admit"

    # Features and target
    X = raw.drop(columns=[target])
    y = raw[target]

    # Remove Serial Number because it is only an ID
    if "Serial No." in X.columns:
        X = X.drop(columns=["Serial No."])

    # Split the data in the same way
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Load trained baseline model
    model_path = "models/random_forest_baseline.pkl"

    if not os.path.exists(model_path):
        print("[ERROR] Baseline model not found:", model_path)
        return

    model = joblib.load(model_path)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Test RMSE: {rmse:.4f} | R2: {r2:.4f}")

    print("[SUCCESS] Evaluation completed.")


if __name__ == "__main__":
    run()