import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run():
    print("[INFO] Starting Admission Preprocessing Pipeline...")

    # Load Admission dataset
    df = pd.read_csv("data/raw/admission.csv")

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Target column
    target = "Chance of Admit"

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Remove Serial Number because it is only an ID
    if "Serial No." in X.columns:
        X = X.drop(columns=["Serial No."])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scale features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create processed directory
    os.makedirs("data/processed", exist_ok=True)

    # Save processed data
    np.save("data/processed/X_train_final.npy", X_train_scaled)
    np.save("data/processed/X_test_final.npy", X_test_scaled)
    np.save("data/processed/y_train.npy", y_train.to_numpy())
    np.save("data/processed/y_test.npy", y_test.to_numpy())

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    print("[SUCCESS] Admission preprocessing pipeline completed.")


if __name__ == "__main__":
    run()