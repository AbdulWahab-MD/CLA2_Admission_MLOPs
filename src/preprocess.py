import os, json, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run():
    print("[INFO] Starting Admission Dataset Preprocessing (CLA1 Reference)...")
    df = pd.read_csv("data/raw/admission.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)

    target = "Chance of Admit"
    X = df.drop(columns=[target])
    y = df[target]

    if "Serial No." in X.columns:
        X = X.drop(columns=["Serial No."])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train)
    X_test_final = scaler.transform(X_test)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy())
    np.save("data/processed/y_test.npy", y_test.to_numpy())

    joblib.dump(scaler, "models/scaler.pkl")

    with open("artifacts/preprocessing_summary_report.json","w") as f:
        json.dump({"dataset":"Admission Dataset","shape":list(df.shape),"target":target}, f, indent=4)

    print("[SUCCESS] Preprocessing completed.")

if __name__ == "__main__":
    run()
