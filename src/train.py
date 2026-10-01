import os, joblib, numpy as np
from sklearn.ensemble import RandomForestRegressor

def run():
    print("[INFO] Starting Baseline Training (Lab 2)...")
    X = np.load("data/processed/X_train_final.npy")
    y = np.load("data/processed/y_train.npy")
    m = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=2, min_samples_leaf=2, random_state=42).fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(m, "models/random_forest_baseline.pkl")
    print("[SUCCESS] Lab 2 Model Saved to models/random_forest_baseline.pkl")

if __name__ == "__main__":
    run()
