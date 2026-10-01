import os, json, numpy as np
from sklearn.ensemble import RandomForestRegressor

def run():
    X_tr, X_te = np.load("data/processed/X_train_final.npy"), np.load("data/processed/X_test_final.npy")
    y_tr, y_te = np.load("data/processed/y_train.npy"), np.load("data/processed/y_test.npy")
    p = {"n_estimators": 100, "max_depth": 20, "random_state": 42}
    p1 = RandomForestRegressor(**p).fit(X_tr, y_tr).predict(X_te)
    p2 = RandomForestRegressor(**p).fit(X_tr, y_tr).predict(X_te)
    res = np.array_equal(p1, p2)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/reproducibility_report.json", "w") as f:
        json.dump({"deterministic": bool(res), "status": "PASSED" if res else "FAILED"}, f, indent=4)
    print(f"[REPRODUCIBILITY]: {'PASSED' if res else 'FAILED'}")

if __name__ == "__main__":
    run()
