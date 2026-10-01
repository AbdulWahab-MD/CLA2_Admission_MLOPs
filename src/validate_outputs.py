import os, json, numpy as np

def run():
    X_tr, X_te = np.load("data/processed/X_train_final.npy"), np.load("data/processed/X_test_final.npy")
    passed = (np.isnan(X_tr).sum() + np.isnan(X_te).sum()) == 0 and (X_tr.shape[1] == X_te.shape[1])
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/preprocessing_summary_report.json", "w") as f:
        json.dump({"passed": bool(passed)}, f, indent=4)
    print(f"[OUTPUT VALIDATION]: {'PASSED' if passed else 'FAILED'}")

if __name__ == "__main__":
    run()
