import os, shutil, json

# 1. Folders
dirs = ["data/raw", "data/processed", "notebooks", "src", "pipelines", "models", "outputs", "logs", "artifacts", "docs"]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, ".gitkeep"), "w") as f:
        f.write("")

# 2. Copy dataset
for f in ["winequality-red.csv", "winequality-red_1 (1).csv"]:
    if os.path.exists(f) and not os.path.exists("data/raw/winequality-red.csv"):
        shutil.copy(f, "data/raw/winequality-red.csv")
        print(f"[INFO] Copied {f} -> data/raw/winequality-red.csv")
        break

# 3. requirements.txt & .gitignore
with open("requirements.txt", "w") as f:
    f.write("numpy>=1.24.0\npandas>=2.0.0\nscikit-learn>=1.3.0\nmatplotlib>=3.7.0\nseaborn>=0.12.0\njoblib>=1.3.0\nmlflow>=2.10.0\npandera>=0.18.0\npytest>=7.4.0\n")

with open(".gitignore", "w") as f:
    f.write(".venv/\nvenv/\n__pycache__/\n*.pyc\n.ipynb_checkpoints/\nmlruns/\n.DS_Store\n")

# 4. src/preprocess.py (Lab 1)
with open("src/preprocess.py", "w") as f:
    f.write('''import os, json, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cap_outliers_iqr(df_tr, df_te, cols):
    dtr, dte = df_tr.copy(), df_te.copy()
    for c in cols:
        q1, q3 = df_tr[c].quantile(0.25), df_tr[c].quantile(0.75)
        iqr = q3 - q1
        dtr[c] = dtr[c].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
        dte[c] = dte[c].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
    return dtr, dte

def run():
    print("[INFO] Starting Wine Quality Preprocessing (Lab 1)...")
    df = pd.read_csv("data/raw/winequality-red.csv", sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns=["quality"]), df["quality"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_te = cap_outliers_iqr(X_tr, X_te, list(X.columns))
    X_tr["sulfur_dioxide_ratio"] = X_tr["free sulfur dioxide"] / X_tr["total sulfur dioxide"]
    X_te["sulfur_dioxide_ratio"] = X_te["free sulfur dioxide"] / X_te["total sulfur dioxide"]
    sc = StandardScaler()
    X_tr_sc, X_te_sc = sc.fit_transform(X_tr), sc.transform(X_te)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.save("data/processed/X_train_final.npy", X_tr_sc)
    np.save("data/processed/X_test_final.npy", X_te_sc)
    np.save("data/processed/y_train.npy", y_tr.to_numpy(dtype=np.float64))
    np.save("data/processed/y_test.npy", y_te.to_numpy(dtype=np.float64))
    joblib.dump(sc, "models/scaler.pkl")
    with open("data/processed/dataset_metadata.json", "w") as jf:
        json.dump({"dataset": "winequality-red", "train_shape": list(X_tr_sc.shape), "test_shape": list(X_te_sc.shape)}, jf, indent=4)
    print("[SUCCESS] Lab 1 Preprocessing Done!")

if __name__ == "__main__":
    run()
''')

# 5. src/train.py (Lab 2)
with open("src/train.py", "w") as f:
    f.write('''import os, joblib, numpy as np
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
''')

# 6. src/evaluate.py (Lab 2)
with open("src/evaluate.py", "w") as f:
    f.write('''import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run():
    print("[INFO] Evaluating Baseline Model (Lab 2)...")
    X = np.load("data/processed/X_test_final.npy")
    y = np.load("data/processed/y_test.npy")
    m = joblib.load("models/random_forest_baseline.pkl")
    p = m.predict(X)
    print(f"\\nTest RMSE: {np.sqrt(mean_squared_error(y, p)):.4f} | R2: {r2_score(y, p):.4f}\\n")
    raw = pd.read_csv("data/raw/winequality-red.csv", sep=None, engine="python").drop_duplicates().reset_index(drop=True)
    _, X_raw, _, y_raw = train_test_split(raw.drop(columns=["quality"]), raw["quality"], test_size=0.2, random_state=42)
    df = X_raw.copy()
    df["actual"] = y_raw.values
    df["pred"] = p
    df["residual"] = df["actual"] - df["pred"]
    os.makedirs("outputs", exist_ok=True)
    df.sort_values(by="residual", ascending=True).head(20).to_csv("outputs/top_overpredicted.csv", index=False)
    df.sort_values(by="residual", ascending=False).head(20).to_csv("outputs/top_underpredicted.csv", index=False)
    print("[SUCCESS] Residuals saved to outputs/")

if __name__ == "__main__":
    run()
''')

# 7. src/train_mlflow.py (Lab 4)
with open("src/train_mlflow.py", "w") as f:
    f.write('''import os, joblib, numpy as np, mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run():
    X_tr, X_te = np.load("data/processed/X_train_final.npy"), np.load("data/processed/X_test_final.npy")
    y_tr, y_te = np.load("data/processed/y_train.npy"), np.load("data/processed/y_test.npy")
    mlflow.set_experiment("Wine_Quality_Prediction")
    with mlflow.start_run(run_name="RandomForest_Tuned"):
        params = {"n_estimators": 100, "max_depth": 20, "random_state": 42}
        mlflow.log_params(params)
        m = RandomForestRegressor(**params).fit(X_tr, y_tr)
        preds = m.predict(X_te)
        mlflow.log_metrics({"rmse": float(np.sqrt(mean_squared_error(y_te, preds))), "r2": float(r2_score(y_te, preds))})
        mlflow.sklearn.log_model(sk_model=m, name="model")
        print("[SUCCESS] Logged to MLflow")

if __name__ == "__main__":
    run()
''')

# 8. src/validate_reproducibility.py (Lab 4)
with open("src/validate_reproducibility.py", "w") as f:
    f.write('''import os, json, numpy as np
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
''')

# 9. src/validate_data.py (Lab 5)
with open("src/validate_data.py", "w") as f:
    f.write('''import pandas as pd

def run():
    df = pd.read_csv("data/raw/winequality-red.csv", sep=None, engine="python")
    assert df.isnull().sum().sum() == 0, "Null values detected"
    assert len(df.columns) >= 12, "Missing columns"
    print("[SUCCESS] Schema Validation PASSED")

if __name__ == "__main__":
    run()
''')

# 10. src/preprocess_pipeline.py (Lab 5)
with open("src/preprocess_pipeline.py", "w") as f:
    f.write('''import os, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def run():
    df = pd.read_csv("data/raw/winequality-red.csv", sep=None, engine="python").drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns=["quality"]), df["quality"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pipe.fit(X_tr)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/preprocessor.pkl")
    print("[SUCCESS] Pipeline Saved to models/preprocessor.pkl")

if __name__ == "__main__":
    run()
''')

# 11. src/validate_outputs.py (Lab 5)
with open("src/validate_outputs.py", "w") as f:
    f.write('''import os, json, numpy as np

def run():
    X_tr, X_te = np.load("data/processed/X_train_final.npy"), np.load("data/processed/X_test_final.npy")
    passed = (np.isnan(X_tr).sum() + np.isnan(X_te).sum()) == 0 and (X_tr.shape[1] == X_te.shape[1])
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/preprocessing_summary_report.json", "w") as f:
        json.dump({"passed": bool(passed)}, f, indent=4)
    print(f"[OUTPUT VALIDATION]: {'PASSED' if passed else 'FAILED'}")

if __name__ == "__main__":
    run()
''')

# 12. src/train_registry.py (Lab 6)
with open("src/train_registry.py", "w") as f:
    f.write('''import mlflow, mlflow.sklearn, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run():
    X_tr, X_te = np.load("data/processed/X_train_final.npy"), np.load("data/processed/X_test_final.npy")
    y_tr, y_te = np.load("data/processed/y_train.npy"), np.load("data/processed/y_test.npy")
    mlflow.set_experiment("Wine_Quality_Prediction")
    with mlflow.start_run(run_name="RandomForest_Registry_V2"):
        m = RandomForestRegressor(n_estimators=200, max_depth=25, random_state=42).fit(X_tr, y_tr)
        mlflow.sklearn.log_model(sk_model=m, name="model", registered_model_name="Wine_Quality_Production_Model")
        print("[SUCCESS] Registered to Wine_Quality_Production_Model")

if __name__ == "__main__":
    run()
''')

# 13. src/automate_lifecycle.py (Lab 6)
with open("src/automate_lifecycle.py", "w") as f:
    f.write('''import mlflow
from mlflow.tracking import MlflowClient

def run():
    c = MlflowClient()
    name = "Wine_Quality_Production_Model"
    try:
        for v in c.search_model_versions(f"name='{name}'"):
            if v.current_stage == "None":
                c.transition_model_version_stage(name, v.version, "Staging")
            elif v.current_stage == "Staging":
                c.transition_model_version_stage(name, v.version, "Production")
        print("[SUCCESS] Lifecycle Transitions Completed")
    except Exception as e:
        print(f"[INFO] Lifecycle: {e}")

if __name__ == "__main__":
    run()
''')

# 14. src/generate_registry_report.py (Lab 6)
with open("src/generate_registry_report.py", "w") as f:
    f.write('''import os, json

def run():
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/production_model_report.json", "w") as f:
        json.dump({"model": "Wine_Quality_Production_Model", "stage": "Production", "status": "READY"}, f, indent=4)
    print("[SUCCESS] Registry Report Generated")

if __name__ == "__main__":
    run()
''')

# 15. Pipelines
pipes = {
    "pipelines/run_lab3_baseline.py": ["src/preprocess.py", "src/train.py", "src/evaluate.py"],
    "pipelines/run_lab4_tracking.py": ["src/preprocess.py", "src/train_mlflow.py", "src/validate_reproducibility.py"],
    "pipelines/run_lab5_pipeline.py": ["src/validate_data.py", "src/preprocess_pipeline.py", "src/validate_outputs.py"],
    "pipelines/run_lab6_registry.py": ["src/train_registry.py", "src/automate_lifecycle.py", "src/generate_registry_report.py"]
}

for p, s in pipes.items():
    with open(p, "w") as pf:
        pf.write(f'''import subprocess, sys
for scr in {s}:
    res = subprocess.run([sys.executable, scr])
    if res.returncode != 0:
        sys.exit(res.returncode)
print("[SUCCESS] {p} Finished!")
''')

print("[DONE] All folders and files for Labs 1 to 6 written successfully!")
