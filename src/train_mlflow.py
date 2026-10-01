import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_and_track(run_name="RandomForest_Tuned", params=None):
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "random_state": 42
        }

    print(f"\n--- Starting MLflow Run: {run_name} ---")

    # 1. Load Processed Data
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')

    # Set Experiment Name
    mlflow.set_experiment("Admission_Prediction")

    with mlflow.start_run(run_name=run_name):
        # 2. Log Hyperparameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForestRegressor")
        mlflow.log_param("task_type", "regression")

        # 3. Train Model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # 4. Predict and Evaluate
        y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2
        }

        # 5. Log Metrics
        mlflow.log_metrics(metrics)
        print(f"Metrics logged: RMSE = {rmse:.4f} | MAE = {mae:.4f} | R^2 = {r2:.4f}")

        # 6. Generate and Log Regression Diagnostic Plots
        os.makedirs("artifacts", exist_ok=True)

        # Plot A: Actual vs Predicted
        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 5))
        ax_scatter.scatter(y_test, y_pred, alpha=0.6, color='teal', edgecolors='k')
        ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax_scatter.set_xlabel("Actual Quality")
        ax_scatter.set_ylabel("Predicted Quality")
        ax_scatter.set_title(f"Actual vs Predicted - {run_name}")
        ax_scatter.grid(True, linestyle="--", alpha=0.5)
        scatter_path = "artifacts/actual_vs_predicted.png"
        fig_scatter.savefig(scatter_path, bbox_inches="tight")
        plt.close(fig_scatter)
        mlflow.log_artifact(scatter_path, artifact_path="plots")

        # Plot B: Residuals Distribution
        residuals = y_test - y_pred
        fig_res, ax_res = plt.subplots(figsize=(6, 5))
        ax_res.hist(residuals, bins=20, color='coral', edgecolor='black')
        ax_res.axvline(0, color='black', linestyle='dashed', linewidth=1.5)
        ax_res.set_xlabel("Residual (Actual - Predicted)")
        ax_res.set_ylabel("Frequency")
        ax_res.set_title(f"Residuals Distribution - {run_name}")
        ax_res.grid(True, linestyle="--", alpha=0.5)
        res_path = "artifacts/residual_distribution.png"
        fig_res.savefig(res_path, bbox_inches="tight")
        plt.close(fig_res)
        mlflow.log_artifact(res_path, artifact_path="plots")

        # 7. Log Preprocessing Metadata for Lineage
        if os.path.exists('data/processed/dataset_metadata.json'):
            mlflow.log_artifact('data/processed/dataset_metadata.json', artifact_path="metadata")

        # 8. Log Model Artifact with explicit skops trusted types for scikit-learn tree models
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        # Save local backup
        joblib.dump(model, 'models/random_forest_model.pkl')
        print(f"[SUCCESS] Run '{run_name}' successfully tracked in MLflow!")

if __name__ == "__main__":
    train_and_track(run_name="RandomForest_Tuned_V1")