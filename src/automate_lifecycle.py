import mlflow
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
