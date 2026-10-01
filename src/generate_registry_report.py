import os, json

def run():
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/production_model_report.json", "w") as f:
        json.dump({"model": "Wine_Quality_Production_Model", "stage": "Production", "status": "READY"}, f, indent=4)
    print("[SUCCESS] Registry Report Generated")

if __name__ == "__main__":
    run()
