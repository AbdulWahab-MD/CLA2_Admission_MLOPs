import subprocess, sys
for scr in ['src/preprocess.py', 'src/train_mlflow.py', 'src/validate_reproducibility.py']:
    res = subprocess.run([sys.executable, scr])
    if res.returncode != 0:
        sys.exit(res.returncode)
print("[SUCCESS] pipelines/run_lab4_tracking.py Finished!")
