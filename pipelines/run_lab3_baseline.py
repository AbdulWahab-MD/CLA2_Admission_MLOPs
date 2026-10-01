import subprocess, sys
for scr in ['src/preprocess.py', 'src/train.py', 'src/evaluate.py']:
    res = subprocess.run([sys.executable, scr])
    if res.returncode != 0:
        sys.exit(res.returncode)
print("[SUCCESS] pipelines/run_lab3_baseline.py Finished!")
