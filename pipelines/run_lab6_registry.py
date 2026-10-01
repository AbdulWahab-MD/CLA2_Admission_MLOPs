import subprocess, sys
for scr in ['src/train_registry.py', 'src/automate_lifecycle.py', 'src/generate_registry_report.py']:
    res = subprocess.run([sys.executable, scr])
    if res.returncode != 0:
        sys.exit(res.returncode)
print("[SUCCESS] pipelines/run_lab6_registry.py Finished!")

