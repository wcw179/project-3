import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / 'src' / 'models' / 'train_lstm_tail.py'

print(f"--- Running LSTM Training Script via Debug Wrapper ---")
print(f"Script path: {TRAIN_SCRIPT}")
print(f"Python executable: {sys.executable}")

command = [
    sys.executable,
    str(TRAIN_SCRIPT),
    '--symbol',
    'EURUSDm'
]

try:
    process = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,  # This will raise an exception for non-zero return codes
        encoding='utf-8'
    )
    print("--- STDOUT ---")
    print(process.stdout)
    print("--- STDERR ---")
    print(process.stderr)
    print("--- Script Finished Successfully ---")

except subprocess.CalledProcessError as e:
    print(f"--- Script Failed with Return Code: {e.returncode} ---")
    print("--- STDOUT ---")
    print(e.stdout)
    print("--- STDERR ---")
    print(e.stderr)

except Exception as e:
    print(f"--- An unexpected error occurred in the wrapper ---")
    print(str(e))

