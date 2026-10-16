import subprocess
import sys
from pathlib import Path

print("Launching script with the current project Python environment...")
script_dir = Path(__file__).resolve().parent
python_path = Path(sys.executable)
subprocess.run([str(python_path), str(script_dir / "test.py")])
