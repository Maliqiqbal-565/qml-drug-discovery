import subprocess
import sys
from pathlib import Path

cmd = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'numpy==1.25.1']
proc = subprocess.run(cmd, capture_output=True, text=True)
output = f'RC={proc.returncode}\nSTDOUT={proc.stdout}\nSTDERR={proc.stderr}\n'
Path('pip_debug.log').write_text(output)
print(output)
