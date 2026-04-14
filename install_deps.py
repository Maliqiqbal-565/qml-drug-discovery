import subprocess
import sys
from pathlib import Path

python_exe = Path(sys.argv[0]).resolve().parent / 'venv' / 'Scripts' / 'python.exe'
print('Using interpreter:', python_exe)
cmd = [
    str(python_exe),
    '-m', 'pip',
    'install',
    '--no-cache-dir',
    '--index-url', 'https://pypi.org/simple',
    '--trusted-host', 'pypi.org',
    '--trusted-host', 'files.pythonhosted.org',
    '--timeout', '120',
    '--retries', '5',
    'numpy==1.25.1',
    'pandas',
    'scikit-learn',
    'scipy',
    'matplotlib',
    'seaborn',
    'plotly',
    'pyyaml',
    'tqdm',
    'streamlit==1.56.0'
]
print('Running:', ' '.join(cmd))
proc = subprocess.run(cmd, capture_output=True, text=True)
print('Return code:', proc.returncode)
print('STDOUT:\n', proc.stdout)
print('STDERR:\n', proc.stderr)
Path('pip_install.log').write_text(f'Return code: {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n')
