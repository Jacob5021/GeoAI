"""One-command start on any machine:  python run.py

Creates ./venv if needed, installs requirements when they change, then starts the server.
The database and uploaded data are created locally in ./data on first start (never committed).

Environment: GEOAI_HOST (default 127.0.0.1), GEOAI_PORT (default 8599), GEOAI_DATA_DIR (default ./data).
"""
import os
import subprocess
import sys
import venv

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, "venv")
PY = os.path.join(VENV, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV, "bin", "python")
REQS = os.path.join(HERE, "requirements.txt")
STAMP = os.path.join(VENV, ".requirements-installed")

if sys.version_info < (3, 10):
    sys.exit("GeoAI Tools needs Python 3.10 or newer")

if not os.path.exists(PY):
    print("Creating virtual environment in ./venv ...")
    venv.create(VENV, with_pip=True)

if not os.path.exists(STAMP) or os.path.getmtime(STAMP) < os.path.getmtime(REQS):
    print("Installing requirements (first run downloads PyTorch, which takes a while) ...")
    subprocess.check_call([PY, "-m", "pip", "install", "-r", REQS])
    open(STAMP, "w").close()

host, port = os.environ.get("GEOAI_HOST", "127.0.0.1"), os.environ.get("GEOAI_PORT", "8599")
data_dir = os.path.abspath(os.environ.get("GEOAI_DATA_DIR", os.path.join(HERE, "data")))
print(f"\n  GeoAI Tools  ->  http://localhost:{port}\n  Data stored in {data_dir}\n")
sys.exit(subprocess.call([PY, "-m", "uvicorn", "server:app", "--host", host, "--port", port], cwd=HERE))
