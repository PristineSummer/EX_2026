"""Single entry point. All generated state stays beneath EX_2026."""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parent
WORKSPACE = PROJECT.parent
CACHE = WORKSPACE / ".cache" / "lab1"
for name in ("tmp", "matplotlib", "pycache"):
    (CACHE / name).mkdir(parents=True, exist_ok=True)
os.environ.update({
    "TEMP": str(CACHE / "tmp"), "TMP": str(CACHE / "tmp"),
    "TMPDIR": str(CACHE / "tmp"), "MPLCONFIGDIR": str(CACHE / "matplotlib"),
    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
    "PYTHONPYCACHEPREFIX": str(CACHE / "pycache"),
})
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if os.name == "nt":
    expected = WORKSPACE.parent / ".venv" / "Scripts" / "python.exe"
    if Path(sys.executable).resolve() != expected.resolve():
        raise SystemExit(f"请使用指定解释器运行：{expected}")

from scripts.experiment import main

if __name__ == "__main__":
    raise SystemExit(main(PROJECT, WORKSPACE))
