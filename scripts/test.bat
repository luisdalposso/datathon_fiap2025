@echo off
setlocal

rem === Go to project root (parent of scripts folder) ===
cd /d "%~dp0\.."

rem === Ensure venv exists ===
if not exist ".venv\Scripts\activate" (
  echo [ERROR] Virtualenv not found at .venv\Scripts\activate
  echo         Create it with: python -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt
  exit /b 1
)

rem === Activate venv ===
call ".venv\Scripts\activate"

rem === Make repo root importable (fix ModuleNotFoundError: src) ===
set "PYTHONPATH=%CD%;%PYTHONPATH%"

rem === Enable coverage if pytest-cov is installed ===
python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('pytest_cov') else 1)"
if %ERRORLEVEL%==0 (
  set "COV_ARGS=--cov=src --cov-report=term-missing"
) else (
  set "COV_ARGS="
)

rem === Run tests (forward any extra args) ===
python -m pytest -q %COV_ARGS% %*

set EXITCODE=%ERRORLEVEL%
endlocal & exit /b %EXITCODE%