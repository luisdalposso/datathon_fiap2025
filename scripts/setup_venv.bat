    @echo off
    REM Create and activate venv, then install requirements
    setlocal
    cd /d %~dp0\..
    if not exist .venv (
        py -3.11 -m venv .venv
    )
    call .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo.
    echo [OK] Ambiente preparado. Ative com: call .venv\Scripts\activate