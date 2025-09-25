\
    @echo off
    setlocal
    cd /d %~dp0\..
    call .venv\Scripts\activate
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
