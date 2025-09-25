    @echo off
    setlocal
    cd /d %~dp0\..
    docker build -f docker/Dockerfile -t decision-match:latest .
    docker run --rm -it -p 8000:8000 --name decision-match decision-match:latest