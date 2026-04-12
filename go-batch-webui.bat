@echo off
chcp 65001 > nul
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "PATH=%SCRIPT_DIR%\runtime;%PATH%"
set "PYTHONIOENCODING=utf-8"

echo ============================================================
echo  GPT-SoVITS Batch Inference WebUI
echo ============================================================
echo.

runtime\python.exe -I webui_batch_inference.py zh_CN

echo.
pause
