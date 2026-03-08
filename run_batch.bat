@echo off
chcp 65001 > nul
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "PATH=%SCRIPT_DIR%\runtime;%PATH%"

echo ============================================================
echo  GPT-SoVITS Batch Inference
echo ============================================================
echo.

runtime\python.exe -I batch_inference.py ^
  --input_list inputs/mod_input/*.list ^
  --output_dir output/ ^
  --speaker_config inputs/speaker_config.yaml ^
  --output_sr 44100 ^
  --output_channels 2 

echo.
echo ============================================================
echo  Finished! Check the output/ folder for results.
echo ============================================================
pause
