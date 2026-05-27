@echo off
set PYTHON_EXE=C:\Users\User\anaconda3\envs\a2b-gru\python.exe
if not exist "%PYTHON_EXE%" (
  echo Could not find %PYTHON_EXE%
  pause
  exit /b 1
)
"%PYTHON_EXE%" evaluate_models.py --loc-index 1 --time-index 0 --epochs 30
if errorlevel 1 pause
pause
