@echo off
set PYTHON_EXE=C:\Users\User\anaconda3\envs\a2b-gru\python.exe
if not exist "%PYTHON_EXE%" (
  echo Could not find %PYTHON_EXE%
  pause
  exit /b 1
)
echo Launching GUI with %PYTHON_EXE%
echo If the terminal stays busy, check your taskbar or Alt+Tab for the window.
"%PYTHON_EXE%" launch_gui.py
if errorlevel 1 pause
