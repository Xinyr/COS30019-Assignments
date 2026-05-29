@echo off
set "PYTHON_EXE="

if defined CONDA_PREFIX (
  if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
)

if not defined PYTHON_EXE (
  if exist "%USERPROFILE%\anaconda3\envs\a2b-gru\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\a2b-gru\python.exe"
)

if not defined PYTHON_EXE (
  echo Could not find the a2b-gru Python environment.
  echo Activate a2b-gru first, or create it from environment.yml.
  pause
  exit /b 1
)
echo Launching GUI with %PYTHON_EXE%
echo If the terminal stays busy, check your taskbar or Alt+Tab for the window.
"%PYTHON_EXE%" launch_gui.py
if errorlevel 1 pause
