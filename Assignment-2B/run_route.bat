@echo off
set PYTHON_EXE=C:\Users\User\anaconda3\envs\a2b-gru\python.exe
if not exist "%PYTHON_EXE%" (
  echo Could not find %PYTHON_EXE%
  pause
  exit /b 1
)
"%PYTHON_EXE%" main.py --origin 4335 --destination 3217 --date 10/15/2006 --time 08:00 --model random_forest --search-method astar --top-k 5
if errorlevel 1 pause
pause
