@echo off
echo ========================
echo  Llama-cpp VLM Web
echo  USE: start.bat [port]
echo  PORT: 5000
echo ========================

set PORT=%1
if "%PORT%"=="" set PORT=5000

echo Starting Llama-cpp VLM Web on port %PORT%...
echo Open http://localhost:%PORT% in your browser
echo.

python main.py %PORT%

pause
