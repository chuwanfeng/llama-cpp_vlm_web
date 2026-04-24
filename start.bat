@echo off
:: Llama-cpp VLM Web 启动脚本
:: 用法: start.bat [port]
:: 默认端口 5555

set PORT=%1
if "%PORT%"=="" set PORT=5555

echo Starting Llama-cpp VLM Web on port %PORT%...
echo Open http://localhost:%PORT% in your browser
echo.

python app.py %PORT%
