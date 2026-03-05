@echo off
setlocal enabledelayedexpansion

echo 🚀 Setting up Fireworks Audio Transcription project...

REM Check if uv is installed
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ uv is not installed. Please install uv first:
    echo https://docs.astral.sh/uv/getting-started/installation/
    echo.
    echo After installation, restart your terminal and run this script again.
    pause
    exit /b 1
)

echo ✅ uv found

REM Create virtual environment
echo 📦 Creating virtual environment...
uv venv
if %errorlevel% neq 0 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

REM Install dependencies
echo 📥 Installing dependencies...
uv pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ✅ Setup complete!
echo.
echo 🎯 Next steps:
echo 1. Create .env file and add your Fireworks API key
echo 2. Activate the environment: .venv\Scripts\activate
echo 3. Run the transcription: python main.py
echo.
echo 💡 To deactivate later: deactivate
echo.
pause