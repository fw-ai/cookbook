@echo off

echo 🚀 Setting up Fireworks Voice Agent development environment with uv...

REM Check if uv is installed
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ uv is not installed. Please install uv first:
    echo https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

echo ✅ uv found

REM Create virtual environment
echo 📦 Creating virtual environment...
uv venv

REM Activate virtual environment and install dependencies
echo 📥 Installing dependencies...
.venv\Scripts\activate && uv pip install -r requirements.txt

echo ✅ Setup complete!
echo.
echo 🎯 Next steps:
echo 1. Set your API key: set FIREWORKS_API_KEY=your_key_here
echo 2. Activate the environment: .venv\Scripts\activate
echo 3. Run the voice agent: python main.py
echo.
echo 💡 To deactivate later: deactivate

pause