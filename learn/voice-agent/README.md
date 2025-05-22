# Fireworks Voice Agent - Python Example

A simple Python example showing how to use Fireworks AI Voice Agents for real-time voice conversations.

## Quick Setup with UV

**macOS/Linux**:
```bash
chmod +x setup.sh
./setup.sh
export FIREWORKS_API_KEY="your_api_key_here"
python main.py
```

**Windows**:
```cmd
setup.bat
set FIREWORKS_API_KEY=your_api_key_here
python main.py
```

## Manual Setup

1. **Create virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # macOS/Linux
   # OR
   .venv\Scripts\activate     # Windows
   ```

2. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Set your API key**:
   - Add a `.env` file in the project directory with the following content:
     ```bash
     FIREWORKS_API_KEY="<YOUR_FIREWORKS_API_KEY>"
     ```
   - If you don't have a Fireworks API key, you can generate one [here](https://fireworks.ai/).

4. **Run the example**:
   ```bash
   python main.py
   ```

## What it does

- Connects to Fireworks Voice Agent API
- Records audio from your microphone
- Sends audio to the voice agent
- Plays back the AI's voice responses
- Press Ctrl+C to quit

## Requirements

- Python 3.8+
- Working microphone and speakers
- Fireworks API key

That's it! Start speaking and the AI will respond with voice.