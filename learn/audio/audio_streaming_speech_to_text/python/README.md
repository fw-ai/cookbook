# [Python] Streaming Audio Transcription

This is a Python project that streams audio to a WebSocket API to receive transcription in real time. It uses TorchAudio for audio processing and communicates with a WebSocket server.

---

## Prerequisites

### Setup

**Windows:**
```cmd
setup.bat
```

**macOS/Linux:**
```bash
chmod +x setup.sh && ./setup.sh
```

This will automatically:
- Install uv package manager (if needed)
- Create virtual environment 
- Install all dependencies

### Add Environment Variables

Create a `.env` file and add your Fireworks API key:
```env
FIREWORKS_API_KEY=your_api_key_here
```
If you don't have a Fireworks API key, you can sign up [here](https://fireworks.ai/).

---

## Run the Script

1. **Activate the virtual environment:**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Run the transcription:**
   ```bash
   python main.py
   ```

---

## More Information

For more information visit [docs.fireworks.ai](https://www.google.com/url?q=https%3A%2F%2Fdocs.fireworks.ai%2Fapi-reference%2Faudio-streaming-transcriptions).

Explore the community or reach out to us in [discord](https://www.google.com/url?q=https%3A%2F%2Fdiscord.gg%2Ffireworks-ai).