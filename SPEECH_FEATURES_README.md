# Fireworks AI Speech & Audio Features

This repository contains a comprehensive collection of speech and audio processing features powered by Fireworks AI. From real-time voice agents to speech-to-text transcription, these examples demonstrate how to build powerful voice-enabled applications.

## 🎯 Overview

The speech features are organized into three main categories:

- **🎙️ Voice Agents** - Real-time conversational AI with voice interaction
- **📝 Speech-to-Text** - Audio transcription capabilities (streaming and prerecorded)
- **🔧 Audio Integration** - Speech features integrated into web applications

---

## 🎙️ Voice Agents

### Real-time Voice Agent
**Location:** `learn/voice-agent/`

A complete Python implementation of a real-time voice agent that demonstrates domain-specific AI conversation. The example creates a dental office receptionist with function calling capabilities.

#### Features
- **Real-time conversation** via WebSocket streaming
- **Function calling** for business automation
- **Audio pipeline** with microphone input and speaker output
- **Domain-specific responses** for business use cases

#### Available Functions
1. **Enroll New Patients** - Collects name and phone number
2. **Schedule Appointments** - Books appointments with date/time
3. **Cancel Appointments** - Cancels existing appointments
4. **Check Availability** - Shows available appointment slots

#### Quick Setup
```bash
# macOS/Linux
cd learn/voice-agent
chmod +x setup.sh
./setup.sh
export FIREWORKS_API_KEY="your_api_key_here"
python main.py

# Windows
cd learn/voice-agent
setup.bat
set FIREWORKS_API_KEY=your_api_key_here
python main.py
```

#### Dependencies
- `sounddevice` - Audio input/output
- `websockets` - Real-time communication
- `numpy` - Audio processing
- `python-dotenv` - Environment management

**💡 Tip:** Use headphones for the best experience to avoid audio feedback.

---

## 📝 Speech-to-Text

### 1. Streaming Speech-to-Text
**Location:** `learn/audio/audio_streaming_speech_to_text/`

Real-time audio transcription using WebSocket streaming. Available in both Python and Node.js implementations.

#### Python Implementation
- **File:** `python/main.py`
- **Features:** Real-time transcription with segment tracking
- **Audio Format:** 16kHz mono, 50ms+ chunks

#### Node.js Implementation
- **File:** `nodejs/index.js`
- **Features:** FFmpeg integration for audio processing
- **Requirements:** Node.js, FFmpeg

#### Setup (Python)
```bash
cd learn/audio/audio_streaming_speech_to_text/python
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate
python main.py
```

#### Setup (Node.js)
```bash
cd learn/audio/audio_streaming_speech_to_text/nodejs
npm install
export FIREWORKS_API_KEY=your_api_key_here
node index.js
```

### 2. Prerecorded Speech-to-Text
**Location:** `learn/audio/audio_prerecorded_speech_to_text/`

Transcribe audio files using the Fireworks API with support for multiple audio formats.

#### Features
- **Multiple formats** supported (FLAC, MP3, WAV, etc.)
- **Advanced options:** Language detection, speaker diarization
- **Timestamp granularity:** Word and segment level
- **Verbose output** with detailed metadata

#### Usage Example
```python
import requests
import os

# API call for prerecorded audio
headers = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}
files = {
    "file": ("audio.flac", audio_bytes),
    "model": (None, "whisper-v3"),
    "language": (None, "en"),
    "response_format": (None, "verbose_json"),
    "timestamp_granularities[]": (None, "word,segment"),
    "diarize": (None, "true"),
}
response = requests.post(
    "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions",
    headers=headers, 
    files=files
)
```

### 3. Concurrent Streaming
**Location:** `learn/audio/audio_concurrent_streaming_speech_to_text/`

Advanced implementation that handles multiple concurrent audio streams for high-throughput transcription scenarios.

#### Features
- **Multiple streams** handled simultaneously
- **TorchAudio integration** for advanced audio processing
- **Real-time processing** with minimal latency
- **Segment-based updates** for continuous transcription

#### Key Components
- Audio chunking and resampling
- WebSocket connection management
- Concurrent stream handling
- Real-time transcript updates

---

## 🔧 Audio Integration

### 1. Functional Chat with Audio
**Location:** `learn/function-calling/project_functional_chat/`

A Next.js web application that combines speech-to-text with function calling capabilities.

#### Features
- **Web-based audio recording** using MediaRecorder API
- **Real-time transcription** integration
- **Function calling** based on voice commands
- **Modern UI** with TypeScript and React

#### Audio Components
- `ChatInferenceModule.tsx` - Main audio recording logic
- `audio-transcription.ts` - API endpoint for transcription
- Integration with Fireworks AI for both speech and LLM processing

#### Setup
```bash
cd learn/function-calling/project_functional_chat
npm install
npm run dev
```

### 2. Transcription Chat Project
**Location:** `learn/inference/project_transcription_chat/`

A comprehensive chat application with integrated audio transcription capabilities.

#### Features
- Document-based Q&A with voice input
- Audio transcription integration
- Chat interface with speech support
- Multi-modal interaction (text + voice)

---

## 🚀 Getting Started

### Prerequisites

1. **Fireworks AI API Key**
   - Sign up at [fireworks.ai](https://fireworks.ai)
   - Get your API key from the dashboard

2. **Environment Setup**
   ```bash
   export FIREWORKS_API_KEY="your_api_key_here"
   # or create a .env file:
   echo "FIREWORKS_API_KEY=your_api_key_here" > .env
   ```

3. **System Requirements**
   - **Python:** 3.8+ for Python examples
   - **Node.js:** 16+ for JavaScript examples
   - **FFmpeg:** Required for Node.js audio processing
   - **Audio drivers:** For microphone/speaker access

### Universal Setup Scripts

Most projects include automated setup scripts:

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

These scripts typically:
- Install package managers (uv for Python, npm for Node.js)
- Create virtual environments
- Install all dependencies
- Set up the development environment

---

## 📊 Feature Comparison

| Feature | Real-time | Prerecorded | Multi-stream | Function Calling | Web Integration |
|---------|-----------|-------------|--------------|------------------|-----------------|
| Voice Agent | ✅ | ❌ | ❌ | ✅ | ❌ |
| Streaming STT | ✅ | ❌ | ❌ | ❌ | ❌ |
| Prerecorded STT | ❌ | ✅ | ❌ | ❌ | ❌ |
| Concurrent STT | ✅ | ❌ | ✅ | ❌ | ❌ |
| Functional Chat | ✅ | ❌ | ❌ | ✅ | ✅ |
| Transcription Chat | ✅ | ✅ | ❌ | ❌ | ✅ |

---

## 🛠️ Technical Details

### Audio Requirements
- **Sample Rate:** 16,000 Hz (16kHz)
- **Channels:** Mono (single channel)
- **Chunk Size:** 50ms minimum for streaming
- **Formats:** FLAC, MP3, WAV, M4A, and more

### API Endpoints
- **Streaming:** `wss://audio-streaming.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions/streaming`
- **Prerecorded:** `https://audio-prod.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions`
- **Voice Agent:** `wss://audio-agent.link.fireworks.ai/v1/audio/agent`

### Models Available
- **Whisper-v3:** High-accuracy speech recognition
- **Voice Models:** Various voice synthesis options
- **Language Support:** Multiple languages with auto-detection

---

## 📚 Additional Resources

- **Documentation:** [docs.fireworks.ai](https://docs.fireworks.ai/api-reference/audio-streaming-transcriptions)
- **Community:** [Discord](https://discord.gg/fireworks-ai)
- **API Reference:** [Fireworks AI API Docs](https://docs.fireworks.ai)

---

## 🤝 Contributing

Each project includes its own setup instructions and dependencies. To contribute:

1. Choose the feature you want to work on
2. Follow the setup instructions for that specific project
3. Make your changes and test thoroughly
4. Submit a pull request with clear documentation

---

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

*For specific implementation details, refer to the README files in each project directory.*
