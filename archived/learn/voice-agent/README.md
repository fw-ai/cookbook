# Fireworks Voice Agent - Python Example

A simple Python example showing how to use Fireworks AI Voice Agents for real-time voice conversations. 
This example creates a dental office receptionist that can handle new patient enrollment and appointment scheduling, 
demonstrating how to build domain-specific voice agents for business automation.

**NOTE: Use headphones for the best experience.**

## Features

- **Real-time voice conversation** with Fireworks AI Voice Agent
- **Function calling capabilities** for business automation
- **Domain-specific AI receptionist** for dental office operations
- **Complete audio pipeline** with microphone input and speaker output

### Available Functions

The voice agent can perform these actions through natural conversation:

1. **Enroll New Patients** - Collects patient name and phone number
   - Example: "I'm a new patient and need to register"
   
2. **Schedule Appointments** - Books appointments for existing patients
   - Example: "I'd like to schedule an appointment for next Tuesday at 2 PM"
   
3. **Check Availability** - Shows available appointment slots for specific dates
   - Example: "What times are available on Friday?"
   

Simply speak naturally to the agent - it will understand your intent and call the appropriate functions automatically.

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

## Function / Tool Calling

To add custom functions to your voice agent:

1. **Define your function**:
   ```python
   def my_custom_function(param1: str, param2: str) -> dict:
       # Your business logic here
       return {"success": True, "message": "Function executed"}
   ```

2. **Add to tool configuration**:
   ```python
   {
       "type": "function",
       "function": {
           "name": "my_custom_function",
           "description": "Brief description of what this function does",
           "parameters": {
               "type": "object",
               "properties": {
                   "param1": {"type": "string", "description": "Parameter description"},
                   "param2": {"type": "string", "description": "Parameter description"}
               },
               "required": ["param1", "param2"]
           }
       }
   }
   ```

3. **Add to function handler**:
   ```python
   elif function_name == "my_custom_function":
       result = my_custom_function(
           param1=arguments.get("param1"),
           param2=arguments.get("param2")
       )
   ```

The voice agent will automatically call your functions when users speak naturally about related tasks.

That's it! Start speaking and the AI will respond with voice.


## What it does

- Connects to Fireworks Voice Agent API
- Records audio from your microphone
- Sends audio to the voice agent
- Plays back the AI's voice responses
- Executes business functions based on voice commands
- Press Ctrl+C to quit

## Requirements

- Python 3.8+
- Working microphone and speakers
- Fireworks API key

That's it! Start speaking and the AI will respond with voice.