import os
import json
from typing import List
from datetime import datetime

# OpenAI
import openai

# SerpApi
from serpapi import GoogleSearch

# FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

# Enable CORS utility
from fastapi.middleware.cors import CORSMiddleware

# Class representing a single message of the conversation between RAG application and user.
class Message(BaseModel):
    role: str
    content: str

# Class representing collection of messages above.
class Messages(BaseModel):
    messages: List[Message]

# Create a FastAPI App instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get today's date
today_date = datetime.today().strftime("%Y-%m-%d")

tools = [
    {
        "type": "function",
        "function": {
            "name": "genre_generator",
            "description": "Extract genre detail from the user prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {
                        "type": "string",
                        "description": " ".join(
                            [
                                "The genre of the video that needs to be recommended to the user.",
                                "Following are some of the known genres: action, adventure, animation, classic, comedy, drama, horror, family, mystery, scifi, fantasy, western.",
                            ]
                        ),
                    },
                    "video_type": {
                        "type": "string",
                        "description": "The type of video: a movie or a tv show that needs to be recommended to the user.",
                    },
                },
                "required": ["genre", "video_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flight_generator",
            "description": "Extract flight details from the user prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure_id": {
                        "type": "string",
                        "description": "This represents the departure airport code in 3 letters. If you find a name of the country from user prompt, locate the most busiest airport and use it's IATA based 3-letter code, else if find an airport in the prompt, use it's IATA based airport code.",
                    },
                    "departure_date": {
                        "type": "string",
                        "description": f"This represents the departure date in YYYY-MM-DD format. If you can not find a date in user prompt, just use {today_date} as the fallback.",
                    },
                    "arrival_id": {
                        "type": "string",
                        "description": "This represents the arrival airport code in 3 letters. If you find a name of the country from user prompt, locate the most busiest airport and use it's IATA based 3-letter code, else if find an airport in the prompt, use it's IATA based airport code.",
                    },
                    "arrival_date": {
                        "type": "string",
                        "description": f"This represents the arrival date in YYYY-MM-DD format. If you can not find a date in user prompt, just use {today_date} as the fallback.",
                    },
                },
                "required": ["departure_id", "departure_date", "arrival_id", "arrival_date"],
            },
        },
    },
]

client = openai.OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"])

@app.post("/chat")
def chat(messages: Messages):
    messages_json = (messages.model_dump())["messages"]
    # Create System Context
    knowledge = "You are a helpful assistant with access to functions. Use them if required."
    messages_json.insert(0, {"role": "system", "content": knowledge})
    # Call Fireworks Function to determine the genre and type of video user asked for
    chat_completion = client.chat.completions.create(
        tools=tools,
        temperature=0.1,
        messages=[messages_json[0], messages_json[-1]],
        model="accounts/fireworks/models/firefunction-v2",
    )
    # Parse the generated function call to obtain the genre and type of video (movie, tv show, etc.)
    generated_tool_call = json.loads(chat_completion.choices[0].message.model_dump_json(include={"tool_calls"}))
    generated_args = generated_tool_call["tool_calls"][0]["function"]["arguments"]
    final_args = json.loads(generated_args)
    params = {
        "api_key": os.environ["SERPAPI_API_KEY"],
    }
    if final_args.get("genre") is not None:
        # Create params for the SerpApi for searching content on YouTube matching the user query
        params.update(
            {
                "engine": "youtube",
                "search_query": f"{final_args['genre']} {final_args['video_type']}",
            }
        )
    else:
        # Create params for the SerpApi for finding flights matching the user query
        params.update(
            {
                "hl": "en",
                "currency": "USD",
                "engine": "google_flights",
                "arrival_id": final_args["arrival_id"],
                "return_date": final_args["arrival_date"],
                "departure_id": final_args["departure_id"],
                "outbound_date": final_args["departure_date"],
            }
        )
    # Obtain the relevant content on YouTube
    search = GoogleSearch(params)
    # Return it as the response to the API endpoint
    resp = search.get_dict()
    if final_args.get("genre") is not None:
        return {
            "videos": [{"href": eachVideo["link"], "alt": eachVideo["description"], "src": eachVideo["thumbnail"]["static"]} for eachVideo in resp["video_results"][:3]],
        }
    else:
        return {
            "flights": [
                {
                    "price": flight["price"],
                    "airline_logo": single_flight["airline_logo"],
                    "arrival_airport_name": single_flight["arrival_airport"]["name"],
                    "arrival_airport_time": single_flight["arrival_airport"]["time"],
                    "departure_airport_name": single_flight["departure_airport"]["name"],
                    "departure_airport_time": single_flight["departure_airport"]["time"],
                }
                for flight in resp["best_flights"]
                for single_flight in flight["flights"][:3]
            ]
        }