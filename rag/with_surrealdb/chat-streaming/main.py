import uuid, os
from typing import List

# FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
## Streaming Response utility
from fastapi.responses import StreamingResponse
## Enable CORS utility
from fastapi.middleware.cors import CORSMiddleware

# Fireworks SDK
import fireworks.client

# SurrealDB Vector Store SDK for LangChain
from langchain_community.vectorstores import SurrealDBStore

# HuggingFace's Embeddings SDK by LangChain
from langchain_fireworks import FireworksEmbeddings

# Class representing the string of messages to be searched and embedded as system context.
class LearningMessages(BaseModel):
    messages: str

# Class representing a single message of the conversation between RAG application and user.
class Message(BaseModel):
    role: str
    content: str

# Class representing collection of messages above.
class Messages(BaseModel):
    messages: List[Message]

# Set the Fireworks API Key
fireworks.client.api_key = os.getenv("FIREWORKS_API_KEY")

# Load the nomic-embed-text-v1.5 embedding models via Langchain Fireworks Integration
embeddings = FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5",fireworks_api_key=os.getenv("FIREWORKS_API_KEY"))


dburl = "ws://localhost:4304/rpc"
db_user = "root"
db_pass = "root"
vector_collection = "vectors"
vector_db = SurrealDBStore(dburl=dburl, db_user=db_user, db_pass=db_pass, collection=vector_collection, embedding_function=embeddings)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to yield content from each choice
def yield_content(response):
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

@app.post('/update')
async def update(messages: LearningMessages):
    messages_json = messages.model_dump()["messages"].split(',')
    # Initialize SurrealDB
    await vector_db.initialize()
    # Create texts to be inserted into the Vector Store (Embeddings are generated automatically)
    metadatas = [{"len": len(t)} for t in messages_json]
    ids = [str(uuid.uuid4()) for _ in messages_json]
    await vector_db.aadd_texts(messages_json, metadatas=metadatas, ids=ids)
    
@app.post("/chat")
async def chat(messages: Messages):
    messages_json = (messages.model_dump())['messages']
    # Initialize SurrealDB
    await vector_db.initialize()
    # Create System Context
    knowledge = "Only answer what you know. If do not know, say it's an unknown. Following are the things you know of:\n"
    relevant_content = await vector_db.asimilarity_search(messages_json[-1]['content'])
    if relevant_content:
        for each_content in relevant_content:
            knowledge += each_content.page_content
    messages_json.insert(0, { "role": "system", "content": knowledge })
    # Create LLAMA Completion Responses
    response = fireworks.client.ChatCompletion.create(
        stream=True,
        prompt_or_messages=messages_json,
        model="accounts/fireworks/models/llama-v3p1-405b-instruct",
    )
    # Stream the response from requests.post
    return StreamingResponse(yield_content(response))