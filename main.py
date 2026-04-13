from fastapi import FastAPI, Request
from anthropic import Anthropic
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = FastAPI()


class ChatRequest(BaseModel):
    messages: list
    system:str


client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@app.post("/chat")
async def chat(body: ChatRequest):

    messages = body.messages
    system= body.system
    temperature = 0.0
    print("Incoming messages:", messages)

    response = client.messages.create(
        model="claude-sonnet-4-0", max_tokens=100, messages=messages, system=system, temperature= temperature
    )

    print("Model response:", response)

    return {"reply": response.content[0].text}
