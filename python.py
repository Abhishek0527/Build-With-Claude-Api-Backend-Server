import os
from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic  # noqa: E402

client = Anthropic()
model = "claude-sonnet-4-0"

messages=[]

def add_user_message(messages, text):
        user_message={"role":"user", "content":text}
        messages.append(user_message)

def add_assistant_message(messages, text):
        assistant_message={"role":"assistant", "content":text}
        messages.append(assistant_message)


def chat(messages):
    message = client.messages.create(
    model=model,
    max_tokens=100,
    messages= messages
)
    return message.content[0].text

add_user_message(messages, "Define quantum computing in one sentence")

answer = chat(messages)

add_assistant_message(messages,answer)

add_user_message(messages, "why we need quantum computers")

final_answer = chat(messages)

add_assistant_message(messages,final_answer)




print(messages)
