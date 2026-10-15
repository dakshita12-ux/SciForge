
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from groq import Groq, APIError, AuthenticationError, RateLimitError


# Load environment variables from the .env file in the project root.
load_dotenv()

# Read the API key without writing the secret directly in this file.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# The AI model used by SciForge.
MODEL_NAME = "openai/gpt-oss-120b"

# Create the FastAPI application.
app = FastAPI(
    title="SciForge API",
    description="AI-powered scientific and engineering workflow assistance",
    version="0.2.0",
)


# Define the structure of a question received from the user.
class Question(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=4000,
        description="A scientific or engineering question",
    )


# Instructions that guide the AI's responses.
SYSTEM_PROMPT = """
You are the AI assistant inside SciForge, a scientific and engineering
workflow assistance project.

Your responsibilities:
1. Understand the user's scientific or engineering problem.
2. Explain concepts in clear, structured language.
3. Suggest possible methods or algorithms when relevant.
4. Explain assumptions, limitations, and important considerations.
5. Never claim that you ran an experiment or verified a result unless
   that actually happened.
6. If the question requires current information or external evidence
   that you do not have, say so clearly.
7. Do not invent citations, experimental results, or scientific facts.

At this stage, answer the user's question directly. Do not claim to
have retrieved documents from a knowledge base or used other agents.
"""


@app.get("/")
def home():
    return {"message": "Welcome to SciForge!"}


@app.get("/about")
def about():
    return {
        "project": "SciForge",
        "purpose": "Scientific and Engineering Workflow Generation",
        "stage": "Stage 2 - LLM Integration",
        "model": MODEL_NAME,
    }


@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello {name}!"}


@app.get("/health")
def health():
    return {"status": "ok", "message": "SciForge backend is running"}


@app.post("/ask")
def ask(data: Question):
    # Check that the API key is configured.
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing. Check the root .env file.",
        )

    try:
        # Create a client that authenticates with Groq.
        client = Groq(api_key=GROQ_API_KEY)

        # Send the system instructions and user's question to the model.
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": data.question,
                },
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        # Extract the generated answer.
        answer = completion.choices[0].message.content

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The AI model returned an empty response.",
            )

        # Send the answer back to the user.
        return {
            "question": data.question,
            "answer": answer,
            "model": MODEL_NAME,
        }

    except AuthenticationError:
        raise HTTPException(
            status_code=502,
            detail="Groq authentication failed. Check your API key.",
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Groq rate limit reached. Please wait and try again.",
        )

    except APIError:
        raise HTTPException(
            status_code=502,
            detail="The Groq API request failed. Check the model and try again.",
        )
