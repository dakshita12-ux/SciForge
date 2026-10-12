from fastapi import FastAPI
from pydantic import BaseModel 

app = FastAPI()

class Question(BaseModel):
    question:str

@app.get("/")
def home():
    return {"message": "Welcome to SciForge!"}

@app.get("/about")
def about():
    return {
        "project": "SciForge",
        "purpose": "Scientific and Engineering Workflow Generation"
    }

@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello, {name}!"}


@app.post("/ask")
def ask(data: Question):
    return {
        "question_received": data.question
    }