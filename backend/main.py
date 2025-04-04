from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import PyPDF2
# Load .env variables
load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow CORS (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 💬 Chat Route (base)
# =========================
class Prompt(BaseModel):
    message: str

@app.post("/chat")
async def chat(prompt: Prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt.message}]
    )
    return {"response": response.choices[0].message.content}

# =========================
# 📄 Upload PDF & Extract
# =========================
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    reader = PyPDF2.PdfReader(file.file)
    text = ""
    print(reader, file)
    for page in reader.pages:
        text += page.extract_text()
    return {"text": text}

# =========================
# 🧠 Generate Embeddings
# =========================
class EmbeddingInput(BaseModel):
    text: str

@app.post("/embed")
async def generate_embeddings(data: EmbeddingInput):
    response = client.embeddings.create(
        input=data.text,
        model="text-embedding-3-small"  # Or use "text-embedding-ada-002"
    )
    return {"embedding": response.data[0].embedding}

# =========================
# 🧠 Assistant API (basic)
# =========================

class QueryRequest(BaseModel):
    question: str
    resume_text: str  # Receive resume text from the frontend

# =========================
# 🧠 Extract Text from PDF

# =========================
# 🧠 Assistant API (based on uploaded PDF text)
# =========================
@app.post("/ask-assistant")
async def ask_assistant(query: QueryRequest):
    resume_text = query.resume_text  # Extracted text from PDF

    # Structure the messages for the OpenAI API
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to custom PDF knowledge."},
        {"role": "user", "content": f"Here is the resume data:\n\n{resume_text}\n\nQuestion: {query.question}"}
    ]

    # Using the client to call the OpenAI API (using the client object)
    response = client.chat.completions.create(
        model="gpt-4",  # Or use gpt-3.5-turbo
        messages=messages,
        max_tokens=150
    )
    print(response)
    # Extract and return the answer from OpenAI's response
    answer = response.choices[0].message.content
    return {"answer": answer}