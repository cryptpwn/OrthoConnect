from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from utils import load_pdf_text, chunk_text, session_history
from pdf_processor import PDFProcessor
from indexer import build_index, retrieve_relevant_chunks
import ollama
import os
import shutil


app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pdf_processor = PDFProcessor()
chunk_index = None


class SessionConfig(BaseModel):
    topic: str
    language: Optional[str] = "en"
    level: Optional[str] = "A2"

class UserResponse(BaseModel):
    answer: str

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global chunk_index

    os.makedirs('data', exist_ok=True)

    file_path = os.path.join('data', file.filename)

    try:
        # Save the uploaded file
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        # Process and index
        text = pdf_processor.upload_and_process_pdf(file_path)
        pdf_processor.pdf_chunks = chunk_text(text, chunk_size=300)
        chunk_index, _ = build_index(pdf_processor.pdf_chunks, pdf_processor.embed_model)

        return {"message": "PDF uploaded, processed, and indexed successfully."}

    except Exception as e:
        return {"error": str(e)}

@app.post("/start-session")
def start_session(config: SessionConfig):
    context = pdf_processor.get_initial_context()
    prompt = f"""
    You are a tutor teaching the topic "{config.topic}" in {config.language} at level {config.level}.
    Use the material below to generate one interactive question or exercise:

    {context}

    Start now with a question.
    """
    
    response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])
    assistant_msg = response['message']['content']
    session_history.append({"role": "assistant", "content": assistant_msg})

    return {"message": "Session started.", "question": assistant_msg}

@app.post("/respond")
def respond(user_input: UserResponse):
    global chunk_index

    session_history.append({"role": "user", "content": user_input.answer})

    context_chunks = retrieve_relevant_chunks(user_input.answer, pdf_processor.embed_model, chunk_index, pdf_processor.pdf_chunks, top_k=3)
    context = "\n".join(context_chunks)

    prompt = [{"role": "system", "content": f"You are a tutor helping a student. Use the context to guide your response:\n\n{context}"}] + session_history

    response = ollama.chat(model="llama3.2:latest", messages=prompt)
    reply = response['message']['content']
    session_history.append({"role": "assistant", "content": reply})

    return {"reply": reply}

# === 4. Reset Session ===
@app.post("/reset-session")
def reset_session():
    global session_history
    session_history = []
    return {"message": "Session reset."}
