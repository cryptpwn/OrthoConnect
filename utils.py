import textwrap

# Global state for session history
session_history = []

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def load_pdf_text(file_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def get_initial_context(chunks, size=2000):
    return textwrap.shorten("\n".join(chunks[:3]), width=size)
