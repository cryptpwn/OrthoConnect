from sentence_transformers import SentenceTransformer
from utils import load_pdf_text, chunk_text
from PyPDF2 import PdfReader

class PDFProcessor:
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pdf_chunks = []

    def upload_and_process_pdf(self,file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def get_initial_context(self, size=2000):
        return " ".join(self.pdf_chunks)[:size]
