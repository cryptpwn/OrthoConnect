import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_index(chunks, embed_model):
    embeddings = embed_chunks(chunks, embed_model)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def embed_chunks(chunks, model):
    return model.encode(chunks)

def retrieve_relevant_chunks(query, model, index, chunks, k=3):
    q_embed = model.encode([query])
    _, I = index.search(np.array(q_embed), k)
    return [chunks[i] for i in I[0]]
