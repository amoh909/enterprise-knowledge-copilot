import os
import pickle
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

DOCUMENT_DIR = Path("documents")
INDEX_PATH = Path("data/index.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

SYSTEM_PROMPT = """
You are an enterprise knowledge assistant.

Answer questions using ONLY the retrieved knowledge base context.

Provide your answer as a single, plain text sentence. Do NOT include markdown formatting, asterisks, bold text (**), bullet points, or list structures. 

If the retrieved context does not contain enough information to
answer the question, clearly state that the information is not
available in the knowledge base.

Do not invent policies, numbers, dates, procedures, or facts.
"""


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40):
    """Split documents by numbered sections, then chunk oversized sections."""
    sections = re.split(r"(?=\b\d+\.\d+\s)", text)

    chunks = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        words = section.split()

        if len(words) <= chunk_size:
            chunks.append(section)
            continue

        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])

            if chunk.strip():
                chunks.append(chunk)

            start += chunk_size - overlap

    return chunks


def build_index():
    """Load documents, chunk them dynamically, generate embeddings, and save."""
    records = []

    for path in DOCUMENT_DIR.iterdir():
        if not path.is_file() or path.suffix not in [".txt", ".md"]:
            continue

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        chunks = chunk_text(text)

        for chunk in chunks:
            records.append(
                {
                    "text": chunk,
                    "source": path.name,
                }
            )

    if not records:
        raise RuntimeError("No documents found in the documents directory.")

    texts = [record["text"] for record in records]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    index = {
        "records": records,
        "embeddings": np.asarray(embeddings),
    }

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(INDEX_PATH, "wb") as file:
        pickle.dump(index, file)

    print(f"Generated {len(records)} vector chunks across your files.")


def load_index():
    """Load the persisted vector index."""
    if not INDEX_PATH.exists():
        raise RuntimeError("Vector index not found. Run: python scripts/ingest.py")

    with open(INDEX_PATH, "rb") as file:
        return pickle.load(file)


def retrieve(question: str, top_k: int = TOP_K):
    """Retrieve the most semantically similar document chunks."""
    index = load_index()

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )[0]

    similarities = index["embeddings"] @ query_embedding

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []

    for index_position in top_indices:
        score = float(similarities[index_position])
        record = index["records"][index_position]

        results.append(
            {
                "text": record["text"],
                "source": record["source"],
                "score": score,
            }
        )

    return results


def answer_question(question: str) -> Dict[str, Any]:
    """Run retrieval-augmented generation for a user question."""
    retrieved_chunks = retrieve(question)

    context = "\n\n".join(
        [
            f"[Source: {chunk['source']}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    user_content = f"Retrieved Knowledge Base Context:\n\n{context}\n\nUser Question:\n{question}"

    chat_completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )

    answer_text = chat_completion.choices[0].message.content.strip()
    sources = list(dict.fromkeys(chunk["source"] for chunk in retrieved_chunks))

    return {
        "answer": answer_text,
        "sources": sources,
    }

