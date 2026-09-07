import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

DOCUMENT_DIR = Path("documents")

SYSTEM_PROMPT = """
You are an enterprise knowledge assistant.

Answer questions using only information retrieved from the provided company knowledge base context.

Provide your answer as a single, plain text sentence. Do not include markdown formatting, bold text (**), bullet points, or document names/sources in your final response.
"""


def load_knowledge_base() -> str:
    """Reads all text files from the documents directory into a single context string."""
    context_blocks = []
    if DOCUMENT_DIR.exists():
        for path in DOCUMENT_DIR.iterdir():
            if path.is_file() and path.suffix in [".txt", ".md"]:
                with open(path, "r", encoding="utf-8") as f:
                    context_blocks.append(f"--- Document: {path.name} ---\n{f.read()}")
    return "\n\n".join(context_blocks)

def answer_question(question: str) -> str:
    context = load_knowledge_base()
    
    user_content = f"Knowledge Base Context:\n{context}\n\nUser Question: {question}"

    chat_completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",  
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0.1,
    )


    return chat_completion.choices[0].message.content
