import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from typing import Dict, Any

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

DOCUMENT_DIR = Path("documents") 

SYSTEM_PROMPT = """
You are an enterprise knowledge assistant.

Answer questions using only information retrieved from the provided company knowledge base context.
If the context does not contain enough information to answer the question, say that the information is not available.

Format your response EXACTLY like this:
<Provide the plain text answer here without including the document names in the sentence text itself.>
[SOURCES]
<Provide a comma-separated list of the exact filename(s) used to answer the question, e.g., leave_policy.txt>
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

def answer_question(question: str) -> Dict[str, Any]:
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

    full_response = chat_completion.choices[0].message.content

    if "[SOURCES]" in full_response:
        answer_part, sources_part = full_response.split("[SOURCES]")
        answer_text = answer_part.strip()
        
        sources_list = [s.strip() for s in sources_part.split(",") if s.strip()]
    else:
        answer_text = full_response.strip()
        sources_list = []

    return {
        "answer": answer_text,
        "sources": sources_list
    }
