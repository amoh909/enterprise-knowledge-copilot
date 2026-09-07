import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DOCUMENT_DIR = Path("documents")


def main():
    vector_store = client.vector_stores.create(
        name="enterprise_knowledge_base"
    )

    print(f"Created vector store: {vector_store.id}")

    for path in DOCUMENT_DIR.iterdir():
        if not path.is_file():
            continue

        with open(path, "rb") as file:
            uploaded_file = client.files.create(
                file=file,
                purpose="assistants"
            )

        client.vector_stores.files.create(
            vector_store_id=vector_store.id,
            file_id=uploaded_file.id
        )

        print(f"Uploaded: {path.name}")

    print()
    print("VECTOR_STORE_ID=" + vector_store.id)


if __name__ == "__main__":
    main()