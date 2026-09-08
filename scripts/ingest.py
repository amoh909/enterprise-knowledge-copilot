import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.rag import build_index

if __name__ == "__main__":
    print("Alright, Starting Semantic Vector Ingestion Pipeline...")
    try:
        build_index()
        print("Great! Successfully built and saved vector index to data/index.pkl!")
    except Exception as e:
        print(f"Ops. Ingestion failed: {e}")
