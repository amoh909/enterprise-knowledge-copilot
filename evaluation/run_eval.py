import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: The 'requests' library is required to run the evaluation.")
    print("Please install it locally by running: pip install requests")
    sys.exit(1)

API_URL = "http://localhost:8000/ask"
EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "test_questions.json"
REPORT_PATH = EVAL_DIR / "eval_report.json"

def run_evaluation():
    if not DATASET_PATH.exists():
        print(f"Error: Could not find evaluation dataset at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Alright! Starting evaluation on {len(test_cases)} test cases against {API_URL}...\n")
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        expected = case["expected"]
        
        print(f"[{i}/{len(test_cases)}] Question: '{question}'")
        
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                actual_answer = data.get("answer", "")
                sources = data.get("sources", [])
                
                print(f"  └─ Expected: {expected}")
                print(f"  └─ Got:      {actual_answer}")
                print(f"  └─ Sources:  {sources}\n")
                
                results.append({
                    "question": question,
                    "expected": expected,
                    "actual": actual_answer,
                    "sources": sources,
                    "status": "SUCCESS"
                })
            else:
                print(f"  Ops, API returned error status: {response.status_code}\n")
                results.append({
                    "question": question,
                    "expected": expected,
                    "actual": f"Error: Status code {response.status_code}",
                    "sources": [],
                    "status": "FAILED"
                })
                
        except requests.exceptions.ConnectionError:
            print("  Ops, Connection Error: Is your Docker container running on port 8000?\n")
            return

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Great! Evaluation complete! Summary report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run_evaluation()
