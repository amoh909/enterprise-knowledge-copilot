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

def is_fact_correct(expected_text: str, actual_text: str) -> bool:
    """
    Normalizes text and checks if all key semantic tokens from the expected 
    fact are present anywhere within the LLM's long-form response.
    """
    exp = expected_text.lower().strip()
    act = actual_text.lower().strip()
    
    replacements = {
        "three": "3",
        "five": "5",
        "twenty": "20"
    }
    
    for word, digit in replacements.items():
        exp = exp.replace(word, digit)
        act = act.replace(word, digit)
            
    expected_tokens = exp.split()
    return all(token in act for token in expected_tokens)


def run_evaluation():
    if not DATASET_PATH.exists():
        print(f"Error: Could not find evaluation dataset at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Alright! Starting smart factual evaluation on {len(test_cases)} test cases against {API_URL}...\n")
    
    results = []
    total_passed = 0
    
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
                
                fact_passed = is_fact_correct(expected, actual_answer)
                has_sources = len(sources) > 0
                
                if fact_passed and has_sources:
                    status = "PASSED"
                    total_passed += 1
                else:
                    status = "FAILED"
                
                print(f"  ├─ Expected Fact: {expected}")
                print(f"  ├─ Got Response:  {actual_answer}")
                print(f"  ├─ Citations:     {sources}")
                print(f"  └─ Status Mark:   [{status}]\n")
                
                results.append({
                    "question": question,
                    "expected_fact": expected,
                    "actual_response": actual_answer,
                    "sources_attributed": sources,
                    "evaluation_status": status,
                    "failure_reason": None if (fact_passed and has_sources) else ("Missing target fact" if not fact_passed else "No sources attributed")
                })
            else:
                print(f"  Ops, API returned error status: {response.status_code}\n")
                results.append({
                    "question": question,
                    "expected_fact": expected,
                    "actual_response": f"Error: Status code {response.status_code}",
                    "sources_attributed": [],
                    "evaluation_status": "FAILED",
                    "failure_reason": f"HTTP Error Status {response.status_code}"
                })
                
        except requests.exceptions.ConnectionError:
            print("  Ops, Connection Error: Is your Docker container running on port 8000?\n")
            return

    # Compute metric ratios
    accuracy_score = (total_passed / len(test_cases)) * 100
    
    summary_report = {
        "metrics": {
            "total_test_cases": len(test_cases),
            "total_passed": total_passed,
            "total_failed": len(test_cases) - total_passed,
            "accuracy_percentage": f"{accuracy_score:.1f}%"
        },
        "results": results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
        
    print("=" * 70)
    print(f"Evaluation Complete! Summary report saved to {REPORT_PATH}")
    print(f"Final System Score: {total_passed}/{len(test_cases)} Passed ({accuracy_score:.1f}% Accuracy)")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
