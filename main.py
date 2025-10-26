import json
from fact_extraction import extract_facts
from validation_and_reasoning import validate_facts_batch
from evidence.web_search import google_search


def main():
    # Read QA pairs from the local project file
    with open("qa_pairs.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", [])
    if not qa_pairs:
        print("No QA pairs found in JSON file.")
        return

    for idx, qa in enumerate(qa_pairs, start=1):
        question = qa.get("question", "")
        answer = qa.get("answer", "")

        print(f"\n===== QA Pair {idx} =====")
        print(f"Q: {question}")
        print(f"A: {answer}")

        # Extract a list of facts (no query mapping)
        facts = extract_facts(answer)
        if not facts:
            print("No factual statements detected.\n")
            continue

        # Build evidence dict: fact -> list of evidence items from web search using the fact as the query
        evidence_dict = {}
        for fact in facts:
            evidence = google_search(fact, n_results=5)
            evidence_dict[fact] = evidence

        # Skip LLM validation if no evidence is found for any fact
        if all(not v for v in evidence_dict.values()):
            for fact in facts:
                print(f"\nFact: {fact}")
                print("Verdict: No evidence")
                print("Reasoning: No relevant web data found.\n")
                print("No supporting citations.")
            continue

        try:
            results = validate_facts_batch(evidence_dict)
        except Exception as e:
            print(f"Error during validation: {e}")
            continue

        for fact, result in results.items():
            print(f"\nFact: {fact}")
            print(f"Verdict: {result.get('verdict', 'Error')}")
            print(f"Reasoning: {result.get('reasoning', '')}")
            su = result.get('supporting_urls') or []
            if su:
                print("Supporting Citations:")
                for url in su:
                    print(f"  - {url}")
            else:
                print("No supporting citations.")

if __name__ == "__main__":
    main()
