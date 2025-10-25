import os
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()


def extract_json_from_response(msg: str):
    """
    Safely extract JSON content from LLM responses.
    Handles code block wrappers, stray backticks, and text before/after JSON.
    """
    msg = msg.strip()

    # Handle ```json ... ``` or ``` ... ```
    if "```json" in msg:
        msg = msg.split("```json")[1]
        if "```" in msg:
            msg = msg.split("```")[0]
    elif "```" in msg:
        parts = msg.split("```")
        if len(parts) >= 3:
            msg = parts[1]
        else:
            msg = parts[-1]

    # Remove accidental prefixes/suffixes
    msg = msg.lstrip("json").strip("`").strip()

    # Try to load JSON safely
    try:
        return json.loads(msg)
    except json.JSONDecodeError as e:
        print("❌ JSON parsing failed. Raw response:")
        print(msg)
        print("Error:", e)
        return None


def validate_facts_batch(
    facts_evidence_dict,
    azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name="o4-mini",
    api_version="2024-12-01-preview"
):
    """
    Batch fact-checking using Azure OpenAI.
    Returns a dict mapping each fact to its verdict, reasoning, and supporting URLs.
    """

    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        azure_deployment=deployment_name,
        api_version=api_version
    )

    # ----- Build the structured multi-fact prompt -----
    facts_text = []
    for idx, (fact, evidence_list) in enumerate(facts_evidence_dict.items(), 1):
        fact_block = f"\n{'='*80}\nFACT #{idx}: {fact}\n{'='*80}\n"
        evidence_blocks = []
        for i, ev in enumerate(evidence_list, 1):
            if isinstance(ev, dict):
                title = ev.get("title", "")
                snippet = ev.get("snippet", "")
                url = ev.get("url", "")
            elif isinstance(ev, (list, tuple)) and len(ev) >= 3:
                title, snippet, url = ev[0], ev[1], ev[2]
            else:
                title = snippet = url = str(ev)

            evidence_blocks.append(
                f"  EVIDENCE {idx}.{i}:\n"
                f"    Title: {title}\n"
                f"    Snippet: {snippet}\n"
                f"    URL: {url}\n"
            )
        fact_block += "\n".join(evidence_blocks)
        facts_text.append(fact_block)

    prompt = f"""
You are an expert fact-checking assistant for UPSC current affairs.

Below are multiple facts with their associated evidence snippets. For each fact:

Determine the verdict as "Supported" or "Refuted", based on the provided evidence. Only choose "Cannot Conclude" if the evidence is truly contradictory, missing, or neutral.

If most evidence supports the fact (even if not fully conclusive), choose "Supported." If most evidence refutes the fact, choose "Refuted."

Use "Cannot Conclude" only when the evidence is ambiguous, insufficient, or directly contradicts itself, making it impossible to reasonably choose "Supported" or "Refuted."

Provide brief reasoning (1-2 sentences) for your verdict.

List ONLY the evidence citations (in format "EVIDENCE X.Y") that directly support your verdict.

Be as decisive as possible. Avoid "Cannot Conclude" unless no clear verdict is reasonable.

{facts_text}
"""

    # ----- Invoke LLM -----
    response = llm.invoke(prompt)
    msg = response.content.strip() if hasattr(response, "content") else str(response)

    # ----- Parse JSON -----
    llm_results = extract_json_from_response(msg)
    if llm_results is None:
        return {
            fact: {"verdict": "Error", "reasoning": "Failed to parse LLM response", "supporting_urls": []}
            for fact in facts_evidence_dict
        }

    # ----- Map citations to URLs -----
    results = {}
    facts_indexed = list(facts_evidence_dict.items())

    for idx, (fact, evidence_list) in enumerate(facts_indexed, 1):
        key = f"fact_{idx}"
        if key not in llm_results:
            results[fact] = {
                "verdict": "Error",
                "reasoning": "No output from LLM",
                "supporting_urls": []
            }
            continue

        llm_fact = llm_results[key]
        verdict = llm_fact.get("verdict", "Uncertain")
        reasoning = llm_fact.get("reasoning", "")
        cited = llm_fact.get("cited_evidence", [])

        supporting_urls = []
        for citation in cited:
            try:
                citation_clean = citation.replace("EVIDENCE", "").strip()
                parts = citation_clean.split(".")
                if len(parts) == 2:
                    fnum, enum = int(parts[0]), int(parts[1])
                    if fnum == idx and 1 <= enum <= len(evidence_list):
                        ev = evidence_list[enum - 1]
                        url = ev.get("url") if isinstance(ev, dict) else (ev[2] if isinstance(ev, (list, tuple)) and len(ev) >= 3 else "")
                        if url:
                            supporting_urls.append(url)
            except Exception:
                continue

        results[fact] = {
            "verdict": verdict,
            "reasoning": reasoning,
            "supporting_urls": list(set(supporting_urls))
        }

    return results


# ---- EXAMPLE USAGE ----
if __name__ == "__main__":
    facts_evidence_dict = {
        "India GDP growth 2025 is 6.8% according to IMF": [
            {
                "title": "IMF projects India's GDP growth at 6.6% for 2025",
                "snippet": "The IMF revised forecast for India's GDP upward for 2025.",
                "url": "https://imf.org/india2025"
            },
            {
                "title": "India's GDP growth set for 2025",
                "snippet": "India will grow at 6.8%, according to International agencies.",
                "url": "https://sample.com/gdp-news"
            }
        ],
        "Kerala literacy rate is over 96%": [
            {
                "title": "Kerala continues to lead with a literacy rate above 96%",
                "snippet": "Latest government data confirms Kerala as India's top state in literacy.",
                "url": "https://indiastats.com/kerala-literacy"
            }
        ],
        "Chandrayaan-4 launched in early 2025": [
            {
                "title": "Chandrayaan-4 mission delayed to 2026",
                "snippet": "ISRO announced Chandrayaan-4 will launch in 2026, not 2025.",
                "url": "https://isro.gov.in/chandrayaan4"
            }
        ]
    }

    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    results = validate_facts_batch(
        facts_evidence_dict,
        azure_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        deployment_name="o4-mini"
    )

    print(json.dumps(results, indent=2))
