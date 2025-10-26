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
Analyze the facts and evidence provided below.

Respond with a single JSON object. Do not include any text before or after the JSON.
The JSON object must have keys "fact_1", "fact_2", ..., corresponding to each fact number.

For each fact key (e.g., "fact_1"), the value must be an object with exactly three keys:
1.  "verdict": (string) Must be one of "Supported", "Refuted", or "Cannot Conclude".
    -Be decisive. Use "Cannot Conclude" ONLY if the evidence is truly    missing, or insufficient.
2.  "reasoning": (string) A brief 1-2 sentence explanation for your verdict.
3.  "cited_evidence": (array of strings) A list of evidence citations (e.g., ["EVIDENCE 1.1", "EVIDENCE 1.2"]) that directly support your verdict. If no evidence is used, return an empty array [].

Here are the facts and evidence:
{facts_text}

Return ONLY the single JSON object.
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
