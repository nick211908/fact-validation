import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()


def google_search(query, api_key=os.environ["GOOGLE_SEARCH_API_KEY"], cx_id=os.environ["GOOGLE_CX_ID"], n_results=3):
    """Search Google Custom Search API, returns list of evidence dicts."""
    url = "https://www.googleapis.com/customsearch/v1"
    
    # Exclude social media and untrustworthy domains
    excluded_domains = [
        "reddit.com", "facebook.com", "instagram.com", "pinterest.com",
        "twitter.com", "x.com", "quora.com", "medium.com",
        "youtube.com", "linkedin.com", "tumblr.com", "tiktok.com"
    ]
    
    # Include only trusted domains (optional - can be empty to allow all except excluded)
    preferred_domains = [
        "vajiramandravieducation.com",  # Vajiram & Ravi
        "visionias.in",                  # Vision IAS
        "nextias.com",                   # Next IAS
        "drishtiias.com",                # Drishti IAS
        "gov.in",                        # Government sites
        "nic.in",                        # NIC sites
        "edu",                           # Educational institutions
        "ac.in",                         # Academic institutions
        "pib.gov.in",                    # Press Information Bureau
        "mygov.in",                      # MyGov India
        "india.gov.in",                  # India Portal
    ]
    
    # Build site exclusion query
    exclude_query = " ".join([f"-site:{domain}" for domain in excluded_domains])
    
    # Optional: Build site inclusion query (use OR logic)
    # include_query = " OR ".join([f"site:{domain}" for domain in preferred_domains])
    
    # Combine query with exclusions
    full_query = f"{query} {exclude_query}"
    
    params = {
        "key": api_key,
        "cx": cx_id,
        "q": full_query,
        "num": n_results,
        "fields": "items(title,snippet,link)"
    }
    
    resp = requests.get(url, params=params, timeout=17)
    data = resp.json()
    results = []
    
    for item in data.get("items", []):
        url_link = item.get("link", "")
        
        # Additional filtering: skip if domain is in excluded list
        if any(domain in url_link.lower() for domain in excluded_domains):
            continue
            
        evidence = {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": url_link
        }
        results.append(evidence)
    
    return results


def get_facts_evidence(fact_queries, n_results=7):
    """Given a dict {fact: query}, returns a dict mapping each fact to its evidence list."""
    out = {}
    for fact, query in fact_queries.items():
        print(f"Fact: {fact}")
        print(f"Optimized query: {query}")
        evidence_list = google_search(query, n_results=n_results)
        out[fact] = evidence_list
    return out


# Demo usage
if __name__ == "__main__":
    fact_queries = {
        "India's GDP growth rate in 2023 is projected at 6.8% by the IMF": "India GDP growth 2023 IMF projection 6.8%",
        "The literacy rate in Kerala is the highest in India, at over 96%": "Kerala literacy rate India highest 96%",
        "Chandrayaan-4 marks India's first sample return mission to the Moon": "Chandrayaan-4 India lunar sample return mission",
        "India's first lunar sample return mission": "India first lunar sample return mission"
    }
    facts_evidence = get_facts_evidence(fact_queries, n_results=5)  # Increased to 5 to account for filtering
    print(json.dumps(facts_evidence, indent=2))
