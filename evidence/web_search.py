import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("TAVILY_API_KEY"))

def google_search(query, n_results=3):
    """
    Search using Tavily API (renamed to maintain compatibility).
    Returns list of evidence dicts with same format as Google Custom Search.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    
    if not api_key:
        print("Error: TAVILY_API_KEY not found in environment variables")
        return []
    
    url = "https://api.tavily.com/search"

    UPSC_EXCLUDE_DOMAINS = [
    "reddit.com",
    "quora.com",
    "facebook.com",
    "youtube.com",
    "instagram.com",
    "pinterest.com",
    "medium.com",
    "stackexchange.com",
    "stackoverflow.com",
    "twitter.com",
    "linkedin.com"
    # Add more as you identify additional unwanted sources
    ]

    
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": n_results,
        "include_answer": "advanced",  # Set to True if you want AI summary
        "include_images": False,
        "include_raw_content": False,
        "exclude_domains": UPSC_EXCLUDE_DOMAINS,
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # print(data)
        results = []
        for item in data.get("results", []):
            # Maintain exact same format as Google Custom Search
            evidence = {
                "title": item.get("title", ""),
                "snippet": item.get("content", ""),
                "url": item.get("url", "")
            }
            results.append(evidence)
        
        return results
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

if __name__ == "__main__":
    result = google_search("Who is the Prime Minister of India?", n_results=3)
    print(json.dumps(result, indent=2))
