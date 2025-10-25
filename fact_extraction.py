import os
from dotenv import load_dotenv
import langextract as lx
from langchain_openai import AzureChatOpenAI

# Load Azure OpenAI credentials from .env file
load_dotenv()

def optimize_search_query(fact, azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), deployment_name="o4-mini", api_version="2024-12-01-preview"):
    """
    Uses Azure OpenAI to optimize a fact into an effective Google search query.
    """
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        azure_deployment=deployment_name,
        api_version=api_version
    )

    prompt = f"""
You are an expert at crafting effective search queries for fact-checking UPSC current affairs.

Given the factual statement: "{fact}"

Create an optimized Google search query that would yield the most relevant, authoritative results. Focus on:
- Key terms and entities
- Specific numbers, dates, or names
- Official sources like government, IMF, WHO, etc.
- Avoid overly broad or vague terms

Return ONLY the search query string, nothing else. Make it concise but comprehensive.

Example:
Input: "India's GDP growth rate in 2023 is projected at 6.8% by the IMF"
Output: "India GDP growth 2023 IMF projection 6.8%"
"""

    response = llm.invoke(prompt)
    optimized_query = response.content.strip() if hasattr(response, "content") else str(response)
    return optimized_query


def extract_facts(answer_text):
    """
    Uses LangExtract to extract factual claims from an answer string,
    and optimizes each fact into a search query.
    Returns a dict {fact: optimized_query}.
    """
    prompt = (
        "Extract all factual claims from the given answer. "
        "Each claim should correspond to a statement that could be verified through authoritative sources such as government, IMF, World Bank reports, or official statistics. "
        "Ignore subjective, philosophical, or rhetorical sentences."
    )

    examples = [
        lx.data.ExampleData(
            text="India's GDP growth rate in 2023 is projected at 6.8% by the IMF.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="Fact",
                    extraction_text="India's GDP growth rate in 2023 is projected at 6.8% by the IMF.",
                    attributes={"source": "IMF"}
                )
            ]
        ),
        lx.data.ExampleData(
            text="The literacy rate in Kerala is the highest in India, at over 96%.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="Fact",
                    extraction_text="The literacy rate in Kerala is the highest in India, at over 96%.",
                    attributes={"region": "Kerala"}
                )
            ]
        )
        # Add more if desired for even tighter format!
    ]
    # Optionally add example extraction if you want more control (few-shot)
    # See LangExtract docs for advanced schema
    result = lx.extract(
        text_or_documents=answer_text,
        prompt_description=prompt,
        model_id="gemini-2.5-flash",
        api_key=os.environ["GOOGLE_API_KEY"],
        examples=examples
    )
    # Get extracted facts
    facts = [ex.extraction_text for ex in result.extractions]

    # Optimize each fact into a search query
    fact_queries = {}
    for fact in facts:
        optimized_query = optimize_search_query(fact)
        fact_queries[fact] = optimized_query

    return fact_queries

# Demo usage
if __name__ == "__main__":
    answer = (
        "India's GDP growth rate in 2023 is projected at 6.8% by the IMF. "
        "Warli and Gond paintings reflect harmony with nature. "
        "The literacy rate in Kerala is the highest in India, at over 96%. "
        "People in tribal communities worship trees and animals. "
        "Diversity is the hallmark of Indian tribal art."
    )
    fact_queries = extract_facts(answer)
    print("Extracted Facts and Optimized Queries:")
    for fact, query in fact_queries.items():
        print(f"Fact: {fact}")
        print(f"Query: {query}")
        print()
