import os
from dotenv import load_dotenv
import langextract as lx

# Load Azure OpenAI credentials from .env file
load_dotenv()



def extract_facts(answer_text):
    """
    Uses LangExtract to extract factual claims from an answer string.
    Returns a list of fact strings.
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
    return facts

# Demo usage
if __name__ == "__main__":
    answer = (
        "India's GDP growth rate in 2023 is projected at 6.8% by the IMF. "
        "Warli and Gond paintings reflect harmony with nature. "
        "The literacy rate in Kerala is the highest in India, at over 96%. "
        "People in tribal communities worship trees and animals. "
        "Diversity is the hallmark of Indian tribal art."
    )
    facts = extract_facts(answer)
    print("Extracted Facts:")
    for fact in facts:
        print(f"- {fact}")
