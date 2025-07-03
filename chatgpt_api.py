"""Simple utility to generate a short report using a local Gemma model."""

import json
from transformers import pipeline

# Configuration
MODEL_NAME = "/Users/solomonchu/PycharmProjects/Project_Donor/gemma-3-4b-pt"
_GENERATOR = pipeline("text-generation", model=MODEL_NAME)


def generate_ai_report(topic, report_type="analysis"):
    """Generate an AI report on a given topic"""

    prompt = (
        f"Write a comprehensive {report_type} report about {topic}\n"
        "Structure the report with clear sections: Executive Summary, Key Findings, Analysis, and Recommendations. "
        "Keep it professional and informative."
    )

    try:
        result = _GENERATOR(prompt, max_new_tokens=1024, do_sample=False)
        text = result[0]["generated_text"]
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

    except Exception as e:
        return f"Error generating report: {e}"


# Example usage
if __name__ == "__main__":
    # Generate a report
    topic = "The Impact of AI on Modern Business"
    report = generate_ai_report(topic)

    # Save to file
    with open("ai_report.txt", "w", encoding="utf-8") as f:
        f.write(f"AI REPORT: {topic.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    print("Report generated and saved to 'ai_report.txt'")
    print(f"Report length: {len(report)} characters")