import requests
import json

# Configuration
GENERATIVE_SERVER_URL = "http://57.129.18.204:51001"


def generate_ai_report(topic, report_type="analysis"):
    """Generate an AI report on a given topic"""

    payload = {
        "input": f"Write a comprehensive {report_type} report about {topic}",
        "instructions": "Structure the report with clear sections: Executive Summary, Key Findings, Analysis, and Recommendations. Keep it professional and informative."
    }

    try:
        response = requests.post(
            f"{GENERATIVE_SERVER_URL}/generate-text",
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        return result["text"]

    except requests.exceptions.RequestException as e:
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