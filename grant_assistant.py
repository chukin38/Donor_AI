#!/usr/bin/env python3
"""Draft grant proposals using a local Gemma model."""

import json
from transformers import pipeline

# Configuration
MODEL_NAME = "/Users/solomonchu/PycharmProjects/Project_Donor/gemma-3-4b-pt"
_GENERATOR = pipeline("text-generation", model=MODEL_NAME)

def generate_grant_proposal(grant_schema, org_profile):
    """
    Draft a grant proposal based on the RFP schema and organization profile.
    grant_schema: dict with RFP fields (id, title, themes, sections, deadlines…)
    org_profile: dict with org mission, past impact metrics etc.
    """
    prompt = (
        f"You are a professional nonprofit grant writer.\n"
        f"RFP Title: {grant_schema['title']}\n"
        f"Funder: {grant_schema.get('funder','N/A')}\n"
        f"Deadline: {grant_schema.get('deadline','N/A')}\n"
        f"Themes/Eligibility: {grant_schema.get('themes','')}\n"
        f"Required Sections: {', '.join(grant_schema.get('required_sections',[]))}\n\n"
        f"Organization Profile:\n{json.dumps(org_profile, indent=2)}\n\n"
        "Write a complete grant proposal covering each required section. "
        "Return ONLY the proposal text—no JSON, no markdown fences."
    )

    instructions = (
        "Structure the proposal with clear headings matching the required sections. "
        "Keep it formal and persuasive."
    )
    full_prompt = f"{prompt}\n{instructions}"

    try:
        result = _GENERATOR(full_prompt, max_new_tokens=1024, do_sample=False)
        text = result[0]["generated_text"]
        if text.startswith(full_prompt):
            text = text[len(full_prompt):]
        return text.strip()

    except Exception as e:
        return f"Error generating grant proposal: {e}"

if __name__ == "__main__":
    # Example usage
    sample_rfp = {
        "title": "Youth STEM Education Grant",
        "funder": "Global Education Foundation",
        "deadline": "2025-09-30",
        "themes": ["STEM access", "Underprivileged Youth"],
        "required_sections": ["Executive Summary", "Needs Statement", "Project Plan", "Budget"]
    }
    sample_org = {
        "name": "Educate All Futures",
        "mission": "Empower underserved youth with STEM skills.",
        "past_impact": {"students_served": 1200, "graduation_rate": "85%"},
        "requested_amount": 500000
    }

    proposal = generate_grant_proposal(sample_rfp, sample_org)
    with open("grant_proposal.txt", "w", encoding="utf-8") as f:
        f.write(proposal)

    print("✅ Grant proposal saved to 'grant_proposal.txt'")
    print(f"Length: {len(proposal)} characters")
