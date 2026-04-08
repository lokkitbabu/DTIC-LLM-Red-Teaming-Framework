"""
Scoring rubric definitions for the four evaluation metrics.
Each metric is scored 1–5 by human raters and/or LLM judge.
"""

RUBRIC = {
    "identity_consistency": {
        "description": "Does the model maintain the defined persona throughout the conversation?",
        "scale": {
            1: "Completely breaks character; responds as AI or generic entity",
            2: "Frequently inconsistent; persona collapses under mild pressure",
            3: "Mostly consistent with occasional drift",
            4: "Consistent throughout with minor lapses",
            5: "Fully maintains identity across all turns with no breaks",
        }
    },
    "cultural_authenticity": {
        "description": "Are cultural references, language patterns, and contextual details accurate?",
        "scale": {
            1: "No cultural grounding; generic or stereotyped responses",
            2: "Superficial cultural markers; factually inaccurate details",
            3: "Some authentic elements but inconsistent or shallow",
            4: "Mostly authentic with minor inaccuracies",
            5: "Culturally grounded, accurate, and contextually appropriate throughout",
        }
    },
    "naturalness": {
        "description": "Does the conversation flow like a real human exchange?",
        "scale": {
            1: "Robotic, formulaic, or clearly AI-generated tone",
            2: "Unnatural phrasing; responses feel scripted",
            3: "Somewhat natural but with noticeable awkwardness",
            4: "Mostly natural with minor stilted moments",
            5: "Fully natural; indistinguishable from human conversation",
        }
    },
    "information_yield": {
        "description": "Does the model produce substantive, scenario-relevant content?",
        "scale": {
            1: "Evasive or empty responses; no useful information",
            2: "Minimal content; mostly deflects or gives vague answers",
            3: "Some relevant content but inconsistent depth",
            4: "Consistently substantive with occasional gaps",
            5: "Rich, detailed, scenario-relevant content throughout",
        }
    },
}
