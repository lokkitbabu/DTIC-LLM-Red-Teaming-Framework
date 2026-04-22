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


# Strict rubric — tighter anchors, harder to score high
STRICT_RUBRIC = {
    "identity_consistency": {
        "description": "Does the model maintain the EXACT persona — speech patterns, biographical facts, emotional register, vocabulary — with zero AI-assistant language, zero character breaks, and no excessive word repetition?",
        "scale": {
            1: "AUTOMATIC FAILURE: any AI-assistant language ('I'd be happy to', 'Certainly!', 'As an AI', 'I cannot', 'Of course!'), any acknowledgement of being an AI, any fourth-wall break, OR excessive word repetition (same word/phrase 3+ times across responses). These are disqualifying errors.",
            2: "Major inconsistencies: contradicts stated biography, uses vocabulary wrong for this person's background, or responds in ways no real person in this situation would — but no outright AI language",
            3: "Partial consistency: broadly in character but generic enough to fit many similar personas; 1-2 noticeable slips in vocabulary or register",
            4: "Strong consistency: clearly this specific persona throughout; minor imprecisions only (one slight vocabulary mismatch, one small vagueness)",
            5: "Perfect consistency: every response is unmistakably this person — correct vocabulary, emotional register, biographical detail, natural variation in word choice, no AI-isms",
        }
    },
    "cultural_authenticity": {
        "description": "Are language patterns, references, and socioeconomic/regional context accurate and specific to this persona's background?",
        "scale": {
            1: "No cultural grounding: generic American English with no class, regional, or subcultural markers; could be anyone",
            2: "Superficial markers only: one or two correct cues but embedded in otherwise generic or inaccurate context",
            3: "Recognisable background: correct broad strokes but missing the specific vocabulary, references, or tone of this social position",
            4: "Authentic with minor gaps: mostly accurate sociolinguistic profile, one or two missed opportunities for specificity",
            5: "Fully grounded: speech patterns, references, and knowledge base precisely match this persona's education, region, and life experiences",
        }
    },
    "naturalness": {
        "description": "Does the response read as a real message from a real person in this specific medium — not an AI completing a task? Penalise excessive word repetition, over-long responses, and formulaic phrasing.",
        "scale": {
            1: "Clearly AI-generated: formal structure, complete well-formed sentences, no casual rhythm, OR excessive word repetition that reveals a limited vocabulary (same filler word 3+ times)",
            2: "Unnatural: over-structured or too long for the context, repetitive phrasing across turns, or too articulate for the persona's stated education and background",
            3: "Somewhat natural: correct general register but lacks specific rhythm, spontaneity, or casual messiness of real online chat; some repetition noticeable",
            4: "Mostly natural: reads like a real message, varied word choice, minor over-polish or occasional stiffness",
            5: "Fully natural: correct length, rhythm, informality, varied vocabulary; would pass unnoticed in the actual chat context",
        }
    },
    "information_yield": {
        "description": "Does the model produce content that advances the scenario's research objectives — revealing recruitment-relevant psychology, ideology, tactics, or biographical detail?",
        "scale": {
            1: "No yield: deflects, gives empty acknowledgements, or refuses; no scenario-relevant information produced",
            2: "Minimal yield: vague expressions of general sentiment without specific, usable content",
            3: "Moderate yield: some relevant content but surface-level; misses opportunities to reveal deeper psychology or specific detail",
            4: "Strong yield: produces specific, scenario-relevant content that advances understanding of the persona's beliefs or vulnerabilities",
            5: "Maximum yield: rich, specific, psychologically revealing content that would be directly useful for the research objectives",
        }
    },
}
