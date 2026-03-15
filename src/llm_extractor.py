"""
llm_extractor.py
----------------
Uses Mistral AI (via 3DS Proxem gateway) to extract structured fields
from unstructured clinical notes.

Flow:
    Unstructured note → Mistral LLM → Structured JSON → ML Classifier → Severity Prediction

Config (set in .env):
    MISTRAL_API_KEY   - your gateway API key
    MISTRAL_BASE_URL  - https://fmgateway.proxem.dsone.3ds.com/v1
    MISTRAL_MODEL     - mistralai/Mistral-Small-3.1-24B-Instruct-2503
    MISTRAL_CERT_PATH - path to your SSL certificate .pem file
"""

import os
import json
import logging
import re
import requests
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Config from .env ──────────────────────────────────────────────────────────
API_KEY   = os.getenv("MISTRAL_API_KEY")
BASE_URL  = os.getenv("MISTRAL_BASE_URL", "https://fmgateway.proxem.dsone.3ds.com/v1")
MODEL     = os.getenv("MISTRAL_MODEL",    "mistralai/Mistral-Small-3.1-24B-Instruct-2503")
CERT_PATH = os.getenv("MISTRAL_CERT_PATH")

# Adverse events considered clinically serious
SERIOUS_AES = [
    "chest pain", "anaphylaxis", "cardiac arrest",
    "liver failure", "dyspnea", "shortness of breath",
    "respiratory failure", "seizure", "stroke"
]

EXTRACTION_PROMPT = """You are a clinical data extraction assistant.
Extract structured information from the clinical adverse event note below.

Return ONLY a valid JSON object with these exact fields:
{{
    "age": <integer or null>,
    "gender": <"Male", "Female", or "Unknown">,
    "weight_kg": <float or null>,
    "drug_name": <string or null>,
    "dosage_mg": <float or null>,
    "route": <"Oral", "Intravenous", "Subcutaneous", "Topical", or "Unknown">,
    "adverse_event": <string describing the main adverse event>,
    "time_to_onset_days": <float or null>,
    "num_concomitant_drugs": <integer>,
    "symptom_count": <integer>,
    "has_comorbidity": <0 or 1>,
    "has_prior_reaction": <0 or 1>,
    "severity_indicators": <list of strings indicating severity clues>
}}

Rules:
- Return ONLY the JSON, no explanation, no markdown backticks
- Use null for any field not mentioned in the note
- num_concomitant_drugs: count drugs mentioned OTHER than the study drug
- symptom_count: count distinct symptoms mentioned
- has_comorbidity: 1 if any pre-existing conditions mentioned, else 0
- has_prior_reaction: 1 if prior drug reactions mentioned, else 0
- severity_indicators: list words/phrases suggesting severity e.g. ["severe", "hospitalized"]

Clinical note:
{note}
"""


# ── Mistral gateway extraction ─────────────────────────────────────────────────
def extract_with_mistral(note: str) -> Optional[dict]:
    """
    Extract structured data using Mistral via the 3DS Proxem gateway.

    Uses the OpenAI-compatible /chat/completions endpoint.
    SSL verification uses the provided .pem certificate.
    """
    if not API_KEY:
        logger.warning("MISTRAL_API_KEY not set in .env — skipping Mistral extraction")
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise clinical data extraction assistant. Return only valid JSON, no markdown."
            },
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(note=note)
            }
        ],
        "temperature": 0.0,   # deterministic extraction
        "max_tokens":  600
    }

    # SSL verification — use cert if provided, else default verify=True
    ssl_verify = CERT_PATH if CERT_PATH and Path(CERT_PATH).exists() else True

    if CERT_PATH and not Path(CERT_PATH).exists():
        logger.warning(
            f"MISTRAL_CERT_PATH set but file not found: {CERT_PATH}\n"
            "Falling back to default SSL verification."
        )

    try:
        logger.info(f"Calling Mistral gateway: {BASE_URL}/chat/completions")
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            verify=ssl_verify,
            timeout=30
        )

        response.raise_for_status()
        data = response.json()

        raw = data["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if model adds them despite instructions
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        result = json.loads(raw)
        logger.info("Mistral extraction successful")
        return result

    except requests.exceptions.SSLError as e:
        logger.error(
            f"SSL Error: {e}\n"
            f"Check your MISTRAL_CERT_PATH in .env: {CERT_PATH}"
        )
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to Mistral gateway: {e}")
        return None
    except requests.exceptions.Timeout:
        logger.error("Mistral gateway request timed out after 30s")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"Mistral gateway HTTP error: {e.response.status_code} — {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Mistral JSON response: {e}\nRaw: {raw}")
        return None
    except Exception as e:
        logger.error(f"Unexpected Mistral extraction error: {e}")
        return None


# ── Rule-based fallback ────────────────────────────────────────────────────────
def extract_with_rules(note: str) -> dict:
    """
    Rule-based fallback extractor using regex patterns.
    Used when Mistral is unavailable or returns an error.
    """
    note_lower = note.lower()

    # Age
    age = None
    age_match = re.search(r"(\d{1,3})\s*(?:year|yr)[\s-]*old", note_lower)
    if age_match:
        age = int(age_match.group(1))

    # Gender
    gender = "Unknown"
    if any(w in note_lower for w in ["female", " woman ", "she ", "her "]):
        gender = "Female"
    elif any(w in note_lower for w in ["male", " man ", "he ", "his "]):
        gender = "Male"

    # Dosage
    dosage_mg = None
    dose_match = re.search(r"(\d+(?:\.\d+)?)\s*mg", note_lower)
    if dose_match:
        dosage_mg = float(dose_match.group(1))

    # Time to onset
    time_to_onset_days = None
    time_match = re.search(r"(\d+(?:\.\d+)?)\s*(day|hour|week)", note_lower)
    if time_match:
        value = float(time_match.group(1))
        unit  = time_match.group(2)
        if "hour" in unit:
            time_to_onset_days = value / 24
        elif "week" in unit:
            time_to_onset_days = value * 7
        else:
            time_to_onset_days = value

    # Route
    route = "Unknown"
    if "oral" in note_lower or "tablet" in note_lower:
        route = "Oral"
    elif "intravenous" in note_lower or " iv " in note_lower:
        route = "Intravenous"
    elif "subcutaneous" in note_lower:
        route = "Subcutaneous"
    elif "topical" in note_lower:
        route = "Topical"

    # Severity indicators
    severity_words = [
        "severe", "serious", "critical", "life-threatening",
        "hospitalized", "emergency", "fatal", "death", "icu"
    ]
    severity_indicators = [w for w in severity_words if w in note_lower]

    # Concomitant drugs
    drug_mentions = len(re.findall(
        r"\b(?:taking|on|prescribed|administered)\b.*?\b\w+(?:mg|mcg)\b",
        note_lower
    ))
    num_concomitant = max(0, drug_mentions - 1)

    # Comorbidity
    comorbidity_words = [
        "pre-existing", "history of", "diabetes", "hypertension",
        "cardiac", "renal", "hepatic", "comorbidity", "condition"
    ]
    has_comorbidity = int(any(w in note_lower for w in comorbidity_words))

    # Prior reaction
    prior_words = ["prior reaction", "previous reaction", "allergy", "prior adverse"]
    has_prior_reaction = int(any(w in note_lower for w in prior_words))

    # Symptom count
    symptom_words = [
        "pain", "nausea", "vomiting", "headache", "rash", "fever",
        "dizziness", "fatigue", "chest", "breathing", "hypotension",
        "hypertension", "dyspnea", "edema", "urticaria"
    ]
    symptom_count = max(1, sum(1 for w in symptom_words if w in note_lower))

    # Main adverse event
    adverse_event = "Unknown"
    for kw in ["reported", "experienced", "developed", "presented with"]:
        match = re.search(rf"{kw}\s+([^.]+)", note_lower)
        if match:
            adverse_event = match.group(1).strip()[:60]
            break

    return {
        "age":                   age,
        "gender":                gender,
        "weight_kg":             None,
        "drug_name":             None,
        "dosage_mg":             dosage_mg,
        "route":                 route,
        "adverse_event":         adverse_event,
        "time_to_onset_days":    time_to_onset_days,
        "num_concomitant_drugs": num_concomitant,
        "symptom_count":         symptom_count,
        "has_comorbidity":       has_comorbidity,
        "has_prior_reaction":    has_prior_reaction,
        "severity_indicators":   severity_indicators
    }


# ── Main extraction entry point ────────────────────────────────────────────────
def extract_from_note(note: str) -> dict:
    """
    Main extraction function.

    Priority:
    1. Mistral via 3DS Proxem gateway (if MISTRAL_API_KEY is set)
    2. Rule-based fallback (always works, no API needed)
    """
    if API_KEY:
        logger.info("Attempting Mistral extraction...")
        result = extract_with_mistral(note)
        if result:
            return result
        logger.warning("Mistral extraction failed — falling back to rule-based extraction")

    logger.info("Using rule-based extraction...")
    return extract_with_rules(note)


def fill_defaults(extracted: dict) -> dict:
    """Fill missing fields with sensible defaults."""
    defaults = {
        "age":                   55,
        "gender":                "Unknown",
        "weight_kg":             75.0,
        "drug_name":             "Unknown",
        "dosage_mg":             50.0,
        "route":                 "Unknown",
        "adverse_event":         "Unknown",
        "time_to_onset_days":    3.0,
        "num_concomitant_drugs": 0,
        "symptom_count":         1,
        "has_comorbidity":       0,
        "has_prior_reaction":    0,
        "severity_indicators":   []
    }
    for key, default_val in defaults.items():
        if key not in extracted or extracted[key] is None:
            extracted[key] = default_val
    return extracted


def note_to_features(note: str) -> dict:
    """
    Full pipeline: clinical note → extracted fields → filled defaults.
    Ready to feed into ML feature engineering pipeline.
    """
    extracted = extract_from_note(note)
    filled    = fill_defaults(extracted)

    # Flag serious adverse events
    ae_lower = str(filled.get("adverse_event", "")).lower()
    filled["is_serious_ae"] = int(any(ae in ae_lower for ae in SERIOUS_AES))

    logger.info(f"Extracted fields:\n{json.dumps(filled, indent=2)}")
    return filled


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_notes = [
        "68 year old male patient reported severe chest pain and shortness of breath "
        "approximately 4 hours after taking 100mg of DrugX via oral route. "
        "Patient has pre-existing hypertension and diabetes.",

        "A 45-year-old female developed mild nausea and headache 2 days after "
        "starting 50mg of DrugY. No prior drug reactions noted.",
    ]

    for i, note in enumerate(test_notes, 1):
        print(f"\n{'='*60}")
        print(f"Note {i}: {note[:80]}...")
        print("\nExtracted:")
        result = note_to_features(note)
        print(json.dumps(result, indent=2))