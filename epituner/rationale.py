# epituner/rationale.py
from typing import Dict, List

def extract_evidence(record: Dict) -> List[str]:
    """
    Return short, verbatim quotes from available fields.
    Keep order stable; skip empty fields.
    """
    keys = ["Chief Complaint", "ChiefComplaint", "Discharge Diagnosis", "DischargeDiagnosis", "Triage Notes", "TriageNotes"]
    out = []
    for k in keys:
        if k in record and record[k]:
            txt = str(record[k]).strip()
            if txt:
                out.append(f'[{k}] "{txt}"')
    return out[:6]  # keep it tight

def rationale_prompt(evidence_quotes: List[str], decided_label: str) -> str:
    evidence_block = "\n".join(f"- {q}" for q in evidence_quotes)
    return (
        "You will explain a clinical classification decision using only the quoted facts.\n"
        "Facts you may use (verbatim quotes):\n"
        f"{evidence_block}\n\n"
        "Write 2–4 short bullet points:\n"
        "- Paraphrase the quotes in plain English.\n"
        "- Do NOT introduce any new facts; only use the quotes.\n"
        f'End with one line: Decision: {decided_label} because <12–20 words>.\n'
    )
