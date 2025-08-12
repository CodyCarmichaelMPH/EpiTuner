# epituner/rationale.py (tight prompt; new chat context each time)
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

def build_rationale_messages(evidence_quotes, decided_label):
    sys = "You explain clinical decisions using only the provided quotes. Do not invent facts."
    user = (
        "Use ONLY these verbatim quotes:\n" +
        "\n".join(f"- {q}" for q in evidence_quotes[:6]) +
        f"\n\nWrite 2–4 short bullets. End with: Decision: {decided_label} because <12–20 words>."
    )
    return [
      {"role":"system","content": sys},
      {"role":"user","content": user},
      {"role":"assistant","content": ""}  # fresh assistant turn
    ]

def create_fallback_rationale(evidence_quotes: List[str], decided_label: str) -> str:
    """Create a simple rationale when model generation fails"""
    if not evidence_quotes:
        return f"Classification: {decided_label} - No detailed evidence available in the medical record."
    
    # Extract key terms from evidence
    evidence_summary = []
    for quote in evidence_quotes[:3]:  # Use first 3 pieces of evidence
        # Extract the field name and a short excerpt
        if '] "' in quote:
            field_part = quote.split('] "')[0] + ']'
            content_part = quote.split('] "')[1].rstrip('"')
            # Take first 50 characters of content
            short_content = content_part[:50] + "..." if len(content_part) > 50 else content_part
            evidence_summary.append(f"{field_part} mentions: {short_content}")
    
    rationale = f"Classification: {decided_label}\n\nEvidence review:\n" + "\n".join(f"• {ev}" for ev in evidence_summary)
    if decided_label != "Unknown":
        rationale += f"\n\nDecision: {decided_label} based on the clinical indicators found in the record."
    else:
        rationale += f"\n\nDecision: {decided_label} due to insufficient clear indicators for classification."
    
    return rationale
