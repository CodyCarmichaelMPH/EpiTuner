# epituner/inference_scoring.py
from typing import Dict, Tuple
import torch

@torch.no_grad()
def score_labels(model, tokenizer, prompt_text: str, labels=("Match", "Not a Match")) -> Dict[str, float]:
    """
    Compute negative log-likelihood for each label by teacher-forcing the label tokens.
    We only score the label suffix; prompt tokens are masked with -100.
    Returns a dict of log-probs (higher is better).
    """
    device = next(model.parameters()).device
    scores = {}
    for label in labels:
        text = prompt_text + label
        toks = tokenizer(text, return_tensors="pt").to(device)
        # Mask prompt tokens; only compute loss on the label portion
        prompt_len = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[-1]
        labels_ids = toks.input_ids.clone()
        labels_ids[:, :prompt_len] = -100
        out = model(**toks, labels=labels_ids)
        # out.loss is mean NLL over label tokens -> convert to log-prob
        nll = out.loss.item()
        scores[label] = -nll
    return scores

def pick_label(scores: Dict[str,float], tau: float = 0.15, temperature: float = 1.0) -> Tuple[str, float, bool]:
    """
    Returns (label, confidence, unknown_flag).
    confidence is softmax over scores/temperature.
    Unknown if (top - second) < tau.
    """
    import math
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top, second = items[0], items[1]
    margin = top[1] - second[1]
    unknown = margin < tau
    # calibrated probability
    exps = [math.exp(s/temperature) for _, s in scores.items()]
    total = sum(exps)
    conf = dict((k, math.exp(v/temperature)/total) for k, v in scores.items())[top[0]]
    return (top[0], conf, unknown)
