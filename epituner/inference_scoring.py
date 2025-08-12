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
    
    try:
        for label in labels:
            text = prompt_text + label
            toks = tokenizer(text, return_tensors="pt").to(device)
            
            # Mask prompt tokens; only compute loss on the label portion
            prompt_toks = tokenizer(prompt_text, return_tensors="pt")
            prompt_len = prompt_toks["input_ids"].shape[-1]
            
            # Create labels for loss computation
            labels_ids = toks.input_ids.clone()
            labels_ids[:, :prompt_len] = -100
            
            # Forward pass to get loss
            out = model(**toks, labels=labels_ids)
            
            # Check if loss is valid
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss):
                print(f"Warning: Invalid loss for label '{label}': {out.loss}")
                scores[label] = float('-inf')  # Very low score for invalid loss
            else:
                nll = out.loss.item()
                scores[label] = -nll
                
    except Exception as e:
        print(f"Error in score_labels: {e}")
        # Fallback to equal scores
        for label in labels:
            scores[label] = 0.0
    
    return scores

def pick_label(scores: Dict[str,float], tau: float = 0.15, temperature: float = 1.0) -> Tuple[str, float, bool]:
    """
    Returns (label, confidence, unknown_flag).
    confidence is softmax over scores/temperature.
    Unknown if (top - second) < tau.
    """
    import math
    
    # Handle edge cases
    if not scores or len(scores) < 2:
        return ("Unknown", 0.5, True)
    
    # Filter out invalid scores
    valid_scores = {k: v for k, v in scores.items() if not (math.isnan(v) or math.isinf(v))}
    
    if len(valid_scores) < 2:
        # Fallback if we don't have enough valid scores
        first_label = list(scores.keys())[0]
        return (first_label, 0.5, True)
    
    items = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)
    top, second = items[0], items[1]
    margin = top[1] - second[1]
    unknown = margin < tau
    
    # Calibrated probability with numerical stability
    try:
        # Normalize scores for numerical stability
        max_score = max(valid_scores.values())
        normalized_scores = {k: v - max_score for k, v in valid_scores.items()}
        
        exps = [math.exp(s/temperature) for s in normalized_scores.values()]
        total = sum(exps)
        
        if total == 0 or math.isnan(total) or math.isinf(total):
            conf = 0.5  # Default fallback
        else:
            top_exp = math.exp(normalized_scores[top[0]]/temperature)
            conf = top_exp / total
            
        # Ensure confidence is valid
        if math.isnan(conf) or math.isinf(conf):
            conf = 0.5
            
    except (OverflowError, ZeroDivisionError, ValueError):
        conf = 0.5  # Fallback for numerical issues
    
    return (top[0], conf, unknown)
