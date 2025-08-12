# epituner/inference_scoring.py (replace your pick_label/score code)

from typing import Dict, Tuple, List
import torch
import torch.nn.functional as F

@torch.no_grad()
def chat_prompt(tokenizer, messages: List[dict]) -> str:
    # Build the exact string the base model expects
    # (don't hand-roll; rely on the tokenizer's chat template)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )  # returns a single string

@torch.no_grad()
def score_labels(model, tokenizer, messages: List[dict], labels=("Match","Not a Match")) -> Dict[str, float]:
    """
    Compute log-prob for each label by teacher-forcing the label *inside* the chat template.
    Only the label tokens contribute to loss (prompt tokens masked with -100).
    """
    device = next(model.parameters()).device
    base = chat_prompt(tokenizer, messages)  # system+user+assistant "Answer:" etc.

    scores = {}
    # NOTE: many LLMs expect a leading space before words; the chat template handles that for us.
    for lab in labels:
        text = base + lab
        toks = tokenizer(text, return_tensors="pt").to(device)

        # Mask out everything except the label span
        prompt_ids = tokenizer(base, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[-1]
        labels_ids = toks.input_ids.clone()
        labels_ids[:, :prompt_len] = -100  # ignore prompt tokens in loss

        out = model(**toks, labels=labels_ids)
        # out.loss is mean NLL over the label tokens → convert to log-prob (neg loss)
        scores[lab] = -float(out.loss)
    return scores

def pick_label(scores: Dict[str, float], tau: float = 0.15, temperature: float = 1.0) -> Tuple[str, float, bool]:
    """
    Numerically stable confidence (no NaNs) + margin-based abstain.
    """
    # pack in fixed order
    labs = list(scores.keys())
    x = torch.tensor([scores[l] for l in labs], dtype=torch.float32) / max(temperature, 1e-6)
    # log-sum-exp trick for stability
    x = x - x.max()
    probs = torch.exp(x) / torch.exp(x).sum()  # stable softmax; no 0/0
    # choose
    top_idx = int(torch.argmax(probs))
    second = float(torch.topk(probs, 2).values[1])
    top = float(probs[top_idx])
    margin = top - second
    unknown = margin < tau
    return labs[top_idx], top, unknown
