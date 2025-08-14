# hybrid_infer.py  (root: C:\SonicAid_clean)
import os, json, numpy as np, torch
from pathlib import Path
from typing import Tuple, List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent
SAVED = ROOT / "saved_models"

# Find saved model folders (with a safety fallback for a common typo)
ROBERTA_CANDIDATES = [
    SAVED / "roberta-goemotion-final",
    SAVED / "robereta_goemotion_final",
    ROOT  / "roberta-goemotion-final",
]
GOEMO_ROBERTA = next((p for p in ROBERTA_CANDIDATES if p.exists()), None)
if GOEMO_ROBERTA is None:
    raise FileNotFoundError("RoBERTa model folder not found. Checked: " + ", ".join(map(str, ROBERTA_CANDIDATES)))

GOEMO_DISTIL = SAVED / "goemotions_distilbert_v3"
if not GOEMO_DISTIL.exists():
    raise FileNotFoundError(f"DistilBERT model not found: {GOEMO_DISTIL}")

DISTRESS_DIR = SAVED / "distress_classifier"
if not DISTRESS_DIR.exists():
    raise FileNotFoundError(f"Distress classifier not found: {DISTRESS_DIR}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

def _load_model_tokenizer(path: Path) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, List[str]]:
    model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(device).eval()
    try:
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=False, local_files_only=True)

    # Prefer explicit label list if available; else read from config; else fallback
    lbl_json = path / "label_classes.json"
    if lbl_json.exists():
        classes = json.loads(lbl_json.read_text(encoding="utf-8"))
        id2label = [classes[i] for i in range(len(classes))]
    else:
        cfg = model.config
        if isinstance(cfg.id2label, dict) and len(cfg.id2label) == cfg.num_labels:
            id2label = [cfg.id2label.get(str(i), cfg.id2label.get(i, f"label_{i}")) for i in range(cfg.num_labels)]
        else:
            id2label = [f"label_{i}" for i in range(cfg.num_labels)]
    return model, tok, id2label

# Load models
rob_model, rob_tok, rob_labels = _load_model_tokenizer(GOEMO_ROBERTA)
dis_model, dis_tok, dis_labels = _load_model_tokenizer(GOEMO_DISTIL)
dis_classifier = AutoModelForSequenceClassification.from_pretrained(str(DISTRESS_DIR)).to(device).eval()
try:
    disclf_tok = AutoTokenizer.from_pretrained(str(DISTRESS_DIR), use_fast=True, local_files_only=True)
except Exception:
    disclf_tok = AutoTokenizer.from_pretrained(str(DISTRESS_DIR), use_fast=False, local_files_only=True)

rob_labels_norm = [_norm(x) for x in rob_labels]
dis_labels_norm = [_norm(x) for x in dis_labels]

# Map Distil classes into RoBERTa’s label index space
name2idx_rob = {lab: i for i, lab in enumerate(rob_labels_norm)}
distil_to_rob = {i: name2idx_rob[l] for i, l in enumerate(dis_labels_norm) if l in name2idx_rob}
missing_distil = [dis_labels[i] for i in range(len(dis_labels)) if i not in distil_to_rob]
if missing_distil:
    print("Distil labels not in RoBERTa space (ignored in ensemble):", missing_distil[:10])

# Optional calibration (defaults work if file absent)
CALIB_PATH = SAVED / "infer_calibration.json"
W_ROBERTA = 0.40
W_DISTIL  = 0.60
NEUTRAL_BACKOFF_TAU = 0.15  # set None to disable

if CALIB_PATH.exists():
    try:
        cfg = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
        W_ROBERTA = float(cfg.get("w_roberta", W_ROBERTA))
        W_DISTIL  = float(cfg.get("w_distil", W_DISTIL))
        tau_val = cfg.get("neutral_backoff_tau", NEUTRAL_BACKOFF_TAU)
        NEUTRAL_BACKOFF_TAU = None if tau_val is None else float(tau_val)
        print(f"[hybrid] calibration loaded: w_rob={W_ROBERTA}, w_dis={W_DISTIL}, tau={NEUTRAL_BACKOFF_TAU}")
    except Exception as e:
        print("Could not read calibration file:", e)

# Fine → coarse emotion buckets (music-oriented)
BASE_FINE_TO_COARSE = {
    "joy": {"amusement","excitement","joy","optimism","pride","approval"},
    "sadness": {"sadness","disappointment","grief","remorse"},
    "anger": {"anger","annoyance","disgust"},
    "fear": {"fear","nervousness"},
    "love": {"love","caring","admiration"},
    "gratitude": {"gratitude"},
    "neutral": {"neutral","curiosity","confusion","realization","surprise","desire","embarrassment","disapproval","relief"},
    "calm": set(),  # filled via rules below
}
BASE_FINE_TO_COARSE = {k: {_norm(v) for v in vals} for k, vals in BASE_FINE_TO_COARSE.items()}

def coarse_from_fine(fine_label: str, distress: bool) -> str:
    """Small rule set to choose a music-friendly coarse bucket."""
    n = _norm(fine_label)

    # Edge cases that read differently under distress
    if n == "surprise":
        return "fear" if distress else "joy"
    if n in {"disapproval", "embarrassment"}:
        return "anger" if distress else "neutral"
    if n == "desire":
        return "sadness" if distress else "love"
    if n in {"relief", "realization"}:
        return "calm"

    # Default mapping
    for coarse, fines in BASE_FINE_TO_COARSE.items():
        if n in fines:
            return coarse
    return "neutral"

# Distress vs non‑distress groups
distress_emotions = {_norm(x) for x in {"anger","fear","grief","remorse","sadness","disappointment","nervousness","disgust","disapproval"}}
non_distress_emotions = {_norm(x) for x in {"joy","love","gratitude","optimism","amusement","excitement","surprise","relief","admiration","approval","caring"}}

@torch.no_grad()
def _logits(model, tok, text: str, max_length: int) -> np.ndarray:
    enc = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    return model(**enc).logits.detach().cpu().numpy()[0]

def _ensemble_logits(text: str, max_length: int = 48) -> np.ndarray:
    """Weighted RoBERTa + Distil logits in RoBERTa index space."""
    rob = _logits(rob_model, rob_tok, text, max_length)
    if dis_model is None:
        return rob
    dis = _logits(dis_model, dis_tok, text, max_length)
    acc = W_ROBERTA * rob.copy()
    for i_distil, i_rob in distil_to_rob.items():
        acc[i_rob] += W_DISTIL * dis[i_distil]
    return acc

@torch.no_grad()
def hybrid_predict(text: str, topk: int = 5, max_length: int = 48) -> Dict:
    # Distress classifier
    di = disclf_tok([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    dlogits = dis_classifier(**di).logits
    dprob = torch.softmax(dlogits, dim=1)
    is_distress = int(torch.argmax(dprob, dim=1).item())
    distress_conf = float(dprob[0, is_distress].item())

    # Emotion ensemble
    logits = _ensemble_logits(text, max_length)
    probs = (torch.softmax(torch.tensor(logits), dim=0)).numpy()

    # If nothing is confident, nudge to neutral
    if NEUTRAL_BACKOFF_TAU is not None:
        neutral_idx = name2idx_rob.get("neutral")
        if neutral_idx is not None and probs.max() < float(NEUTRAL_BACKOFF_TAU):
            tmp = np.zeros_like(probs); tmp[neutral_idx] = 1.0
            probs = tmp

    # Top‑k labels (RoBERTa space)
    k = min(topk, len(probs))
    top_idx = np.argsort(probs)[::-1][:k]
    top_emotions = [(rob_labels[i], float(probs[i])) for i in top_idx]
    top_norm = [(_norm(rob_labels[i]), float(probs[i])) for i in top_idx]

    # Keep emotions consistent with distress state
    filtered = [(lab, p) for lab, p in top_norm if (lab in distress_emotions if is_distress else lab in non_distress_emotions)]
    if not filtered:
        filtered = [top_norm[0]]

    chosen_fine_norm, chosen_prob = filtered[0]
    # Map back to original label casing
    try:
        idx = rob_labels_norm.index(chosen_fine_norm)
        chosen_fine = rob_labels[idx]
    except ValueError:
        chosen_fine = chosen_fine_norm
    chosen_coarse = coarse_from_fine(chosen_fine, bool(is_distress))

    return {
        "text": text,
        "distress": bool(is_distress),
        "distress_conf": round(distress_conf, 4),
        "top_emotions": [(rob_labels[i], round(float(probs[i]), 4)) for i in top_idx],
        "filtered_emotions": [(lab, round(p,4)) for lab, p in filtered],
        "chosen_emotion_fine": chosen_fine,
        "chosen_emotion_coarse": chosen_coarse,
        "chosen_prob": round(float(chosen_prob), 4),
    }

# Quick manual check
if __name__ == "__main__":
    s = "I feel so hopeless and tired of everything."
    print(hybrid_predict(s))
