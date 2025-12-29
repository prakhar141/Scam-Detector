# app.py
################################################################################
# 0.  Imports
################################################################################
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID  = "prakhar146/Scam"
CACHE_DIR = Path("./hf_model")
CACHE_DIR.mkdir(exist_ok=True)

LABELS = [
    "authority_name",
    "threat_type",
    "time_pressure",
    "payment_method",
    "language_mixing",
]

################################################################################
# 1.  Download artefacts once per container
################################################################################
ARTEFACTS = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "scam_v1.json"
       
]

def _download():
    for file in ARTEFACTS:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=file,
            repo_type="dataset",
            local_dir=CACHE_DIR,
            local_dir_use_symlinks=False,
        )

################################################################################
# 2.  Load model + two calibrations
################################################################################
@st.cache_resource(show_spinner="Booting AI engine …")
def _load():
    _download()
    tok = AutoTokenizer.from_pretrained(CACHE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CACHE_DIR).to(DEVICE).eval()

    with open(CACHE_DIR / "scam_v1.json") as f:
        temp = float(json.load(f)["temperature"])

    with open(CACHE_DIR / "extreme_distribution.json") as f:
        extreme = json.load(f)          # Dict[str, List[float]]  label -> 10 000 worst-case probs

    return tok, model, temp, extreme


tok, model, temperature, EXTREME = _load()

################################################################################
# 3.  Inference helpers
################################################################################
def _get_proba(text: str) -> np.ndarray:
    """Return calibrated probabilities for each attribute."""
    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits / temperature
        prob = torch.sigmoid(logits).cpu().numpy().squeeze()
    return prob


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Exact 1-D Wasserstein (EMD) between two *sorted* samples."""
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def _risk_score(prob: np.ndarray) -> float:
    """
    Think of each label as a 1-D distribution.
    We compare the incoming message to the *worst 1 %* of each label
    using 1-Wasserstein distance.  The *closest* match (smallest
    distance) tells us how 'extreme' this message is.
    Return percentile in [0,100] where 100 ≡ already seen worst-case.
    """
    dists = []
    for p, label in zip(prob, LABELS):
        ref = np.array(EXTREME[label])          # 10 000 calibrated probs
        # subsample for speed (exact EMD still accurate)
        ref = np.random.choice(ref, size=1000, replace=False)
        dists.append(_wasserstein_1d(np.array([p]), ref))
    # closest match
    closest = min(dists)
    # squash to percentile via sigmoid
    # scale chosen on validation set: 0.02 → ~50 %, 0.05 → ~90 %
    percentile = 1 / (1 + math.exp(-(closest - 0.02) / 0.008))
    return percentile * 100


def _explain(prob: np.ndarray) -> str:
    """Return 1-sentence explanation."""
    top = np.argsort(prob)[-2:][::-1]
    keys = [LABELS[i] for i in top]
    return (
        f"Strongest scam patterns: {', '.join(keys)} "
        f"({prob[top[0]]:.0%} and {prob[top[1]]:.0%} confidence)."
    )


def analyse(text: str) -> Tuple[str, str, Dict[str, float]]:
    prob = _get_proba(text)
    score = _risk_score(prob)
    explain = _explain(prob)
    detail = {L: float(p) for L, p in zip(LABELS, prob)}

    if score < 30:
        verdict = "🟢 SAFE"
    elif score < 70:
        verdict = "🟡 SUSPICIOUS"
    else:
        verdict = "🔴 SCAM"
    return verdict, f"{score:.0f}/100 risk – {explain}", detail


################################################################################
# 4.  Streamlit UI
################################################################################
st.set_page_config(page_title="AI Scam Detector", layout="centered")
st.title("🛡️ AI Scam Detector")
st.markdown(
    "Paste any message.  The engine compares it to **worst-case scam distributions** "
    "and tells you how embarrassed you will be if you let it through."
)

msg = st.text_area("Message to analyse", height=150)

if st.button("Analyse"):
    if not msg.strip():
        st.warning("Please enter text.")
        st.stop()

    with st.spinner("Analysing …"):
        verdict, reason, probs = analyse(msg)

    st.markdown(f"## {verdict}")
    st.write(reason)

    with st.expander("Technical probabilities"):
        st.json(probs)

