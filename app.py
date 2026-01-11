# ============================================================
# IMPORTS
# ============================================================
import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/scam-detection"
LOCAL_DIR = Path("./hf_cpaft_core")
LOCAL_DIR.mkdir(exist_ok=True)

CP_AFT_LABELS = [
    "AUTHORITY","URGENCY","FEAR","GREED",
    "SOCIAL_PROOF","SCARCITY","OBEDIENCE","TRUST"
]

# ============================================================
# MODEL LOADER
# ============================================================
def load_model():
    for f in ["config.json", "model.safetensors", "tokenizer.json", "scam_v1.json"]:
        hf_hub_download(REPO_ID, f, local_dir=LOCAL_DIR, repo_type="dataset")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR)
    model.to(DEVICE).eval()

    with open(LOCAL_DIR / "scam_v1.json") as f:
        cal = json.load(f)

    return (
        tokenizer,
        model,
        float(cal["temperature"]),
        np.array(cal["thresholds"])
    )

# ============================================================
# PURE MODEL-ONLY INFERENCE
# ============================================================
def analyze_text(text: str) -> dict:
    tokenizer, model, T, thresholds = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits / T
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    detected = probs > thresholds

    # --- risk computation (model-only) ---
    if detected.any():
        risk = probs[detected].mean()
    else:
        risk = probs.max() * 0.25

    # --- risk level ---
    if risk < 0.25:
        level = "SAFE"
    elif risk < 0.5:
        level = "CAUTION"
    elif risk < 0.75:
        level = "SUSPICIOUS"
    else:
        level = "SCAM"

    return {
        "risk_score": round(float(risk * 100), 2),
        "risk_level": level,
        "confidence": round(float((1 - np.std(probs)) * 100), 2),
        "triggers": {
            label: round(float(p), 3)
            for label, p, d in zip(CP_AFT_LABELS, probs, detected)
            if d
        },
        "raw_probs": {
            label: round(float(p), 3)
            for label, p in zip(CP_AFT_LABELS, probs)
        }
    }

# ============================================================
# EXAMPLE
# ============================================================
if __name__ == "__main__":
    text = "Your bank account will be frozen in 2 hours. Click link now."
    result = analyze_text(text)
    print(json.dumps(result, indent=2))
