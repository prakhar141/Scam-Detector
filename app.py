import streamlit as st
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from pathlib import Path

# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(page_title="📛 Scam Detector", layout="centered")
st.title("📛 Scam Detection System")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/Scam"        # DATASET repo
REPO_TYPE = "dataset"

LABELS = [
    "authority_name",
    "threat_type",
    "time_pressure",
    "payment_method",
    "language_mixing"
]

LOCAL_DIR = Path("./hf_model")
LOCAL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Download required files
# --------------------------------------------------
REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "scam_v1.json"
]

def download_files():
    for file in REQUIRED_FILES:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=file,
            repo_type=REPO_TYPE,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )

# --------------------------------------------------
# Load model, tokenizer, calibration
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_all():
    download_files()

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR)

    model.to(DEVICE)
    model.eval()

    with open(LOCAL_DIR / "scam_v1.json", "r") as f:
        cal = json.load(f)

    temperature = float(cal.get("temperature", 1.0))
    thresholds = np.array(
        cal.get("thresholds", [0.5] * model.config.num_labels)
    )

    return model, tokenizer, temperature, thresholds


model, tokenizer, temperature, base_thresholds = load_all()

# --------------------------------------------------
# Senior ML adaptive thresholding
# --------------------------------------------------
def adaptive_thresholds(probs, base):
    """
    Signal-aware thresholding:
    - Strong signal → stricter
    - Weak but structured → looser
    """
    mean_conf = probs.mean(axis=1, keepdims=True)

    dynamic = np.where(
        mean_conf > 0.45, base + 0.12,
        np.where(mean_conf < 0.18, base - 0.15, base)
    )

    return np.clip(dynamic, 0.25, 0.8)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits / temperature
        probs = torch.sigmoid(logits).cpu().numpy()

    thresholds = adaptive_thresholds(probs, base_thresholds)
    preds = (probs > thresholds).astype(int)

    detected = [LABELS[i] for i, v in enumerate(preds[0]) if v == 1]

    if len(detected) == 0:
        verdict = "🟢 SAFE"
    elif len(detected) <= 2:
        verdict = "🟡 SUSPICIOUS"
    else:
        verdict = "🔴 SCAM"

    return verdict, detected, probs[0]

# --------------------------------------------------
# UI
# --------------------------------------------------
user_text = st.text_area(
    "Enter SMS / WhatsApp / Email text",
    height=150,
    placeholder="Paste message here..."
)

if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter text.")
    else:
        with st.spinner("Analyzing..."):
            verdict, detected, probs = predict(user_text)

        st.subheader("Result")
        st.markdown(f"### {verdict}")
        st.write("**Detected Dimensions:**", detected if detected else "None")

        st.write("**Probabilities:**")
        for lbl, p in zip(LABELS, probs):
            st.write(f"- {lbl}: {p:.3f}") 
