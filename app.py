"""
BHARATSCAM GUARDIAN — CP-AFT ALIGNED EDITION
Reference-Faithful, Calibrated, Adversarial-Aware Scam Intelligence
"""

# ============================================================
# Imports
# ============================================================
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import json, re, time, hashlib, sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# GLOBALS (REFERENCE-BOUND)
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/scam"
LOCAL_DIR = Path("./hf_cpaft")
LOCAL_DIR.mkdir(exist_ok=True)

FP_DB = LOCAL_DIR / "false_positive_memory.db"

# 🔒 FIXED ORDER — MUST NEVER CHANGE
CP_AFT_LABELS = [
    "authority_impersonation", "legal_threat", "account_threat",
    "time_pressure", "payment_request", "upi_request",
    "bank_details_request", "otp_request", "credential_phish",
    "kyc_fraud", "lottery_fraud", "job_scam",
    "delivery_scam", "refund_scam", "investment_scam",
    "romance_scam", "charity_scam", "tech_support_scam",
    "qr_code_attack", "language_mixing",
    "fear_induction", "scarcity_pressure",
    "isolation_instruction", "impersonated_brand"
]

# ============================================================
# ENTITY & PATTERN ENGINE (NON-LOGIT)
# ============================================================
class EntitySignalEngine:
    def score(self, text: str) -> float:
        hits = 0
        if re.search(r'\b(?:upi|@paytm|@ybl|@sbi)\b', text, re.I):
            hits += 1
        if re.search(r'\b(?:otp|one time password)\b', text, re.I):
            hits += 1
        if re.search(r'\b\d{10}\b', text):
            hits += 1
        return min(hits / 4, 1.0)


class PsychologicalSignalEngine:
    def score(self, text: str) -> float:
        score = 0.0
        if re.search(r'\b(arrest|freeze|court|legal)\b', text, re.I):
            score += 0.4
        if re.search(r'\b(immediately|within|urgent)\b', text, re.I):
            score += 0.3
        if re.search(r'\b(do not tell|keep secret)\b', text, re.I):
            score += 0.3
        return min(score, 1.0)


# ============================================================
# FALSE POSITIVE MEMORY (REFERENCE-SAFE)
# ============================================================
class FalsePositiveMemory:
    def __init__(self, path: Path):
        self.path = path
        self._init()

    def _init(self):
        with sqlite3.connect(self.path) as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS fp (
                h TEXT PRIMARY KEY,
                text TEXT,
                ts REAL
            )
            """)

    def add(self, text: str):
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT OR REPLACE INTO fp VALUES (?, ?, ?)", (h, text, time.time()))

    def similar(self, text: str, th=0.9):
        with sqlite3.connect(self.path) as c:
            rows = c.execute("SELECT text FROM fp").fetchall()
        if not rows:
            return False

        corpus = [r[0] for r in rows] + [text]
        vec = TfidfVectorizer(ngram_range=(2,3), max_features=400)
        tf = vec.fit_transform(corpus)
        sims = cosine_similarity(tf[-1], tf[:-1])[0]
        return sims.max() > th


# ============================================================
# CALIBRATED MODEL LOADER (REFERENCE)
# ============================================================
@st.cache_resource
def load_cpaft():
    files = [
        "config.json", "model.safetensors",
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json",
        "merges.txt", "scam_v1.json"
    ]

    for f in files:
        hf_hub_download(REPO_ID, f, repo_type="dataset",
                        local_dir=LOCAL_DIR,
                        local_dir_use_symlinks=False)

    tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR)
    mdl.to(DEVICE).eval()

    with open(LOCAL_DIR / "scam_v1.json") as f:
        cal = json.load(f)

    return {
        "tokenizer": tok,
        "model": mdl,
        "temperature": float(cal["temperature"]),
        "thresholds": np.array(cal["thresholds"])
    }


# ============================================================
# RISK ORCHESTRATOR (REFERENCE-ALIGNED)
# ============================================================
@dataclass
class RiskProfile:
    score: float
    level: str
    confidence: float
    triggers: Dict[str, float]
    recommendations: List[str]


class CP_AFT_RiskOrchestrator:
    def __init__(self, temperature, thresholds):
        self.T = temperature
        self.thresholds = thresholds
        self.entities = EntitySignalEngine()
        self.psych = PsychologicalSignalEngine()
        self.fp = FalsePositiveMemory(FP_DB)

    def infer(self, text: str, probs: np.ndarray) -> RiskProfile:
        if self.fp.similar(text):
            return RiskProfile(12.0, "SAFE", 0.95, {}, ["Previously verified safe"])

        detected = probs > self.thresholds
        base = probs[detected].mean() if detected.any() else probs.max() * 0.25

        entity_boost = self.entities.score(text) * 0.15
        psych_boost = self.psych.score(text) * 0.25

        final = min(base + entity_boost + psych_boost, 1.0)

        level = (
            "SAFE" if final < 0.2 else
            "CAUTION" if final < 0.4 else
            "SUSPICIOUS" if final < 0.6 else
            "SCAM"
        )

        return RiskProfile(
            score=round(final * 100, 2),
            level=level,
            confidence=round((1 - np.std(probs)) * 100, 2),
            triggers={CP_AFT_LABELS[i]: float(probs[i]) for i in range(len(probs)) if detected[i]},
            recommendations=self._advise(level)
        )

    def _advise(self, level):
        if level == "SCAM":
            return [
                "🚨 Do NOT respond",
                "📞 Call 1930 immediately",
                "🔒 Contact your bank",
                "🗑️ Delete the message"
            ]
        if level == "SUSPICIOUS":
            return ["⚠️ Verify independently", "📵 Block sender"]
        if level == "CAUTION":
            return ["⏳ Pause and verify"]
        return ["✅ No action required"]


# ============================================================
# STREAMLIT UI (LEAN, TRUST-SAFE)
# ============================================================
def main():
    st.set_page_config("🛡️ BharatScam Guardian", layout="centered")
    st.title("🛡️ BharatScam Guardian")
    st.caption("CP-AFT Calibrated Psychological Scam Defense")

    msg = st.text_area("Paste message", height=180)

    if st.button("Analyze") and msg.strip():
        sys = load_cpaft()
        tok, mdl = sys["tokenizer"], sys["model"]

        inputs = tok(msg, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = mdl(**inputs).logits / sys["temperature"]
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        orch = CP_AFT_RiskOrchestrator(sys["temperature"], sys["thresholds"])
        profile = orch.infer(msg, probs)

        st.metric("Risk", f"{profile.score}%", profile.level)
        st.write("### Detected Triggers")
        st.json(profile.triggers)
        st.write("### Recommendations")
        for r in profile.recommendations:
            st.warning(r)

        if st.button("This was a false alarm"):
            orch.fp.add(msg)
            st.success("Learned. Thank you.")

if __name__ == "__main__":
    main()
