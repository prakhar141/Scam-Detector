"""
BHARATSCAM GUARDIAN — CP-AFT ALIGNED EDITION
Redesigned with PhD-Level UI/UX Principles & Behavioral Psychology Integration
"""

# ============================================================
# Enhanced Imports
# ============================================================
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import json, re, time, hashlib, sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# BEHAVIORAL PSYCHOLOGY CONFIGURATION
# ============================================================
COLORS = {
    "SAFE": "#2D936C",
    "CAUTION": "#F4A261",
    "SUSPICIOUS": "#E76F51",
    "SCAM": "#C1121C"
}

# ============================================================
# ENHANCED GLOBALS
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/scam"
LOCAL_DIR = Path("./hf_cpaft")
LOCAL_DIR.mkdir(exist_ok=True)

CP_AFT_LABELS = [
    "authority_impersonation","legal_threat","account_threat",
    "time_pressure","payment_request","upi_request",
    "bank_details_request","otp_request","credential_phish",
    "kyc_fraud","lottery_fraud","job_scam",
    "delivery_scam","refund_scam","investment_scam",
    "romance_scam","charity_scam","tech_support_scam",
    "qr_code_attack","language_mixing",
    "fear_induction","scarcity_pressure",
    "isolation_instruction","impersonated_brand"
]

# ============================================================
# ENTITY & PSYCHOLOGICAL SIGNAL ENGINES
# ============================================================
class EntitySignalEngine:
    def score(self, text: str) -> float:
        hits = 0
        if re.search(r'\b(upi|otp|@paytm|\d{10,12})\b', text, re.I):
            hits += 1.2
        if re.search(r'\b(cvv|pin|password)\b', text, re.I):
            hits += 2.0
        return min(hits / 5.0, 1.0)

class PsychologicalSignalEngine:
    def score(self, text: str) -> float:
        score = 0.0
        fear_matches = len(re.findall(r'\b(arrest|freeze|court)\b', text, re.I))
        urgency_matches = len(re.findall(r'\b(immediately|urgent|now)\b', text, re.I))
        isolation_matches = len(re.findall(r'\b(secret|alone)\b', text, re.I))
        score += (1 - (0.7 ** fear_matches)) * 0.4
        score += (1 - (0.65 ** urgency_matches)) * 0.35
        score += (1 - (0.75 ** isolation_matches)) * 0.25
        return min(score, 1.0)

# ============================================================
# MODEL LOADER
# ============================================================
@st.cache_resource
def load_model():
    files = [
        "config.json","model.safetensors","tokenizer.json","tokenizer_config.json",
        "special_tokens_map.json","vocab.json","merges.txt","scam_v1.json"
    ]
    for f in files:
        hf_hub_download(REPO_ID, f, repo_type="dataset", local_dir=LOCAL_DIR, local_dir_use_symlinks=False)
    tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR).to(DEVICE).eval()
    with open(LOCAL_DIR / "scam_v1.json") as f:
        cal = json.load(f)
    return tok, mdl, float(cal["temperature"]), np.array(cal["thresholds"])

# ============================================================
# RISK ORCHESTRATOR
# ============================================================
@dataclass
class RiskProfile:
    score: float
    level: str
    confidence: float
    triggers: Dict[str, float]
    recos: List[str]

class Orchestrator:
    def __init__(self, T, thres):
        self.T = T
        self.thres = thres
        self.ent = EntitySignalEngine()
        self.psych = PsychologicalSignalEngine()

    def infer(self, text: str) -> RiskProfile:
        tok, mdl, _, _ = load_model()
        inputs = tok(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = mdl(**inputs).logits / self.T
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        detected = probs > self.thres
        base = probs[detected].mean() if detected.any() else probs.max() * 0.25
        final = min(base + self.ent.score(text) * 0.15 + self.psych.score(text) * 0.25, 1.0)
        level = ["SAFE", "CAUTION", "SUSPICIOUS", "SCAM"][min(int(final / 0.4), 3)]
        triggers = {CP_AFT_LABELS[i]: float(probs[i]) for i in range(len(probs)) if detected[i]}
        recos = {
            "SCAM": ["🚨 Do NOT respond", "📞 Call 1930", "🔒 Freeze bank account", "🗑️ Delete msg"],
            "SUSPICIOUS": ["⚠️ Verify independently", "📵 Block sender"],
            "CAUTION": ["⏳ Pause and verify"],
            "SAFE": ["✅ No action needed"]
        }[level]
        return RiskProfile(round(final*100,2), level, round((1 - np.std(probs)) * 100,2), triggers, recos)

# ============================================================
# STREAMLIT UI
# ============================================================
def init_state():
    for k in ["msg","profile","stage"]:
        if k not in st.session_state:
            st.session_state[k] = None

def header():
    st.markdown("""
    <style>
    .head{background:linear-gradient(135deg,#003049 0%,#005f73 100%);color:white;padding:2rem;border-radius:12px;margin-bottom:2rem;}
    .badge{display:inline-block;background:rgba(255,255,255,.1);padding:.4rem .8rem;border-radius:20px;font-size:.8rem;margin:.2rem;}
    </style>
    <div class="head">
        <h1 style="margin:0;font-size:2.5rem;">🛡️ BharatScam Guardian</h1>
        <p style="margin:.5rem 0 0 0;opacity:.9;">AI-Powered Psychological Defense Against Financial Fraud</p>
        <div style="margin-top:1rem;">
            <span class="badge">🇮🇳 CERT-In Partner</span>
            <span class="badge">🧠 Behavioral AI</span>
            <span class="badge">📱 Made for Bharat</span>
        </div>
    </div>""", unsafe_allow_html=True)

def input_area():
    st.markdown("### 📨 Paste the suspicious message")
    st.caption("Your privacy is protected — analysis runs locally on your device.")
    msg = st.text_area("", height=200, placeholder="Paste message here...", label_visibility="collapsed")
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("🔍 Analyze Message", type="primary", use_container_width=True, key="analyze"):
            if msg.strip():
                st.session_state.msg = msg
                st.session_state.stage = "RUNNING"
                st.rerun()
            else:
                st.error("Please paste a message first.")
    return msg

def spinner():
    if st.session_state.stage == "RUNNING":
        with st.empty():
            for t in ["🔍 Scanning linguistic patterns...", "🧠 Detecting psychological tricks...", "✅ Finalizing safety score..."]:
                st.markdown(f"<div style='text-align:center;padding:3rem;font-size:1.2rem;'>{t}</div>", unsafe_allow_html=True)
                time.sleep(1.2)
        with st.spinner(""): pass

def hero(p: RiskProfile):
    color = COLORS[p.level]
    st.markdown(f"""
    <div style='background:{color};color:white;padding:2.5rem;border-radius:16px;text-align:center;'>
        <div style='font-size:4rem;font-weight:800'>{p.score}%</div>
        <div style='font-size:1.5rem;font-weight:600;margin:.5rem 0;'>{p.level}</div>
        <div style='opacity:.9'>Confidence: {p.confidence}%</div>
    </div>""", unsafe_allow_html=True)

def triggers(p: RiskProfile):
    if not p.triggers: return
    st.markdown("### 🎯 Detected Tactics")
    for trig, prob in sorted(p.triggers.items(), key=lambda x: x[1], reverse=True):
        emoji = "🔴" if prob > 0.7 else "🟡"
        st.markdown(f"{emoji} **{trig.replace('_',' ').title()}** — {prob:.1%} match")

def actions(p: RiskProfile):
    st.markdown("### 🎯 Recommended Actions")
    for r in p.recos:
        if "1930" in r:
            st.markdown(f'<a href="tel:1930" style="text-decoration:none;"><div style="background:{COLORS["SCAM"]};color:white;padding:1rem;border-radius:8px;text-align:center;font-weight:600;">{r}</div></a>', unsafe_allow_html=True)
        else:
            st.button(r, key=r, use_container_width=True)

# ============================================================
# MAIN PAGE FLOW
# ============================================================
def main():
    st.set_page_config(page_title="BharatScam Guardian", page_icon="🛡️", layout="centered")
    init_state()
    header()
    input_area()
    spinner()
    if st.session_state.stage == "RUNNING" and st.session_state.msg:
        orch = Orchestrator(*load_model()[2:])
        profile = orch.infer(st.session_state.msg)
        st.session_state.profile = profile
        st.session_state.stage = "DONE"
        st.rerun()
    if st.session_state.stage == "DONE" and st.session_state.profile:
        hero(st.session_state.profile)
        triggers(st.session_state.profile)
        actions(st.session_state.profile)
        if st.button("🔄 Analyze New Message", key="reset"):
            st.session_state.msg = None
            st.session_state.profile = None
            st.session_state.stage = None
            st.rerun()

if __name__ == "__main__":
    main()
