# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import torch, torch.nn.functional as F
import numpy as np
import re, time, json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
# >>> WHISPER START <<<
import whisper, tempfile, io
import streamlit_toggle as tog   # pip install streamlit-toggle-switch-pkg
# >>> WHISPER END <<<

# ============================================================
# GLOBAL CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/scam-detection"
LOCAL_DIR = Path("./hf_cpaft_core")
LOCAL_DIR.mkdir(exist_ok=True)

COLORS = {
    "SAFE": "#2D936C",
    "CAUTION": "#F4A261",
    "SUSPICIOUS": "#E76F51",
    "SCAM": "#C1121C"
}

# NEW (must match Cell-6/8 exactly)
CP_AFT_LABELS = ["AUTHORITY","URGENCY","FEAR","GREED",
                 "SOCIAL_PROOF","SCARCITY","OBEDIENCE","TRUST"]
LEGITIMATE_PATTERNS = {
    "bank_official": r'\b(?:HDFC|ICICI|SBI|AXIS|KOTAK|BOB|PNB)[\s]*(?:Bank|Ltd|Limited)\b|\bRBI\b|\bNPCI\b|\bIRDAI\b',
    "govt_official": r'\b(?:UIDAI|ITA|GST|EPFO|CBDT|MCA|CEIR)\b|\b(?:gov\.in|nic\.in|ac\.in)\b',
    "verifiable_ref": r'\b(?:UTR|Ref|Reference|Txn|Transaction)[\s]*[No|ID|Number]*[:#]?\s*[A-Z0-9]{8,20}\b',
    "official_contact": r'\b(?:1800|1860)[\s]*-?\d{3}[\s]*-?\d{4}\b|\b(?:91|0)?\s*\d{8}\b',
    "secure_url": r'\bhttps?://(?:www\.)?(?:hdfcbank\.com|icicibank\.com|sbi\.co\.in|axisbank\.com|paytm\.com|amazon\.in|flipkart\.com)[/\w.-]*\b'
}

SCAM_PATTERNS = {
    "urgency_vague": r'\b(immediately|now|urgent|within\s+\d+\s+hours?)\b(?!.*\b(fraud|unauthorized)\b)',
    "authority_impersonation": r'\b(?:fake|fraud|spoof|impersonat).*(?:RBI|Bank|Govt|Police|CIBIL|IT Dept)\b',
    "unverifiable_sender": r'\b(?:Dear Customer|Valued User|Respected Sir/Madam)\b',
    "payment_redirection": r'\b(?:pay|transfer|send).*?(?:UPI|Wallet|Account).*?(?:new|alternate|other)\b'
}

# ============================================================
# DATACLASSES
# ============================================================
@dataclass
class Claim:
    text: str
    type: str  # financial, temporal, identity, action
    verifiability: float = 0.0

@dataclass
class RiskProfile:
    score: float
    level: str
    confidence: float
    triggers: Dict[str,float]
    recos: List[str]
    legitimacy_proof: List[str]
    claim_analysis: List[str]
    coherence_issues: List[str]

# ============================================================
# ENGINES
# ============================================================
class TrustAnchorEngine:
    """Score messages based on official trust anchors"""
    def score(self, text: str) -> Tuple[float, List[str]]:
        score, hits = 0.0, []
        for name, pat in LEGITIMATE_PATTERNS.items():
            matches = re.findall(pat, text, re.I)
            if matches:
                hits.append(f"✓ {name.replace('_',' ').title()}: {len(matches)}")
                weights = {
                    "bank_official": 0.35,
                    "govt_official": 0.35,
                    "verifiable_ref": 0.3,
                    "official_contact": 0.25,
                    "secure_url": 0.35
                }
                score += min(len(matches) * weights.get(name,0.2), weights.get(name,0.2))
        return min(score, 1.0), hits

class VerifiableClaimsEngine:
    """Decompose text into verifiable claims"""
    def extract_claims(self, text:str) -> List[Claim]:
        claims = []
        for m in re.findall(r'\b(?:₹|Rs\.?|INR)\s*[\d,]+|\b\d{6,}\b', text):
            claims.append(Claim(m,"financial"))
        for m in re.findall(r'\b(?:today|tomorrow|yesterday|within\s+\d+\s+(?:hour|day|week)s?)\b', text):
            claims.append(Claim(m,"temporal"))
        for m in re.findall(r'\b(?:RBI|NPCI|UIDAI|IT Department|HDFC|ICICI|SBI|AXIS|KOTAK|Government|Police|CIBIL)\b', text):
            claims.append(Claim(m,"identity"))
        for m in re.findall(r'\b(?:click|pay|transfer|send|share|update|verify)\s+(?:link|amount|money|details|OTP|UPI|account)\b', text):
            claims.append(Claim(m,"action"))
        return claims

    def score_verifiability(self, claims:List[Claim]) -> Tuple[float, List[str]]:
        details, verified = [], 0
        for c in claims:
            if c.type=="financial" and re.search(r'\d{6,}',c.text):
                c.verifiability = 0.8; verified+=1
                details.append(f"💰 '{c.text}' – financial claim verifiable")
            elif c.type=="temporal":
                c.verifiability = 0.3
                details.append(f"⏰ '{c.text}' – temporal claim low verifiability")
            elif c.type=="identity":
                c.verifiability = 0.7 if re.search(r'\b(?:RBI|NPCI|UIDAI|IT Department)\b',c.text) else 0.1
                if c.verifiability>0.5: verified+=1
                details.append(f"🏛️ '{c.text}' – identity claim verifiability={c.verifiability}")
            elif c.type=="action":
                c.verifiability = 0.6 if any(w in c.text.lower() for w in ['app','portal','website','official']) else 0.0
                if c.verifiability>0.5: verified+=1
                details.append(f"✅ '{c.text}' – action claim verifiability={c.verifiability}")
        return verified/len(claims) if claims else 0.0, details

class SemanticCoherenceEngine:
    """Detects confusion tactics"""
    def score(self,text:str) -> Tuple[float,List[str]]:
        score, issues = 0.0, []
        urgencies = set(re.findall(r'\b(immediately|now|within\s+\d+|asap|by\s+\d+)\b',text))
        if len(urgencies)>2: score+=0.3; issues.append(f"🕒 Conflicting urgencies: {urgencies}")
        auths = re.findall(r'\b(RBI|Government|Police|Bank|IT Dept|Court)\b',text)
        if len(auths)>=3: score+=0.25; issues.append(f"🏛️ Multiple authorities: {auths}")
        if any(len(s.split())>25 for s in re.split(r'[.!?]',text)): score+=0.15; issues.append("📜 Long/confusing sentences")
        emotion = len(re.findall(r'\b(urgent|immediately|freeze|arrest|cancel|terminate)\b',text))
        factual = len(re.findall(r'\b(reference|transaction|account|number|date|time)\b',text)) or 1
        if emotion>factual*2: score+=0.3; issues.append(f"😱 Emotion vs facts imbalance: {emotion}/{factual}")
        return min(score,1.0), issues

# ============================================================
# MODEL LOADER
# ============================================================
@st.cache_resource
def load_model():
    for f in ["config.json","model.safetensors","tokenizer.json","scam_v1.json"]:
        hf_hub_download(REPO_ID,f,local_dir=LOCAL_DIR,repo_type="dataset")
    tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR).to(DEVICE).eval()
    with open(LOCAL_DIR/"scam_v1.json") as f: cal=json.load(f)
    return tok, mdl, float(cal["temperature"]), np.array(cal["thresholds"])
# >>> WHISPER START <<<
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")
# >>> WHISPER END <<<

# ============================================================
# CORE ORCHESTRATOR
# ============================================================
class CoreOrchestrator:
    def __init__(self,T,thres):
        self.T, self.thres = T, thres
        self.trust = TrustAnchorEngine()
        self.claims = VerifiableClaimsEngine()
        self.coherence = SemanticCoherenceEngine()
    
    # ------------- MESSAGE-SPECIFIC ACTIONS -------------
    def _build_actions(self, rp:RiskProfile, leg_score:float, incoh_score:float) -> List[str]:
        t = rp.triggers
        actions = []
        # high-risk
        if rp.level=="SCAM" or rp.score>75:
            actions.append("🚨 **Do NOT reply / click / pay** – highest risk")
            actions.append("📞 Cross-check via official customer-care number printed on card/bank-statement")
            actions.append("🗑️ Delete message & report as spam")
            if "URGENCY" in t: actions.append("⏱️ Slow-down – scares are designed to rush you")
            if "AUTHORITY" in t: actions.append("🏛️ Real RBI/Bank never threaten on WhatsApp/SMS")
            return actions
        # medium-risk
        if rp.level=="SUSPICIOUS" or rp.score>50:
            actions.append("⏳ Pause – do nothing for 10 minutes")
            actions.append("🔍 Can you verify without replying? (official app / website)")
            if "GREED" in t: actions.append("💸 If it looks too good to be true, it is")
            if incoh_score>0.3: actions.append("🧠 Confusing language = red flag")
            actions.append("📞 Call bank on printed number & ask")
            return actions
        # low-risk but some triggers
        if rp.level=="CAUTION" and t:
            actions.append("🟡 Double-check sender ID & spelling of links")
            if "SOCIAL_PROOF" in t: actions.append("👥 Random testimonials may be fake")
            actions.append("🔒 Keep UPI autopay limits low")
            return actions
        # safe
        if leg_score>0.6:
            actions.append("✅ Official anchors detected – likely safe")
            actions.append("📲 Still, verify in your bank app before acting")
            return actions
        # default
        return ["✅ Looks clean – always use common sense"]

    def infer(self,text:str) -> RiskProfile:
        tok, mdl, _, _ = load_model()
        text = text.strip()
        if not text: text="blank"
        inputs = tok(text,return_tensors="pt",truncation=True,padding=True).to(DEVICE)
        with torch.no_grad():
            logits = mdl(**inputs).logits/self.T
            probs  = torch.sigmoid(logits).cpu().numpy()[0]
        detected = probs>self.thres 
        scam_signals = probs[detected].mean() if detected.any() else probs.max()*0.25
        
        leg_score, leg_proof = self.trust.score(text)
        claims_list = self.claims.extract_claims(text)
        ver_score, claim_details = self.claims.score_verifiability(claims_list)
        incoh_score, incoh_issues = self.coherence.score(text)
        
        risk = scam_signals*(1-leg_score)**2*(1-ver_score)*(1+0.5*incoh_score)
        base_thresh = np.array([0.25,0.5,0.75])
        adaptive_thresh = base_thresh*(1-leg_score)*(1-0.5*ver_score)+0.2*incoh_score
        if risk<adaptive_thresh[0]: level="SAFE"
        elif risk<adaptive_thresh[1]: level="CAUTION"
        elif risk<adaptive_thresh[2]: level="SUSPICIOUS"
        else: level="SCAM"
        
        conf = (1-np.std(probs))*100
        triggers = {label:float(p) for label,p,det in zip(CP_AFT_LABELS,probs,detected) if det}
        recos = self._build_actions(RiskProfile(0,level,0,triggers,[],[],[],[]), leg_score, incoh_score)
        
        return RiskProfile(round(float(risk*100),2),level,round(float(conf),2),
                           triggers,recos,leg_proof,claim_details,incoh_issues)

# ============================================================
# BHARATSCAM GUARDIAN  –  UNIQUE LIGHT-THEME UI
# ============================================================
import streamlit as st
import time
import tempfile
import whisper
import torch
import pandas as pd
import numpy as np

# ---------- colour palette (clean Indian summer) ----------
THEME = {
    "bg": "#FDFBF8",
    "card": "#FFFFFF",
    "accent": "#FF8F00",
    "safe": "#2E7D32",
    "caution": "#F57C00",
    "suspicious": "#D32F2F",
    "scam": "#B71C1C",
    "text": "#3E2723",
    "subtle": "#8D6E63"
}

# ---------- inject css ----------
def local_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp {{background:{THEME["bg"]};color:{THEME["text"]};font-family:'Inter',sans-serif;}}
    .card {{background:{THEME["card"]};border-radius:16px;padding:24px;margin-bottom:24px;
            box-shadow:0 2px 8px rgba(0,0,0,.06);border:1px solid #F5F0EB;}}
    div.stButton > button {{
        border:none;color:#FFF;
        background:linear-gradient(90deg,{THEME["accent"]} 0%,{THEME["caution"]} 100%);
        font-weight:600;border-radius:12px;height:52px;font-size:18px;
    }}
    h1,h2,h3 {{font-weight:700;letter-spacing:-0.5px;}}
    .subtle {{color:{THEME["subtle"]};font-size:14px;}}
    </style>
    """, unsafe_allow_html=True)

# ---------- helpers ----------
def init_state():
    for k in ["msg", "profile", "stage", "mode"]:
        if k not in st.session_state:
            st.session_state[k] = None


# ---------- page ----------
def main():
    st.set_page_config(page_title="BharatScam Guardian", page_icon="🛡️", layout="centered")
    local_css()
    init_state()

    # ---- hero ----
    st.markdown("""
    <div style="text-align:center;margin-top:-60px;margin-bottom:40px;">
        <h1 style="font-size:52px;background:-webkit-linear-gradient(45deg,#FF8F00,#FF6F00);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        BharatScam Guardian</h1>
        <p class="subtle">AI that smells a rat — but sometimes barks at shadows 🤖</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- toggle ----------
    if st.button(
        label="🎤 Speak" if not st.session_state.mode else "⌨️ Type",
        help="Switch input mode"
    ):
        st.session_state.mode = not st.session_state.mode
        st.rerun()

    # ---------- input ----------
    if st.session_state.mode:  # SPEECH MODE
        st.markdown("""
        <div style="background:#fff8e1;border-left:5px solid #ff8f00;
        border-radius:12px;padding:14px 18px;margin-bottom:12px;">
        👂 <b>I’m listening.</b> Press START and speak.
        </div>
        """, unsafe_allow_html=True)

        audio = st.audio_input("Record")
        if audio:
            raw = audio.read()
            audio_hash = hash(raw)

            if st.session_state.get("_last_audio_hash") != audio_hash:
                st.session_state._last_audio_hash = audio_hash
                with st.spinner("🧠 Turning your voice into words…"):
                    model = load_whisper()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(raw)
                        path = tmp.name
                    try:
                        result = model.transcribe(path, fp16=False)
                        text = result.get("text", "").strip()
                        if text:
                            st.session_state.msg = text
                            st.rerun()
                    finally:
                        import os
                        if os.path.exists(path):
                            os.remove(path)

    # ---------- unified text box (FIXED) ----------
    st.text_area(
        "",
        key="msg",
        placeholder="🎙️ Speak or 💬 paste a message here…",
        height=180,
        label_visibility="collapsed"
    )

    # ---------- analyze ----------
    if st.button("🛡️ Guard This Message", use_container_width=True):
        if st.session_state.msg and st.session_state.msg.strip():
            st.session_state.stage = "RUNNING"
            st.rerun()

    # ---------- running ----------
    if st.session_state.stage == "RUNNING":
        with st.spinner("🔍 Reading between the lines…"):
            time.sleep(1)
            st.session_state.stage = "DONE"
            st.rerun()

    # ---------- footer ----------
    st.markdown("""
    <div style="text-align:center;margin-top:40px;color:#8D6E63;font-size:14px;">
        Built with ❤️ by <b>Prakhar Mathur</b>. BITS Pilani
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

