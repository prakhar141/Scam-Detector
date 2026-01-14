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
                details.append(f"💰 '{c.text}' financial claim verifiable")
            elif c.type=="temporal":
                c.verifiability = 0.3
                details.append(f"⏰ '{c.text}' temporal claim low verifiability")
            elif c.type=="identity":
                c.verifiability = 0.7 if re.search(r'\b(?:RBI|NPCI|UIDAI|IT Department)\b',c.text) else 0.1
                if c.verifiability>0.5: verified+=1
                details.append(f"🏛️ '{c.text}' identity claim verifiability={c.verifiability}")
            elif c.type=="action":
                c.verifiability = 0.6 if any(w in c.text.lower() for w in ['app','portal','website','official']) else 0.0
                if c.verifiability>0.5: verified+=1
                details.append(f"✅ '{c.text}' action claim verifiability={c.verifiability}")
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
    # tiny ≈ 75 MB, fastest for CPU Streamlit-cloud
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
    
    def infer(self,text:str) -> RiskProfile:
        tok, mdl, _, _ = load_model()
        text = text.strip()
        if not text:
            text="blank"
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
        if leg_score>0.6:
            recos = ["✅ Official trust anchors detected","📞 Verify on official portal","🔍 Check reference numbers"]
        elif risk>0.5:
            recos = ["🚨 DO NOT respond","📞 Call official numbers","🔒 Enable transaction limits","🗑️ Delete after reporting"]
        else:
            recos = ["⏳ Pause before acting","🤔 Can I verify without replying?"]
        
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
    "bg": "#FDFBF8",               # warm paper-white
    "card": "#FFFFFF",
    "accent": "#FF8F00",           # saffron accent
    "safe": "#2E7D32",             # Indian-flag green
    "caution": "#F57C00",          # soft amber
    "suspicious": "#D32F2F",       # brick red
    "scam": "#B71C1C",             # deep maroon
    "text": "#3E2723",             # espresso brown
    "subtle": "#8D6E63"            # warm grey
}

# ---------- inject css ----------
def local_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp {{background: {THEME["bg"]}; color: {THEME["text"]}; font-family: 'Inter', sans-serif;}}
    .card {{background: {THEME["card"]}; border-radius: 16px; padding: 24px; margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,.06); border: 1px solid #F5F0EB;}}
    .stProgress > div > div > div > div {{background: linear-gradient(90deg,{THEME["accent"]} 0%, {THEME["caution"]} 100%);}}
    div.stButton > button {{border: none; color: #FFF; background: linear-gradient(90deg,{THEME["accent"]} 0%, {THEME["caution"]} 100%);
                            font-weight: 600; border-radius: 12px; height: 52px; font-size: 18px;}}
    div.stButton > button:hover {{transform: scale(1.02);}}
    h1,h2,h3 {{font-weight: 700; letter-spacing: -0.5px;}}
    .subtle {{color: {THEME["subtle"]}; font-size: 14px;}}
    </style>
    """, unsafe_allow_html=True)

# ---------- helpers ----------
def init_state():
    for k in ["msg","profile","stage"]:
        if k not in st.session_state:
            st.session_state[k]=None

def risk_badge(level:str) -> str:
    color = {"SAFE":THEME["safe"],"CAUTION":THEME["caution"],"SUSPICIOUS":THEME["suspicious"],"SCAM":THEME["scam"]}[level]
    return f'<span style="background:{color}22;color:{color};padding:6px 16px;border-radius:999px;font-weight:600;">{level}</span>'

# ---------- page ----------
def main():
    st.set_page_config(page_title="BharatScam Guardian", page_icon="🛡️", layout="centered")
    local_css()
    init_state()

    # ---- hero ----
    st.markdown(f"""
        <div style="text-align:center;margin-top:-60px;margin-bottom:40px;">
        <h1 style="font-size:52px;background:-webkit-linear-gradient(45deg,{THEME["accent"]},#FF6F00);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">BharatScam Guardian</h1>
        <p class="subtle">AI that smells a rat — but sometimes barks at shadows 🤖</p>
        </div>
        """, unsafe_allow_html=True)

    # ---- input ----
    # ---------- UNIQUE TOGGLE ----------
    st.markdown("""
    <style>
      .toggle-pill{
        display:inline-flex;align-items:center;border-radius:999px;
        padding:6px 14px;font-weight:600;cursor:pointer;
        transition:all .3s ease;
      }
      .off{background:#e0e0e0;color:#333}
      .on{background:#ff8f00;color:#fff}
    </style>
    """, unsafe_allow_html=True)

    if "mode" not in st.session_state:
        st.session_state.mode = False

    # click detector
    if st.button(
        label=f"{'🎤'} Speak" if st.session_state.mode else f"{'⌨️'} Type",
        key="pill_toggle",
        help="Switch input mode"
    ):
        st.session_state.mode = not st.session_state.mode
        st.rerun()

    mode = st.session_state.mode

    # ---------- unified text box ----------
    if st.session_state.get("mode"):          # SPEECH MODE
        st.markdown(
    """
    <div style="
        background: linear-gradient(135deg,#fff8e1 0%,#ffecb3 100%);
        border-left:5px solid #ff8f00;
        border-radius:12px;
        padding:14px 18px;
        font-size:17px;
        color:#3e2723;
        box-shadow:0 2px 6px rgba(0,0,0,.07);
    ">
    👂 <b>I’m listening. </b> Press START and tell me.
    </div>
    """,
    unsafe_allow_html=True
)
        audio_bytes = st.audio_input("Record", key="mic")
        if audio_bytes is not None:
            # run Whisper only once per new recording
            if st.session_state.get("_last_audio_hash") != hash(audio_bytes):
                st.session_state._last_audio_hash = hash(audio_bytes)
                with st.spinner("Transcribing…"):
                    model = load_whisper()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_bytes.read())
                        tmp.flush()
                        result = model.transcribe(tmp.name, language="en", fp16=False)
                    st.session_state["msg"] = result["text"].strip()
        msg = st.text_area("", value=st.session_state.get("msg", ""),
                           placeholder="Your speech will appear here…",
                           height=180, label_visibility="collapsed")
    else:                                     # TYPE MODE
        msg = st.text_area("", value=st.session_state.get("msg", ""),
                           placeholder="Paste the suspicious message here…",
                           height=180, label_visibility="collapsed")

    # keep single source of truth
    st.session_state.msg = msg

    if st.button("🛡️ Analyze Message", use_container_width=True) and msg.strip():
        st.session_state.msg = msg
        st.session_state.stage = "RUNNING"
        st.rerun()

    # ---- running ----
    if st.session_state.stage=="RUNNING":
        with st.container():
            st.markdown('<div class="card"><h4>🤖 AI is thinking …</h4>', unsafe_allow_html=True)
            bar = st.progress(0)
            for i in range(100):
                bar.progress(i+1)
                time.sleep(0.005)
            orch = CoreOrchestrator(*load_model()[2:])  # replace with your real orchestrator
            st.session_state.profile = orch.infer(st.session_state.msg)
            st.session_state.stage="DONE"
            st.markdown('</div>', unsafe_allow_html=True)
            st.rerun()

    # ---- results ----
    if st.session_state.stage=="DONE" and st.session_state.profile:
        p = st.session_state.profile

        # top card with personality hint
        st.markdown(f'<div class="card"><h3>Risk Score: {p.score}% {risk_badge(p.level)}</h3>', unsafe_allow_html=True)
        st.progress(float(p.score)/100.0)
        st.markdown(f'<p class="subtle">Confidence: {p.confidence}% &nbsp; • &nbsp; '
                    f'<small>🔍 <b>Tip:</b> I can over-call triggers; use your own judgement too.</small></p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # triggers (always show, even if empty)
        st.markdown('<div class="card"><h4>🎯 Detected Scam Triggers</h4>', unsafe_allow_html=True)
        if p.triggers:
            for k,v in p.triggers.items():
                st.error(f"{k.replace('_',' ').title()}: {float(v):.1%}")
        else:
            st.info("No specific triggers fired — message looks clean on this axis.")
        st.markdown('</div>', unsafe_allow_html=True)

        # other sections
        sections = [
            ("✅ Legitimacy Anchors", p.legitimacy_proof, "success"),
            ("🔬 Claim Verifiability", p.claim_analysis, "info"),
            ("⚠️ Coherence Issues", p.coherence_issues, "warning"),
            ("💡 Recommended Actions", p.recos, None)
        ]
        for title, items, flag in sections:
            if items:
                st.markdown(f'<div class="card"><h4>{title}</h4>', unsafe_allow_html=True)
                for x in items:
                    if flag=="success": st.success(x)
                    elif flag=="info": st.info(x)
                    elif flag=="warning": st.warning(x)
                    else: st.write(f"- {x}")
                st.markdown('</div>', unsafe_allow_html=True)

        # reset
        if st.button("🔄 Analyze New Message", use_container_width=True):
            st.session_state.update({"msg":None,"profile":None,"stage":None})
            st.rerun()

    # ---- persistent footer ----
    st.markdown(
        """
        <div style="
            text-align:center;
            margin-top:40px;
            padding:16px 0;
            color:#8D6E63;
            font-size:14px;
        ">
            Built with ❤️ by <b>Prakhar Mathur</b>. BITS Pilani<br/>
            <span style="font-size:13px;">
                Contact us: 
                <a href="mailto:prakhar.mathur2020@gmail.com" 
                   style="color:#FF8F00;text-decoration:none;font-weight:600;">
                   Meet my devloper
                </a>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__=="__main__": 
    main()
