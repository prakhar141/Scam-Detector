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

# ============================================================
# GLOBAL CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/scam"
LOCAL_DIR = Path("./hf_cpaft_core")
LOCAL_DIR.mkdir(exist_ok=True)

COLORS = {
    "SAFE": "#2D936C",
    "CAUTION": "#F4A261",
    "SUSPICIOUS": "#E76F51",
    "SCAM": "#C1121C"
}

CP_AFT_LABELS = [
    "account_threat","time_pressure","payment_request","credential_phish",
    "kyc_fraud","lottery_fraud","job_scam","delivery_scam","refund_scam",
    "investment_scam","romance_scam","charity_scam","qr_code_attack","fear_induction"
]

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
                # Scoring weights
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
        # Financial
        for m in re.findall(r'\b(?:₹|Rs\.?|INR)\s*[\d,]+|\b\d{6,}\b', text):
            claims.append(Claim(m,"financial"))
        # Temporal
        for m in re.findall(r'\b(?:today|tomorrow|yesterday|within\s+\d+\s+(?:hour|day|week)s?)\b', text):
            claims.append(Claim(m,"temporal"))
        # Identity
        for m in re.findall(r'\b(?:RBI|NPCI|UIDAI|IT Department|HDFC|ICICI|SBI|AXIS|KOTAK|Government|Police|CIBIL)\b', text):
            claims.append(Claim(m,"identity"))
        # Action
        for m in re.findall(r'\b(?:click|pay|transfer|send|share|update|verify)\s+(?:link|amount|money|details|OTP|UPI|account)\b', text):
            claims.append(Claim(m,"action"))
        return claims

    def score_verifiability(self, claims:List[Claim]) -> Tuple[float, List[str]]:
        details = []
        verified = 0
        for c in claims:
            if c.type=="financial" and re.search(r'\d{6,}',c.text):
                c.verifiability = 0.8
                verified+=1
                details.append(f"💰 '{c.text}' financial claim verifiable")
            elif c.type=="temporal":
                c.verifiability = 0.3
                details.append(f"⏰ '{c.text}' temporal claim low verifiability")
            elif c.type=="identity":
                c.verifiability = 0.7 if re.search(r'\b(?:RBI|NPCI|UIDAI|IT Department)\b',c.text) else 0.1
                details.append(f"🏛️ '{c.text}' identity claim verifiability={c.verifiability}")
                if c.verifiability>0.5: verified+=1
            elif c.type=="action":
                c.verifiability = 0.6 if any(w in c.text.lower() for w in ['app','portal','website','official']) else 0.0
                details.append(f"✅ '{c.text}' action claim verifiability={c.verifiability}")
                if c.verifiability>0.5: verified+=1
        return verified/len(claims) if claims else 0.0, details

class SemanticCoherenceEngine:
    """Detects confusion tactics"""
    def score(self,text:str) -> Tuple[float,List[str]]:
        score, issues = 0.0, []
        # Conflicting urgencies
        urgencies = set(re.findall(r'\b(immediately|now|within\s+\d+|asap|by\s+\d+)\b',text))
        if len(urgencies)>2:
            score+=0.3; issues.append(f"🕒 Conflicting urgencies: {urgencies}")
        # Authority overload
        auths = re.findall(r'\b(RBI|Government|Police|Bank|IT Dept|Court)\b',text)
        if len(auths)>=3: score+=0.25; issues.append(f"🏛️ Multiple authorities: {auths}")
        # Long sentences
        if any(len(s.split())>25 for s in re.split(r'[.!?]',text)): score+=0.15; issues.append("📜 Long/confusing sentences")
        # Emotional imbalance
        emotion = len(re.findall(r'\b(urgent|immediately|freeze|arrest|cancel|terminate)\b',text))
        factual = len(re.findall(r'\b(reference|transaction|account|number|date|time)\b',text))
        if factual==0: factual=1
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
        inputs = tok(text,return_tensors="pt",truncation=True,padding=True).to(DEVICE)
        with torch.no_grad():
            logits = mdl(**inputs).logits/self.T
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        # Scam signals
        detected = probs>self.thres
        scam_signals = probs[detected].mean() if detected.any() else probs.max()*0.25
        # Legitimacy
        leg_score, leg_proof = self.trust.score(text)
        # Claims
        claims_list = self.claims.extract_claims(text)
        ver_score, claim_details = self.claims.score_verifiability(claims_list)
        # Coherence
        incoh_score, incoh_issues = self.coherence.score(text)
        # Multiplicative risk
        risk = scam_signals*(1-leg_score)**2*(1-ver_score)*(1+0.5*incoh_score)
        # Dynamic, normalized 0-1 scaling for 4 levels
        thresholds = np.array([0.25,0.5,0.75])
        if risk<thresholds[0]: level="SAFE"
        elif risk<thresholds[1]: level="CAUTION"
        elif risk<thresholds[2]: level="SUSPICIOUS"
        else: level="SCAM"
        # Confidence
        conf = (1-np.std(probs))*100
        # Triggers
        triggers = {label:float(p) for label,p,det in zip(CP_AFT_LABELS,probs,detected) if det}
        # Recommendations
        if leg_score>0.6:
            recos = ["✅ Official trust anchors detected","📞 Verify on official portal","🔍 Check reference numbers"]
        elif risk>0.5:
            recos = ["🚨 DO NOT respond","📞 Call official numbers","🔒 Enable transaction limits","🗑️ Delete after reporting"]
        else:
            recos = ["⏳ Pause before acting","🤔 Can I verify without replying?"]
        return RiskProfile(round(risk*100,2),level,round(conf,2),triggers,recos,leg_proof,claim_details,incoh_issues)

# ============================================================
# STREAMLIT UI
# ============================================================
def init_state():
    for k in ["msg","profile","stage"]:
        if k not in st.session_state: st.session_state[k]=None

def main():
    st.set_page_config(page_title="BharatScam Guardian",page_icon="🛡️",layout="centered")
    init_state()
    st.header("🛡️ BharatScam Guardian — Legitimacy First, Risk Second")
    msg = st.text_area("Paste any message here",height=200)
    if st.button("Analyze") and msg.strip():
        st.session_state.msg = msg
        st.session_state.stage="RUNNING"
        st.rerun()
    
    if st.session_state.stage=="RUNNING":
        progress=st.progress(0)
        for i in range(100): progress.progress(i+1); time.sleep(0.01)
        orch = CoreOrchestrator(*load_model()[2:])
        profile = orch.infer(st.session_state.msg)
        st.session_state.profile=profile
        st.session_state.stage="DONE"
        st.rerun()
    
    if st.session_state.stage=="DONE" and st.session_state.profile:
        p=st.session_state.profile
        st.markdown(f"### Risk Score: {p.score}% — {p.level}")
        st.progress(p.score/100)
        if p.legitimacy_proof: st.markdown("#### ✅ Legitimacy Anchors"); [st.markdown(f"- {x}") for x in p.legitimacy_proof]
        if p.claim_analysis: st.markdown("#### 🔬 Claim Verifiability"); [st.markdown(f"- {x}") for x in p.claim_analysis]
        if p.coherence_issues: st.markdown("#### ⚠️ Coherence Issues"); [st.markdown(f"- {x}") for x in p.coherence_issues]
        if p.triggers: st.markdown("#### 🎯 Detected Scam Triggers"); [st.markdown(f"- {k}: {v:.1%}") for k,v in p.triggers.items()]
        st.markdown("#### Recommended Actions"); [st.markdown(f"- {r}") for r in p.recos]
        if st.button("Analyze New Message"): st.session_state.update({"msg":None,"profile":None,"stage":None}); st.rerun()

if __name__=="__main__":
    main()
