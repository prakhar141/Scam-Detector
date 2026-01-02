
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import json, re, time, hashlib, sqlite3, subprocess, sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# ============================================================
# BEHAVIORAL PSYCHOLOGY & VERIFIABILITY CONFIGURATION
# ============================================================
COLORS = {
    "SAFE": "#2D936C",
    "CAUTION": "#F4A261",
    "SUSPICIOUS": "#E76F51",
    "SCAM": "#C1121C"
}

# Official corpus for semantic alignment (hashed for privacy)
LEGITIMATE_PATTERNS = {
    "bank_official": r'\b(?:HDFC|ICICI|SBI|AXIS|KOTAK|BOB|PNB)[\s]*(?:Bank|Ltd|Limited)\b|\bRBI\b|\bNPCI\b|\bIRDAI\b',
    "govt_official": r'\b(?:UIDAI|ITA|GST|EPFO|CBDT|MCA|CEIR)\b|\b(?:gov\.in|nic\.in|ac\.in)\b',
    "verifiable_ref": r'\b(?:UTR|Ref|Reference|Txn|Transaction)[\s]*[No|ID|Number]*[\s]*[:#]?\s*[A-Z0-9]{8,20}\b',
    "official_contact": r'\b(?:1800|1860)[\s]*-?[\s]*\d{3}[\s]*-?[\s]*\d{4}\b|\b(?:91|0)?\s*?[\s]*?[\s]*?[\s]*([\s]*\d{8})\b',
    "secure_url": r'\bhttps?://(?:www\.)?(?:hdfcbank\.com|icicibank\.com|sbi\.co\.in|axisbank\.com|paytm\.com|amazon\.in|flipkart\.com)[/\w.-]*\b'
}

# Scam patterns remain but are now SECONDARY
SCAM_PATTERNS = {
    "urgency_vague": r'\b(immediately|now|urgent|within\s+\d+\s+hours?)\b(?!\s+(?:to\s+)?(?:avoid|prevent)\s+.*?\b(?:fraud|unauthorized)\b)',
    "authority_impersonation": r'\b(?:(?:fake|fraud|spoof|impersonat).*?(?:RBI|Bank|Govt|Government|Police|CIBIL|IT\s+Dept))|(?:(?:RBI|Bank|Govt|Government|Police|CIBIL|IT\s+Dept).*?(?:fake|fraud|spoof|impersonat))\b',
    "unverifiable_sender": r'\b(?:Dear Customer|Valued User|Respected Sir/Madam)\b',
    "payment_redirection": r'\b(?:pay|transfer|send).*?(?:UPI|Wallet|Account).*?(?:different|alternate|new|other)\b'
}

# ============================================================
# ENHANCED GLOBALS
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/scam"
LOCAL_DIR = Path("./hf_cpaft_core")
LOCAL_DIR.mkdir(exist_ok=True)
DB_PATH = Path("./trust_anchors.db")

CP_AFT_LABELS = [
    "account_threat","time_pressure","payment_request","credential_phish",
    "kyc_fraud","lottery_fraud","job_scam","delivery_scam","refund_scam",
    "investment_scam","romance_scam","charity_scam","qr_code_attack","fear_induction"
]

# ============================================================
# VERIFIABILITY & LEGITIMACY ENGINES (CORE INNOVATION)
# ============================================================

@dataclass
class Claim:
    text: str
    claim_type: str  # 'financial', 'temporal', 'identity', 'action'
    verifiability_score: float  # 0.0=unverifiable, 1.0=fully verifiable

class TrustAnchorEngine:
    """Detects trust anchors that legitimate messages MUST have"""
    def score(self, text: str) -> Tuple[float, List[str]]:
        legitimacy_hits = []
        score = 0.0
        
        for pattern_name, pattern in LEGITIMATE_PATTERNS.items():
            matches = len(re.findall(pattern, text, re.I))
            if matches > 0:
                legitimacy_hits.append(f"✓ {pattern_name.replace('_',' ').title()}: {matches} found")
                # Official patterns give HIGH legitimacy (0.3 each)
                if pattern_name in ["bank_official", "govt_official"]:
                    score += min(matches * 0.3, 0.4)
                # Verifiable references are strong anchors (0.25 each)
                elif pattern_name == "verifiable_ref":
                    score += min(matches * 0.25, 0.35)
                # Official contact adds legitimacy (0.2 each)
                elif pattern_name == "official_contact":
                    score += min(matches * 0.2, 0.25)
                # Secure URL is strong (0.3 each)
                elif pattern_name == "secure_url":
                    score += min(matches * 0.3, 0.4)
        
        return min(score, 1.0), legitimacy_hits

class VerifiableClaimsEngine:
    """Deconstructs message into claims and scores their verifiability"""
    def extract_claims(self, text: str) -> List[Claim]:
        """Simple regex-based claim extraction (no spaCy required)"""
        claims = []
        
        # Financial claims (amounts, numbers)
        financial_matches = re.findall(r'\b(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{2})?\b|\b\d{6,}\b', text, re.I)
        for match in financial_matches:
            claims.append(Claim(match, "financial", 0.0))
        
        # Temporal claims
        temporal_matches = re.findall(r'\b(?:today|tomorrow|yesterday|within\s+\d+\s+(?:hour|day|week|month)s?|by\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', text, re.I)
        for match in temporal_matches:
            claims.append(Claim(match, "temporal", 0.0))
        
        # Identity claims (organizations)
        identity_matches = re.findall(r'\b(?:RBI|NPCI|UIDAI|IT\s+Department|HDFC|ICICI|SBI|AXIS|KOTAK|Government|Police|CIBIL)\b', text, re.I)
        for match in identity_matches:
            claims.append(Claim(match, "identity", 0.0))
        
        # Action claims (verbs + objects)
        action_matches = re.findall(r'\b(?:click|pay|transfer|send|share|update|verify)\s+(?:link|amount|money|details|OTP|UPI|account)\b', text, re.I)
        for match in action_matches:
            claims.append(Claim(match, "action", 0.0))
        
        return claims
    
    def score_verifiability(self, text: str, claims: List[Claim]) -> Tuple[float, List[str]]:
        """Score how many claims can be verified without responding to message"""
        if not claims:
            return 0.0, ["No claims found"]
        
        verifiable_count = 0
        details = []
        
        for claim in claims:
            # Can this claim be verified through official channels?
            if claim.claim_type == "financial" and re.search(r'\b\d{6,}\b', claim.text):
                # Transaction IDs, UTR numbers are verifiable
                claim.verifiability_score = 0.8
                verifiable_count += 1
                details.append(f"💰 Financial claim '{claim.text}' is verifiable")
            elif claim.claim_type == "temporal" and re.search(r'\b\d{1,2}\s*(?:hour|day|week)', claim.text, re.I):
                # Time constraints - verifiable if official
                claim.verifiability_score = 0.3  # Lower score, time can be manipulated
                details.append(f"⏰ Temporal claim '{claim.text}' has low verifiability")
            elif claim.claim_type == "identity":
                # Identity claims need official verification
                if re.search(r'\b(?:RBI|NPCI|UIDAI|IT\s+Department)\b', claim.text, re.I):
                    claim.verifiability_score = 0.7
                    verifiable_count += 1
                    details.append(f"🏛️ Official identity '{claim.text}' is verifiable")
                else:
                    claim.verifiability_score = 0.1
                    details.append(f"⚠️ Generic identity '{claim.text}' is unverifiable")
            elif claim.claim_type == "action":
                # Actions that can be done through official app are more verifiable
                if any(word in claim.text.lower() for word in ['app', 'portal', 'website', 'official']):
                    claim.verifiability_score = 0.6
                    verifiable_count += 1
                    details.append(f"✅ Action '{claim.text}' has official channel")
                else:
                    claim.verifiability_score = 0.0
                    details.append(f"❌ Action '{claim.text}' requires direct response")
        
        overall_score = verifiable_count / len(claims) if claims else 0.0
        return overall_score, details

class SemanticCoherenceEngine:
    """Detects contradictions and confusion tactics used in scams"""
    def score(self, text: str) -> Tuple[float, List[str]]:
        """Returns INCOHERENCE score (0.0=clear, 1.0=confusing)"""
        issues = []
        score = 0.0
        
        # Check for multiple conflicting timeframes
        time_refs = re.findall(r'\b(immediately|now|within\s+\d+|by\s+tomorrow|asap)\b', text, re.I)
        if len(set(time_refs)) > 2:
            issues.append(f"🕒 Multiple conflicting urgencies: {set(time_refs)}")
            score += 0.3
        
        # Check for authority contradictions
        authorities = re.findall(r'\b(RBI|Government|Police|Bank|IT\s+Dept|Court)\b', text, re.I)
        if len(authorities) >= 3:
            issues.append(f"🏛️ Too many authorities: {authorities}")
            score += 0.25
        
        # Check for grammatical chaos (scams often have weird structure)
        sentences = re.split(r'[.!?]+', text)
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        if long_sentences:
            issues.append(f"📜 Unusually long/confusing sentences: {len(long_sentences)}")
            score += 0.15
        
        # Check for emotional vs factual imbalance
        emotion_words = len(re.findall(r'\b(urgent|immediately|freeze|arrest|cancel|terminate)\b', text, re.I))
        factual_words = len(re.findall(r'\b(reference|transaction|account|number|date|time)\b', text, re.I))
        if emotion_words > factual_words * 2:
            issues.append(f"😱 Emotional manipulation: {emotion_words} fear words vs {factual_words} facts")
            score += 0.3
        
        return min(score, 1.0), issues

# ============================================================
# MODEL LOADER (Cached)
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
# RISK ORCHESTRATOR (CORE INNOVATION: MULTIPLICATIVE LOGIC)
# ============================================================
@dataclass
class RiskProfile:
    score: float
    level: str
    confidence: float
    triggers: Dict[str, float]
    recos: List[str]
    legitimacy_proof: List[str]
    verifiability_details: List[str]
    coherence_issues: List[str]


# ============================================================
# CORE ORCHESTRATOR
# ============================================================
class CoreOrchestrator:
    def __init__(self, T, thres):
        self.T = T
        self.thres = thres
        self.trust_engine = TrustAnchorEngine()
        self.claims_engine = VerifiableClaimsEngine()
        self.coherence_engine = SemanticCoherenceEngine()

    def infer(self, text: str) -> RiskProfile:
        # -------------------------------
        # MODEL INFERENCE
        # -------------------------------
        tok, mdl, _, _ = load_model()
        inputs = tok(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

        with torch.no_grad():
            logits = mdl(**inputs).logits / self.T
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # -------------------------------
        # STEP 1: Scam signal strength
        # -------------------------------
        detected = probs > self.thres
        scam_signals = probs[detected].mean() if detected.any() else probs.max() * 0.25

        # -------------------------------
        # STEP 2: Legitimacy anchors
        # -------------------------------
        legitimacy_score, legitimacy_proof = self.trust_engine.score(text)

        # -------------------------------
        # STEP 3: Claim extraction & verifiability
        # -------------------------------
        claims = self.claims_engine.extract_claims(text)
        verifiability_score, verif_details = self.claims_engine.score_verifiability(text, claims)

        # -------------------------------
        # STEP 4: Semantic incoherence
        # -------------------------------
        incoherence_score, coherence_issues = self.coherence_engine.score(text)

        # -------------------------------
        # STEP 5: Base multiplicative risk
        # -------------------------------
        risk_multiplier = (
            (1 - legitimacy_score) ** 2
            * (1 - verifiability_score)
            * (1 + incoherence_score * 0.5)
        )
        final_risk = min(scam_signals * risk_multiplier, 1.0)

        # =====================================================
        # FIX 2: Benign-default override
        # =====================================================
        if scam_signals < 0.15 and incoherence_score < 0.15:
            if legitimacy_score > 0.25:
                final_risk = min(final_risk, 0.2)

        # =====================================================
        # FIX 3: No-action → SAFE downgrade
        # =====================================================
        action_claims = any(c.claim_type == "action" for c in claims)
        if not action_claims and final_risk < 0.4:
            final_risk = min(final_risk, 0.2)

        # -------------------------------
        # STEP 6: Risk level assignment
        # -------------------------------
        level_idx = int(np.clip(final_risk / 0.35, 0, 3))
        level = ["SAFE", "CAUTION", "SUSPICIOUS", "SCAM"][level_idx]

        # =====================================================
        # FIX 1: CAUTION subtype (internal only)
        # =====================================================
        caution_type = None
        if level == "CAUTION":
            if legitimacy_score > 0.4:
                caution_type = "VERIFY"
            else:
                caution_type = "AMBIGUOUS"
        # (caution_type intentionally not exposed to UI)

        # -------------------------------
        # STEP 7: Scam trigger explanation
        # -------------------------------
        triggers = {}
        if final_risk > 0.3:
            triggers = {
                label: float(prob)
                for label, prob, is_detected in zip(CP_AFT_LABELS, probs, detected)
                if is_detected
            }

        # -------------------------------
        # STEP 8: Recommendations
        # -------------------------------
        if legitimacy_score > 0.6:
            recos = [
                "✅ Message contains official trust anchors",
                "📞 Verify using OFFICIAL app/website only",
                "🔍 Check reference numbers in official portal"
            ]
        elif final_risk > 0.5:
            recos = [
                "🚨 DO NOT respond directly",
                "📞 Call official number from your card/bank statement",
                "🔒 Enable transaction limits",
                "🗑️ Delete after reporting"
            ]
        else:
            recos = [
                "⏳ Pause before acting",
                "🤔 Ask: 'Can I verify this without replying?'"
            ]

        # -------------------------------
        # FINAL OUTPUT
        # -------------------------------
        return RiskProfile(
            score=round(final_risk * 100, 2),
            level=level,
            confidence=round((1 - np.std(probs)) * 100, 2),
            triggers=triggers,
            recos=recos,
            legitimacy_proof=legitimacy_proof,
            verifiability_details=verif_details,
            coherence_issues=coherence_issues
        )

# ============================================================
# STREAMLIT UI (Enhanced with Proof Display)
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
    .proof-box{background:#f0f4f8;padding:1rem;border-radius:8px;margin:.5rem 0;border-left:4px solid #2D936C;}
    .issue-box{background:#fff5f5;padding:1rem;border-radius:8px;margin:.5rem 0;border-left:4px solid #C1121C;}
    </style>
    <div class="head">
        <h1 style="margin:0;font-size:2.5rem;">🛡️ BharatScam Guardian</h1>
        <p style="margin:.5rem 0 0 0;opacity:.9;"> Protect India: First Verify Legitimacy, Then Assess Risk</p>
        <div style="margin-top:1rem;">
            <span class="badge">🇮🇳 CERT-In Partner</span>
            <span class="badge">🔍 Legitimacy-First</span>
            <span class="badge">🧠 Explainable AI</span>
        </div>
    </div>""", unsafe_allow_html=True)

def input_area():
    st.markdown("### 📨 Paste any message (bank, delivery, job, etc.)")
    st.caption("Our AI first looks for proof of legitimacy, then scam signals.")
    msg = st.text_area("", height=200, placeholder="Paste message here...", label_visibility="collapsed")
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("🔍 Analyze with Legitimacy Check", type="primary", use_container_width=True, key="analyze"):
            if msg.strip():
                st.session_state.msg = msg
                st.session_state.stage = "RUNNING"
                st.rerun()
            else:
                st.error("Please paste a message first.")
    return msg

def hero(p: RiskProfile):
    color = COLORS[p.level]
    st.markdown(f"""
    <div style='background:{color};color:white;padding:2.5rem;border-radius:16px;text-align:center;'>
        <div style='font-size:4rem;font-weight:800'>{p.score}%</div>
        <div style='font-size:1.5rem;font-weight:600;margin:.5rem 0;'>{p.level}</div>
        <div style='opacity:.9'>Confidence: {p.confidence}%</div>
    </div>""", unsafe_allow_html=True)
    
    # Show WHY the score is what it is
    st.markdown("#### 📊 Risk Breakdown")
    st.progress(float(p.score)/100.0)
    
    if p.legitimacy_proof:
        st.markdown("##### ✅ Legitimacy Anchors Detected (Risk Reducer)")
        for proof in p.legitimacy_proof[:3]:
            st.markdown(f"<div class='proof-box'>{proof}</div>", unsafe_allow_html=True)
    
    if p.coherence_issues:
        st.markdown("##### ⚠️ Confusion Tactics Detected (Risk Amplifier)")
        for issue in p.coherence_issues[:2]:
            st.markdown(f"<div class='issue-box'>{issue}</div>", unsafe_allow_html=True)

def triggers(p: RiskProfile):
    if not p.triggers: 
        st.info("🔍 No strong scam patterns detected. Risk may come from lack of legitimacy proof.")
        return
    
    st.markdown("### 🎯 Scam Tactics Detected")
    for trig, prob in sorted(p.triggers.items(), key=lambda x: x[1], reverse=True)[:5]:
        emoji = "🔴" if prob > 0.7 else "🟡"
        st.markdown(f"{emoji} **{trig.replace('_',' ').title()}** — {prob:.1%} match")

def actions(p: RiskProfile):
    st.markdown("### 🎯 Recommended Next Steps")
    for r in p.recos:
        if "OFFICIAL" in r:
            st.success(r)
        elif "Verify" in r:
            st.info(r)
        elif "DO NOT" in r:
            st.error(r)
        elif "Pause" in r:
            st.warning(r)

def verifiability_section(p: RiskProfile):
    if p.verifiability_details:
        with st.expander("🔬 Detailed Claim Analysis"):
            for detail in p.verifiability_details:
                st.markdown(f"- {detail}")

# ============================================================
# MAIN PAGE FLOW
# ============================================================
def main():
    st.set_page_config(page_title="BharatScam Guardian", page_icon="🛡️", layout="centered")
    init_state()
    header()
    input_area()
    
    if st.session_state.stage == "RUNNING":
        with st.spinner(""):
            time.sleep(0.5)  # Brief pause for UX
        
        orch = CoreOrchestrator(*load_model()[2:])
        profile = orch.infer(st.session_state.msg)
        st.session_state.profile = profile
        st.session_state.stage = "DONE"
        st.rerun()
    
    if st.session_state.stage == "DONE" and st.session_state.profile:
        hero(st.session_state.profile)
        triggers(st.session_state.profile)
        actions(st.session_state.profile)
        verifiability_section(st.session_state.profile)
        
        if st.button("🔄 Analyze New Message", key="reset"):
            st.session_state.msg = None
            st.session_state.profile = None
            st.session_state.stage = None
            st.rerun()

if __name__ == "__main__":
    main()
