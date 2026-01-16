# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import torch, torch.nn.functional as F
import numpy as np
import re, time, json, hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import whisper, tempfile, io
from datetime import datetime

# ============================================================
# GLOBAL CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/scam-detection"
LOCAL_DIR = Path("./hf_cpaft_core")
LOCAL_DIR.mkdir(exist_ok=True)

# Theme colors (same as original)
THEME = {
    "bg": "#FDFBF8", "card": "#FFFFFF", "accent": "#FF8F00",
    "safe": "#2E7D32", "caution": "#F57C00", "suspicious": "#D32F2F", "scam": "#B71C1C",
    "text": "#3E2723", "subtle": "#8D6E63"
}

# ============================================================
# ANTI-GAMING & CREDIBILITY ENGINE
# ============================================================
class AntiGamingEngine:
    """Detects if user is testing the system with nonsense/gaming attempts"""
    
    def __init__(self):
        self.test_patterns = [
            r'^(test|asdf|qwerty|hello|hi|random|blah|foo|bar|spam|scam|message|abc|xyz|123)',
            r'(.)\1{4,}',  # repeated characters
            r'\b(lorem ipsum|dummy text|sample message)\b',
            r'^[a-z]{1,3}$',  # too short
        ]
        self.suspicious_senders = set()
        self.message_timestamps = []
        
    def check_gaming(self, text: str) -> Tuple[bool, str, float]:
        """
        Returns: (is_gaming, reason, penalty_score)
        """
        text_lower = text.lower().strip()
        
        # Check for gibberish
        if len(text) < 10:
            return True, "Message too short to analyze meaningfully", 0.9
        
        # Check test patterns
        for pattern in self.test_patterns:
            if re.search(pattern, text_lower, re.I):
                return True, "Detected as test/gaming attempt", 0.85
        
        # Check randomness (too many unique characters vs length)
        char_ratio = len(set(text_lower)) / len(text_lower) if text_lower else 1
        if char_ratio > 0.8 and len(text) < 50:
            return True, "Appears to be random characters", 0.75
        
        # Check for repeated submission patterns (same message hash)
        msg_hash = hashlib.md5(text_lower.encode()).hexdigest()
        if "message_history" not in st.session_state:
            st.session_state.message_history = set()
        
        if msg_hash in st.session_state.message_history:
            return True, "Repeated message detected", 0.6
        
        st.session_state.message_history.add(msg_hash)
        
        # Check submission frequency
        now = datetime.now()
        if "submission_times" not in st.session_state:
            st.session_state.submission_times = []
        
        st.session_state.submission_times = [t for t in st.session_state.submission_times 
                                           if (now - t).total_seconds() < 300]  # 5 minutes
        st.session_state.submission_times.append(now)
        
        if len(st.session_state.submission_times) > 5:
            return True, "Too many submissions in short time", 0.5
        
        return False, "", 0.0

# ============================================================
# ENHANCED PATTERN DETECTION
# ============================================================
class TrustAnchorEngine:
    """Score messages based on official trust anchors with sophistication detection"""
    
    def __init__(self):
        self.legitimate_patterns = {
            "bank_official": (r'\b(?:HDFC|ICICI|SBI|AXIS|KOTAK|BOB|PNB)[\s]*(?:Bank|Ltd|Limited)\b|\bRBI\b|\bNPCI\b|\bIRDAI\b', 0.35),
            "govt_official": (r'\b(?:UIDAI|ITA|GST|EPFO|CBDT|MCA|CEIR)\b|\b(?:gov\.in|nic\.in|ac\.in)\b', 0.35),
            "verifiable_ref": (r'\b(?:UTR|Ref|Reference|Txn|Transaction)[\s]*[No|ID|Number]*[:#]?\s*[A-Z0-9]{8,20}\b', 0.3),
            "official_contact": (r'\b(?:1800|1860)[\s]*-?\d{3}[\s]*-?\d{4}\b|\b(?:91|0)?\s*\d{8}\b', 0.25),
            "secure_url": (r'\bhttps?://(?:www\.)?(?:hdfcbank\.com|icicibank\.com|sbi\.co\.in|axisbank\.com|paytm\.com|amazon\.in|flipkart\.com)[/\w.-]*\b', 0.35),
            "official_email": (r'\b[A-Za-z0-9._%+-]+@(?:hdfcbank\.com|icicibank\.com|sbi\.co\.in|axisbank\.com|rbi\.org\.in)\b', 0.4)
        }
    
    def score(self, text: str) -> Tuple[float, List[str], Dict[str, List[str]]]:
        score, hits = 0.0, []
        detailed_matches = defaultdict(list)
        
        for name, (pat, weight) in self.legitimate_patterns.items():
            matches = re.findall(pat, text, re.I)
            if matches:
                unique_matches = set(matches)
                detailed_matches[name].extend(unique_matches)
                hits.append(f"{name.replace('_',' ').title()}: {len(unique_matches)} found")
                score += min(len(unique_matches) * weight, weight)
        
        return min(score, 1.0), hits, dict(detailed_matches)

class VerifiableClaimsEngine:
    """Decompose text into claims with verifiability scoring"""
    
    def extract_claims(self, text: str) -> List[Tuple[str, str, float, str]]:
        claims = []
        
        # Financial claims
        for m in re.finditer(r'\b(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{2})?|\b\d{6,}\b', text):
            amount = m.group()
            context = text[max(0, m.start()-50):min(len(text), m.end()+50)]
            claims.append((amount, "financial", 0.8, context))
        
        # Temporal claims
        for m in re.finditer(r'\b(?:immediately|now|within\s+\d+\s+(?:hour|day|min)s?|today|tomorrow|yesterday|asap|urgent)\b', text, re.I):
            urgency = m.group()
            context = text[max(0, m.start()-50):min(len(text), m.end()+50)]
            claims.append((urgency, "temporal", 0.3, context))
        
        # Identity claims
        for m in re.finditer(r'\b(?:RBI|NPCI|UIDAI|IT Department|HDFC Bank|ICICI Bank|SBI|AXIS Bank|Government|Police|CIBIL|TRAI|DOT)\b', text):
            entity = m.group()
            context = text[max(0, m.start()-50):min(len(text), m.end()+50)]
            verifiability = 0.8 if entity in ["RBI", "NPCI", "UIDAI", "IT Department"] else 0.4
            claims.append((entity, "identity", verifiability, context))
        
        # Action claims
        for m in re.finditer(r'\b(?:click|pay|transfer|send|share|update|verify|download|install)\s+(?:here|link|amount|money|details|OTP|UPI|account|app|apk)\b', text, re.I):
            action = m.group()
            context = text[max(0, m.start()-50):min(len(text), m.end()+50)]
            verifiability = 0.7 if any(w in context.lower() for w in ['official', 'bank', 'portal']) else 0.2
            claims.append((action, "action", verifiability, context))
        
        return claims
    
    def score_verifiability(self, claims: List[Tuple[str, str, float, str]]) -> Tuple[float, List[Dict], float]:
        if not claims:
            return 0.0, [], 0.0
        
        total_score = sum(c[2] for c in claims)
        avg_score = total_score / len(claims)
        
        high_verif = [c for c in claims if c[2] >= 0.7]
        low_verif = [c for c in claims if c[2] < 0.4]
        
        details = []
        for claim, ctype, score, context in claims:
            details.append({
                "claim": claim,
                "type": ctype,
                "score": score,
                "context": context[:100] + "..." if len(context) > 100 else context,
                "verifiable": score >= 0.6
            })
        
        # Red flag ratio - too many unverifiable claims
        red_flag_ratio = len(low_verif) / len(claims) if claims else 0
        
        return avg_score, details, red_flag_ratio

class SemanticCoherenceEngine:
    """Detects confusion tactics and emotional manipulation"""
    
    def score(self, text: str) -> Tuple[float, List[str], Dict[str, float]]:
        issues = []
        sub_scores = {"urgency": 0.0, "authority": 0.0, "emotion": 0.0, "complexity": 0.0}
        
        # Urgency analysis
        urgency_words = re.findall(r'\b(immediately|now|within\s+\d+|asap|by\s+\d+|urgent|hurry|act\s+fast)\b', text, re.I)
        if len(urgency_words) > 2:
            sub_scores["urgency"] = min(len(urgency_words) * 0.15, 0.4)
            issues.append(f"⏰ Repeated urgency: {set(urgency_words)}")
        
        # Authority mixing
        authorities = re.findall(r'\b(RBI|Government|Police|Bank|IT Dept|CIBIL|Court|Supreme Court|ED|CBI)\b', text)
        if len(set(authorities)) >= 3:
            sub_scores["authority"] = 0.3
            issues.append(f"🏛️ Conflicting authorities: {set(authorities)}")
        
        # Sentence complexity (confusion tactic)
        sentences = re.split(r'[.!?]', text)
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        if len(long_sentences) > 0:
            sub_scores["complexity"] = min(len(long_sentences) * 0.1, 0.2)
            issues.append(f"📜 {len(long_sentences)} confusingly long sentences")
        
        # Emotion vs facts imbalance
        emotion_words = re.findall(r'\b(urgent|immediately|freeze|arrest|cancel|terminate|suspend|block|deactivate|fraud|illegal)\b', text, re.I)
        factual_words = re.findall(r'\b(reference|transaction|account|number|date|time|amount|ID|UTR)\b', text, re.I)
        
        emotion_score = len(emotion_words)
        factual_score = max(len(factual_words), 1)
        
        if emotion_score > factual_score * 2:
            sub_scores["emotion"] = min((emotion_score / factual_score) * 0.15, 0.35)
            issues.append(f"😱 Emotion manipulation: {emotion_score} emotional vs {factual_score} factual words")
        
        total_score = sum(sub_scores.values())
        return min(total_score, 1.0), issues, sub_scores

class TriggerSynergyAnalyzer:
    """Analyzes how triggers work together to create scam narratives"""
    
    def __init__(self):
        self.cp_aft_labels = ["AUTHORITY","URGENCY","FEAR","GREED","SOCIAL_PROOF","SCARCITY","OBEDIENCE","TRUST"]
        self.synergy_patterns = {
            "impersonation_rbi": {"triggers": ["AUTHORITY", "URGENCY", "FEAR"], "threshold": 0.6, "narrative": "Classic RBI/Bank impersonation: Pretends to be authority, creates urgency, threatens consequences"},
            "lottery_scam": {"triggers": ["GREED", "SOCIAL_PROOF", "SCARCITY"], "threshold": 0.5, "narrative": "Lottery/prize scam: Promises huge money, shows fake winners, creates scarcity"},
            "digital_arrest": {"triggers": ["AUTHORITY", "FEAR", "URGENCY", "OBEDIENCE"], "threshold": 0.7, "narrative": "Digital arrest scam: Impersonates police/RBI, threatens arrest, demands immediate compliance"},
            "fake_investment": {"triggers": ["GREED", "TRUST", "SOCIAL_PROOF"], "threshold": 0.6, "narrative": "Fake investment: Promises high returns, shows fake testimonials, builds trust"},
            "urgent_kyc": {"triggers": ["AUTHORITY", "URGENCY", "TRUST"], "threshold": 0.5, "narrative": "Urgent KYC scam: Claims to be bank, threatens account suspension, asks for details"},
        }
    
    def analyze_synergies(self, triggers: Dict[str, float], text: str) -> List[Dict[str, any]]:
        """
        Returns list of detected scam patterns with narratives
        """
        detected_patterns = []
        
        for pattern_name, config in self.synergy_patterns.items():
            required_triggers = config["triggers"]
            trigger_scores = [triggers.get(t, 0) for t in required_triggers]
            
            if all(score > 0.3 for score in trigger_scores):  # All triggers present
                avg_score = sum(trigger_scores) / len(trigger_scores)
                if avg_score >= config["threshold"]:
                    detected_patterns.append({
                        "name": pattern_name.replace("_", " ").title(),
                        "score": round(avg_score, 2),
                        "narrative": config["narrative"],
                        "primary_triggers": required_triggers,
                        "severity": "HIGH" if avg_score > 0.7 else "MEDIUM"
                    })
        
        return detected_patterns

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

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

# ============================================================
# ENHANCED CORE ORCHESTRATOR
# ============================================================
class CoreOrchestrator:
    def __init__(self,T,thres):
        self.T, self.thres = T, thres
        self.trust = TrustAnchorEngine()
        self.claims = VerifiableClaimsEngine()
        self.coherence = SemanticCoherenceEngine()
        self.synergy = TriggerSynergyAnalyzer()
        self.antigaming = AntiGamingEngine()
    
    def calculate_risk_score(self, probs: np.ndarray, leg_score: float, 
                           ver_score: float, incoh_score: float, 
                           red_flag_ratio: float, gaming_penalty: float) -> float:
        """
        Multi-dimensional risk calculation specific to Indian scams
        """
        # Base scam signal from model
        detected = probs > self.thres
        scam_signals = probs[detected].mean() if detected.any() else probs.max() * 0.25
        
        # Penalize heavily if gaming detected
        if gaming_penalty > 0:
            return min(scam_signals * (1 + gaming_penalty), 1.0)
        
        # Legitimacy anchors provide strong protection
        legitimacy_factor = (1 - leg_score) ** 1.5  # Non-linear penalty
        
        # Verifiability is critical - low verifiability = high risk
        verifiability_factor = (1 - ver_score) ** 1.2
        
        # Coherence issues amplify risk
        coherence_factor = 1 + (incoh_score * 0.6)
        
        # Red flag ratio (many unverifiable claims)
        red_flag_factor = 1 + (red_flag_ratio * 0.8)
        
        # Weight triggers based on Indian scam prevalence
        trigger_weights = {
            "AUTHORITY": 1.3, "URGENCY": 1.2, "FEAR": 1.4, "GREED": 1.1,
            "SOCIAL_PROOF": 0.9, "SCARCITY": 1.0, "OBEDIENCE": 1.3, "TRUST": 1.2
        }
        
        weighted_triggers = sum(probs[i] * trigger_weights.get(label, 1.0) 
                               for i, label in enumerate(self.synergy.cp_aft_labels))
        
        # Combine all factors
        risk = (scam_signals * weighted_triggers * 
                legitimacy_factor * verifiability_factor * 
                coherence_factor * red_flag_factor)
        
        # Adaptive thresholding based on context
        base_risk = scam_signals * 0.7  # Base model confidence
        contextual_risk = risk * 0.3    # Contextual factors
        
        return min(base_risk + contextual_risk, 1.0)
    
    def generate_narrative(self, text: str, triggers: Dict[str, float], 
                          synergy_patterns: List[Dict], leg_matches: Dict[str, List[str]],
                          coherence_issues: List[str]) -> str:
        """
        Generate human-readable explanation of why message is risky
        """
        if not triggers:
            return "This message appears legitimate. No scam patterns detected."
        
        story_parts = []
        
        # Start with dominant pattern if found
        if synergy_patterns:
            dominant = max(synergy_patterns, key=lambda x: x["score"])
            story_parts.append(f"**Primary Pattern:** {dominant['narrative']}")
        else:
            # Fallback to individual triggers
            top_triggers = sorted(triggers.items(), key=lambda x: x[1], reverse=True)[:2]
            story_parts.append(f"**Key Concerns:** High {top_triggers[0][0].replace('_', ' ').lower()} ({top_triggers[0][1]:.0%})")
            if len(top_triggers) > 1:
                story_parts.append(f"and {top_triggers[1][0].replace('_', ' ').lower()} ({top_triggers[1][1]:.0%}) signals")
        
        # Add legitimacy context
        if leg_matches:
            story_parts.append(f"\n**Legitimacy Check:** Found {len(leg_matches)} official references, but context matters.")
        else:
            story_parts.append(f"\n**Legitimacy Check:** No official bank/government identifiers found - be cautious.")
        
        # Add coherence issues
        if coherence_issues:
            story_parts.append(f"\n**Language Analysis:** {len(coherence_issues)} manipulation tactics detected.")
        
        return " ".join(story_parts)
    
    def build_specific_actions(self, synergy_patterns: List[Dict], triggers: Dict[str, float],
                              leg_score: float, incoh_score: float, text: str) -> List[str]:
        """
        Build message-specific actions based on detected patterns
        """
        actions = []
        
        # Pattern-specific actions
        for pattern in synergy_patterns:
            if pattern["name"] == "Impersonation Rbi":
                actions.extend([
                    "🚨 **DO NOT CALL** any number in this message - it's fake",
                    "📞 **CALL YOUR BANK DIRECTLY** using number on your passbook/card (NOT this message)",
                    "🏛️ **RBI NEVER** contacts citizens via WhatsApp/SMS for personal matters"
                ])
            elif pattern["name"] == "Lottery Scam":
                actions.extend([
                    "💸 **NOBODY GIVES FREE MONEY** - especially not via random messages",
                    "🚫 **NEVER PAY** 'processing fee' or 'tax' to claim prizes",
                    "📢 **REAL LOTTERIES** don't notify winners via WhatsApp"
                ])
            elif pattern["name"] == "Digital Arrest":
                actions.extend([
                    "⚖️ **POLICE NEVER** arrest via video call or demand money to 'settle'",
                    "🚔 **CALL 100** or visit local police station if threatened",
                    "💰 **NEVER TRANSFER** money to 'avoid arrest' - it's always a scam"
                ])
            elif pattern["name"] == "Fake Investment":
                actions.extend([
                    "📊 **VERIFY SEBI REGISTRATION** of any investment scheme",
                    "⏳ **IF TOO GOOD TO BE TRUE**, it is. High returns = high risk",
                    "👥 **TESTIMONIALS CAN BE FAKED** - check independent reviews"
                ])
            elif pattern["name"] == "Urgent Kyc":
                actions.extend([
                    "🔐 **BANKS NEVER ASK** for KYC via links in messages",
                    "📱 **USE OFFICIAL APP ONLY** - type URL yourself, don't click links",
                    "🛡️ **YOUR ACCOUNT WON'T BE BLOCKED** for not clicking a link"
                ])
        
        # If no specific pattern, use trigger-based actions
        if not actions:
            if "FEAR" in triggers and triggers["FEAR"] > 0.6:
                actions.append("😰 **SCARE TACTICS DETECTED** - Real authorities don't threaten via SMS")
            if "GREED" in triggers and triggers["GREED"] > 0.6:
                actions.append("💰 **TOO GOOD TO BE TRUE** - Verify independently before believing")
            if "URGENCY" in triggers and triggers["URGENCY"] > 0.6:
                actions.append("⏱️ **URGENCY IS A RED FLAG** - Scammers rush you. Take 10 minutes to think")
            if "AUTHORITY" in triggers and triggers["AUTHORITY"] > 0.6:
                actions.append("🏛️ **VERIFY AUTHORITY** - Call official numbers, don't use message details")
            
            # Generic safe actions if no high triggers
            if not actions:
                if leg_score > 0.6:
                    actions.extend([
                        "✅ **Looks relatively safe**, but verify in official app",
                        "🔍 **Double-check sender** - spoofing is common"
                    ])
                else:
                    actions.append("🟡 **Use common sense** - When in doubt, delete")
        
        # Always add these universal actions
        actions.extend([
            "📞 **Customer Care**: Use number on your bank statement/card",
            "🗑️ **When in doubt**: Delete and block sender",
            "📢 **Report**: Forward to 1930 (National Cyber Crime Helpline)"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for a in actions:
            action_key = a.split("**")[1].strip() if "**" in a else a
            if action_key not in seen:
                seen.add(action_key)
                unique_actions.append(a)
        
        return unique_actions
    
    def infer(self, text: str) -> Dict:
        # Anti-gaming check
        is_gaming, reason, penalty = self.antigaming.check_gaming(text)
        if is_gaming:
            return {
                "risk_score": round(penalty * 100, 2),
                "level": "SCAM",
                "confidence": 95.0,
                "narrative": f"System detected gaming attempt: {reason}",
                "triggers": {"ANTI-GAMING": penalty},
                "synergy_patterns": [],
                "claim_analysis": [],
                "legitimacy_proof": [],
                "coherence_issues": [reason],
                "recos": ["🎮 This appears to be a test. Please use real messages for meaningful analysis."],
                "is_gaming": True
            }
        
        # Load model
        tok, mdl, _, _ = load_model()
        text = text.strip()
        
        # Get model predictions
        inputs = tok(text,return_tensors="pt",truncation=True,padding=True).to(DEVICE)
        with torch.no_grad():
            logits = mdl(**inputs).logits/self.T
            probs  = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Analyze components
        leg_score, leg_proof, leg_matches = self.trust.score(text)
        claims_list = self.claims.extract_claims(text)
        ver_score, claim_details, red_flag_ratio = self.claims.score_verifiability(claims_list)
        incoh_score, incoh_issues, incoh_subscores = self.coherence.score(text)
        
        # Detect triggers
        detected = probs > self.thres
        triggers = {label: float(p) for label, p, det in zip(self.synergy.cp_aft_labels, probs, detected) if det}
        
        # Analyze synergies
        synergy_patterns = self.synergy.analyze_synergies(triggers, text)
        
        # Calculate risk
        risk = self.calculate_risk_score(probs, leg_score, ver_score, incoh_score, red_flag_ratio, 0.0)
        
        # Determine level with adaptive thresholds
        base_thresh = np.array([0.2, 0.45, 0.7])  # More stringent
        adaptive_thresh = base_thresh * (1 - leg_score) * (1 - 0.3 * ver_score) + 0.15 * incoh_score
        
        if risk < adaptive_thresh[0]:
            level = "SAFE"
        elif risk < adaptive_thresh[1]:
            level = "CAUTION"
        elif risk < adaptive_thresh[2]:
            level = "SUSPICIOUS"
        else:
            level = "SCAM"
        
        # Confidence based on model certainty and context clarity
        conf = (1 - np.std(probs)) * 100 * (1 - incoh_score * 0.3)
        
        # Generate narrative
        narrative = self.generate_narrative(text, triggers, synergy_patterns, leg_matches, incoh_issues)
        
        # Build specific actions
        recos = self.build_specific_actions(synergy_patterns, triggers, leg_score, incoh_score, text)
        
        return {
            "risk_score": round(float(risk * 100), 2),
            "level": level,
            "confidence": round(float(conf), 2),
            "narrative": narrative,
            "triggers": triggers,
            "synergy_patterns": synergy_patterns,
            "claim_analysis": claim_details,
            "legitimacy_proof": leg_proof,
            "coherence_issues": incoh_issues,
            "recos": recos,
            "is_gaming": False
        }

# ============================================================
# UI HELPERS
# ============================================================
def init_state():
    for k in ["msg","result","stage","mode","message_history","submission_times"]:
        if k not in st.session_state:
            st.session_state[k] = [] if k in ["message_history","submission_times"] else None

def risk_badge(level: str) -> str:
    color = THEME.get(level.lower(), THEME["subtle"])
    return f'<span style="background:{color}22;color:{color};padding:8px 18px;border-radius:999px;font-weight:600;">{level}</span>'

def draw_risk_score(result: Dict):
    color = THEME.get(result["level"].lower(), THEME["subtle"])
    narrative_preview = result["narrative"][:150] + "..." if len(result["narrative"]) > 150 else result["narrative"]
    
    st.markdown(f"""
    <div style="text-align:center;background:#fff;border-radius:20px;padding:35px;margin-bottom:25px;
                box-shadow: 0 4px 18px rgba(0,0,0,.07);">
        <div style="font-size:22px;color:{THEME["subtle"]};margin-bottom:8px;">Risk Assessment</div>
        <div style="font-size:72px;font-weight:700;color:{color};line-height:1">{result["risk_score"]}<span style="font-size:36px">%</span></div>
        <div style="margin-top:15px;">{risk_badge(result["level"])}</div>
        <div class="subtle" style="margin-top:12px;">Confidence {result["confidence"]}%</div>
        <div style="margin-top:20px;padding-top:20px;border-top:1px solid #eee;font-size:15px;color:{THEME["text"]};">
            {narrative_preview}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(float(result["risk_score"])/100.0)

def draw_synergy_patterns(patterns: List[Dict]):
    if not patterns:
        st.success("✅ No coordinated scam patterns detected")
        return
    
    st.markdown('<div style="font-size:24px;font-weight:600;margin:25px 0 15px 0;">🎯 Scam Pattern Analysis</div>', unsafe_allow_html=True)
    
    for pattern in patterns:
        severity_color = "#D32F2F" if pattern["severity"] == "HIGH" else "#F57C00"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#fff8e1 0%,#ffecb3 100%);border-left:5px solid {severity_color};
                    border-radius:16px;padding:18px 22px;margin-bottom:18px;box-shadow:0 2px 8px rgba(0,0,0,.07);">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <span style="font-size:20px;font-weight:600;color:{THEME["text"]};">{pattern["name"]}</span>
                <span style="background:{severity_color}22;color:{severity_color};padding:5px 12px;border-radius:999px;font-weight:600;">
                    {pattern["severity"]}
                </span>
            </div>
            <div style="color:{THEME["text"]};font-size:16px;line-height:1.5;">
                {pattern["narrative"]}
            </div>
            <div style="margin-top:12px;font-size:14px;color:{THEME["subtle"]};">
                <b>Primary Triggers:</b> {', '.join(pattern["primary_triggers"])}
            </div>
        </div>
        """, unsafe_allow_html=True)

def draw_claim_cards(claims: List[Dict]):
    if not claims:
        st.info("No specific claims found in the message.")
        return
    
    st.markdown('<div style="font-size:24px;font-weight:600;margin:25px 0 15px 0;">🔬 Claim Analysis</div>', unsafe_allow_html=True)
    
    type_colors = {
        "financial": ("💰", "#2E7D32", "High verifiability"),
        "temporal": ("⏰", "#F57C00", "Low verifiability"),
        "identity": ("🏛️", "#1976D2", "Context-dependent"),
        "action": ("✅", "#388E3C", "Action needed")
    }
    
    for claim in claims:
        icon, color, status = type_colors.get(claim["type"], ("•", THEME["subtle"], "Unknown"))
        verif_status = "✓ Verifiable" if claim["verifiable"] else "⚠ Unverifiable"
        
        st.markdown(f"""
        <div style="background:{color}11;border-left:5px solid {color};border-radius:14px;padding:16px 20px;margin-bottom:14px;">
            <div style="display:flex;align-items:center;margin-bottom:8px;">
                <span style="font-size:24px;margin-right:10px;">{icon}</span>
                <span style="font-weight:600;font-size:18px;color:{THEME["text"]};">{claim["claim"]}</span>
                <span style="margin-left:auto;background:{color}22;color:{color};padding:4px 10px;border-radius:999px;font-size:13px;font-weight:600;">
                    {verif_status}
                </span>
            </div>
            <div style="color:{THEME["subtle"]};font-size:14px;margin-bottom:8px;">
                <b>Type:</b> {claim["type"].title()} | <b>Verifiability:</b> {claim["score"]:.0%}
            </div>
            <div style="color:{THEME["text"]};font-size:15px;background:#fff;padding:8px 12px;border-radius:8px;border-left:3px solid {color};">
                Context: "{claim["context"]}"
            </div>
        </div>
        """, unsafe_allow_html=True)

def draw_section(title: str, items: List[str], icon: str, is_warning: bool = False):
    if not items:
        return
    
    color = "#D32F2F" if is_warning else THEME["safe"]
    st.markdown(f'<div style="font-size:24px;font-weight:600;margin:25px 0 15px 0;">{icon} {title}</div>', unsafe_allow_html=True)
    
    for item in items:
        st.markdown(f"""
        <div style="background:{color}11;border-left:5px solid {color};border-radius:8px;padding:12px 16px;margin-bottom:10px;">
            {item}
        </div>
        """, unsafe_allow_html=True)

def draw_actions(recos: List[str]):
    if not recos:
        return
    
    st.markdown('<div style="font-size:24px;font-weight:600;margin:25px 0 15px 0;">💡 What You Should Do</div>', unsafe_allow_html=True)
    
    for i, reco in enumerate(recos[:6], 1):  # Show max 6 actions
        st.markdown(f"""
        <div style="background:#FFFFFF;border:2px solid {THEME["accent"]}33;border-radius:14px;padding:16px 20px;margin-bottom:14px;
                    box-shadow: 0 2px 6px rgba(0,0,0,.05);font-size:17px;display:flex;align-items:center;">
            <span style="background:{THEME["accent"]};color:#fff;width:28px;height:28px;border-radius:50%;display:inline-flex;
                         align-items:center;justify-content:center;font-weight:600;margin-right:12px;font-size:14px;">{i}</span>
            <span>{reco}</span>
        </div>
        """, unsafe_allow_html=True)

def draw_narrative(narrative: str):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{THEME["accent"]}11 0%,{THEME["caution"]}11 100%);border-radius:16px;padding:20px 24px;
                margin:25px 0;border:1px solid {THEME["accent"]}33;">
        <div style="font-size:18px;font-weight:600;margin-bottom:12px;color:{THEME["text"]};">🧠 Why This Analysis?</div>
        <div style="color:{THEME["text"]};line-height:1.6;font-size:15px;">
            {narrative}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    st.set_page_config(page_title="BharatScam Guardian", page_icon="🛡️", layout="centered")
    
    # CSS
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
    
    init_state()
    
    # Hero
    st.markdown(f"""
        <div style="text-align:center;margin-top:-40px;margin-bottom:40px;">
        <h1 style="font-size:52px;background:-webkit-linear-gradient(45deg,{THEME["accent"]},#FF6F00);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">BharatScam Guardian</h1>
        <p class="subtle">AI that understands Indian scam patterns 🛡️</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Mode toggle state
    if "mode" not in st.session_state:
        st.session_state.mode = False  # False = Text, True = Speech
    
    # Mode switcher
    col_mode1, col_mode2, col_mode3 = st.columns([1,2,1])
    with col_mode2:
        mode_text = "🎤 Voice Mode" if st.session_state.mode else "⌨️ Text Mode"
        if st.button(mode_text, use_container_width=True, key="mode_toggle"):
            st.session_state.mode = not st.session_state.mode
            st.rerun()
    
    # Input handling
    msg = None
    if st.session_state.mode:  # Speech mode
        st.markdown("""
        <div style="background:linear-gradient(135deg,#fff8e1 0%,#ffecb3 100%);border-left:5px solid #ff8f00;
                    border-radius:12px;padding:14px 18px;font-size:17px;color:#3e2723;margin-bottom:15px;">
        👂 <b>I'm listening...</b> Record your message below.
        </div>
        """, unsafe_allow_html=True)
        
        audio_bytes = st.audio_input("Record", key="mic")
        if audio_bytes is not None:
            audio_raw = audio_bytes.read()
            audio_hash = hashlib.md5(audio_raw).hexdigest()
            
            if st.session_state.get("_last_audio_hash") != audio_hash:
                st.session_state._last_audio_hash = audio_hash
                with st.spinner("🧠 Transcribing your voice..."):
                    model = load_whisper()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_raw)
                        tmp_path = tmp.name
                    
                    try:
                        result = model.transcribe(tmp_path, fp16=False)
                        text = result.get("text", "").strip()
                        st.session_state.msg = text
                        st.rerun()
                    finally:
                        import os
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
    else:  # Text mode
        st.markdown("""
        <div style="background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);border-left:5px solid #4CAF50;
                    border-radius:12px;padding:14px 18px;font-size:17px;color:#3e2723;margin-bottom:15px;">
        💬 <b>Paste your message</b> - Hindi, English, or Hinglish supported
        </div>
        """, unsafe_allow_html=True)
        
        msg = st.text_area("", key="msg_input", placeholder="Paste suspicious message here...", height=180, 
                          label_visibility="collapsed")
    
    # Use either voice or text input
    final_msg = st.session_state.get("msg", "") if st.session_state.mode else msg
    
    # Analyze button
    col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
    with col_btn2:
        if st.button("🛡️ Analyze Message", use_container_width=True, key="analyze", 
                    type="primary", disabled=not final_msg):
            if final_msg and len(final_msg.strip()) > 5:
                st.session_state.stage = "RUNNING"
                st.session_state.msg_cache = final_msg
                st.rerun()
    
    # Processing
    if st.session_state.stage == "RUNNING":
        with st.container():
            st.markdown('<div class="card"><h4>🔍 Analyzing message patterns...</h4>', unsafe_allow_html=True)
            bar = st.progress(0)
            for i in range(100):
                bar.progress(i+1)
                time.sleep(0.008)
            
            orch = CoreOrchestrator(*load_model()[2:])
            st.session_state.result = orch.infer(st.session_state.msg_cache)
            st.session_state.stage = "DONE"
            st.markdown('</div>', unsafe_allow_html=True)
            st.rerun()
    
    # Results
    if st.session_state.stage == "DONE" and st.session_state.result:
        result = st.session_state.result
        
        # Risk score and badge
        draw_risk_score(result)
        
        # Narrative explanation
        draw_narrative(result["narrative"])
        
        # Synergy patterns (if any)
        draw_synergy_patterns(result["synergy_patterns"])
        
        # Individual triggers (compact view)
        if result["triggers"]:
            st.markdown('<div style="font-size:20px;font-weight:600;margin:20px 0 12px 0;">📊 Trigger Signals</div>', unsafe_allow_html=True)
            trigger_cols = st.columns(len(result["triggers"]))
            for i, (trigger, score) in enumerate(result["triggers"].items()):
                with trigger_cols[i]:
                    st.markdown(f"""
                    <div style="background:{THEME["accent"]}22;border-radius:12px;padding:12px;text-align:center;">
                        <div style="font-size:18px;font-weight:600;">{trigger.replace('_',' ').title()}</div>
                        <div style="font-size:24px;color:{THEME["accent"]};">{score:.0%}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Claims analysis
        draw_claim_cards(result["claim_analysis"])
        
        # Legitimacy and issues
        draw_section("✅ Legitimacy Anchors", result["legitimacy_proof"], "✅")
        draw_section("⚠️ Coherence Issues", result["coherence_issues"], "⚠️", is_warning=True)
        
        # Actions
        draw_actions(result["recos"])
        
        # Reset
        if st.button("🔄 Analyze Another Message", use_container_width=True, key="reset"):
            for key in ["msg", "result", "stage", "msg_cache", "msg_input"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Footer
    st.markdown("""
        <div style="text-align:center;margin-top:40px;padding:16px 0;color:#8D6E63;font-size:14px;">
            Built for Indian users 🇮🇳 | 
            <a href="mailto:prakhar.mathur2020@gmail.com" style="color:#FF8F00;text-decoration:none;font-weight:600;">
                Contact Developer
            </a>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
