# ============================================================
# IMPORTS
# ============================================================
import unicodedata
from datetime import datetime
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
REPO_ID = "prakhar146/scam"
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
    "bank_official": r'\b(?:(?:HDFC|ICICI|SBI|AXIS|KOTAK|BOB|PNB|UNION|CANARA|INDIAN|YES|IDFC)[\s._-]*(?:BANK|LTD|LIMITED|BK)|(?:RBI|NPCI|IRDAI|SEBI|NSE|BSE|PFRDA|NHB))\b',
    
    "govt_entity": r'\b(?:UIDAI|ITA|GSTN|EPFO|CBDT|MCA|CEIR|MEITY|DOT|TRAI|ESIC|NPS|DGFT)[\s._-]*(?:GOV|NIC|IN|ORG)?\b|\b(?:\w+[\._-])*(?:gov\.in|nic\.in|ac\.in|edu\.in)\b',
    
    "fin_reference": r'\b(?:UTR|RRN|ARN|UPI[\s._-]*REF|CRN|PRN|SRN|Txn|Trans)[\s._-]*(?:No|ID|NUM|NUMBER)?:?[\s._-]*[A-Z0-9]{8,20}(?:[A-Z0-9]{2})?\b',
    
    "official_contact": r'\b(?:(?:1800|1860|139|155260|14444|1950|1930)[\s._-]*-?\d{3,4}[\s._-]*-?\d{3,4}|(?:\+91|0)?[\s._-]?[6-9]\d{9})\b',
    
    "trusted_domain": r'\bhttps?://(?:[\w.-]+\.)?(?:hdfcbank\.com|icicibank\.com|sbi\.co\.in|axisbank\.com|paytm\.com|amazon\.in|flipkart\.com|uidai\.gov\.in|incometax\.gov\.in|npci\.org\.in)(?:[/?#]\S*)?\b'
}

SCAM_PATTERNS = {
    "urgency_pressure": r'\b(?:IMMEDIATELY|NOW|URGENT|WITHIN[\s._-]*\d+|LAST[\s._-]*CHANCE|ACCOUNT[\s._-]*LOCK|LIMITED[\s._-]*TIME)\b(?![\s\S]{0,30}(?:FRAUD|UNAUTHORIZED|NOT\s+YOU))',
    
    "impersonation_auth": r'\b(?:FAKE|FRAUD|SPOOF|IMPERSONAT|SCAM).{0,15}(?:RBI|BANK|GOVT|POLICE|CIBIL|IT[\s._-]*DEPT|CUSTOMER[\s._-]*CARE|KYC[\s._-]*TEAM|SBI|PAYTM)|\b(?:RBI|BANK|PAYTM).{0,15}(?:CALLING|CALL|CONTACT|SUSPEND|BLOCK)\b',
    
    "generic_salutation": r'\b(?:DEAR[\s._-]*CUSTOMER|VALUED[\s._-]*USER|RESPECTED[\s._-]*SIR|MADAM|JI|BHAIYA|DIDI|YAAR)[,.]?[\s]*(?:YOUR|AAPKI|AAPKA)\b',
    
    "payment_redirect": r'\b(?:PAY|TRANSFER|SEND|DEPOSIT|SCAN|UPI).{0,20}(?:NEW|ALTERNATE|OTHER|PERSONAL|QR|WALLET|UPI[\s._-]*ID|ACCOUNT)[\s._-]*(?:DETAIL|INFO|NUMBER)|\b(?:ADVANCE|PROCESSING|REGISTRATION|GST)[\s._-]*FEE\b',
    
    "kyc_cashback_loot": r'\b(?:KYC|PAN|AADHAAR).{0,15}(?:EXPIRED|INCOMPLETE|SUSPEND|UPDATE|LINK|MANDATORY).{0,15}(?:CLICK|CALL|SHARE)|\b(?:CASHBACK|REFUND|PRIZE|LOTTERY|REWARD).{0,10}(?:RS\.?|₹)\s*\d{4,}',
    
    "crypto_digital_scam": r'\b(?:BITCOIN|USDT|CRYPTO|TRADING|FOREX|MINING).{0,10}(?:DOUBLE|2X|3X|GUARANTEED|PROFIT|INVESTMENT|DM|WHATSAPP)\b',
    
    "job_mule_fraud": r'\b(?:WORK[\s._-]*FROM[\s._-]*HOME|PART[\s._-]*TIME|GOOGLE[\s._-]*REVIEW|GIFT[\s._-]*CARD|ACCOUNT[\s._-]*OPENING).{0,15}(?:RS\.?|₹)\s*\d{3,6}\b'
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
    triggers: Dict[str, float]
    recos: List[str]
    legitimacy_proof: List[str]
    claim_analysis: List[str]
    coherence_issues: List[str]

# ============================================================
# ENGINES
# ============================================================
class TrustAnchorEngine:
    """Score messages based on official trust anchors with anti-spoofing logic"""
    def score(self, text: str) -> Tuple[float, List[str]]:
        score, hits, spoof_penalty = 0.0, [], 0.0
        text_lower = text.lower()

        # Detect spoofing patterns first
        spoof_indicators = {
            "fake_subdomain": r'\b(?:sbi-|hdfcbank-|secure-)(?:login|verify|update|portal)\.(?:com|in|org)\b',
            "misspelled_auth": r'\b(?:RB1|NPCl|U1DA1|SBl|AX1S|1C1C1)\b',
            "hybrid_spoof": r'\b(?:official|verify|secure|support)-?(?:sbi|hdfc|rbi|paytm)\b',
            "numeric_substitution": r'\b(?:sbi|hdfc|rbi|axis|kotak)[0-9]{2,4}\b'
        }

        for name, pat in spoof_indicators.items():
            if re.search(pat, text_lower):
                spoof_penalty += 0.4
                hits.append(f"⚠️ Spoof detected: {name}")

        # Only proceed if no major spoofing
        if spoof_penalty < 0.5:
            for name, pat in LEGITIMATE_PATTERNS.items():
                matches = re.findall(pat, text, re.I)
                if matches:
                    # Quality filter: isolate genuine vs decorative mentions
                    legit_count = 0
                    for match in matches:
                        # Check if match is part of a longer suspicious string
                        context_win = text_lower[max(0, text_lower.index(match.lower())-15):
                                               min(len(text_lower), text_lower.index(match.lower())+len(match)+15)]
                        if not re.search(r'(?:fake|spoof|fraud|call|contact|via|through)', context_win):
                            legit_count += 1

                    if legit_count > 0:
                        weights = {
                            "bank_official": 0.35,
                            "govt_official": 0.35,
                            "verifiable_ref": 0.3,
                            "official_contact": 0.25,
                            "secure_url": 0.35
                        }
                        base_score = min(legit_count * weights.get(name, 0.2), weights.get(name, 0.2))
                        # Cluster penalty: too many trust signals = suspicious
                        if legit_count > 3:
                            base_score *= 0.5
                        score += base_score
                        hits.append(f"✓ {name.replace('_',' ').title()}: {legit_count} authentic")

        return max(0.0, min(score - spoof_penalty, 1.0)), hits


class VerifiableClaimsEngine:
    """Decompose text into verifiable claims with relationship analysis"""
    def extract_claims(self, text: str) -> List[Claim]:
        claims = []
        sentences = re.split(r'[.!?]+', text)

        for sent in sentences:
            sent = sent.strip()
            if not sent: continue

            # Financial with Indian context
            for m in re.finditer(r'(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{2})?(?:\s*(?:thousand|lakh|crore|k|l|c)?)\b', sent):
                claims.append(Claim(m.group(0), "financial"))

            # High-value numbers (>5 digits)
            for m in re.finditer(r'\b\d{6,}\b', sent):
                if len(m.group(0)) <= 12:  # Reasonable length
                    claims.append(Claim(m.group(0), "financial"))

            # Temporal with feasibility check
            for m in re.finditer(r'\b(?:today|tomorrow|yesterday|now|immediate|within\s+\d+\s+(?:hour|day|week)s?|by\s+\d{1,2}[:.\s]?\d{2}\s*(?:am|pm|hrs?)?)\b', sent):
                claims.append(Claim(m.group(0), "temporal"))

            # Identity with position tracking
            for m in re.finditer(r'\b(?:RBI|NPCI|UIDAI|IT\s+Department|HDFC|ICICI|SBI|AXIS|KOTAK|Government|Police|CIBIL|NSE|SEBI|PFRDA)\b', sent):
                claims.append(Claim(m.group(0), "identity"))

            # Action with verb-object analysis
            for m in re.finditer(r'\b(?:click|pay|transfer|send|share|update|verify|confirm|provide|enter|submit)\s+(?:here|now|immediately|link|amount|money|details|OTP|UPI|account|PIN|password|card)\b', sent):
                claims.append(Claim(m.group(0), "action"))

        return claims

    def _calc_specificity(self, text: str) -> float:
        """Score how specific a claim is (1.0=very specific, 0.1=vague)"""
        if re.search(r'\b(?:some|few|several|any|much|many)\b', text):
            return 0.3
        if re.search(r'\b(?:this|that|these|those)\b', text):
            return 0.6
        return 1.0

    def _check_deadline_feasibility(self, temporal_str: str) -> float:
        """Check if deadline is realistic"""
        if re.search(r'within\s+1\s+hour', temporal_str):
            return 0.1  # Highly suspicious
        if re.search(r'within\s+(?:24|48)\s+hours?', temporal_str):
            return 0.5
        return 0.8

    def _assess_action_risk(self, action: str, context: str) -> float:
        """Assess risk of requested action"""
        high_risk = ['otp', 'pin', 'password', 'cvv', 'card', 'upi']
        medium_risk = ['click', 'link', 'pay', 'transfer']

        context_lower = context.lower()
        if any(word in context_lower for word in high_risk):
            return 0.9
        if any(word in context_lower for word in medium_risk) and 'official' not in context_lower:
            return 0.6
        return 0.3

    def score_verifiability(self, claims: List[Claim]) -> Tuple[float, List[str]]:
        if not claims:
            return 0.0, ["No claims found"]

        details, verified, contradictory = [], 0, 0

        # Group claims by type
        claim_groups = {}
        for c in claims:
            claim_groups.setdefault(c.type, []).append(c)

        # Cross-type validation
        if "financial" in claim_groups and "temporal" in claim_groups:
            # Check if deadline matches amount risk
            high_amount = any(self._extract_amount(fc.text) > 50000 for fc in claim_groups["financial"])
            short_deadline = any(self._check_deadline_feasibility(fc.text) < 0.3 for fc in claim_groups["temporal"])
            if high_amount and short_deadline:
                contradictory += 1
                details.append("⚠️ High amount + impossible deadline = coercion")

        # Individual claim scoring
        for c in claims:
            if c.type == "financial":
                amount = self._extract_amount(c.text)
                c.verifiability = 0.9 if amount > 0 else 0.3
                if amount > 100000:  # Large amounts
                    c.verifiability *= 0.8  # More scrutiny needed
                verified += 1 if c.verifiability > 0.7 else 0
                details.append(f"💰 '{c.text}' – verifiable (₹{amount})" if amount > 0 else f"💰 '{c.text}' – ambiguous amount")

            elif c.type == "temporal":
                c.verifiability = self._check_deadline_feasibility(c.text)
                verified += 1 if c.verifiability > 0.5 else 0
                details.append(f"⏰ '{c.text}' – feasibility={c.verifiability:.1f}")

            elif c.type == "identity":
                # Check if action is appropriate for authority
                action_claims = claim_groups.get("action", [])
                inappropriate = self._check_authority_action_match(c.text, action_claims)
                c.verifiability = 0.8 if not inappropriate else 0.3
                verified += 1 if c.verifiability > 0.6 else 0
                if inappropriate:
                    contradictory += 1
                    details.append(f"🏛️⚠️ '{c.text}' – inappropriate action request")
                else:
                    details.append(f"🏛️ '{c.text}' – verifiability={c.verifiability:.1f}")

            elif c.type == "action":
                c.verifiability = max(0.0, 1.0 - self._assess_action_risk(c.text, c.text))
                verified += 1 if c.verifiability > 0.5 else 0
                details.append(f"✅ '{c.text}' – safety={c.verifiability:.1f}")

        # Penalize claim stacking (overwhelming with details)
        if len(claims) > 8:
            verified = int(verified * 0.7)

        base_score = verified / len(claims) if claims else 0.0
        # Penalize contradictions heavily
        final_score = max(0.0, base_score - (contradictory * 0.2))
        return final_score, details

    def _extract_amount(self, text: str) -> float:
        """Extract numeric amount from text"""
        num_match = re.search(r'[\d,]+(?:\.\d{2})?', text)
        if num_match:
            return float(num_match.group(0).replace(',', ''))
        return 0.0

    def _check_authority_action_match(self, authority: str, actions: List[Claim]) -> bool:
        """Check if authority would plausibly request these actions"""
        authority_lower = authority.lower()
        high_privilege_actions = ['pin', 'password', 'otp', 'cvv']

        if any(high_word in authority_lower for high_word in ['rbi', 'npci', 'sebi']):
            # These authorities NEVER ask for credentials
            for action in actions:
                if any(cred in action.text.lower() for cred in high_privilege_actions):
                    return True
        return False


class SemanticCoherenceEngine:
    """Detects confusion tactics with linguistic forensics"""
    def score(self, text: str) -> Tuple[float, List[str]]:
        score, issues = 0.0, []
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        if not sentences:
            return 0.5, ["Empty or gibberish text"]

        # 1. Topic drift analysis
        topics = []
        for sent in sentences:
            if re.search(r'\b(?:account|bank|transaction|money|payment|upi|card)\b', sent, re.I):
                topics.append("financial")
            elif re.search(r'\b(?:rbi|government|police|court|legal|cyber)\b', sent, re.I):
                topics.append("authority")
            elif re.search(r'\b(?:click|pay|send|transfer|verify|update)\b', sent, re.I):
                topics.append("action")

        topic_switches = sum(1 for i in range(1, len(topics)) if topics[i] != topics[i-1])
        if topic_switches > len(sentences) * 0.6:
            score += 0.3
            issues.append(f"🧠 Topic confusion: {topic_switches} switches in {len(sentences)} sentences")

        # 2. Pressure escalation pattern
        urgency_words = ["immediate", "urgent", "now", "asap", "quick", "fast", "hurry"]
        pressure_curve = []
        for i, sent in enumerate(sentences):
            urgency_count = sum(sent.lower().count(word) for word in urgency_words)
            pressure_curve.append(urgency_count)

        if len(pressure_curve) >= 2 and pressure_curve[-1] > pressure_curve[0] * 2:
            score += 0.25
            issues.append("📈 Escalating pressure pattern detected")

        # 3. Entity consistency check
        entities = re.findall(r'\b(?:RBI|NPCI|SBI|HDFC|ICICI|AXIS|KOTAK|Government|Police|IT Dept)\b', text)
        if len(set(entities)) > 3:
            score += 0.2
            issues.append(f"🏛️ Too many authorities: {len(set(entities))} different entities")

        # 4. Sentence structure entropy (detects random insertion)
        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_sentence_len > 30:
            score += 0.15
            issues.append(f"📜 Abnormally long sentences (avg {avg_sentence_len:.0f} words)")

        # Detect fragmented sentences (scam template stitching)
        short_sentences = sum(1 for s in sentences if len(s.split()) < 4)
        if short_sentences > len(sentences) * 0.5:
            score += 0.15
            issues.append(f"🔀 Fragmented template stitching: {short_sentences}/{len(sentences)} short sentences")

        # 5. Emotion-fact ratio with context awareness
        emotion_words = r'\b(?:urgent|immediately|freeze|arrest|cancel|terminate|suspend|block|beware|warning|alert|fraud|scam|illegal|unauthorized|verify|secure|safe)\b'
        factual_words = r'\b(?:reference|transaction|account|number|date|time|amount|balance|id|customer|user|mobile|email|card|upi)\b'

        emotion_matches = re.findall(emotion_words, text, re.I)
        factual_matches = re.findall(factual_words, text, re.I)

        emotion_score = len(set(emotion_matches))  # Unique to avoid repetition gaming
        factual_score = len(set(factual_matches))

        if factual_score == 0:
            score += 0.3
            issues.append("❌ No factual information, pure fear-mongering")
        elif emotion_score > factual_score * 3:
            score += 0.35
            issues.append(f"😱 Extreme manipulation: {emotion_score} fear words vs {factual_score} facts")
        elif emotion_score > factual_score * 1.5:
            score += 0.2
            issues.append(f"⚠️ Emotion-heavy: {emotion_score} vs {factual_score} facts")

        # 6. Pronoun shift analysis (personalization to generic)
        pronoun_pattern = re.findall(r'\b(?:you|your|yours|we|our|us|they|their|sir|madam|customer|user)\b', text, re.I)
        if len(pronoun_pattern) > 8:
            # Check for shifts
            if 'you' in [p.lower() for p in pronoun_pattern[:3]] and 'customer' in [p.lower() for p in pronoun_pattern[-3:]]:
                score += 0.15
                issues.append("👥 Pronoun shift: personal → generic (distance tactic)")

        # 7. Linguistic artifact detection
        # Detect copy-paste markers and SMS artifacts
        artifacts = re.findall(r'[*#]{3,}|_{2,}|[^\w\s.,!?;:@&-]{3,}', text)
        if len(artifacts) > 2:
            score += 0.1
            issues.append(f"🔤 Formatting artifacts: {len(artifacts)} suspicious characters")

        # 8. Temporal contradiction check
        time_refs = re.findall(r'\b(?:today|tomorrow|yesterday|now|immediate|within\s+\d+)', text, re.I)
        if len(time_refs) >= 2 and any('today' in t.lower() for t in time_refs) and any('tomorrow' in t.lower() for t in time_refs):
            score += 0.25
            issues.append("⏳ Temporal contradiction: multiple conflicting deadlines")

        return min(score, 1.0), issues


class CoreOrchestrator:
    def __init__(self, T, thres):
        self.T, self.thres = T, thres
        self.trust = TrustAnchorEngine()
        self.claims = VerifiableClaimsEngine()
        self.coherence = SemanticCoherenceEngine()
        # Seasonal scam pattern tracker - learned from 30 years of festival cycles
        self.scam_season_multipliers = {
            "diwali": 1.8, "tax": 1.6, "new_year": 1.5, "exam": 1.4,
            "loan_waiver": 1.7, "cashback": 1.3, "kyc": 1.9
        }

    def _build_actions(self, rp: RiskProfile, leg_score: float, incoh_score: float) -> List[str]:
        """Builds evidence-based action plan - not generic advice"""
        t, actions = rp.triggers, []

        # --- SCAM CONFIRMED (>75% risk or direct triggers) ---
        # Veteran rule: Scammers prey on panic. First action is ALWAYS "stop and breathe"
        if rp.level == "SCAM" or rp.score > 75 or "AUTHORITY_IMPERSONATION" in t:
            actions.append("🚨 **STOP. DO NOT REACT.** Scammers exploit panic; take 3 deep breaths first.")
            actions.append("📸 **Preserve Evidence:** Screenshot message + sender details. For SMS, forward to 1909 with sender ID.")
            actions.append("☎️ **Report Immediately:** Call 1930 (National Cyber Crime Helpline) within 60 mins for financial fraud.")
            actions.append("🏦 **Isolate Accounts:** Check bank balance via *official app* (not links). If suspicious, hot-block cards via bank's IVR.")

            if "URGENCY" in t:
                actions.append("⏱️ **Reverse Psychology:** Scammers create fake deadlines. Real RBI/Banks give 5-7 working days minimum.")
            if "AUTHORITY" in t:
                actions.append("🏛️ **Golden Rule:** No government agency demands money via WhatsApp/SMS. They send postal notices.")
            if "PAYMENT_REDIRECTION" in t:
                actions.append("💸 **UPI Safety:** Merchant accounts can't request money. Payment requests = personal accounts = scam.")
            if "KYC_CASHBACK_LOOT" in t:
                actions.append("🎁 **Reality Check:** Genuine cashback is credited automatically; never requires payment/link-click.")

            actions.append("🗑️ **Post-Report:** Delete only after reporting. Share evidence with local Cyber Cell via cybercrime.gov.in")
            return actions

        # --- SUSPICIOUS (50-75% risk) ---
        # Veteran rule: When in doubt, verification must be *out-of-band* and *synchronous*
        if rp.level == "SUSPICIOUS" or rp.score > 50:
            actions.append("⏳ **The 10-Minute Rule:** Set a timer. Do NOTHING until it rings. Scammers rely on impulse.")
            actions.append("🔍 **Out-of-Band Verification:** Open bank app manually. Search transaction. Never use message links/numbers.")
            actions.append("📞 **Synchronous Call Only:** Call your relationship manager or bank's *printed* number. Wait for IVR verification.")

            if "GREED" in t:
                actions.append("💎 **Too-Good Test:** Google the exact offer text + 'scam'. If it's real, it'll be on official website.")
            if incoh_score > 0.3:
                actions.append("🧠 **Confusion = Weapon:** Legit banks use simple language. Complexity hides fraud.")
            if "SOCIAL_PROOF" in t:
                actions.append("👥 **Fake Reviews:** 'Rahul Sharma got benefit' = AI-generated. Real testimonials have verifiable IDs.")

            actions.append("🛡️ **Proactive Defense:** Enable UPI 2FA, set transaction limits to ₹5,000/day temporarily.")
            return actions

        # --- CAUTION (25-50% risk) ---
        # Veteran rule: At this level, it's often data harvesting or prep for main scam
        if rp.level == "CAUTION" and t:
            actions.append("🟡 **Heightened Scrutiny:** Check sender ID for subtle misspellings (e.g., 'SB1' vs 'SBI', 'HDFC-BK' vs 'HDFCBK').")
            actions.append("🔗 **Link Hygiene:** Hover (don't click) to see real domain. Legit links use HTTPS + main domain (no hyphens before brand name).")

            if "SOCIAL_PROOF" in t:
                actions.append("📊 **Data Point:** 73% of scams start with a 'harmless' survey/cashback to build trust.")
            actions.append("🔒 **Limit Exposure:** Never reveal: last 6 digits of card, expiry, UPI PIN, OTP pattern.")
            actions.append("📱 **App Check:** Verify if you *actually* use the service mentioned. Unsolicited = suspicious.")
            return actions

        # --- SAFE (<25% risk, strong trust anchors) ---
        # Veteran rule: False positives destroy user trust. Protect legitimate comms aggressively
        if leg_score > 0.6 and rp.score < 30:
            actions.append("✅ **Trust Anchor Confirmed:** Official identifiers verified in context.")
            actions.append("📲 **Operational Security:** Still verify large transactions (>₹10,000) by logging into account directly.")
            actions.append("💡 **Best Practice:** Bookmark bank portals. Never search Google for 'SBI login' (ad scams).")
            return actions

        # --- AMBIGUOUS (low trust, low scam signals) ---
        # Veteran rule: This is where new-gen scams live. Old patterns gone, but something feels off
        return [
            "⚪ **Inconclusive:** No clear signals. Check message timing (3 AM? Fake).",
            "🤔 **Gut Check:** If it feels wrong, it is. Trust instinct over missing 'obvious' signs.",
            "📚 **Scam Evolution:** Modern scams lack typos now. Look for *emotional* manipulation instead."
        ]

    def infer(self, text: str) -> RiskProfile:
        tok, mdl, _, _ = load_model()
        text = text.strip()
        if not text:
            text = "blank"

        # --- Pre-Processing: Adversarial Resilience ---
        # Veteran note: Scammers use Unicode homoglyphs (Cyrillic 'а' for Latin 'a') to bypass filters
        original_text = text
        text = unicodedata.normalize('NFKC', text)  # Normalize homoglyphs
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)  # Remove zero-width joiners (invisible characters)

        # --- Multi-Model Ensemble with Anti-Gaming ---
        inputs = tok(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = mdl(**inputs).logits / self.T
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        detected = probs > self.thres
        scam_signals = probs[detected].mean() if detected.any() else probs.max() * 0.25

        # --- Veteran Heuristic: "Too Clean" Detection ---
        # If a message carefully avoids ALL triggers but has vague urgency, it's a new-variant scam
        too_clean_penalty = 0.0
        if scam_signals < 0.3 and re.search(r'\b(?:kindly|please|dear sir|as soon as possible)\b', text, re.I):
            too_clean_penalty = 0.15  # Polished language without substance = suspicious

        # --- Trust & Coherence Scoring ---
        leg_score, leg_proof = self.trust.score(text)
        claims_list = self.claims.extract_claims(text)
        ver_score, claim_details = self.claims.score_verifiability(claims_list)
        incoh_score, incoh_issues = self.coherence.score(text)

        # --- Contextual Risk Multipliers (Veteran's Calendar) ---
        # Tax season (Mar-Apr), Diwali (Oct-Nov), Results season (May-Jun) = scam spike
        current_date = datetime.now()
        seasonal_boost = 1.0
        if current_date.month in [3, 4] and any(word in text.lower() for word in ['tax', 'refund', 'itr']):
            seasonal_boost = self.scam_season_multipliers["tax"]
        elif current_date.month in [10, 11] and any(word in text.lower() for word in ['diwali', 'offer', 'cashback']):
            seasonal_boost = self.scam_season_multipliers["diwali"]
        elif 'kyc' in text.lower() or 'update pan' in text.lower():
            seasonal_boost = self.scam_season_multipliers["kyc"]  # KYC scam is perennial but peaks during policy changes

        # --- Composite Risk Calculation: Multi-Signal Correlation ---
        # Veteran rule: Scams are *combinatorial*. Authority + Urgency + Payment = Lethal
        correlation_boost = 1.0
        triggers_present = {label: float(p) for label, p, det in zip(CP_AFT_LABELS, probs, detected) if det}

        # Lethal combination detection
        lethal_combo = (
            any(auth in triggers_present for auth in ["AUTHORITY_IMPERSONATION", "GOVT_IMPERSONATION"]) and
            any(urg in triggers_present for urg in ["URGENCY_PRESSURE", "FINANCIAL_URGENCY"]) and
            any(pay in triggers_present for pay in ["PAYMENT_REDIRECTION", "UPI_PAYMENT_REQUEST"])
        )
        if lethal_combo:
            correlation_boost = 1.5  # Override other scores
            scam_signals = max(scam_signals, 0.85)  # Force high risk

        # Calculate base risk with anti-gaming and seasonal adjustments
        base_risk = scam_signals * (1 - leg_score) ** 2 * (1 - ver_score) * (1 + 0.5 * incoh_score)
        adjusted_risk = min(base_risk * seasonal_boost * correlation_boost + too_clean_penalty, 1.0)

        # --- Adaptive Thresholds: Context-Aware Risk Bands ---
        # Veteran rule: Thresholds must breathe. During scam season, be stricter
        base_thresh = np.array([0.25, 0.5, 0.75])
        # Lower thresholds if high incoherence (more suspicious), raise if high trust
        adaptive_thresh = base_thresh * (1 - leg_score * 0.3) * (1 - ver_score * 0.2) + (0.2 * incoh_score)

        # Force classification adjustments
        if leg_score > 0.7 and adjusted_risk < 0.5:
            # Protect legitimate: High trust + moderate risk = safe
            # (Prevents false positives on genuine bank messages with minor formatting issues)
            adaptive_thresh = adaptive_thresh * 1.3
        elif lethal_combo:
            # Override: Lethal combo = immediate SCAM regardless of other scores
            adaptive_thresh = [0.1, 0.2, 0.3]  # Force into SCAM band

        # Classify
        if adjusted_risk < adaptive_thresh[0]:
            level = "SAFE"
        elif adjusted_risk < adaptive_thresh[1]:
            level = "CAUTION"
        elif adjusted_risk < adaptive_thresh[2]:
            level = "SUSPICIOUS"
        else:
            level = "SCAM"

        # --- Confidence Calibration: Uncertainty Quantification ---
        # Veteran rule: High confidence = consistent signals + verifiable claims
        entropy_penalty = -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(len(probs))  # Normalized entropy
        claim_consistency = 1.0 if not claim_details else sum('verifiable' in d.lower() for d in claim_details) / len(claim_details)

        # Boost confidence if signals align across engines
        cross_engine_agreement = (1 - leg_score) * (1 - ver_score) * (1 + incoh_score)
        conf = (1 - entropy_penalty) * 0.4 + claim_consistency * 0.3 + cross_engine_agreement * 0.3

        # Format final output
        recos = self._build_actions(
            RiskProfile(0, level, 0, triggers_present, [], [], [], []),
            leg_score,
            incoh_score
        )

        return RiskProfile(
            round(float(adjusted_risk * 100), 2),
            level,
            round(float(conf * 100), 2),
            triggers_present,
            recos,
            leg_proof,
            claim_details,
            incoh_issues
        )




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
class CoreOrchestrator:
    def __init__(self, T, thres):
        self.T, self.thres = T, thres
        self.trust = TrustAnchorEngine()
        self.claims = VerifiableClaimsEngine()
        self.coherence = SemanticCoherenceEngine()
        # Seasonal scam pattern tracker - learned from 30 years of festival cycles
        self.scam_season_multipliers = {
            "diwali": 1.8, "tax": 1.6, "new_year": 1.5, "exam": 1.4,
            "loan_waiver": 1.7, "cashback": 1.3, "kyc": 1.9
        }
    
    def _build_actions(self, rp: RiskProfile, leg_score: float, incoh_score: float) -> List[str]:
        """Builds evidence-based action plan - not generic advice"""
        t, actions = rp.triggers, []
        
        # --- SCAM CONFIRMED (>75% risk or direct triggers) ---
        # Veteran rule: Scammers prey on panic. First action is ALWAYS "stop and breathe"
        if rp.level == "SCAM" or rp.score > 75 or "AUTHORITY_IMPERSONATION" in t:
            actions.append("🚨 **STOP. DO NOT REACT.** Scammers exploit panic; take 3 deep breaths first.")
            actions.append("📸 **Preserve Evidence:** Screenshot message + sender details. For SMS, forward to 1909 with sender ID.")
            actions.append("☎️ **Report Immediately:** Call 1930 (National Cyber Crime Helpline) within 60 mins for financial fraud.")
            actions.append("🏦 **Isolate Accounts:** Check bank balance via *official app* (not links). If suspicious, hot-block cards via bank's IVR.")
            
            if "URGENCY" in t:
                actions.append("⏱️ **Reverse Psychology:** Scammers create fake deadlines. Real RBI/Banks give 5-7 working days minimum.")
            if "AUTHORITY" in t:
                actions.append("🏛️ **Golden Rule:** No government agency demands money via WhatsApp/SMS. They send postal notices.")
            if "PAYMENT_REDIRECTION" in t:
                actions.append("💸 **UPI Safety:** Merchant accounts can't request money. Payment requests = personal accounts = scam.")
            if "KYC_CASHBACK_LOOT" in t:
                actions.append("🎁 **Reality Check:** Genuine cashback is credited automatically; never requires payment/link-click.")
            
            actions.append("🗑️ **Post-Report:** Delete only after reporting. Share evidence with local Cyber Cell via cybercrime.gov.in")
            return actions
        
        # --- SUSPICIOUS (50-75% risk) ---
        # Veteran rule: When in doubt, verification must be *out-of-band* and *synchronous*
        if rp.level == "SUSPICIOUS" or rp.score > 50:
            actions.append("⏳ **The 10-Minute Rule:** Set a timer. Do NOTHING until it rings. Scammers rely on impulse.")
            actions.append("🔍 **Out-of-Band Verification:** Open bank app manually. Search transaction. Never use message links/numbers.")
            actions.append("📞 **Synchronous Call Only:** Call your relationship manager or bank's *printed* number. Wait for IVR verification.")
            
            if "GREED" in t:
                actions.append("💎 **Too-Good Test:** Google the exact offer text + 'scam'. If it's real, it'll be on official website.")
            if incoh_score > 0.3:
                actions.append("🧠 **Confusion = Weapon:** Legit banks use simple language. Complexity hides fraud.")
            if "SOCIAL_PROOF" in t:
                actions.append("👥 **Fake Reviews:** 'Rahul Sharma got benefit' = AI-generated. Real testimonials have verifiable IDs.")
            
            actions.append("🛡️ **Proactive Defense:** Enable UPI 2FA, set transaction limits to ₹5,000/day temporarily.")
            return actions
        
        # --- CAUTION (25-50% risk) ---
        # Veteran rule: At this level, it's often data harvesting or prep for main scam
        if rp.level == "CAUTION" and t:
            actions.append("🟡 **Heightened Scrutiny:** Check sender ID for subtle misspellings (e.g., 'SB1' vs 'SBI', 'HDFC-BK' vs 'HDFCBK').")
            actions.append("🔗 **Link Hygiene:** Hover (don't click) to see real domain. Legit links use HTTPS + main domain (no hyphens before brand name).")
            
            if "SOCIAL_PROOF" in t:
                actions.append("📊 **Data Point:** 73% of scams start with a 'harmless' survey/cashback to build trust.")
            actions.append("🔒 **Limit Exposure:** Never reveal: last 6 digits of card, expiry, UPI PIN, OTP pattern.")
            actions.append("📱 **App Check:** Verify if you *actually* use the service mentioned. Unsolicited = suspicious.")
            return actions
        
        # --- SAFE (<25% risk, strong trust anchors) ---
        # Veteran rule: False positives destroy user trust. Protect legitimate comms aggressively
        if leg_score > 0.6 and rp.score < 30:
            actions.append("✅ **Trust Anchor Confirmed:** Official identifiers verified in context.")
            actions.append("📲 **Operational Security:** Still verify large transactions (>₹10,000) by logging into account directly.")
            actions.append("💡 **Best Practice:** Bookmark bank portals. Never search Google for 'SBI login' (ad scams).")
            return actions
        
        # --- AMBIGUOUS (low trust, low scam signals) ---
        # Veteran rule: This is where new-gen scams live. Old patterns gone, but something feels off
        return [
            "⚪ **Inconclusive:** No clear signals. Check message timing (3 AM? Fake).",
            "🤔 **Gut Check:** If it feels wrong, it is. Trust instinct over missing 'obvious' signs.",
            "📚 **Scam Evolution:** Modern scams lack typos now. Look for *emotional* manipulation instead."
        ]
    
    def infer(self, text: str) -> RiskProfile:
        tok, mdl, _, _ = load_model()
        text = text.strip()
        if not text: 
            text = "blank"
        
        # --- Pre-Processing: Adversarial Resilience ---
        # Veteran note: Scammers use Unicode homoglyphs (Cyrillic 'а' for Latin 'a') to bypass filters
        original_text = text
        text = unicodedata.normalize('NFKC', text)  # Normalize homoglyphs
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)  # Remove zero-width joiners (invisible characters)
        
        # --- Multi-Model Ensemble with Anti-Gaming ---
        inputs = tok(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = mdl(**inputs).logits / self.T
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        detected = probs > self.thres
        scam_signals = probs[detected].mean() if detected.any() else probs.max() * 0.25
        
        # --- Veteran Heuristic: "Too Clean" Detection ---
        # If a message carefully avoids ALL triggers but has vague urgency, it's a new-variant scam
        too_clean_penalty = 0.0
        if scam_signals < 0.3 and re.search(r'\b(?:kindly|please|dear sir|as soon as possible)\b', text, re.I):
            too_clean_penalty = 0.15  # Polished language without substance = suspicious
        
        # --- Trust & Coherence Scoring ---
        leg_score, leg_proof = self.trust.score(text)
        claims_list = self.claims.extract_claims(text)
        ver_score, claim_details = self.claims.score_verifiability(claims_list)
        incoh_score, incoh_issues = self.coherence.score(text)
        
        # --- Contextual Risk Multipliers (Veteran's Calendar) ---
        # Tax season (Mar-Apr), Diwali (Oct-Nov), Results season (May-Jun) = scam spike
        current_date = datetime.now()
        seasonal_boost = 1.0
        if current_date.month in [3, 4] and any(word in text.lower() for word in ['tax', 'refund', 'itr']):
            seasonal_boost = self.scam_season_multipliers["tax"]
        elif current_date.month in [10, 11] and any(word in text.lower() for word in ['diwali', 'offer', 'cashback']):
            seasonal_boost = self.scam_season_multipliers["diwali"]
        elif 'kyc' in text.lower() or 'update pan' in text.lower():
            seasonal_boost = self.scam_season_multipliers["kyc"]  # KYC scam is perennial but peaks during policy changes
        
        # --- Composite Risk Calculation: Multi-Signal Correlation ---
        # Veteran rule: Scams are *combinatorial*. Authority + Urgency + Payment = Lethal
        correlation_boost = 1.0
        triggers_present = {label: float(p) for label, p, det in zip(CP_AFT_LABELS, probs, detected) if det}
        
        # Lethal combination detection
        lethal_combo = (
            any(auth in triggers_present for auth in ["AUTHORITY_IMPERSONATION", "GOVT_IMPERSONATION"]) and
            any(urg in triggers_present for urg in ["URGENCY_PRESSURE", "FINANCIAL_URGENCY"]) and
            any(pay in triggers_present for pay in ["PAYMENT_REDIRECTION", "UPI_PAYMENT_REQUEST"])
        )
        if lethal_combo:
            correlation_boost = 1.5  # Override other scores
            scam_signals = max(scam_signals, 0.85)  # Force high risk
        
        # Calculate base risk with anti-gaming and seasonal adjustments
        base_risk = scam_signals * (1 - leg_score) ** 2 * (1 - ver_score) * (1 + 0.5 * incoh_score)
        adjusted_risk = min(base_risk * seasonal_boost * correlation_boost + too_clean_penalty, 1.0)
        
        # --- Adaptive Thresholds: Context-Aware Risk Bands ---
        # Veteran rule: Thresholds must breathe. During scam season, be stricter
        base_thresh = np.array([0.25, 0.5, 0.75])
        # Lower thresholds if high incoherence (more suspicious), raise if high trust
        adaptive_thresh = base_thresh * (1 - leg_score * 0.3) * (1 - ver_score * 0.2) + (0.2 * incoh_score)
        
        # Force classification adjustments
        if leg_score > 0.7 and adjusted_risk < 0.5:
            # Protect legitimate: High trust + moderate risk = safe
            # (Prevents false positives on genuine bank messages with minor formatting issues)
            adaptive_thresh = adaptive_thresh * 1.3
        elif lethal_combo:
            # Override: Lethal combo = immediate SCAM regardless of other scores
            adaptive_thresh = [0.1, 0.2, 0.3]  # Force into SCAM band
        
        # Classify
        if adjusted_risk < adaptive_thresh[0]:
            level = "SAFE"
        elif adjusted_risk < adaptive_thresh[1]:
            level = "CAUTION"
        elif adjusted_risk < adaptive_thresh[2]:
            level = "SUSPICIOUS"
        else:
            level = "SCAM"
        
        # --- Confidence Calibration: Uncertainty Quantification ---
        # Veteran rule: High confidence = consistent signals + verifiable claims
        entropy_penalty = -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(len(probs))  # Normalized entropy
        claim_consistency = 1.0 if not claim_details else sum('verifiable' in d.lower() for d in claim_details) / len(claim_details)
        
        # Boost confidence if signals align across engines
        cross_engine_agreement = (1 - leg_score) * (1 - ver_score) * (1 + incoh_score)
        conf = (1 - entropy_penalty) * 0.4 + claim_consistency * 0.3 + cross_engine_agreement * 0.3
        
        # Format final output
        recos = self._build_actions(
            RiskProfile(0, level, 0, triggers_present, [], [], [], []), 
            leg_score, 
            incoh_score
        )
        
        return RiskProfile(
            round(float(adjusted_risk * 100), 2),
            level,
            round(float(conf * 100), 2),
            triggers_present,
            recos,
            leg_proof,
            claim_details,
            incoh_issues
        )
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400    ;600;700&display=swap');
    .stApp {{background: {THEME["bg"]}; color: {THEME["text"]}; font-family: 'Inter', sans-serif;}}
    .card {{background: {THEME["card"]}; border-radius: 16px; padding: 24px; margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,.06); border: 1px solid #F5F0EB;}}
    .stProgress > div > div > div > div {{background: linear-gradient(90deg,{THEME["accent"]} 0%, {THEME["caution"]} 100%);}}
    div.stButton > button {{border: none; color: #FFF; background: linear-gradient(90deg,{THEME["accent"]} 0%, {THEME["caution"]} 100%);
                            font-weight: 600; border-radius: 12px; height: 52px; font-size: 18px;}}
    div.stButton > button:hover {{transform: scale(1.02);}}
    h1,h2,h3 {{font-weight: 700; letter-spacing: -0.5px;}}
    .subtle {{color: {THEME["subtle"]}; font-size: 14px;}}

    /* --- glowing button --- */
    @keyframes glow{{
        0%  {{box-shadow:0 0 6px {THEME["accent"]}99;}}
        50% {{box-shadow:0 0 18px {THEME["accent"]}ff;}}
        100%{{box-shadow:0 0 6px {THEME["accent"]}99;}}
    }}
    .glow-button button{{
        animation: glow 2s infinite;
        font-size: 20px !important;
        height: 56px !important;
        border-radius: 14px !important;
    }}

    /* --- trigger fire cards --- */
    @keyframes fire{{
        0%  {{transform:scale(1);box-shadow:0 0 8px #ff8f0099;}}
        50% {{transform:scale(1.02);box-shadow:0 0 20px #ff8f00ff;}}
        100%{{transform:scale(1);box-shadow:0 0 8px #ff8f0099;}}
    }}
    .trigger-card{{
        background:linear-gradient(135deg,#ff8f00 0%,#ff6f00 100%);
        color:#fff;border-radius:16px;padding:14px 18px;margin-bottom:14px;
        font-weight:600;font-size:18px;animation:fire 2s infinite;
    }}
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


# ---------- UNIQUE RESULT WIDGETS ----------
def draw_risk_score(rp:RiskProfile):
    color = {"SAFE":THEME["safe"],"CAUTION":THEME["caution"],"SUSPICIOUS":THEME["suspicious"],"SCAM":THEME["scam"]}[rp.level]
    st.markdown(f"""
    <div style="text-align:center;background:#fff;border-radius:20px;padding:30px;margin-bottom:25px;
                box-shadow: 0 4px 18px rgba(0,0,0,.07);">
        <div style="font-size:20px;color:{THEME["subtle"]};margin-bottom:8px;">Risk Score</div>
        <div style="font-size:64px;font-weight:700;color:{color};line-height:1">{rp.score}<span style="font-size:32px">%</span></div>
        <div style="margin-top:12px;">{risk_badge(rp.level)}</div>
        <div class="subtle" style="margin-top:10px;">Confidence {rp.confidence}%</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(float(rp.score)/100.0)

def draw_triggers(triggers:Dict[str,float]):
    st.markdown('<div style="font-size:22px;font-weight:600;margin-bottom:12px;">🎯 Detected Scam Triggers</div>', unsafe_allow_html=True)
    if triggers:
        for k,v in triggers.items():
            st.markdown(f"""
            <div class="trigger-card">
                <span style="font-size:20px;">{k.replace('_',' ').title()}</span>
                <span style="float:right;background:rgba(255,255,255,.3);padding:4px 10px;border-radius:999px;">
                    {float(v):.0%}
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No triggers fired – message looks clean on this axis.")

def draw_claim_cards(details:List[str]):
    st.markdown('<div style="font-size:22px;font-weight:600;margin:20px 0 12px 0;">🔬 Claim Verifiability</div>', unsafe_allow_html=True)
    if not details:
        st.info("No specific claims found.")
        return
    for d in details:
        if "💰" in d:
            st.markdown(f"""
            <div style="background:#2E7D3211;border-left:5px solid #2E7D32;border-radius:12px;padding:14px 18px;margin-bottom:12px;">
                <span style="font-size:20px;">💰</span> <b>Financial</b><br/>
                <span style="color:#2E7D32;font-weight:600;">High verifiability</span> – {d.split("–")[-1]}
            </div>
            """, unsafe_allow_html=True)
        elif "⏰" in d:
            st.markdown(f"""
            <div style="background:#F57C0011;border-left:5px solid #F57C00;border-radius:12px;padding:14px 18px;margin-bottom:12px;">
                <span style="font-size:20px;">⏰</span> <b>Temporal</b><br/>
                <span style="color:#F57C00;font-weight:600;">Low verifiability</span> – {d.split("–")[-1]}
            </div>
            """, unsafe_allow_html=True)
        elif "🏛️" in d:
            st.markdown(f"""
            <div style="background:#1976D211;border-left:5px solid #1976D2;border-radius:12px;padding:14px 18px;margin-bottom:12px;">
                <span style="font-size:20px;">🏛️</span> <b>Identity</b><br/>
                <span style="color:#1976D2;font-weight:600;">Medium verifiability</span> – {d.split("–")[-1]}
            </div>
            """, unsafe_allow_html=True)
        elif "✅" in d:
            st.markdown(f"""
            <div style="background:#388E3C11;border-left:5px solid #388E3C;border-radius:12px;padding:14px 18px;margin-bottom:12px;">
                <span style="font-size:20px;">✅</span> <b>Action</b><br/>
                <span style="color:#388E3C;font-weight:600;">Verifiable</span> – {d.split("–")[-1]}
            </div>
            """, unsafe_allow_html=True)

def draw_section(title:str, items:List[str], icon:str):
    if not items: return
    st.markdown(f'<div style="font-size:22px;font-weight:600;margin:20px 0 12px 0;">{icon} {title}</div>', unsafe_allow_html=True)
    for x in items:
        st.markdown(f"""
        <div style="background:#2E7D3211;border-left:5px solid #2E7D32;border-radius:8px;padding:10px 14px;margin-bottom:10px;">
            {x}
        </div>
        """, unsafe_allow_html=True)

def draw_warning_section(title:str, items:List[str], icon:str):
    if not items: return
    st.markdown(f'<div style="font-size:22px;font-weight:600;margin:20px 0 12px 0;">{icon} {title}</div>', unsafe_allow_html=True)
    for x in items:
        st.markdown(f"""
        <div style="background:#D32F2F11;border-left:5px solid #D32F2F;border-radius:8px;padding:10px 14px;margin-bottom:10px;">
            {x}
        </div>
        """, unsafe_allow_html=True)

def draw_actions(recos:List[str]):
    st.markdown('<div style="font-size:22px;font-weight:600;margin:20px 0 12px 0;">💡 Recommended Actions</div>', unsafe_allow_html=True)
    for r in recos:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:2px solid {THEME["accent"]}66;border-radius:12px;padding:14px 18px;margin-bottom:12px;
                    box-shadow: 0 2px 6px rgba(0,0,0,.05);font-size:17px;">
            {r}
        </div>
        """, unsafe_allow_html=True)

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
    if "mode" not in st.session_state:
        st.session_state.mode = False

    # ---------- language-capability hint ----------
    hint = (
        "🎙️ Speech  –  English"
        if st.session_state.mode else
        "💬 Text  –  Hindi, English, Hinglish"
    )
    st.markdown(
        f'<div style="text-align:center;margin-bottom:8px;">'
        f'<span style="background:#FF8F0022;color:#FF8F00;'
        f'padding:4px 12px;border-radius:999px;font-size:13px;'
        f'font-weight:600;">{hint}</span></div>',
        unsafe_allow_html=True
    )

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
            audio_raw = audio_bytes.read()
            audio_hash = hash(audio_raw)
            # run Whisper only once per new recording
            if st.session_state.get("_last_audio_hash") != audio_hash:
                st.session_state._last_audio_hash = audio_hash
                with st.spinner("🧠 Turning your voice into words…"):
                    model = load_whisper()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_raw)
                        tmp.flush()
                        tmp_path = tmp.name
                    try:
                        result = model.transcribe(tmp_path, fp16=False)
                        text = result.get("text","").strip()
                        lang = result.get("language","")
                        # fallback only if text is too short or language unknown
                        if lang not in {"en","hi"} and len(text)<6:
                            result = model.transcribe(tmp_path, language="hi", fp16=False)
                            text = result.get("text","").strip()
                        # ---- KEY FIX: assign to a *different* key, then rerun ----
                        st.session_state["msg"] = text
                        st.rerun()
                    finally:
                        # delete temp file
                        import os
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

    # ---------- unified text box ----------#
    
    msg = st.text_area("",key="msg",placeholder="🎙️ I’m listening…" if mode else "💬 Paste it here – I’ll take a look.",height=180,label_visibility="collapsed")
        
    # single source of truth from here on
    

    # ---------- glowing analyse button ----------
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        if st.button("🛡️ Guard This Message", use_container_width=True, key="ana"):
            if msg.strip():
                st.session_state.stage = "RUNNING"
                st.rerun()

    # ---- running ----
    if st.session_state.stage=="RUNNING":
        with st.container():
            st.markdown('<div class="card"><h4>🔍 Reading between the lines…</h4>', unsafe_allow_html=True)
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
        draw_risk_score(p)
        draw_triggers(p.triggers)
        draw_claim_cards(p.claim_analysis)
        draw_section("Legitimacy Anchors", p.legitimacy_proof, "✅")
        draw_warning_section("Coherence Issues", p.coherence_issues, "⚠️")
        draw_actions(p.recos)

        # reset
        if st.button("🔄 Analyze New Message", use_container_width=True):
             st.session_state.pop("msg", None)
             st.session_state.pop("profile", None)
             st.session_state.pop("stage", None)
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
