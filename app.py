"""
BHARATSCAM GUARDIAN — META-CONSENSUS PROTOCOL EDITION
Utilizes Adversarial Validation & Reputational Cryptoeconomics
"""

# ============================================================
# Core Protocol Imports
# ============================================================
import streamlit as st, torch, torch.nn.functional as F, numpy as np, json, re, time, hashlib, sqlite3, uuid
from pathlib import Path; from dataclasses import dataclass; from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime; from collections import defaultdict; from functools import lru_cache

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Cryptoeconomic Constants (Tamper-evident)
# ============================================================
GENESIS_HASH = "bharat_scam_guardian_genesis_2024_v3"
REPUTATION_SLASH = 0.35  # Malicious reporter penalty
CONSENSUS_THRESHOLD = 0.68  # 68% agreement required
REPORTING_STAKE = 50  # Minimum reputation to report
DECAY_CONSTANT = 0.002  # Exponential decay per hour

# ============================================================
# Zero-Knowledge Pattern Extraction
# ============================================================
def zk_pattern_extract(text: str) -> bytes:
    """Extracts non-reversible pattern fingerprint"""
    # Remove PII while preserving linguistic structure
    sanitized = re.sub(r'\d{6,}', '[NUM]', re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text))
    # Generate structural n-gram signature (3-5 char windows)
    ngrams = [''.join(y) for y in zip(*[sanitized[i:] for i in range(4)])]
    # Create frequency distribution hash
    freq_hash = hashlib.blake2b(str(sorted([(g, ngrams.count(g)) for g in set(ngrams)])).encode(), digest_size=16).digest()
    return freq_hash

# ============================================================
# Reputation-Based Distributed Memory
# ============================================================
@dataclass
class ConsensusEntry:
    pattern_hash: bytes
    first_reported: float
    reporter_ids: List[str]
    reputation_snapshot: List[float]
    challenge_responses: List[Dict[str, bool]]
    cumulative_weight: float
    
class DistributedConsensusLedger:
    def __init__(self, path: Path):
        self.path = path
        self._init_schema()
    
    def _init_schema(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consensus (
                    zk_hash BLOB PRIMARY KEY,
                    first_seen REAL NOT NULL,
                    reporters TEXT NOT NULL,  # JSON array of session IDs
                    reputations TEXT NOT NULL,  # JSON array of scores
                    challenges TEXT NOT NULL,  # JSON array of challenge responses
                    weight REAL DEFAULT 0
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reputation (
                    session_id TEXT PRIMARY KEY,
                    score REAL DEFAULT 25.0,
                    last_active REAL NOT NULL
                )""")
    
    def get_reputation(self, session_id: str) -> float:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT score FROM reputation WHERE session_id=?", (session_id,)).fetchone()
            if not row:
                conn.execute("INSERT INTO reputation VALUES(?, 25.0, ?)", (session_id, time.time()))
                return 25.0
            return row[0]
    
    def update_reputation(self, session_id: str, delta: float):
        with sqlite3.connect(self.path) as conn:
            score = max(0.0, min(100.0, self.get_reputation(session_id) + delta))
            conn.execute("UPDATE reputation SET score=?, last_active=? WHERE session_id=?", 
                        (score, time.time(), session_id))
    
    def submit_challenge(self, zk_hash: bytes, session_id: str, 
                         challenge_verified: Dict[str, bool]) -> bool:
        """Submit proof-of-analysis challenge"""
        reputation = self.get_reputation(session_id)
        if reputation < REPORTING_STAKE:
            return False  # Insufficient stake
            
        with sqlite3.connect(self.path) as conn:
            existing = conn.execute("SELECT * FROM consensus WHERE zk_hash=?", (zk_hash,)).fetchone()
            
            if existing:
                # Update consensus
                reporters = json.loads(existing[2])
                reputations = json.loads(existing[3])
                challenges = json.loads(existing[4])
                
                if session_id in reporters:  # Already reported
                    return False
                    
                reporters.append(session_id)
                reputations.append(reputation)
                challenges.append(challenge_verified)
                
                # Calculate time-decayed weight
                age_hours = (time.time() - existing[1]) / 3600
                decay = np.exp(-DECAY_CONSTANT * age_hours)
                cumulative = sum(r * (1 if c["valid"] else REPUTATION_SLASH) 
                               for r, c in zip(reputations, challenges)) * decay
                
                conn.execute("""
                    UPDATE consensus 
                    SET reporters=?, reputations=?, challenges=?, weight=?
                    WHERE zk_hash=?
                """, (json.dumps(reporters), json.dumps(reputations), 
                      json.dumps(challenges), cumulative, zk_hash))
            else:
                # Create new entry
                conn.execute("""
                    INSERT INTO consensus VALUES(?, ?, ?, ?, ?, ?)
                """, (zk_hash, time.time(), json.dumps([session_id]),
                      json.dumps([reputation]), json.dumps([challenge_verified]),
                      reputation))
        
        return True
    
    def is_validated_safe(self, zk_hash: bytes) -> bool:
        """Check if pattern has reached consensus"""
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT weight, first_seen FROM consensus WHERE zk_hash=?", 
                             (zk_hash,)).fetchone()
            if not row:
                return False
                
            age_hours = (time.time() - row[1]) / 3600
            decayed_weight = row[0] * np.exp(-DECAY_CONSTANT * age_hours)
            return decayed_weight >= CONSENSUS_THRESHOLD * 100  # Normalized

# ============================================================
# Challenge Generation Engine
# ============================================================
class ChallengeEngine:
    def __init__(self, model_outputs: Dict[str, float], text: str):
        self.triggers = {k: v for k, v in model_outputs.items() if v > 0.4}
        self.text = text
        
    def generate_challenge(self) -> Tuple[Dict[str, bool], List[str]]:
        """Generate adversarial challenge questions"""
        # Extract actual scam signals present
        real_signals = {}
        decoy_signals = {}
        
        # Authority indicators
        real_signals["authority"] = bool(re.search(r'\b(government|sbi|rbi|irs|police|court)\b', self.text, re.I))
        decoy_signals["authority"] = ["bank", "official", "department"]
        
        # Urgency indicators
        real_signals["urgency"] = bool(re.search(r'\b(immediately|within 24h|urgent|now|last chance)\b', self.text, re.I))
        decoy_signals["urgency"] = ["soon", "please", "whenever"]
        
        # Financial request
        real_signals["financial"] = bool(re.search(r'\b(upi|paytm|otp|cvv|deposit)\b', self.text, re.I))
        decoy_signals["financial"] = ["account", "balance", "statement"]
        
        # Isolation
        real_signals["isolation"] = bool(re.search(r'\b(do not tell|secret|alone|confidential)\b', self.text, re.I))
        decoy_signals["isolation"] = ["private", "personal", "yourself"]
        
        # Build challenge: user must correctly identify which signals ACTUALLY exist
        challenge_map = {}
        questions = []
        
        for signal_type, exists in real_signals.items():
            challenge_map[signal_type] = exists
            # Mix real and decoy examples
            examples = [signal_type] if exists else decoy_signals[signal_type]
            questions.append(f"Does this message contain '{examples[0]}' language?")
            
        return challenge_map, questions

# ============================================================
# Model Loader (Deterministic)
# ============================================================
@st.cache_resource
def load_cpaft():
    files = ["config.json", "model.safetensors", "tokenizer.json", 
             "tokenizer_config.json", "special_tokens_map.json", 
             "vocab.json", "merges.txt", "scam_v1.json"]
    
    LOCAL_DIR = Path("./hf_cpaft_v3")
    LOCAL_DIR.mkdir(exist_ok=True)
    
    for f in files:
        hf_hub_download(REPO_ID := "prakhar146/scam", f, repo_type="dataset",
                        local_dir=LOCAL_DIR, local_dir_use_symlinks=False)
    
    tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR)
    mdl.to(DEVICE := "cuda" if torch.cuda.is_available() else "cpu").eval()
    
    with open(LOCAL_DIR / "scam_v1.json") as f:
        cal = json.load(f)
    
    return tok, mdl, float(cal["temperature"]), np.array(cal["thresholds"])

# ============================================================
# Risk Orchestrator with Meta-Validation
# ============================================================
@dataclass
class RiskProfile:
    score: float
    level: str
    confidence: float
    triggers: Dict[str, float]
    recommendations: List[str]
    zk_hash: bytes
    challenge: Optional[ChallengeEngine] = None

class MetaConsensusOrchestrator:
    def __init__(self, temperature, thresholds):
        self.T = temperature
        self.thresholds = thresholds
        self.ledger = DistributedConsensusLedger(Path("./consensus_v3.db"))
        
    def infer(self, text: str) -> RiskProfile:
        zk_hash = zk_pattern_extract(text)
        
        # Check meta-consensus first
        if self.ledger.is_validated_safe(zk_hash):
            return RiskProfile(
                score=5.0,  # Near-zero but non-zero for transparency
                level="SAFE",
                confidence=99.7,
                triggers={},
                recommendations=["✅ Community-verified safe pattern (consensus achieved)"],
                zk_hash=zk_hash
            )
        
        # Standard inference
        tok, mdl, _, _ = load_cpaft()
        inputs = tok(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        
        with torch.no_grad():
            logits = mdl(**inputs).logits / self.T
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        detected = probs > self.thresholds
        base = probs[detected].mean() if detected.any() else probs.max() * 0.25
        
        # Composite scoring
        entity_boost = self._entity_score(text) * 0.18
        psych_boost = self._psych_score(text) * 0.28
        
        final = min(base + entity_boost + psych_boost, 1.0)
        level = ("SAFE" if final < 0.2 else "CAUTION" if final < 0.4 
                else "SUSPICIOUS" if final < 0.6 else "SCAM")
        
        triggers = {label: float(probs[i]) for i, label in enumerate(CP_AFT_LABELS) if detected[i]}
        
        return RiskProfile(
            score=round(final * 100, 2),
            level=level,
            confidence=round((1 - np.std(probs)) * 100, 2),
            triggers=triggers,
            recommendations=self._generate_recos(level, final),
            zk_hash=zk_hash,
            challenge=ChallengeEngine(triggers, text) if final > 0.15 else None
        )
    
    def _entity_score(self, text: str) -> float:
        return min(len(re.findall(r'\b(upi|otp|@paytm|cvv|\d{10,12})\b', text, re.I)) / 5.0, 1.0)
    
    def _psych_score(self, text: str) -> float:
        fear = len(re.findall(r'\b(arrest|freeze|court|terminate)\b', text, re.I))
        urgency = len(re.findall(r'\b(immediately|24h|urgent|last.chance)\b', text, re.I))
        isolation = len(re.findall(r'\b(do.not.tell|secret|confidential)\b', text, re.I))
        return min((fear * 0.4 + urgency * 0.35 + isolation * 0.25) / 4, 1.0)
    
    def _generate_recos(self, level: str, score: float):
        recos = {
            "SCAM": [("🚨 CRITICAL: Do NOT engage", "primary"), 
                    ("📞 Emergency: Dial 1930", "emergency"),
                    ("🔒 Freeze all financial accounts", "secondary")],
            "SUSPICIOUS": [("⚠️ Verify via official channels only", "primary"),
                          ("📵 Block sender immediately", "secondary")],
            "CAUTION": [("⏸️ Pause and independently verify", "primary")],
            "SAFE": [("✅ No action required", "information")]
        }
        return recos[level]

# ============================================================
# Session Identity Manager
# ============================================================
def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.sha256(
            f"{st.runtime.scriptrunner.get_remote_ip()}|{time.time()}".encode()
        ).hexdigest()[:16]
    return st.session_state.session_id

# ============================================================
# Streamlit UI (Invisible Complexity)
# ============================================================
COLORS = {"SAFE": "#2D936C", "CAUTION": "#F4A261", 
          "SUSPICIOUS": "#E76F51", "SCAM": "#C1121C"}

def main():
    st.set_page_config(page_title="BharatScam Guardian", page_icon="🛡️", 
                      layout="centered", initial_sidebar_state="collapsed")
    
    if "ledger" not in st.session_state:
        st.session_state.ledger = DistributedConsensusLedger(Path("./consensus_v3.db"))
    
    session_id = get_session_id()
    reputation = st.session_state.ledger.get_reputation(session_id)
    
    # Header with reputation
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#003049 0%,#005f73 100%);color:white;padding:2rem;border-radius:12px;'>
        <h1>🛡️ BharatScam Guardian</h1>
        <p>Reputation Score: {reputation:.1f}/100</p>
    </div>""", unsafe_allow_html=True)
    
    # Input
    msg = st.text_area("📨 Paste message", height=200, 
                      placeholder="Paste suspicious message here...")
    
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not msg.strip():
            st.error("Message required")
            return
        st.session_state.msg = msg
        st.session_state.stage = "ANALYZING"
        st.rerun()
    
    # Analysis flow
    if st.session_state.get("stage") == "ANALYZING":
        with st.spinner(""):
            time.sleep(1.5)  # Simulated processing
            
        orch = MetaConsensusOrchestrator(*load_cpaft()[2:])
        profile = orch.infer(st.session_state.msg)
        st.session_state.profile = profile
        st.session_state.stage = "RESULTS"
        st.rerun()
    
    # Results
    if st.session_state.get("stage") == "RESULTS":
        p = st.session_state.profile
        
        # Hero display
        color = COLORS[p.level]
        st.markdown(f"""
        <div style='background:{color};color:white;padding:2.5rem;border-radius:16px;text-align:center;'>
            <div style='font-size:4rem;font-weight:800'>{p.score}%</div>
            <div style='font-size:1.5rem;font-weight:600'>{p.level}</div>
            <div>Confidence: {p.confidence}%</div>
        </div>""", unsafe_allow_html=True)
        
        # Triggers
        if p.triggers:
            st.markdown("### 🎯 Detected Tactics")
            for trig, prob in sorted(p.triggers.items(), key=lambda x: x[1], reverse=True):
                st.markdown(f"{'🔴' if prob > 0.7 else '🟡'} **{trig}** — {prob:.1%}")
        
        # Actions
        st.markdown("### 🎯 Recommended Actions")
        for text, type in p.recommendations:
            if type == "emergency":
                st.markdown(f'<a href="tel:1930" style="text-decoration:none;"><div style="background:{COLORS["SCAM"]};color:white;padding:1rem;border-radius:8px;text-align:center;font-weight:600;">{text}</div></a>', unsafe_allow_html=True)
            else:
                st.button(text, key=text, use_container_width=True)
        
        # False positive reporting with challenge
        if p.level != "SAFE" and p.challenge:
            with st.expander("🤔 This is NOT a scam? Prove it"):
                st.info("To prevent abuse, you must pass a validation challenge.")
                
                # Display challenge questions
                correct_map, questions = p.challenge.generate_challenge()
                
                # Create checkboxes for each signal type
                user_responses = {}
                for q in questions:
                    signal_type = q.split("'")[1]
                    user_responses[signal_type] = st.checkbox(f"❓ {q}", key=f"chk_{signal_type}")
                
                if st.button("✅ Submit Proof", disabled=reputation < REPORTING_STAKE):
                    if reputation < REPORTING_STAKE:
                        st.error(f"Reputation {reputation:.1f} < Required {REPORTING_STAKE}")
                        return
                    
                    # Validate responses
                    is_correct = all(user_responses[k] == v for k, v in correct_map.items())
                    
                    if is_correct:
                        # Submit to ledger
                        success = st.session_state.ledger.submit_challenge(
                            p.zk_hash, session_id, {"valid": True}
                        )
                        if success:
                            st.session_state.ledger.update_reputation(session_id, +8)  # Reward
                            st.success("✅ Consensus recorded! +8 reputation")
                            time.sleep(2)
                            st.rerun()
                    else:
                        st.session_state.ledger.update_reputation(session_id, -15)  # Slash
                        st.error("❌ Incorrect analysis. -15 reputation. Please re-read the message.")
        
        if st.button("🔄 New Analysis"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
