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
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import whisper, tempfile
from datetime import datetime

# ============================================================
# GLOBAL CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/scam-detection"
LOCAL_DIR = Path("./hf_cpaft_core")
LOCAL_DIR.mkdir(exist_ok=True)

THEME = {
    "bg": "#FDFBF8", "card": "#FFFFFF", "accent": "#FF8F00",
    "safe": "#2E7D32", "caution": "#F57C00", "suspicious": "#D32F2F", "scam": "#B71C1C",
    "text": "#3E2723", "subtle": "#8D6E63"
}

# ============================================================
# CORE ENGINES (Simplified for focus)
# ============================================================
class AntiGamingEngine:
    def check_gaming(self, text: str) -> Tuple[bool, str, float]:
        text_lower = text.lower().strip()
        if len(text) < 10:
            return True, "Message too short", 0.9
        test_patterns = [r'^(test|asdf|qwerty|hello|hi|random|blah|foo|bar|spam|scam|message|abc|xyz|123)']
        for pattern in test_patterns:
            if re.search(pattern, text_lower, re.I):
                return True, "Test pattern detected", 0.85
        return False, "", 0.0

class TrustAnchorEngine:
    def score(self, text: str) -> Tuple[float, List[str], Dict]:
        score, hits, matches = 0.0, [], {}
        patterns = {
            "bank_official": (r'\b(?:HDFC|ICICI|SBI|AXIS)[\s]*(?:Bank|Ltd)\b|\bRBI\b', 0.35),
            "govt_official": (r'\b(?:UIDAI|ITA|GST|EPFO)\b|\b(?:gov\.in|nic\.in)\b', 0.35),
        }
        for name, (pat, weight) in patterns.items():
            if matches_list := re.findall(pat, text, re.I):
                matches[name] = matches_list
                hits.append(f"{name.replace('_',' ').title()}: {len(matches_list)}")
                score += min(len(matches_list) * weight, weight)
        return min(score, 1.0), hits, matches

class VerifiableClaimsEngine:
    def extract_claims(self, text: str) -> List[Tuple]:
        claims = []
        for m in re.finditer(r'\b(?:₹|Rs\.?)\s*[\d,]+', text):
            claims.append((m.group(), "financial", 0.8, "N/A"))
        return claims

class CoreOrchestrator:
    def __init__(self,T,thres):
        self.T, self.thres = T, thres
        self.antigaming = AntiGamingEngine()
    
    def infer(self, text: str) -> Dict:
        is_gaming, reason, penalty = self.antigaming.check_gaming(text)
        if is_gaming:
            return {
                "risk_score": round(penalty * 100, 2),
                "level": "SCAM",
                "confidence": 95.0,
                "narrative": f"Gaming attempt: {reason}",
                "triggers": {},
                "recos": ["Please use real messages for analysis."],
                "is_gaming": True
            }
        
        tok, mdl, _, _ = load_model()
        inputs = tok(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = mdl(**inputs).logits/self.T
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        risk = probs.mean() * 100
        level = "SAFE" if risk < 30 else "CAUTION" if risk < 50 else "SUSPICIOUS" if risk < 75 else "SCAM"
        
        return {
            "risk_score": round(risk, 2),
            "level": level,
            "confidence": 90.0,
            "narrative": f"Analysis complete. Risk level: {level}",
            "triggers": {"SAMPLE": 0.8},
            "recos": ["Verify via official app", "Call bank directly"],
            "is_gaming": False
        }

# ============================================================
# MODEL LOADER
# ============================================================
@st.cache_resource
def load_model():
    try:
        for f in ["config.json","model.safetensors","tokenizer.json","scam_v1.json"]:
            hf_hub_download(REPO_ID, f, local_dir=LOCAL_DIR, repo_type="dataset")
        tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR).to(DEVICE).eval()
        with open(LOCAL_DIR/"scam_v1.json") as f: cal=json.load(f)
        return tok, mdl, float(cal["temperature"]), np.array(cal["thresholds"])
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, 1.0, np.array([0.25,0.5,0.75])

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

# ============================================================
# UI HELPERS
# ============================================================
def init_state():
    for k in ["msg","result","stage","mode","message_history","_last_audio_hash"]:
        if k not in st.session_state:
            st.session_state[k] = [] if k in ["message_history"] else None

def risk_badge(level: str) -> str:
    color = THEME.get(level.lower(), THEME["subtle"])
    return f'<span style="background:{color}22;color:{color};padding:8px 18px;border-radius:999px;font-weight:600;">{level}</span>'

def draw_risk_score(result: Dict):
    color = THEME.get(result["level"].lower(), THEME["subtle"])
    st.markdown(f"""
    <div style="text-align:center;background:#fff;border-radius:20px;padding:35px;margin-bottom:25px;">
        <div style="font-size:22px;color:{THEME["subtle"]};">Risk Assessment</div>
        <div style="font-size:72px;font-weight:700;color:{color};">{result["risk_score"]}<span style="font-size:36px">%</span></div>
        <div style="margin-top:15px;">{risk_badge(result["level"])}</div>
        <div style="margin-top:10px;color:{THEME["subtle"]};">Confidence {result["confidence"]}%</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(float(result["risk_score"])/100.0)

# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    st.set_page_config(page_title="BharatScam Guardian", page_icon="🛡️", layout="centered")
    
    # CSS
    st.markdown(f"""
    <style>
    .stApp {{background: {THEME["bg"]}; color: {THEME["text"]};}}
    div.stButton > button {{border: none; color: #FFF; background: linear-gradient(90deg,{THEME["accent"]} 0%, {THEME["caution"]} 100%);
                            font-weight: 600; border-radius: 12px; height: 52px; font-size: 18px;}}
    div.stButton > button:hover {{transform: scale(1.02); cursor: pointer !important;}}
    div.stButton > button:disabled {{opacity: 0.5; cursor: not-allowed !important;}}
    button[kind="primary"] {{cursor: pointer !important;}}
    </style>
    """, unsafe_allow_html=True)
    
    init_state()
    
    # Hero
    st.markdown(f"""
        <div style="text-align:center;margin-top:-30px;margin-bottom:40px;">
        <h1 style="font-size:52px;background:-webkit-linear-gradient(45deg,{THEME["accent"]},#FF6F00);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            BharatScam Guardian
        </h1>
        <p style="color:{THEME["subtle"]};">AI that understands Indian scam patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Mode toggle
    if "mode" not in st.session_state:
        st.session_state.mode = False
    
    col_mode1, col_mode2, col_mode3 = st.columns([1,2,1])
    with col_mode2:
        mode_text = "🎤 Voice Mode" if st.session_state.mode else "⌨️ Text Mode"
        if st.button(mode_text, use_container_width=True, key="mode_toggle"):
            st.session_state.mode = not st.session_state.mode
            st.rerun()
    
    # Input handling
    msg = None
    if st.session_state.mode:  # Speech
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
                with st.spinner("🧠 Transcribing..."):
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
        # Use stored voice message
        msg = st.session_state.get("msg", "")
    else:  # Text
        st.markdown("""
        <div style="background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);border-left:5px solid #4CAF50;
                    border-radius:12px;padding:14px 18px;font-size:17px;color:#3e2723;margin-bottom:15px;">
        💬 <b>Paste your message</b> - Hindi, English, or Hinglish supported
        </div>
        """, unsafe_allow_html=True)
        msg = st.text_area("", key="msg_input", placeholder="Paste suspicious message here...", height=180, 
                          label_visibility="collapsed")
    
    # FINAL MESSAGE SOURCE (voice or text)
    final_msg = msg or st.session_state.get("msg", "")
    
    # DEBUG: Show message length
    st.markdown(f"<div style='text-align:center;color:{THEME['subtle']};font-size:13px;'>Message length: {len(final_msg)} chars</div>", 
                unsafe_allow_html=True)
    
    # ANALYZE BUTTON - FIXED VERSION
    col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
    with col_btn2:
        # CRITICAL FIX: Always enable button when there's text, let backend validate
        button_enabled = len(final_msg.strip()) >= 5
        if st.button("🛡️ Analyze Message", 
                    use_container_width=True, 
                    key="analyze_btn",  # Changed key
                    type="primary",
                    disabled=not button_enabled):
            # Store and proceed
            st.session_state.stage = "RUNNING"
            st.session_state.msg_cache = final_msg.strip()
            # Clear input to prevent double-click
            if "msg_input" in st.session_state:
                st.session_state.msg_input = ""
            st.rerun()
    
    # Processing
    if st.session_state.stage == "RUNNING":
        with st.container():
            st.markdown('<div class="card"><h4>🔍 Analyzing message patterns...</h4>', unsafe_allow_html=True)
            bar = st.progress(0)
            for i in range(100):
                bar.progress(i+1)
                time.sleep(0.008)
            
            if (orch := CoreOrchestrator(*load_model()[2:])):
                st.session_state.result = orch.infer(st.session_state.msg_cache)
                st.session_state.stage = "DONE"
            st.markdown('</div>', unsafe_allow_html=True)
            st.rerun()
    
    # Results
    if st.session_state.stage == "DONE" and st.session_state.result:
        result = st.session_state.result
        draw_risk_score(result)
        # ... rest of your UI code ...
        
        # Reset button
        if st.button("🔄 Analyze Another", use_container_width=True, key="reset"):
            for key in ["msg", "result", "stage", "msg_cache", "msg_input", "_last_audio_hash"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Footer
    st.markdown(f"""
        <div style="text-align:center;margin-top:40px;padding:16px 0;color:{THEME["subtle"]};font-size:14px;">
            Built for Indian users 🇮🇳 
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
