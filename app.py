import streamlit as st
import torch
import json
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

# --------------------------------------------------
# App Configuration & Indian Context Setup
# --------------------------------------------------
st.set_page_config(
    page_title="🛡️ BharatScam Guard - Advanced Fraud Detector", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

# Indian-specific scam patterns and keywords
INDIAN_SCAM_PATTERNS = {
    'bank_kyc': {
        'patterns': [
            r'kyc.*expire|expire.*kyc',
            r'(?:sbi|hdfc|icici|axis|paytm|phonepe|google pay).*(?:update|verify|suspend)',
            r'(?:account|wallet).*blocked.*update',
            r'dear customer.*(?:bank|paytm)'
        ],
        'weight': 2.5,
        'description': 'Bank/Payment KYC Scam'
    },
    'digital_arrest': {
        'patterns': [
            r'digital arrest',
            r'(?:cbi|narcotics|cyber crime|fedex).*(?:officer|case|id)',
            r'parcel.*(?:drug|illegal|customs)',
            r'video call.*(?:investigation|officer)'
        ],
        'weight': 3.0,
        'description': 'Digital Arrest Impersonation'
    },
    'lottery_prize': {
        'patterns': [
            r'(?:crore|lakh).*lottery',
            r'(?:kbc|kaun banega crorepati|whatsapp lottery)',
            r'(?:winner|prize).*amount',
            r'(?:govt|government).*lottery'
        ],
        'weight': 2.0,
        'description': 'Fake Lottery/Prize Scam'
    },
    'otp_fraud': {
        'patterns': [
            r'(?:share|send|provide|otp|password|cvv)',
            r'(?:never share|do not share).*otp',
            r'(?:code|otp).*verification.*immediately'
        ],
        'weight': 2.8,
        'description': 'OTP/Credentials Phishing'
    },
    'job_fraud': {
        'patterns': [
            r'(?:work from home|part time job|earn.*(?:thousand|lakh))',
            r'(?:data entry|typing job).*advance',
            r'(?:refund|registration|training).*fee'
        ],
        'weight': 2.2,
        'description': 'Fake Job Offer Scam'
    },
    'government_impersonation': {
        'patterns': [
            r'(?:pm modi|MODI|govt of india).*scheme',
            r'(?:income tax|pf|epfo|govt subsidy)',
            r'(?:1[0-9]{11,12})',  # Fake Aadhar pattern
            r'aadhar.*(?:link|update|suspend)'
        ],
        'weight': 2.7,
        'description': 'Government Authority Scam'
    }
}

# Entity patterns for Indian context
ENTITY_PATTERNS = {
    'phone': r'(?:\+91|0)?[6-9]\d{9}',
    'urgency_words': r'(?:immediately|urgent|now|within.*(?:minutes|hours)|last chance|final notice)',
    'payment_urgency': r'(?:pay|transfer|send|deposit).*immediately',
    'upi_id': r'[\w.-]+@[\w.-]+',
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/Scam"
LABELS = [
    "authority_name",
    "threat_type", 
    "time_pressure",
    "payment_method",
    "language_mixing"
]

# Color scheme for Indian theme
COLORS = {
    'SAFE': '#28a745',
    'CAUTION': '#ffc107', 
    'SUSPICIOUS': '#fd7e14',
    'SCAM': '#dc3545',
    'primary': '#007bff',
    'secondary': '#6c757d'
}

# --------------------------------------------------
# Advanced Model & Pattern Engine
# --------------------------------------------------
class BharatScamDetector:
    def __init__(self, model, tokenizer, temperature: float, base_thresholds: np.ndarray):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.base_thresholds = base_thresholds
        self.risk_cache = {}

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities = {}
        text_lower = text.lower()
        for entity_type, pattern in ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        return entities

    def calculate_pattern_score(self, text: str) -> Tuple[float, List[Dict]]:
        text_lower = text.lower()
        pattern_matches = []
        total_score = 0
        for scam_type, data in INDIAN_SCAM_PATTERNS.items():
            for pattern in data['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    total_score += data['weight']
                    pattern_matches.append({
                        'type': scam_type,
                        'description': data['description'],
                        'weight': data['weight']
                    })
                    break
        return total_score, pattern_matches

    def adaptive_thresholding(self, probs: np.ndarray) -> np.ndarray:
        mean_conf = probs.mean()
        std_conf = probs.std()
        dynamic = self.base_thresholds.copy()
        if mean_conf > 0.4:
            dynamic -= 0.1
        elif mean_conf < 0.15:
            dynamic += 0.05
        if std_conf > 0.3:
            dynamic -= 0.08
        return np.clip(dynamic, 0.2, 0.75)

    def calculate_combination_multiplier(self, detected_labels: List[str]) -> float:
        label_set = set(detected_labels)
        if 'authority_name' in label_set and 'threat_type' in label_set:
            return 1.5
        if 'time_pressure' in label_set and 'payment_method' in label_set:
            return 1.4
        if 'authority_name' in label_set and 'time_pressure' in label_set and 'payment_method' in label_set:
            return 1.8
        return 1.0

    def predict(self, text: str) -> Dict:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.risk_cache:
            return self.risk_cache[text_hash]

        # Model inference
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits / self.temperature
            base_probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Define mean_conf to fix NameError
        mean_conf = base_probs.mean()

        # Adaptive thresholding
        thresholds = self.adaptive_thresholding(base_probs)
        detected = [LABELS[i] for i, v in enumerate(base_probs > thresholds) if v]

        # Pattern detection
        pattern_score, pattern_matches = self.calculate_pattern_score(text)
        entities = self.extract_entities(text)

        # Base risk calculation
        if len(detected) == 0:
            base_risk = 0
        elif len(detected) == 1:
            base_risk = np.max(base_probs) * 30
        else:
            base_risk = (np.sum(base_probs) / len(base_probs)) * 50

        # Combination multiplier
        combo_multiplier = self.calculate_combination_multiplier(detected)

        # Pattern bonus
        pattern_bonus = min(pattern_score * 12, 40)

        # Urgency penalty
        urgency_penalty = 0
        if re.search(ENTITY_PATTERNS['urgency_words'], text, re.IGNORECASE):
            urgency_penalty = 15

        # Final risk score
        risk_score = (base_risk * combo_multiplier) + pattern_bonus + urgency_penalty
        risk_score = min(risk_score, 100)

        # Verdict
        if risk_score < 25:
            verdict = "🟢 SAFE"
            risk_level = "SAFE"
        elif risk_score < 45:
            verdict = "🟡 CAUTION"
            risk_level = "CAUTION"
        elif risk_score < 70:
            verdict = "🟠 SUSPICIOUS"
            risk_level = "SUSPICIOUS"
        else:
            verdict = "🔴 CONFIRMED SCAM"
            risk_level = "SCAM"

        # Confidence calculation using mean_conf
        confidence = min((mean_conf * 100) + (pattern_score * 10), 95) if detected else mean_conf * 100

        result = {
            'verdict': verdict,
            'risk_level': risk_level,
            'risk_score': round(risk_score, 2),
            'confidence': round(confidence, 2),
            'detected_labels': detected,
            'probabilities': {lbl: float(p) for lbl, p in zip(LABELS, base_probs)},
            'pattern_matches': pattern_matches,
            'entities': entities,
            'thresholds_used': {lbl: float(t) for lbl, t in zip(LABELS, thresholds)}
        }

        self.risk_cache[text_hash] = result
        return result

# --------------------------------------------------
# Download & Load Model
# --------------------------------------------------
@st.cache_resource(show_spinner="🚀 Initializing BharatScam Guard...")
def load_detector():
    LOCAL_DIR = Path("./hf_model")
    LOCAL_DIR.mkdir(exist_ok=True)
    
    REQUIRED_FILES = [
        "config.json", "model.safetensors", "tokenizer.json", 
        "tokenizer_config.json", "special_tokens_map.json", 
        "vocab.json", "merges.txt", "scam_v1.json"
    ]
    
    for file in REQUIRED_FILES:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=file,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
    
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR)
    model.to(DEVICE)
    model.eval()
    
    with open(LOCAL_DIR / "scam_v1.json", "r") as f:
        cal = json.load(f)
    
    temperature = float(cal.get("temperature", 1.0))
    thresholds = np.array(cal.get("thresholds", [0.5] * model.config.num_labels))
    
    return BharatScamDetector(model, tokenizer, temperature, thresholds)

# --------------------------------------------------
# UI Components
# --------------------------------------------------
def display_risk_score(score: float, level: str):
    """Display risk score with visual indicator"""
    color = COLORS[level]
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {color} {score}%, #f0f2f6 {score}%); 
                padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h2 style="color: white; text-align: center; margin: 0;">
            Risk Score: {score}% ({level})
        </h2>
    </div>
    """, unsafe_allow_html=True)

def display_dimension_analysis(detected: List[str], probs: Dict, thresholds: Dict):
    """Show detailed breakdown of each dimension"""
    st.markdown("### 🔍 Dimension Analysis")
    
    for label in LABELS:
        prob = probs[label]
        threshold = thresholds[label]
        is_detected = label in detected
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            status = "🚨 Detected" if is_detected else "✅ Not Found"
            st.write(f"**{label.replace('_', ' ').title()}** - {status}")
        
        with col2:
            st.progress(min(prob, 1.0))
        
        with col3:
            st.write(f"{prob:.3f} (threshold: {threshold:.3f})")
        
        if is_detected:
            st.info(get_dimension_explanation(label))

def get_dimension_explanation(label: str) -> str:
    """Explain what each dimension means in Indian context"""
    explanations = {
        'authority_name': "Claims to be from RBI, SBI, CBI, or Police",
        'threat_type': "Threatens legal action, account block, or arrest",
        'time_pressure': "Creates urgency ('within 24 hours', 'immediately')",
        'payment_method': "Asks for UPI, wallet transfer, or gift cards",
        'language_mixing': "Mixes English with Hindi/regional languages"
    }
    return explanations.get(label, "")

def get_action_recommendations(result: Dict) -> List[Dict]:
    """Provide tailored action steps based on risk level"""
    risk_level = result['risk_level']
    
    if risk_level == 'SAFE':
        return [{"action": "No action needed", "priority": "low"}]
    
    elif risk_level == 'CAUTION':
        return [
            {"action": "Verify sender identity independently", "priority": "medium"},
            {"action": "Do not click any links", "priority": "high"},
            {"action": "Check for spelling/grammar errors", "priority": "low"}
        ]
    
    elif risk_level == 'SUSPICIOUS':
        return [
            {"action": "⛔ DO NOT respond or click links", "priority": "critical"},
            {"action": "Block the sender immediately", "priority": "high"},
            {"action": "Verify through official website/branch", "priority": "high"},
            {"action": "Never share OTP/passwords", "priority": "critical"}
        ]
    
    else:  # SCAM
        return [
            {"action": "🚨 IMMEDIATELY DELETE & BLOCK", "priority": "critical"},
            {"action": "Report to Cyber Crime (1930)", "priority": "critical"},
            {"action": "Report to NFCS @ cybercrime.gov.in", "priority": "high"},
            {"action": "Warn family & friends about this pattern", "priority": "medium"}
        ]

def main():
    # Sidebar
    with st.sidebar:
        st.title("🇮🇳 BharatScam Guard")
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "Advanced AI detector trained on real Indian scam patterns. "
            "Uses deep learning + pattern matching to identify fraud."
        )
        st.markdown("### Emergency Contacts")
        st.error(
            "🚨 **Cyber Crime Helpline: 1930**\n\n"
            "📧 **NFCS Portal: cybercrime.gov.in**"
        )
        
        # Example selector
        st.markdown("### Try Examples")
        examples = {
            "KYC Scam": "Dear Customer, Your SBI KYC has expired. Click link to verify or account will be blocked in 24 hrs.",
            "Digital Arrest": "I am Inspector Rajesh from CBI. Your Aadhar linked to drug case. Pay 50,000 fine or face digital arrest.",
            "Safe Message": "Hi, dinner at 8 PM? Let me know if you can make it.",
            "Job Scam": "Earn 50,000/month from home! Data entry job. Pay 2000 registration fee to start immediately."
        }
        
        selected_example = st.selectbox("Load Example", ["Select..."] + list(examples.keys()))
        if selected_example != "Select...":
            st.session_state['example_text'] = examples[selected_example]

    # Main UI
    st.markdown("### 🛡️ Enter Message for Analysis")
    
    # Text area with example support
    default_text = st.session_state.get('example_text', '')
    user_text = st.text_area(
        "Paste SMS, WhatsApp, or Email message:",
        value=default_text,
        height=150,
        placeholder="e.g., 'Dear Customer, Your Paytm KYC has expired. Click here to update...'",
        key="message_input"
    )
    
    # Clear example if user types
    if user_text != st.session_state.get('example_text', ''):
        st.session_state.pop('example_text', None)

    # Analyze button
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        analyze_clicked = st.button("🔍 Analyze Message", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_text.strip() or len(user_text) < 10:
            st.warning("⚠️ Please enter a meaningful message (at least 10 characters).")
            return
        
        # Process message
        with st.spinner("🤖 Analyzing message patterns..."):
            detector = load_detector()
            result = detector.predict(user_text.strip())
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Risk Score
        display_risk_score(result['risk_score'], result['risk_level'])
        
        # Verdict
        st.markdown(f"""
        <div style="background-color: {COLORS[result['risk_level']]}20; 
                    border-left: 5px solid {COLORS[result['risk_level']]};
                    padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3 style="margin: 0; color: {COLORS[result['risk_level']]};">
                {result['verdict']} (Confidence: {result['confidence']}%)
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Columns for details
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("### 📈 Risk Breakdown")
            st.metric("ML Model Score", f"{np.mean(list(result['probabilities'].values())):.3f}")
            st.metric("Pattern Score", f"{sum([m['weight'] for m in result['pattern_matches']]):.1f}")
            st.metric("Final Risk", f"{result['risk_score']}%")
        
        with col_right:
            st.markdown("### 🎯 Key Indicators")
            for match in result['pattern_matches']:
                st.error(f"🚨 {match['description']} (+{match['weight']})")
            
            if 'phone' in result['entities']:
                st.warning(f"📞 Phone found: {result['entities']['phone'][:3]}")
            if 'upi_id' in result['entities']:
                st.warning(f"💳 UPI ID found: {result['entities']['upi_id'][:2]}")
        
        # Dimension analysis
        with st.expander("🔬 Detailed Dimension Analysis", expanded=True):
            display_dimension_analysis(
                result['detected_labels'], 
                result['probabilities'], 
                result['thresholds_used']
            )
        
        # Action recommendations
        with st.expander("⚡ What Should You Do?", expanded=True):
            actions = get_action_recommendations(result)
            for action in actions:
                if action['priority'] == 'critical':
                    st.error(f"**{action['action']}**")
                elif action['priority'] == 'high':
                    st.warning(f"**{action['action']}**")
                else:
                    st.info(f"**{action['action']}**")
        
        # Raw probabilities for advanced users
        with st.expander("📊 Technical Details (Experts)"):
            st.json({
                'probabilities': result['probabilities'],
                'thresholds': result['thresholds_used'],
                'detected_labels': result['detected_labels']
            })

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #6c757d;'>"
        "🛡️ BharatScam Guard - Protecting India from Digital Fraud<br>"
        "Built with advanced AI & Indian scam intelligence"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
