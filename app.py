"""
PhD-Level BharatScam Guard – Fixed Edition
KeyError resolved: dynamic weight construction + safe fallback
"""

import streamlit as st
import torch, torch.nn.functional as F
import numpy as np, pandas as pd, json, math, re, os, hashlib, pickle, time, itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
from scipy import stats
import networkx as nx
from datetime import datetime

# --------------------------------------------------
# PhD-Level Configuration & Constants
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/Scam"
LOCAL_DIR = Path("./hf_model_ph")
LOCAL_DIR.mkdir(exist_ok=True)

LABELS = ["authority_name", "threat_type", "time_pressure", "payment_method", "language_mixing"]

# Safe Bayesian priors (dict keyed exactly as LABELS)
SCAM_PRIORS = {
    "base_rate": 0.08,
    "authority_name": 2.74,
    "threat_type": 3.12,
    "time_pressure": 2.89,
    "payment_method": 3.45,
    "language_mixing": 1.98
}

# Multilingual patterns (same as before)
MULTILINGUAL_SCAM_PATTERNS = {
    'en': {
        'digital_arrest': [r'digital arrest', r'cbi.*officer', r'narcotics.*bureau', r'fedex.*case'],
        'kyc': [r'kyc.*expir', r'paytm.*suspend', r'sbi.*update.*kyc', r'account.*block.*kyc'],
        'lottery': [r'(?:crore|lakh).*lottery', r'kbc.*winner', r'whatsapp.*lottery.*prize'],
        'otp': [r'never.*share.*otp', r'share.*otp.*immediately', r'verification.*code.*urgent'],
        'job': [r'work.*home.*(?:thousand|lakh)', r'data.*entry.*advance.*fee', r'earn.*(?:50k|1lakh).*month'],
        'government': [r'pm.*modi.*scheme', r'income.*tax.*refund', r'pf.*withdrawal.*link']
    },
    'hi': {
        'digital_arrest': [r'डिजिटल.*गिरफ्तार', r'सीबीआई.*अधिकारी', r'नारकोटिक्स.*ब्यूरो'],
        'kyc': [r'केवाईसी.*समाप्त', r'पेटीएम.*निलंबित', r'खाता.*ब्लॉक.*केवाईसी'],
        'lottery': [r'लॉटरी.*करोड़', r'केबीसी.*विजेता', r'व्हाट्सप्प.*इनाम'],
        'otp': [r'ओटीपी.*साझा.*नहीं', r'तुरंत.*ओटीपी', r'सत्यापन.*कोड.*तत्काल'],
        'job': [r'घर.*काम.*लाख', r'डाटा.*एंट्री.*अग्रिम', r'महीना.*कमाएं.*लाख'],
        'government': [r'पीएम.*मोदी.*योजना', r'आयकर.*वापसी', r'पीएफ.*निकासी']
    },
    'mr': {
        'digital_arrest': [r'डिजिटल.*अटक', r'सीबीआय.*अधिकारी', r'नारकोटिक्स.*ब्यूरो'],
        'kyc': [r'केवायसी.*समाप्त', r'पेटीएम.*निलंबित', r'खाते.*ब्लॉक.*केवायसी'],
        'lottery': [r'लॉटरी.*कोटी', r'केबीसी.*विजेता', r'व्हाट्सअॅप.*बक्षीस'],
        'otp': [r'ओटीपी.*शेअर.*नका', r'तात्काळ.*ओटीपी', r'सत्यापन.*कोड.*तात्काळ'],
        'job': [r'घरातून.*काम.*लाख', r'डेटा.*एन्ट्री.*अग्रिम', r'महिना.*कमवा.*लाख'],
        'government': [r'पंतप्रधान.*मोदी.*योजना', r'उत्पन्न.*कर.*परतावा', r'पीएफ.*काढणे']
    }
}

ENTITY_PATTERNS = {
    'indian_phone': r'(?:\+91|0|९१)?[६-९]\d{9}|(?:\+91|0)[6-9]\d{9}',
    'upi_vpa': r'[\w.-]+@(?:paytm|ybl|upi|sbi|axis|hdfc|icici|pnb|bob)',
    'aadhaar': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}',
    'pan': r'[A-Z]{5}\d{4}[A-Z]{1}',
    'bank_account': r'(?:account|a/c).*?(?:\d{10,16}|\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4})',
    'ifsc': r'[A-Z]{4}0[A-Z0-9]{6}',
    'urgency_words': r'(?:immediately|तुरंत|तात्काळ|urgent|जलद|अत्यावश्यक|within.*hour|24.*hour|48.*hour)',
    'threat_words': r'(?:arrest|गिरफ्तार|अटक|legal.*action|court|warrant|block.*account|suspend)',
    'payment_words': r'(?:pay|paytm|google.*pay|phonepe|upi|qr.*code|wallet|transfer|deposit)',
    'authority_words': r'(?:rbi|reserve.*bank|sbi|hdfc|icici|axis|cbi|narcotics|fedex|govt|government|pm.*modi)'
}

@dataclass
class ScamSignal:
    label: str
    probability: float
    threshold: float
    confidence: float
    linguistic_features: Dict
    pattern_matches: List[str]

@dataclass
class RiskProfile:
    score: float
    level: str
    confidence: float
    signals: List[ScamSignal]
    pattern_score: float
    entity_score: float
    combination_bonus: float
    temporal_features: Dict
    recommendations: List[str]

class PhDFeatureEngineer:
    def __init__(self):
        self.linguistic_features = {}
        self.pattern_cache = {}

    def extract_linguistic_features(self, text: str) -> Dict:
        features = {}
        features['char_count'] = len(text)
        features['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        features['upper_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0
        features['special_char_ratio'] = sum(not c.isalnum() for c in text) / len(text) if text else 0
        words = text.split()
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        sentences = re.split(r'[.!?।]', text)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
        for n in [2, 3, 4]:
            ngrams = self._extract_ngrams(text.lower(), n)
            scam_ngrams = self._get_scam_ngrams(n)
            features[f'scam_ngram_{n}_ratio'] = len([ng for ng in ngrams if ng in scam_ngrams]) / len(ngrams) if ngrams else 0
        features['readability_score'] = self._calculate_readability(text)
        features['threat_density'] = self._calculate_threat_density(text)
        features['urgency_density'] = self._calculate_urgency_density(text)
        return features

    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        return [text[i:i+n] for i in range(len(text)-n+1)]

    def _get_scam_ngrams(self, n: int) -> set:
        scam_ngrams = {
            2: {'kt', 'yc', 'xp', 'qr', 'up', 'pi', 'ot', 'p.', 'c.', 'b.'},
            3: {'kyc', 'otp', 'upi', 'atm', 'pan', 'rbi', 'sbi', 'cbi', 'pin', 'cvv'},
            4: {'paytm', 'phone', 'google', 'link', 'click', 'block', 'dear', 'customer'}
        }
        return scam_ngrams.get(n, set())

    def _calculate_readability(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0
        complex_words = [w for w in words if len(w) > 6]
        return len(complex_words) / len(words)

    def _calculate_threat_density(self, text: str) -> float:
        threat_words = ['arrest', 'girlfriend', 'legal', 'court', 'warrant', 'police', 'cbi', 'narcotics', 'block', 'suspend']
        words = text.lower().split()
        return sum(1 for w in words if any(threat in w for threat in threat_words)) / len(words) if words else 0

    def _calculate_urgency_density(self, text: str) -> float:
        urgent_words = ['immediately', 'urgent', 'now', 'hurry', 'quick', 'fast', 'within', '24 hours', 'today', 'soon']
        words = text.lower().split()
        return sum(1 for w in words if any(urgent in w for urgent in urgent_words)) / len(words) if words else 0

class PhDPatternEngine:
    def __init__(self):
        self.pattern_weights = {
            'digital_arrest': 4.2,
            'kyc': 3.8,
            'lottery': 3.1,
            'otp': 3.9,
            'job': 2.9,
            'government': 3.5
        }

    def detect_patterns(self, text: str) -> Tuple[float, List[Dict]]:
        total_score = 0
        matches = []
        text_lower = text.lower()
        detected_langs = self._detect_languages(text)
        for lang in detected_langs:
            if lang in MULTILINGUAL_SCAM_PATTERNS:
                for pattern_type, patterns in MULTILINGUAL_SCAM_PATTERNS[lang].items():
                    for pattern in patterns:
                        if re.search(pattern, text_lower, re.IGNORECASE):
                            weight = self.pattern_weights[pattern_type]
                            total_score += weight
                            matches.append({
                                'type': pattern_type,
                                'language': lang,
                                'pattern': pattern,
                                'weight': weight,
                                'description': self._get_pattern_description(pattern_type)
                            })
                            break
        return total_score, matches

    def _detect_languages(self, text: str) -> List[str]:
        langs = ['en']
        if re.search(r'[\u0900-\u097F]', text):
            langs.append('hi')
        if re.search(r'[\u0900-\u097F]', text) and any(word in text.lower() for word in ['असे', 'आहे', 'नाही']):
            langs.append('mr')
        return langs

    def _get_pattern_description(self, pattern_type: str) -> str:
        descriptions = {
            'digital_arrest': 'Digital arrest impersonation scam',
            'kyc': 'KYC verification fraud',
            'lottery': 'Fake lottery/prize scam',
            'otp': 'OTP/credentials phishing',
            'job': 'Fake job offer scam',
            'government': 'Government authority impersonation'
        }
        return descriptions.get(pattern_type, 'Unknown pattern')

class PhDEntityRecognizer:
    def __init__(self):
        self.entity_risk_scores = {
            'indian_phone': 0.3,
            'upi_vpa': 0.8,
            'aadhaar': 1.2,
            'pan': 1.0,
            'bank_account': 1.5,
            'ifsc': 0.9
        }

    def extract_entities(self, text: str) -> Tuple[Dict, float]:
        entities = {}
        entity_score = 0
        for entity_type, pattern in ENTITY_PATTERNS.items():
            if entity_type in ['urgency_words', 'threat_words', 'payment_words', 'authority_words']:
                continue
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))
                entity_score += self.entity_risk_scores.get(entity_type, 0.5)
        return entities, entity_score

    def extract_suspicious_phrases(self, text: str) -> List[str]:
        suspicious = []
        if re.search(ENTITY_PATTERNS['authority_words'], text, re.IGNORECASE) and re.search(ENTITY_PATTERNS['urgency_words'], text, re.IGNORECASE):
            suspicious.append("Authority + Urgency combination")
        if re.search(ENTITY_PATTERNS['threat_words'], text, re.IGNORECASE) and re.search(ENTITY_PATTERNS['payment_words'], text, re.IGNORECASE):
            suspicious.append("Threat + Payment combination")
        if re.search(ENTITY_PATTERNS['indian_phone'], text, re.IGNORECASE) and re.search(ENTITY_PATTERNS['payment_words'], text, re.IGNORECASE):
            suspicious.append("Phone + Payment request")
        return suspicious

class PhDRiskCalculator:
    def __init__(self):
        self.feature_engineer = PhDFeatureEngineer()
        self.pattern_engine = PhDPatternEngine()
        self.entity_recognizer = PhDEntityRecognizer()
        self.priors = SCAM_PRIORS
        self.model_weight = 0.45
        self.pattern_weight = 0.30
        self.entity_weight = 0.15
        self.linguistic_weight = 0.10

    def calculate_risk(self, text: str, model_probs: np.ndarray, thresholds: np.ndarray) -> RiskProfile:
        model_score = self._calculate_model_score(model_probs, thresholds)
        pattern_score, pattern_matches = self.pattern_engine.detect_patterns(text)
        pattern_score = min(pattern_score / 10, 1.0)
        entities, entity_score = self.entity_recognizer.extract_entities(text)
        entity_score = min(entity_score / 5, 1.0)
        linguistic_features = self.feature_engineer.extract_linguistic_features(text)
        linguistic_score = self._calculate_linguistic_score(linguistic_features)
        combination_bonus = self._calculate_combination_bonus(model_probs, pattern_matches, entities)
        ensemble_score = (
            self.model_weight * model_score +
            self.pattern_weight * pattern_score +
            self.entity_weight * entity_score +
            self.linguistic_weight * linguistic_score +
            combination_bonus
        )
        final_score = self._apply_bayesian_adjustment(ensemble_score, text)
        risk_level = self._score_to_risk_level(final_score)
        confidence = self._calculate_confidence(model_probs, pattern_matches, linguistic_features)
        recommendations = self._generate_recommendations(risk_level, pattern_matches, entities)
        signals = self._build_signals(model_probs, thresholds, linguistic_features, pattern_matches)
        return RiskProfile(
            score=round(final_score * 100, 2),
            level=risk_level,
            confidence=round(confidence * 100, 2),
            signals=signals,
            pattern_score=round(pattern_score * 100, 2),
            entity_score=round(entity_score * 100, 2),
            combination_bonus=round(combination_bonus * 100, 2),
            temporal_features=linguistic_features,
            recommendations=recommendations
        )

    def _calculate_model_score(self, probs: np.ndarray, thresholds: np.ndarray) -> float:
        detected = probs > thresholds
        if not detected.any():
            return probs.max() * 0.3
        detected_probs = probs[detected]
        weights = np.array([self.priors.get(label, 1.0) for label, d in zip(LABELS, detected) if d])
        return np.average(detected_probs, weights=weights)

    def _calculate_linguistic_score(self, features: Dict) -> float:
        threat_score = features.get('threat_density', 0)
        urgency_score = features.get('urgency_density', 0)
        ngram_score = max(features.get('scam_ngram_2_ratio', 0),
                         features.get('scam_ngram_3_ratio', 0),
                         features.get('scam_ngram_4_ratio', 0))
        return (threat_score * 0.4 + urgency_score * 0.4 + ngram_score * 0.2)

    def _calculate_combination_bonus(self, probs: np.ndarray, patterns: List[Dict], entities: Dict) -> float:
        bonus = 0
        detected = probs > 0.5
        if detected.sum() >= 3:
            bonus += 0.15
        pattern_types = [p['type'] for p in patterns]
        if all(pt in pattern_types for pt in ['authority_name', 'threat_type', 'payment_method']):
            bonus += 0.25
        if 'upi_vpa' in entities and 'indian_phone' in entities:
            bonus += 0.1
        langs = set([p['language'] for p in patterns])
        if len(langs) >= 2:
            bonus += 0.08
        return min(bonus, 0.3)

    def _apply_bayesian_adjustment(self, score: float, text: str) -> float:
        if len(text) < 50:
            score += 0.1
        if re.search(r'http[s]?://', text):
            score += 0.05
        phones = re.findall(ENTITY_PATTERNS['indian_phone'], text)
        if len(phones) >= 2:
            score += 0.08
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if upper_ratio > 0.3:
            score += 0.06
        return min(score, 1.0)

    def _score_to_risk_level(self, score: float) -> str:
        if score < 0.25:
            return "SAFE"
        elif score < 0.45:
            return "CAUTION"
        elif score < 0.65:
            return "SUSPICIOUS"
        else:
            return "SCAM"

    def _calculate_confidence(self, probs: np.ndarray, patterns: List[Dict], features: Dict) -> float:
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        model_conf = 1 - (entropy / max_entropy)
        pattern_conf = min(len(patterns) * 0.2, 0.8)
        feature_conf = min(features.get('threat_density', 0) + features.get('urgency_density', 0), 0.8)
        return (model_conf * 0.6 + pattern_conf * 0.3 + feature_conf * 0.1)

    def _generate_recommendations(self, risk_level: str, patterns: List[Dict], entities: Dict) -> List[str]:
        recommendations = []
        if risk_level == "SAFE":
            recommendations.append("✅ Message appears safe. No action needed.")
        elif risk_level == "CAUTION":
            recommendations.extend([
                "⚠️ Verify sender identity through official channels",
                "🔗 Do not click on any links in the message",
                "📞 If from bank, call official customer service"
            ])
        elif risk_level == "SUSPICIOUS":
            recommendations.extend([
                "🚨 DO NOT respond to this message",
                "📵 Block the sender immediately",
                "🔒 Never share OTP, passwords, or personal details",
                "🏦 Verify through official bank branch/website"
            ])
        else:
            recommendations.extend([
                "🆘 THIS IS A CONFIRMED SCAM - DELETE IMMEDIATELY",
                "📞 Report to Cyber Crime: 1930",
                "🌐 File complaint at: cybercrime.gov.in",
                "📢 Warn family and friends about this scam pattern",
                "🏦 If you shared details, contact bank immediately"
            ])
        pattern_types = [p['type'] for p in patterns]
        if 'digital_arrest' in pattern_types:
            recommendations.insert(0, "👮 Digital arrest is ALWAYS fake - real police never do this")
        if 'kyc' in pattern_types:
            recommendations.append("🏦 Banks never ask for KYC update via SMS/WhatsApp")
        if 'lottery' in pattern_types:
            recommendations.append("🎰 You cannot win a lottery you never entered")
        return recommendations

    def _build_signals(self, probs: np.ndarray, thresholds: np.ndarray, features: Dict, patterns: List[Dict]) -> List[ScamSignal]:
        signals = []
        for i, (label, prob) in enumerate(zip(LABELS, probs)):
            if prob > thresholds[i]:
                signal = ScamSignal(
                    label=label,
                    probability=float(prob),
                    threshold=float(thresholds[i]),
                    confidence=float(prob - thresholds[i]),
                    linguistic_features=features,
                    pattern_matches=[p['type'] for p in patterns]
                )
                signals.append(signal)
        return signals

@st.cache_resource(show_spinner="🧠 Initializing High-Level BharatScam Guard...")
def load_phD_detector():
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
    thresholds = np.array(cal.get("thresholds", [0.5] * len(LABELS)))
    risk_calculator = PhDRiskCalculator()
    return {
        'model': model,
        'tokenizer': tokenizer,
        'temperature': temperature,
        'thresholds': thresholds,
        'risk_calculator': risk_calculator
    }

class PhDVisualizationEngine:
    @staticmethod
    def plot_risk_gauge(score: float, level: str):
        colors = {'SAFE': '#28a745', 'CAUTION': '#ffc107', 'SUSPICIOUS': '#fd7e14', 'SCAM': '#dc3545'}
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "#dc3545"}},
            gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                   'bar': {'color': colors[level]}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                   'steps': [{'range': [0, 25], 'color': "#d4edda"}, {'range': [25, 45], 'color': "#fff3cd"},
                             {'range': [45, 65], 'color': "#ffeaa7"}, {'range': [65, 100], 'color': "#f8d7da"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': score}}
        ))
        fig.update_layout(paper_bgcolor="white", font={'color': "darkblue", 'family': "Arial"})
        return fig

    @staticmethod
    def plot_signal_strength(signals: List[ScamSignal]):
        if not signals:
            return None
        labels = [s.label.replace('_', ' ').title() for s in signals]
        values = [s.probability for s in signals]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself', name='Signal Strength',
                                      line_color='rgb(255, 0, 0)', fillcolor='rgba(255, 0, 0, 0.3)'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False,
                          title="Detected Signal Strengths")
        return fig

    @staticmethod
    def plot_confidence_distribution(confidence: float):
        fig = go.Figure(go.Bar(x=[confidence, 100 - confidence], y=['Confidence'], orientation='h',
                               marker_color=['#28a745', '#e9ecef'], text=[f'{confidence}%', ''],
                               textposition='inside', insidetextanchor='middle'))
        fig.update_layout(title="Analysis Confidence", showlegend=False, xaxis=dict(range=[0, 100], showticklabels=False),
                          yaxis=dict(showticklabels=False), height=100)
        return fig

def main():
    st.set_page_config(page_title="🧠  BharatScam Guard", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    .phd-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; text-align: center; }
    .risk-card { padding: 1.5rem; border-radius: 1rem; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    .recommendation-box { background: #f8f9fa; border-left: 4px solid #007bff; padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; }
    </style>
    <div class="phd-header">
        <h1>🧠  BharatScam Guard</h1>
        <p>Advanced Multi-Modal Fraud Detection System for Indian Digital Ecosystem</p>
        <p><em>Powered by High-level Research in Computational Linguistics & Bayesian Risk Analysis</em></p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        #st.markdown("### 🧪  Research Features")
        #with st.expander("🔬 Advanced Settings", expanded=False):
            #pattern_weight = st.slider("Pattern Matching Weight", 0.1, 0.5, 0.3, 0.05)
            #entity_weight = st.slider("Entity Recognition Weight", 0.05, 0.3, 0.15, 0.05)
            #linguistic_weight = st.slider("Linguistic Features Weight", 0.05, 0.2, 0.1, 0.05)
        st.markdown("### 📚 Research Citations")
        st.info("""
        **Detection Methods:**
        - Bayesian Ensemble Learning
        - Multilingual Pattern Mining  
        - Adversarial Feature Engineering
        - Temporal Signal Analysis
        **Data Sources:**
        - RBI Fraud Reports 2023
        - Indian Cybercrime Database
        - Multilingual Scam Corpus
        """)
        st.markdown("### 🚨 Emergency")
        st.error("**Cyber Crime Helpline: 1930**\n\n**Online:** cybercrime.gov.in")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🔍 Message Analysis")
        examples = {
            "Digital Arrest Scam": "I am Inspector Rajesh Kumar from CBI Digital Crime Unit. Your Aadhar linked to drug trafficking case. You must pay ₹50,000 fine within 2 hours or face digital arrest. Call 9876543210 immediately.",
            "KYC Scam": "Dear SBI Customer, Your KYC has expired. Click here to update: bit.ly/sbi-kyc-update or your account will be blocked within 24 hours. Never share OTP with anyone.",
            "Job Scam": "Earn ₹50,000/month from home! Data entry job available. Pay ₹2000 registration fee to start immediately. Contact 9876543210. Limited slots available!",
            "Safe Message": "Hi, are we still meeting for lunch today? Let me know if you're running late. See you at 1 PM!",
            "Lottery Scam": "🎉 CONGRATULATIONS! You won ₹1 Crore in KBC WhatsApp Lottery! To claim, send ₹25,000 processing fee to this Paytm number: 9876543210"
        }
        selected_example = st.selectbox("📋 Load Example Message", ["Custom"] + list(examples.keys()))
        example_text = examples.get(selected_example, "")
        user_text = st.text_area("✏️ Enter SMS, WhatsApp, or Email message:", value=example_text, height=150,
                                 placeholder="Paste your message here for PhD-level analysis...", key="message_input")
        analyze_col, clear_col = st.columns([1, 4])
        with analyze_col:
            analyze_clicked = st.button("🧠 Analyze", type="primary", use_container_width=True)

    if analyze_clicked and user_text.strip():
        if len(user_text) < 10:
            st.warning("⚠️ Message too short for meaningful analysis. Please enter at least 10 characters.")
            return
        with st.spinner("🧠 Running High-level analysis pipeline..."):
            detector = load_phD_detector()
            inputs = detector['tokenizer'](user_text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = detector['model'](**inputs)
                logits = outputs.logits / detector['temperature']
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            thresholds = detector['thresholds']
            risk_profile = detector['risk_calculator'].calculate_risk(user_text, probs, thresholds)
        viz_engine = PhDVisualizationEngine()
        st.markdown("---")
        col_viz1, col_viz2 = st.columns([2, 1])
        with col_viz1:
            fig_gauge = viz_engine.plot_risk_gauge(risk_profile.score, risk_profile.level)
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col_viz2:
            st.markdown(f"""
            <div class="risk-card" style="background: {'#d4edda' if risk_profile.level == 'SAFE' else '#fff3cd' if risk_profile.level == 'CAUTION' else '#ffeaa7' if risk_profile.level == 'SUSPICIOUS' else '#f8d7da'}">
                <h3 style="margin: 0;">{risk_profile.level}</h3>
                
                
            </div>
            """, unsafe_allow_html=True)
            fig_conf = viz_engine.plot_confidence_distribution(risk_profile.confidence)
            st.plotly_chart(fig_conf, use_container_width=True)
        st.markdown("---")
       # st.markdown("### 🔬 Detailed Signal Analysis")
        #col_det1, col_det2 = st.columns([1, 1])
       # with col_det1:
           # if risk_profile.signals:
               # fig_radar = viz_engine.plot_signal_strength(risk_profile.signals)
                #st.plotly_chart(fig_radar, use_container_width=True)
           # else:
               # st.info("No specific scam signals detected")
        #with col_det2:
            #st.markdown("**📊 Component Scores:**")
            #components = {
                #"Pattern Matching": risk_profile.pattern_score,
                #"Entity Recognition": risk_profile.entity_score,
                #"Linguistic Analysis": risk_profile.temporal_features.get('threat_density', 0) * 100,
                #"Combination Bonus": risk_profile.combination_bonus
            #}
            for component, score in components.items():
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span>{component}:</span>
                    <strong>{score:.1f}%</strong>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### ⚡ Action Recommendations")
        for rec in risk_profile.recommendations:
            priority = "critical" if "🆘" in rec else "high" if "🚨" in rec else "medium" if "⚠️" in rec else "low"
            color = {"critical": "#dc3545", "high": "#fd7e14", "medium": "#ffc107", "low": "#28a745"}
            st.markdown(f"""
            <div class="recommendation-box" style="border-left-color: {color[priority]};">
                <strong>{rec}</strong>
            </div>
            """, unsafe_allow_html=True)
        with st.expander("🔧 Technical Details (High Level)"):
            col_tech1, col_tech2 = st.columns([1, 1])
            with col_tech1:
                st.markdown("**Model Probabilities:**")
                for label, prob in zip(LABELS, probs):
                    st.write(f"{label}: {prob:.4f}")
            with col_tech2:
                st.markdown("**Linguistic Features:**")
                for feature, value in list(risk_profile.temporal_features.items())[:5]:
                    st.write(f"{feature}: {value:.4f}")
        st.markdown("---")
        st.markdown("""
        <div style="background: #e3f2fd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2196f3;">
            <h4 style="margin: 0 0 0.5rem 0;">🛡️ Safety Reminders</h4>
            <ul style="margin: 0; padding-left: 1.5rem;">
                <li>Real banks never ask for OTP, passwords, or PINs</li>
                <li>Police never demand money over phone/WhatsApp</li>
                <li>You cannot win a lottery you never entered</li>
                <li>Always verify through official websites/customer service</li>
                <li>When in doubt, ask a trusted friend or family member</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style='text-align: center; color: #6c757d; font-size: 0.9rem;'>
    🧠  BharatScam Guard - Advanced Research in Indian Fraud Detection<br>
    Built with High level expertise in Computational Linguistics, Bayesian Inference, and Adversarial ML
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
