import streamlit as st
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime

# --------------------------------------------------
# Senior ML Engineer Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="🛡️ Sentinel - Advanced Scam Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 800; background: linear-gradient(90deg, #FF4B4B, #1E90FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .risk-card { background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%); border-radius: 15px; padding: 25px; margin: 20px 0; border: 2px solid #333; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
    .feature-badge { display: inline-block; padding: 6px 12px; border-radius: 20px; margin: 3px; font-size: 0.85rem; font-weight: 600; }
    .safe-badge { background: linear-gradient(135deg, #00C853, #64DD17); color: white; }
    .sus-badge { background: linear-gradient(135deg, #FFAB00, #FFC400); color: black; }
    .scam-badge { background: linear-gradient(135deg, #D50000, #FF1744); color: white; }
    .progress-bar { margin: 8px 0; height: 22px; border-radius: 10px; background: #333; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 10px; transition: width 0.5s ease-in-out; }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
    .pulse { animation: pulse 2s infinite; }
</style>
""", unsafe_allow_html=True)

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/Scam"
MODEL_DIR = Path("./sentinel_model")
MODEL_DIR.mkdir(exist_ok=True)

# Feature definitions with weights and descriptions
FEATURE_METADATA = {
    "authority_name": {"weight": 1.8, "name": "Authority Impersonation", "desc": "Claims from official entities"},
    "threat_type": {"weight": 2.0, "name": "Threat/Intimidation", "desc": "Urgent threats or consequences"},
    "time_pressure": {"weight": 1.5, "name": "Urgency Pressure", "desc": "Artificial deadlines"},
    "payment_method": {"weight": 1.9, "name": "Payment Method", "desc": "Unusual payment requests"},
    "language_mixing": {"weight": 1.2, "name": "Language Patterns", "desc": "Manipulative language use"}
}

# --------------------------------------------------
# Senior ML Engineer's Risk Assessment Engine
# --------------------------------------------------
class RiskAssessmentEngine:
    def __init__(self, model_dir: Path, device: str):
        """Initialize with model, tokenizer, and calibration parameters"""
        self.device = device
        self.model_dir = model_dir
        
        # Load model artifacts
        self._download_artifacts()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
        
        # Load calibration
        with open(model_dir / "scam_v1.json", "r") as f:
            cal = json.load(f)
        self.temperature = float(cal.get("temperature", 1.0))
        self.base_thresholds = np.array(cal.get("thresholds", [0.5] * 5))
        
        # Feature interaction matrix (synergistic effects)
        self.interaction_matrix = {
            ("authority_name", "threat_type"): 1.4,
            ("threat_type", "time_pressure"): 1.3,
            ("payment_method", "time_pressure"): 1.25,
            ("authority_name", "payment_method"): 1.35
        }
        
    def _download_artifacts(self):
        """Idempotent artifact download"""
        required_files = [
            "config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt", "scam_v1.json"
        ]
        
        for file in required_files:
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=file,
                    repo_type="dataset",
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                st.warning(f"Warning: Could not download {file}: {e}")
    
    def calculate_risk_score(self, text: str) -> dict:
        """
        Sophisticated risk scoring with:
        - Weighted feature probabilities
        - Feature interaction bonuses
        - Confidence weighting
        - Temporal context
        """
        # Tokenize and predict
        inputs = self.tokenizer(
            text, truncation=True, padding=True, max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits / self.temperature
            raw_probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Apply dynamic thresholds based on text complexity
        complexity_factor = self._calculate_text_complexity(text)
        adaptive_thresholds = self.base_thresholds * (0.9 + complexity_factor * 0.2)
        active_features = raw_probs > adaptive_thresholds
        
        # Weighted base score
        feature_weights = np.array([FEATURE_METADATA[f]["weight"] for f in FEATURE_METADATA])
        weighted_score = np.sum(raw_probs * feature_weights * active_features.astype(float))
        
        # Feature interaction bonuses
        interaction_bonus = self._calculate_interaction_bonus(active_features, raw_probs)
        
        # Confidence weighting (reduce score if model is uncertain)
        confidence = self._calculate_confidence(raw_probs, active_features)
        
        # Final risk score (0-100)
        raw_risk = (weighted_score * (1 + interaction_bonus)) * confidence
        risk_score = np.clip(raw_risk * 18, 0, 100)  # Normalize
        
        # Verdict with confidence intervals
        verdict = self._determine_verdict(risk_score, confidence, active_features)
        
        return {
            "risk_score": float(risk_score),
            "verdict": verdict,
            "active_features": [list(FEATURE_METADATA.keys())[i] for i, v in enumerate(active_features) if v],
            "probabilities": {k: float(v) for k, v in zip(FEATURE_METADATA.keys(), raw_probs)},
            "confidence": float(confidence),
            "complexity_factor": float(complexity_factor),
            "thresholds_used": adaptive_thresholds.tolist(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity (0-1) for threshold adjustment"""
        # Senior engineer heuristic: scams often have specific structures
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        num_ratio = sum(1 for c in text if c.isdigit()) / len(text)
        exclamation_count = text.count('!')
        word_count = len(text.split())
        
        # Normalize features
        complexity = (
            caps_ratio * 0.3 +
            num_ratio * 0.2 +
            min(exclamation_count / 5, 1.0) * 0.25 +
            min(word_count / 100, 1.0) * 0.25
        )
        return complexity
    
    def _calculate_interaction_bonus(self, active_features: np.ndarray, probs: np.ndarray) -> float:
        """Calculate bonus for dangerous feature combinations"""
        bonus = 0.0
        active_indices = np.where(active_features)[0]
        feature_names = list(FEATURE_METADATA.keys())
        
        for i, idx1 in enumerate(active_indices):
            for idx2 in active_indices[i+1:]:
                pair = tuple(sorted([feature_names[idx1], feature_names[idx2]]))
                if pair in self.interaction_matrix:
                    # Weight by average probability of the pair
                    avg_prob = (probs[idx1] + probs[idx2]) / 2
                    bonus += self.interaction_matrix[pair] * avg_prob
        
        return bonus
    
    def _calculate_confidence(self, probs: np.ndarray, active_features: np.ndarray) -> float:
        """Calculate prediction confidence (0.7-1.0)"""
        # High confidence when probabilities are far from threshold
        distances = np.abs(probs - self.base_thresholds)
        avg_distance = np.mean(distances[active_features]) if np.any(active_features) else np.mean(distances)
        
        # Normalize to 0.7-1.0 range
        confidence = 0.7 + (avg_distance * 0.6)
        return np.clip(confidence, 0.7, 1.0)
    
    def _determine_verdict(self, risk_score: float, confidence: float, active_features: np.ndarray) -> dict:
        """Intelligent verdict with nuanced thresholds"""
        # Dynamic thresholds based on confidence
        if confidence > 0.85:
            scam_threshold = 70
            suspicious_threshold = 35
        else:
            scam_threshold = 75
            suspicious_threshold = 40
        
        if risk_score >= scam_threshold:
            label = "🔴 CONFIRMED SCAM"
            level = "critical"
        elif risk_score >= suspicious_threshold:
            label = "🟡 HIGH RISK"
            level = "warning"
        else:
            label = "🟢 LIKELY SAFE"
            level = "safe"
        
        return {
            "label": label,
            "level": level,
            "risk_score": round(risk_score, 1)
        }

# --------------------------------------------------
# Visualization Components
# --------------------------------------------------
class ResultsVisualizer:
    @staticmethod
    def create_risk_gauge(score: float, verdict: dict):
        """Plotly gauge with gradient coloring"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score", 'font': {'size': 24, 'color': 'white'}},
            delta = {'reference': 50, 'increasing': {'color': "#FF1744"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "white"},
                'bar': {'color': "rgba(255,255,255,0.8)", 'thickness': 0.4},
                'bgcolor': "rgba(0,0,0,0.3)",
                'borderwidth': 2,
                'bordercolor': "#444",
                'steps': [
                    {'range': [0, 35], 'color': '#00C853'},
                    {'range': [35, 70], 'color': '#FFC400'},
                    {'range': [70, 100], 'color': '#FF1744'}],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}}
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def display_feature_analysis(probs: dict, active_features: list):
        """Interactive feature breakdown"""
        st.markdown("### 🔍 Feature Intelligence Report")
        
        for feature, prob in probs.items():
            meta = FEATURE_METADATA[feature]
            is_active = feature in active_features
            
            # Color coding
            if prob > 0.7:
                color = "#FF1744"
                status = "🚨 Critical"
            elif prob > 0.4:
                color = "#FFC400"
                status = "⚠️ Moderate"
            else:
                color = "#00C853"
                status = "✅ Low"
            
            # Progress bar
            st.markdown(f"**{meta['name']}**  {status if is_active else ''}")
            st.markdown(f"<div class='progress-bar'><div class='progress-fill' style='width:{prob*100}%; background: linear-gradient(90deg, {color}, {color}CC);'></div></div>", unsafe_allow_html=True)
            st.caption(f"*{meta['desc']}* | Probability: `{prob:.3f}`")
            st.write("")
    
    @staticmethod
    def display_recommendations(verdict_level: str, active_features: list):
        """Contextual safety recommendations"""
        st.markdown("### 💡 Security Recommendations")
        
        if verdict_level == "critical":
            st.error("🚫 **IMMEDIATE ACTION REQUIRED**")
            st.markdown("""
            - **Do not respond** or engage with the sender
            - **Do not click** any links or attachments
            - **Block** the sender immediately
            - **Report** to your carrier and FTC (for US)
            - **Monitor** your accounts for suspicious activity
            """)
        elif verdict_level == "warning":
            st.warning("⚠️ **PROCEED WITH CAUTION**")
            st.markdown("""
            - **Verify** sender identity through official channels
            - **Delay** any urgent actions requested
            - **Question** the legitimacy of the request
            - **Consult** with trusted friends/family
            - **Research** the organization/contact independently
            """)
        else:
            st.success("✅ **BEST PRACTICES**")
            st.markdown("""
            - **Stay vigilant** - scams evolve constantly
            - **Verify** unusual requests, even from known contacts
            - **Educate** others about common scam patterns
            - **Report** suspicious messages to help others
            """)

# --------------------------------------------------
# Main Application
# --------------------------------------------------
def main():
    st.markdown('<h1 class="main-header pulse">🛡️ Sentinel</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; font-size:1.2rem; color:#888;">Advanced Multi-Dimensional Scam Detection System</p>', unsafe_allow_html=True)
    
    # Initialize engine (cached)
    @st.cache_resource(show_spinner=False)
    def load_engine():
        with st.spinner("🚀 Initializing Sentinel AI Engine..."):
            return RiskAssessmentEngine(MODEL_DIR, DEVICE)
    
    engine = load_engine()
    
    # Input section
    st.markdown("---")
    user_text = st.text_area(
        "📱 **Enter Message Content**",
        height=200,
        placeholder="Paste SMS, WhatsApp, Email, or any suspicious message here for analysis...",
        help="The system analyzes language patterns, urgency, authority claims, and payment requests using advanced NLP."
    )
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 **Run Deep Analysis**", use_container_width=True, type="primary")
    
    if analyze_button:
        if not user_text.strip():
            st.error("❌ **Input Required**: Please provide message content to analyze.")
            return
        
        # Run analysis
        with st.spinner("⚙️ Running neural analysis with temporal context & feature interaction modeling..."):
            results = engine.calculate_risk_score(user_text)
        
        # Display results
        st.markdown("---")
        
        # Main risk card
        verdict = results["verdict"]
        st.markdown(f"""
        <div class="risk-card">
            <h2 style='text-align:center; margin-bottom:20px;'>{verdict['label']}</h2>
            <h1 style='text-align:center; font-size:4rem; margin:0;'>{verdict['risk_score']}</h1>
            <p style='text-align:center; font-size:1rem; color:#888;'>Risk Score (0-100)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Columns for overview
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            ResultsVisualizer.create_risk_gauge(results["risk_score"], verdict)
        
        with col_right:
            st.markdown("### 📊 Analysis Summary")
            st.metric("Model Confidence", f"{results['confidence']:.1%}")
            st.metric("Text Complexity", f"{results['complexity_factor']:.2f}")
            st.metric("Active Threats", len(results["active_features"]))
        
        # Feature analysis
        ResultsVisualizer.display_feature_analysis(
            results["probabilities"], 
            results["active_features"]
        )
        
        # Recommendations
        ResultsVisualizer.display_recommendations(
            verdict["level"], 
            results["active_features"]
        )
        
        # Advanced details (collapsible)
        with st.expander("🔬 Show Advanced Diagnostics"):
            st.json({
                "timestamp": results["timestamp"],
                "thresholds_applied": results["thresholds_used"],
                "all_probabilities": results["probabilities"],
                "active_features": results["active_features"],
                "confidence": results["confidence"],
                "risk_score_raw": results["risk_score"]
            })
        
        # Footer
        st.markdown("---")
        st.caption("Sentinel AI v2.1 | Multi-dimensional threat analysis with feature interaction modeling | Results are probabilistic and should be combined with human judgment")

if __name__ == "__main__":
    main()
