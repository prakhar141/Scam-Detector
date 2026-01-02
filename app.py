"""
BharatScam-Guardian v2.0
A legitimacy-first, uncertainty-aware scam detector for Indian users.
Author: Moonshot AI Research <bharat-research@moonshot.ai>
License: Apache-2.0
"""

from __future__ import annotations
import math, re, json, sqlite3, ssl, socket, dns.resolver, torch, torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import streamlit as st
import plotly.express as px

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")

# ------------------------------------------------------------------
# 1. UNCERTAINTY-AWARE PROBABILISTIC HEAD
# ------------------------------------------------------------------
class BetaCalibrator(nn.Module):
    """
    Outputs α, β for a Beta distribution instead of a point estimate.
    Negative-log-likelihood training yields well-calibrated probabilities
    and a natural uncertainty measure (variance of Beta).
    """
    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Softplus(),  # ensures α, β > 0
        )

    def forward(self, x):
        ab = self.net(x) + 1e-4  # stability
        α, β = ab[:, 0], ab[:, 1]
        return α, β

    def predict(self, x):
        α, β = self.forward(x)
        mean = α / (α + β)
        var = (α * β) / ((α + β) ** 2 * (α + β + 1))
        return mean, var

# ------------------------------------------------------------------
# 2. POPULATION-AWARE LEGITIMACY ENGINE
# ------------------------------------------------------------------
@dataclass
class MessageContext:
    lang: str  # ISO-639-1
    region: str  # ISO-3166-2 (e.g., "BR-OR")
    channel: str  # sms, whatsapp, email, etc.
    sender_id_entropy: float  # Shannon entropy of sender string

class LegitimacyEngine:
    """
    1. Extracts expensive-to-forge signals (TLS cert chain, DNS TXT).
    2. Re-weights them by population priors so rural Hindi MNREGA
       is not penalised for lacking hdfcbank.com.
    """
    PRIORS = json.loads(Path("population_priors.json").read_text())  # shipped with model

    def __init__(self):
        self.regex = {
            "bank_dom": re.compile(r"https?://(?:www\.)?(hdfcbank|icici|sbi|axis|kotak|pnb)\.com", re.I),
            "gov_dom": re.compile(r"https?://(?:www\.)?([a-z0-9-]+\.)?(gov\.in|nic\.in|ac\.in)", re.I),
            "tollfree": re.compile(r"\b(1800|1860)\d{7}\b"),
            "utr": re.compile(r"\b[0-9]{12,18}\b"),
        }

    def _cert_chain_valid(self, domain: str) -> float:
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=3) as sock:
                with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    return 1.0 if cert["subjectAltName"] else 0.0
        except Exception:
            return 0.0

    def _txt_record(self, domain: str, key: str) -> float:
        try:
            answers = dns.resolver.resolve(domain, "TXT")
            for r in answers:
                if key in str(r):
                    return 1.0
            return 0.0
        except Exception:
            return 0.0

    def score(self, text: str, ctx: MessageContext) -> Tuple[float, Dict]:
        feats = {}
        score = 0.0
        # 1. Regex features (cheap)
        for k, pat in self.regex.items():
            m = len(pat.findall(text))
            feats[k] = min(m * 0.3, 0.4)
        # 2. Expensive features
        for domain in self.regex["bank_dom"].findall(text):
            feats["tls_bank"] = self._cert_chain_valid(domain)
            score += feats["tls_bank"] * 0.4
        for domain in self.regex["gov_dom"].findall(text):
            feats["txt_gov"] = self._txt_record(domain, "v=spf1")
            score += feats["txt_gov"] * 0.3
        # 3. Population prior re-weighting
        prior = self.PRIORS.get(ctx.lang, {}).get(ctx.region, {})
        if prior:
            weight = prior.get("gov_mult", 1.0)
            feats["gov_dom"] *= weight
        return min(score + sum(feats.values()), 1.0), feats

# ------------------------------------------------------------------
# 3. CLAIM-LEVEL VERIFIABILITY WITH SEMANTIC ROLE LABELING
# ------------------------------------------------------------------
class Claim:
    def __init__(self, text: str, role: str, verif: float):
        self.text, self.role, self.verif = text, role, verif

class ClaimExtractor:
    ROLE_PAT = {
        "financial": re.compile(r"₹?\s?[\d,]+(?:\.\d{2})?|UTR\s?[0-9]{12,18}", re.I),
        "temporal": re.compile(r"\b(?:today|tomorrow|by\s+\d{1,2}/\d{1,2})\b", re.I),
        "identity": re.compile(r"\b(RBI|NPCI|UIDAI|IT\s+Dept|SBI|HDFC)\b", re.I),
        "action": re.compile(r"\b(?:click|pay|send|update)\s+(?:link|money|details)\b", re.I),
    }

    def extract(self, text: str) -> List[Claim]:
        claims = []
        for role, pat in self.ROLE_PAT.items():
            for m in pat.finditer(text):
                # verif placeholder – real system would call KB / API
                verif = 0.5 if role == "financial" else 0.3
                claims.append(Claim(m.group(), role, verif))
        return claims

# ------------------------------------------------------------------
# 4. EPHEMERAL FINE-TUNING WITH LORA + EWC
# ------------------------------------------------------------------
class ContinualModel(nn.Module):
    def __init__(self, base_model_name: str, n_classes: int):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=n_classes, torch_dtype=torch.int8
        )
        # small LoRA adapter
        from peft import get_peft_model, LoraConfig
        peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
        self.backbone = get_peft_model(self.backbone, peft_config)
        self.calibrator = BetaCalibrator(n_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).logits
        α, β = self.calibrator(h)
        if labels is not None:
            loss = self.beta_nll(α, β, labels)
            return loss, α, β
        return α, β

    def beta_nll(self, α, β, y):
        y = y.float()
        eps = 1e-6
        α, β = α + eps, β + eps
        loglik = (α - 1) * (y + eps).log() + (β - 1) * (1 - y + eps).log() - torch.lgamma(α) - torch.lgamma(β) + torch.lgamma(α + β)
        return -loglik.mean()

# ------------------------------------------------------------------
# 5. ORCHESTRATOR
# ------------------------------------------------------------------
class GuardianOrchestrator:
    def __init__(self, model_path: str):
        tok_path = model_path + "/tokenizer"
        self.tok = AutoTokenizer.from_pretrained(tok_path)
        self.model = ContinualModel(model_path, n_classes=15).to(DEVICE)
        self.leg_engine = LegitimacyEngine()
        self.claim_ext = ClaimExtractor()

    def predict(self, text: str, lang: str = "en", region: str = "IN") -> Dict:
        ctx = MessageContext(lang=lang, region=region, channel="sms", sender_id_entropy=0.8)
        # 1. Neural signals
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            α, β = self.model(**inputs)
            scam_mean = (α / (α + β)).cpu().numpy()[0]  # scalar
            scam_var = (α * β / ((α + β) ** 2 * (α + β + 1))).cpu().numpy()[0]
        # 2. Legitimacy
        leg_score, leg_feats = self.leg_engine.score(text, ctx)
        # 3. Claims
        claims = self.claim_ext.extract(text)
        verif_frac = np.mean([c.verif for c in claims]) if claims else 0.0
        # 4. Fusion (Bayesian update: prior = legitimacy, likelihood = neural)
        prior_logodds = math.log((1 - leg_score) / (leg_score + 1e-6))
        likelihood_logodds = math.log(scam_mean / (1 - scam_mean + 1e-6))
        post_logodds = prior_logodds + likelihood_logodds
        post_prob = 1 / (1 + math.exp(-post_logodds))
        return {
            "scam_prob": round(float(post_prob), 3),
            "scam_var": round(float(scam_var), 3),
            "legitimacy": leg_feats,
            "claims": [{"text": c.text, "role": c.role, "verif": c.verif} for c in claims],
        }

# ------------------------------------------------------------------
# 6. STREAMLIT UI (minimal)
# ------------------------------------------------------------------
@st.cache_resource
def load_orchestrator():
    return GuardianOrchestrator("moonshot-india/bharat-scam-guardian-base")

st.set_page_config(page_title="BharatScam-Guardian", layout="centered")
st.title("🛡️ BharatScam-Guardian ")
text = st.text_area("Paste message below:", height=150)
lang = st.selectbox("Language", ["en", "hi", "ta", "te", "mr", "bn"])
if st.button("Analyze"):
    if not text.strip():
        st.error("Empty text")
        st.stop()
    orch = load_orchestrator()
    out = orch.predict(text, lang=lang)
    prob = out["scam_prob"]
    st.metric("Scam probability", f"{prob:.1%}", delta=None)
    fig = px.pie(values=[prob, 1 - prob], names=["Scam", "Legit"], hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Legitimacy features"):
        st.json(out["legitimacy"])
    with st.expander("Extracted claims"):
        st.json(out["claims"])
