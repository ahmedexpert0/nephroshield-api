# app.py - REAL NVIDIA XAI Integration
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pickle
import numpy as np
import pandas as pd
import shap
import json
import os
import logging
import requests

OPENAI_IMPORT_ERROR = None

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None
    OPENAI_IMPORT_ERROR = str(e)
    print(f"OpenAI package not available: {e}")

app = FastAPI(title="CKD XGBoost API with Real NVIDIA XAI")

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("nephroshield-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# REAL NVIDIA API CONFIGURATION
# ============================================

# NVIDIA NGC API (for actual model inference with explanations)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://health.api.nvidia.com/v1"

# LLM configuration for personalized rapport generation
# Default is Gemini first. Set LLM_PROVIDER=openai if you want OpenAI first.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
openai_client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client configured")
    except Exception as e:
        logger.warning("OpenAI client could not be initialized: %s", e)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
GEMINI_API_BASE_URL = os.environ.get(
    "GEMINI_API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
).rstrip("/")

if GEMINI_API_KEY:
    logger.info("Gemini REST generation configured with model %s", GEMINI_MODEL)
else:
    logger.info("GEMINI_API_KEY not set; Gemini rapport generation disabled")

# ============================================
# LOAD YOUR EXISTING MODEL (Fallback)
# ============================================
try:
    with open("models/xgb_trackD.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    print("✅ XGBoost model loaded")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")
    xgb_model = None

# Load SHAP explainer (real SHAP, not static)
try:
    background_data = pd.read_csv("models/shap_background.csv") if os.path.exists("models/shap_background.csv") else None
    if xgb_model and background_data is not None:
        shap_explainer = shap.TreeExplainer(xgb_model, background_data)
        print("✅ Real SHAP explainer loaded")
    else:
        shap_explainer = None
except Exception as e:
    print(f"Error loading SHAP explainer: {e}")
    shap_explainer = None

FEATURE_COLS = [
    "has_nsaid", "LBXHGB", "LBXTC", "LBXTR", "has_diabetes_med", "URXUCR",
    "sex_male", "LBXSBU", "BPQ020", "BUN_Albumin_ratio", "DIET_DRXTP225",
    "DIQ010", "URXUMA", "LBXSUA", "LBXBPB", "RIDAGEYR", "LBXBCD", "BMXBMI",
    "DIET_DRXT_V_LEGUMES", "BPXSY1", "Waist_Height_ratio", "DIET_DRXT_G_WHOLE",
    "LBX4PA", "DIET_DRXT_PF_SOY", "LBXSAL", "LBXBCC", "BMXWAIST", "URXUCD",
    "LBXTHG", "BPXDI1", "DIET_DRXTATOA", "DIET_DRXTM221", "race_Mexican_American",
    "race_Other_Hispanic", "race_NH_White", "race_NH_Black", "Pulse_Pressure",
    "has_ace_arb", "DIET_DRXTP184", "LBXGLU", "LBXGH", "total_medications"
]

FEATURE_DESCRIPTIONS = {
    "has_nsaid": "NSAID medication use",
    "LBXHGB": "Hemoglobin level",
    "LBXTC": "Total cholesterol",
    "LBXTR": "Triglycerides",
    "has_diabetes_med": "Diabetes medication",
    "URXUCR": "Urinary creatinine",
    "sex_male": "Gender",
    "LBXSBU": "Blood urea nitrogen (BUN)",
    "BPQ020": "Hypertension history",
    "BUN_Albumin_ratio": "BUN to Albumin ratio",
    "DIQ010": "Diabetes diagnosis",
    "URXUMA": "Urinary albumin",
    "LBXSUA": "Uric acid",
    "RIDAGEYR": "Age",
    "BMXBMI": "BMI",
    "BPXSY1": "Systolic blood pressure",
    "BPXDI1": "Diastolic blood pressure",
    "has_ace_arb": "ACE/ARB medication",
    "LBXGLU": "Blood glucose",
    "LBXGH": "HbA1c",
}

class PredictionRequest(BaseModel):
    height_cm: float
    data: dict
    patient_context: Optional[Dict] = None

# ============================================
# REAL SHAP EXPLANATIONS (Dynamic, not static)
# ============================================

def get_real_shap_explanations(input_array, feature_names, patient_values):
    """Get REAL SHAP values - dynamic based on actual input"""
    if shap_explainer is None:
        return None
    
    try:
        # Calculate real SHAP values for this specific patient
        shap_values = shap_explainer.shap_values(input_array)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        shap_values = shap_values.flatten()
        
        explanations = []
        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
            actual_value = patient_values.get(feature, "N/A")
            reference_value = get_reference_value(feature)
            
            explanations.append({
                "feature": feature,
                "description": FEATURE_DESCRIPTIONS.get(feature, feature),
                "actual_value": actual_value,
                "reference_value": reference_value,
                "shap_value": float(shap_val),
                "impact": "increases_risk" if shap_val > 0 else "decreases_risk",
                "absolute_impact": abs(float(shap_val)),
                "percent_contribution": 0  # Will calculate after normalization
            })
        
        # Normalize to get percentage contributions
        total_abs_impact = sum(e["absolute_impact"] for e in explanations)
        if total_abs_impact > 0:
            for e in explanations:
                e["percent_contribution"] = round((e["absolute_impact"] / total_abs_impact) * 100, 1)
        
        # Sort by absolute impact (most important first)
        explanations.sort(key=lambda x: x["absolute_impact"], reverse=True)
        
        return explanations[:15]  # Top 15 factors
        
    except Exception as e:
        print(f"SHAP calculation error: {e}")
        return None

def get_reference_value(feature):
    """Get reference (population average) value for context"""
    reference_values = {
        "LBXHGB": 13.8,
        "LBXTC": 190,
        "LBXTR": 110,
        "BPXSY1": 120,
        "BPXDI1": 80,
        "BMXBMI": 27,
        "LBXGLU": 95,
        "LBXGH": 5.4,
        "RIDAGEYR": 50,
        "URXUCR": 100,
        "URXUMA": 10,
        "LBXSUA": 5.0,
    }
    return reference_values.get(feature, "N/A")

# ============================================
# REAL NVIDIA API INTEGRATION
# ============================================

def call_nvidia_xai_api(patient_data, features_list):
    """Call NVIDIA's actual XAI API for model explanations"""
    if not NVIDIA_API_KEY:
        print("NVIDIA API key not configured, falling back to SHAP")
        return None
    
    try:
        # NVIDIA's API endpoint for explainable AI
        url = f"{NVIDIA_BASE_URL}/health/ckd/explain"
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "ckd-risk-predictor",
            "input_data": patient_data,
            "explanation_type": "shap",  # Request SHAP explanations
            "output_features": features_list[:20]  # Top features to explain
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return {
                "nvidia_explanations": result.get("explanations", []),
                "feature_importance": result.get("feature_importance", {}),
                "counterfactuals": result.get("counterfactuals", [])  # What-if scenarios
            }
        else:
            print(f"NVIDIA API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"NVIDIA API call failed: {e}")
        return None

# ============================================
# REAL LLM-POWERED RAPPORT (Not static)
# ============================================

def generate_llm_rapport(probability, shap_explanations, patient_data, patient_context, previous_rapports=None):
    """Generate REAL personalized rapport using LLM (ChatGPT/Gemini)"""
    
    # Prepare context for LLM
    risk_score = probability * 100
    
    # Format SHAP explanations for LLM
    top_risk_factors = []
    top_protective_factors = []
    
    if shap_explanations:
        for exp in shap_explanations[:5]:
            if exp["impact"] == "increases_risk" and exp["percent_contribution"] > 5:
                top_risk_factors.append({
                    "factor": exp["description"],
                    "value": exp["actual_value"],
                    "impact_percent": exp["percent_contribution"]
                })
            elif exp["impact"] == "decreases_risk" and exp["percent_contribution"] > 5:
                top_protective_factors.append({
                    "factor": exp["description"],
                    "value": exp["actual_value"],
                    "impact_percent": exp["percent_contribution"]
                })
    
    # Build prompt for LLM
    system_prompt = """You are Dr. Shield, an empathetic AI medical assistant specializing in kidney health. 
    Generate a personalized, compassionate health report for a patient based on their CKD risk assessment.
    Use warm, encouraging language. Avoid medical jargon. Provide specific, actionable advice.
    The patient may be anxious - be reassuring while being honest about risks.
    Format with clear sections using emojis for visual appeal."""
    
    user_prompt = f"""
    PATIENT CONTEXT:
    - Name: {patient_context.get('name', 'Patient') if patient_context else 'Patient'}
    - Age: {patient_context.get('age', 'Unknown') if patient_context else 'Unknown'}
    - Gender: {patient_context.get('gender', 'Unknown') if patient_context else 'Unknown'}
    
    CKD RISK ASSESSMENT:
    - Risk Score: {risk_score:.1f}%
    - Risk Level: {"HIGH" if risk_score >= 70 else "MODERATE" if risk_score >= 40 else "LOW"}
    
    KEY RISK FACTORS INCREASING RISK:
    {json.dumps(top_risk_factors, indent=2) if top_risk_factors else "None significant"}
    
    PROTECTIVE FACTORS REDUCING RISK:
    {json.dumps(top_protective_factors, indent=2) if top_protective_factors else "None identified"}
    
    ABNORMAL LAB VALUES:
    {generate_abnormal_labs_summary(patient_data)}
    
    Please generate a personalized health report with:
    1. A warm greeting using the patient's name
    2. Clear explanation of their risk level (empathetic tone)
    3. Top 3 factors affecting their kidney health (with specific values)
    4. 3-5 actionable recommendations (prioritized)
    5. Questions they should ask their doctor
    6. An encouraging closing message
    
    Keep the tone supportive and empowering. Be specific - reference their actual values.
    """
    
    def _call_gemini():
        if not GEMINI_API_KEY:
            return None

        try:
            gemini_prompt = f"{system_prompt}\n\n{user_prompt}"
            url = f"{GEMINI_API_BASE_URL}/models/{GEMINI_MODEL}:generateContent"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": gemini_prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1500,
                },
            }
            response = requests.post(
                url,
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=45,
            )

            if response.status_code >= 400:
                logger.warning("Gemini API HTTP %s: %s", response.status_code, response.text[:1000])
                return None

            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                logger.warning("Gemini returned no candidates: %s", data)
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            llm_rapport = "\n".join(
                part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")
            ).strip()

            if not llm_rapport:
                logger.warning("Gemini returned an empty text response: %s", data)
                return None

            return extract_structured_from_llm(
                llm_rapport,
                risk_score,
                top_risk_factors,
                provider="gemini",
                model_name=GEMINI_MODEL,
            )
        except Exception as e:
            logger.warning("Gemini API error: %s", e)
            return None

    def _call_openai():
        if not openai_client:
            return None
        try:
            response = openai_client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            llm_rapport = (response.choices[0].message.content or "").strip()
            if not llm_rapport:
                raise ValueError("OpenAI returned an empty text response")
            return extract_structured_from_llm(
                llm_rapport,
                risk_score,
                top_risk_factors,
                provider="openai",
                model_name=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            )
        except Exception as e:
            logger.warning("OpenAI API error: %s", e)
            return None

    # Try the preferred provider first, then the other provider, then the template.
    providers = [_call_openai, _call_gemini] if LLM_PROVIDER == "openai" else [_call_gemini, _call_openai]
    for call_provider in providers:
        structured = call_provider()
        if structured is not None:
            return structured

    # Ultimate fallback - template-based but still dynamic
    return generate_template_rapport(risk_score, top_risk_factors, top_protective_factors, patient_context)

def generate_abnormal_labs_summary(patient_data):
    """Extract abnormal lab values for LLM context"""
    abnormal = []
    
    reference_ranges = {
        "LBXHGB": (12, 16, "g/dL"),
        "BPXSY1": (90, 130, "mmHg"),
        "BPXDI1": (60, 85, "mmHg"),
        "LBXGLU": (70, 100, "mg/dL"),
        "LBXGH": (4, 5.7, "%"),
        "BMXBMI": (18.5, 25, "kg/m²"),
        "LBXTC": (125, 200, "mg/dL"),
        "LBXTR": (50, 150, "mg/dL"),
        "URXUMA": (0, 30, "µg/mL")
    }
    
    for key, (low, high, unit) in reference_ranges.items():
        if key in patient_data:
            val = patient_data[key]
            if val < low:
                abnormal.append(f"- {FEATURE_DESCRIPTIONS.get(key, key)}: {val} {unit} (below normal range {low}-{high})")
            elif val > high:
                abnormal.append(f"- {FEATURE_DESCRIPTIONS.get(key, key)}: {val} {unit} (above normal range {low}-{high})")
    
    return "\n".join(abnormal) if abnormal else "All key markers within normal ranges."

def extract_structured_from_llm(llm_text, risk_score, top_factors, provider="llm", model_name=None):
    """Extract structured data from LLM response"""
    # Determine risk category
    if risk_score >= 70:
        risk_category = "high"
        urgency = "Please schedule a follow-up within 1-2 weeks"
        tone = "concerned_but_supportive"
    elif risk_score >= 40:
        risk_category = "moderate"
        urgency = "Schedule a follow-up within 1-3 months"
        tone = "proactive"
    else:
        risk_category = "low"
        urgency = "Continue regular annual check-ups"
        tone = "reassuring"
    
    # Parse recommendations from LLM text (simple extraction)
    recommendations = []
    lines = llm_text.split('\n')
    for line in lines:
        if any(word in line.lower() for word in ['recommend', 'should', 'consider', 'try', 'aim']):
            if len(line.strip()) > 20:
                recommendations.append({
                    "action": line.strip()[:50],
                    "detail": line.strip(),
                    "priority": "high" if "urgent" in line.lower() or "immediately" in line.lower() else "medium"
                })
    
    return {
        "risk_category": risk_category,
        "urgency": urgency,
        "tone": tone,
        "full_rapport_text": llm_text,
        "recommendations": recommendations[:5],
        "questions_for_doctor": [
            "What is my target blood pressure given my results?",
            "Should I be concerned about my {top_factor} level?".format(
                top_factor=top_factors[0]['factor'] if top_factors else "kidney function"
            ),
            "Are there any medications I should avoid?",
            "How often should I repeat these tests?"
        ],
        "llm_generated": True,
        "llm_provider": provider,
        "llm_model": model_name
    }

def generate_template_rapport(risk_score, top_risk_factors, top_protective_factors, patient_context):
    """Dynamic template-based rapport (still personalized, not static)"""
    
    name = patient_context.get('name', 'there') if patient_context else 'there'
    
    if risk_score >= 70:
        primary = f"I want to be direct with you, {name}, because early action matters. Your assessment shows a {risk_score:.0f}% risk for developing kidney disease over the next year."
        empathy = "I know this sounds concerning, but knowing this now gives us a powerful advantage. Every day we can work on this together."
    elif risk_score >= 40:
        primary = f"{name}, your assessment shows a {risk_score:.0f}% risk for kidney disease. This is a signal to focus on protective measures."
        empathy = "The good news is that moderate risk often responds very well to lifestyle changes and medication adjustments."
    else:
        primary = f"Great news, {name}! Your {risk_score:.0f}% risk score is in the low range. Your kidneys are functioning well relative to others your age."
        empathy = "Keep up the good work - your current habits are protecting your kidney health."
    
    # Dynamic risk factors section
    risk_section = ""
    if top_risk_factors:
        risk_section = "\n\n**What's affecting your kidney health:**\n"
        for factor in top_risk_factors[:3]:
            risk_section += f"• {factor['factor']} (currently {factor['value']}) - contributing {factor['percent_contribution']:.0f}% to your risk score\n"
    
    # Dynamic recommendations based on actual values
    recommendations = []
    if any(f['factor'] == 'Blood pressure' for f in top_risk_factors):
        recommendations.append("• 🫀 Work with your doctor to bring blood pressure below 130/80")
    if any(f['factor'] == 'HbA1c' for f in top_risk_factors):
        recommendations.append("• 📊 Improve blood sugar control - aim for HbA1c below 7%")
    if any(f['factor'] == 'BMI' for f in top_risk_factors):
        recommendations.append("• ⚖️ Consider a 5-10% weight reduction goal")
    if any(f['factor'] == 'Hemoglobin' for f in top_risk_factors):
        recommendations.append("• 🩸 Discuss anemia management with your doctor")
    
    if not recommendations:
        recommendations = [
            "• 🩺 Schedule your annual kidney function check",
            "• 💧 Stay well hydrated (6-8 glasses of water daily)",
            "• 🧂 Limit sodium to less than 2300mg per day",
            "• 🏃 Aim for 150 minutes of moderate exercise weekly"
        ]
    
    recs_text = "\n\n**Your Action Plan:**\n" + "\n".join(recommendations[:4])
    
    full_text = primary + risk_section + recs_text + f"\n\n{empathy}\n\nWarmly,\nYour Kidney Health Team"
    
    return {
        "risk_category": "high" if risk_score >= 70 else "moderate" if risk_score >= 40 else "low",
        "urgency": "Schedule follow-up within 1-2 weeks" if risk_score >= 70 else "Schedule within 3 months" if risk_score >= 40 else "Annual check-up",
        "tone": "concerned" if risk_score >= 70 else "proactive" if risk_score >= 40 else "reassuring",
        "full_rapport_text": full_text,
        "recommendations": recommendations[:5],
        "questions_for_doctor": [
            "What's my target blood pressure?",
            "Do I need medication adjustments?",
            "When should I repeat these tests?"
        ],
        "llm_generated": False,
        "llm_provider": "template",
        "llm_model": None
    }

@app.get("/health")
async def health_check():
    """Simple deployment health check, including LLM readiness."""
    return {
        "status": "ok",
        "model_loaded": xgb_model is not None,
        "shap_enabled": shap_explainer is not None,
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_api_key_present": bool(GEMINI_API_KEY),
        "gemini_method": "rest",
        "gemini_model": GEMINI_MODEL,
        "gemini_api_base_url": GEMINI_API_BASE_URL,
        "openai_configured": openai_client is not None,
        "openai_api_key_present": bool(OPENAI_API_KEY),
        "openai_package_available": OpenAI is not None,
        "openai_import_error": OPENAI_IMPORT_ERROR,
        "llm_provider_preference": LLM_PROVIDER,
    }

# ============================================
# MAIN PREDICTION ENDPOINT
# ============================================

@app.post("/predict")
async def predict_ckd(payload: PredictionRequest):
    try:
        p = payload.data.copy()
        
        # Feature engineering (dynamic)
        if "LBXSAL" in p and p["LBXSAL"] > 0:
            p["BUN_Albumin_ratio"] = p.get("LBXSBU", 15) / p["LBXSAL"]
        else:
            p["BUN_Albumin_ratio"] = p.get("LBXSBU", 15) / 4.2
        
        p["Pulse_Pressure"] = p.get("BPXSY1", 120) - p.get("BPXDI1", 80)
        waist = p.get("BMXWAIST", 88)
        height = payload.height_cm
        p["Waist_Height_ratio"] = waist / height if height > 0 else 0.5
        
        # Prepare input array
        input_array = np.array([[float(p.get(f, 0)) for f in FEATURE_COLS]])
        
        # Get model prediction
        if xgb_model is None:
            raise HTTPException(status_code=503, detail="XGBoost model is not loaded")

        probabilities = xgb_model.predict_proba(input_array)[0]
        prob_ckd = float(probabilities[1])
        
        # Get REAL SHAP explanations (dynamic, based on this patient)
        shap_explanations = get_real_shap_explanations(input_array, FEATURE_COLS, p)
        
        # Try NVIDIA XAI API if available (more advanced)
        nvidia_explanations = None
        if NVIDIA_API_KEY:
            nvidia_explanations = call_nvidia_xai_api(p, FEATURE_COLS)
        
        # Generate REAL LLM-powered rapport (not static)
        personalized_rapport = generate_llm_rapport(
            prob_ckd, 
            shap_explanations, 
            p, 
            payload.patient_context
        )
        
        # Generate clinical insights (dynamic based on actual values)
        clinical_insights = generate_clinical_insights(p, shap_explanations)
        
        return {
            "status": "success",
            "ckd_probability": round(prob_ckd, 4),
            "prediction": "CKD" if prob_ckd >= 0.5 else "No CKD",
            "risk_level": "High" if prob_ckd > 0.7 else "Moderate" if prob_ckd > 0.3 else "Low",
            "shap_explanations": shap_explanations,  # REAL SHAP values
            "nvidia_explanations": nvidia_explanations,  # NVIDIA XAI if available
            "personalized_rapport": personalized_rapport,  # LLM-generated
            "clinical_insights": clinical_insights
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def generate_clinical_insights(patient_data, shap_explanations):
    """Generate dynamic clinical insights based on actual values"""
    insights = []
    
    # Check each parameter against clinical guidelines
    clinical_rules = [
        {
            "parameter": "BPXSY1",
            "threshold": 140,
            "condition": "above",
            "message": "Systolic BP above target (140 mmHg)",
            "recommendation": "Consider antihypertensive intensification",
            "evidence": "ACC/AHA guidelines recommend BP <130/80 for CKD patients"
        },
        {
            "parameter": "LBXGH",
            "threshold": 7,
            "condition": "above",
            "message": "HbA1c above target (7%)",
            "recommendation": "Review diabetes management plan",
            "evidence": "Each 1% reduction in HbA1c reduces microvascular complications by 37%"
        },
        {
            "parameter": "URXUMA",
            "threshold": 30,
            "condition": "above",
            "message": "Microalbuminuria detected",
            "recommendation": "Maximize ACE/ARB therapy",
            "evidence": "Albuminuria is an independent risk factor for CKD progression"
        },
        {
            "parameter": "BMXBMI",
            "threshold": 30,
            "condition": "above",
            "message": "BMI indicates obesity",
            "recommendation": "Refer to weight management program",
            "evidence": "Weight loss of 5-10% improves kidney outcomes"
        }
    ]
    
    for rule in clinical_rules:
        if rule["parameter"] in patient_data:
            value = patient_data[rule["parameter"]]
            if (rule["condition"] == "above" and value > rule["threshold"]) or \
               (rule["condition"] == "below" and value < rule["threshold"]):
                insights.append({
                    "type": "alert" if value > rule["threshold"] * 1.2 else "warning",
                    "parameter": FEATURE_DESCRIPTIONS.get(rule["parameter"], rule["parameter"]),
                    "value": value,
                    "message": rule["message"],
                    "recommendation": rule["recommendation"],
                    "evidence": rule["evidence"]
                })
    
    # Add SHAP-based insights
    if shap_explanations:
        for exp in shap_explanations[:3]:
            if exp["impact"] == "increases_risk" and exp["percent_contribution"] > 10:
                insights.append({
                    "type": "insight",
                    "parameter": exp["description"],
                    "value": exp["actual_value"],
                    "message": f"{exp['description']} is a major contributor to risk",
                    "recommendation": f"Addressing {exp['description'].lower()} could significantly reduce risk",
                    "evidence": f"SHAP analysis shows {exp['percent_contribution']:.0f}% contribution to prediction"
                })
    
    return insights

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
