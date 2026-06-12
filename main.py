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
import requests
from openai import OpenAI
import google.generativeai as genai

app = FastAPI(title="CKD XGBoost API with Real NVIDIA XAI")

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

# OpenAI/ChatGPT for rapport generation
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# OR Google Gemini (alternative)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')

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
    
    # Try OpenAI first
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            llm_rapport = response.choices[0].message.content
            
            # Also extract structured data
            structured = extract_structured_from_llm(llm_rapport, risk_score, top_risk_factors)
            return structured
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
    
    # Fallback to Gemini
    if gemini_model:
        try:
            response = gemini_model.generate_content(user_prompt)
            llm_rapport = response.text
            structured = extract_structured_from_llm(llm_rapport, risk_score, top_risk_factors)
            return structured
        except Exception as e:
            print(f"Gemini API error: {e}")
    
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

def extract_structured_from_llm(llm_text, risk_score, top_factors):
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
        "llm_generated": True
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
        "llm_generated": False
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