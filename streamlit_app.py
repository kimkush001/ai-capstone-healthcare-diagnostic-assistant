# ============================================================
# Streamlit Demo — Healthcare Diagnostic Assistant
# Fix 11: interactive front-end for live presentation.
#
# Standalone by design: wires the same six modules app.py does,
# but does not import app.py itself, so it isn't coupled to
# app.py's console-report formatting or any bugs there.
# ============================================================

import streamlit as st

from modules.agent import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner import TreatmentPlanner

st.set_page_config(page_title="Healthcare Diagnostic Assistant", page_icon="🏥", layout="wide")


# ── System wiring (cached so training only happens once per server run) ──
@st.cache_resource(show_spinner="Training models and wiring modules (first run only)...")
def load_system():
    agent = HealthcareDiagnosticAgent()
    kb = MedicalKnowledgeBase()
    modules = {
        'KnowledgeBase': kb,
        'BayesianNet':   SimpleBayesianDiagnostics(),
        'MLClassifier':  MLDiagnosticClassifier(),
        'NeuralNetwork': NeuralDiagnosticModel(),
        'Fuzzy':         FuzzySeverityAssessor(),
        'Planner':       TreatmentPlanner(),
    }
    for name, module in modules.items():
        agent.register_module(name, module)
    return agent, kb


def resolve_kb_explanation(kb: MedicalKnowledgeBase, diagnosis: str):
    """
    Same logic as app.py's Fix 9 helper: the winning diagnosis is a
    plain label (e.g. "flu"), but KB's rule conclusions still use
    "_confirmed"/"_suspected" suffixes. Try each candidate against
    KB's real rules rather than assuming a match.
    """
    for candidate in (f"{diagnosis}_confirmed", f"{diagnosis}_suspected", diagnosis):
        for conditions, conclusion, cf in kb.rules:
            if conclusion == candidate:
                return kb.get_explanation(candidate)
    return None


URGENCY_COLORS = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🔵",
    "LOW":      "🟢",
}

# Full symptom vocabulary the ML/NN/Bayesian/KB modules actually recognize.
# Pulled from MLDiagnosticClassifier's own class attribute rather than
# retyped by hand, so the checkbox list can't drift out of sync with it.
ALL_SYMPTOMS = MLDiagnosticClassifier.SYMPTOM_FEATURES


def main():
    st.title("🏥 Intelligent Healthcare Diagnostic Assistant")
    st.caption("Live demo — same agent and modules as `app.py`, run against symptoms you choose below.")

    agent, kb = load_system()

    with st.sidebar:
        st.header("Patient Info")
        patient_id = st.text_input("Patient ID", value="DEMO-001")
        age = st.number_input("Age", min_value=0, max_value=120, value=35)
        temperature = st.slider("Temperature (°C)", 35.0, 42.0, 37.0, 0.1)
        heart_rate = st.slider("Heart Rate (bpm)", 40, 180, 80)
        blood_pressure = st.text_input("Blood Pressure", value="120/80")

    st.subheader("Symptoms")
    st.caption("Check all symptoms the patient is currently experiencing.")

    cols = st.columns(3)
    selected_symptoms = []
    for i, symptom in enumerate(ALL_SYMPTOMS):
        label = symptom.replace('_', ' ').title()
        if cols[i % 3].checkbox(label, key=f"symptom_{symptom}"):
            selected_symptoms.append(symptom)

    run_clicked = st.button("🩺 Run Diagnosis", type="primary", disabled=not selected_symptoms)
    if not selected_symptoms:
        st.info("Select at least one symptom to run the diagnostic agent.")

    if run_clicked:
        patient = PatientPercept(
            patient_id=patient_id,
            symptoms=selected_symptoms,
            age=age,
            temperature=temperature,
            heart_rate=heart_rate,
            blood_pressure=blood_pressure,
        )

        report = agent.run(patient)
        module_results = agent.memory.diagnosis_history[-1]
        explanation = resolve_kb_explanation(kb, report['diagnosis'])

        st.divider()
        st.subheader(f"Results for {report['patient_id']}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Diagnosis", report['diagnosis'])
        c2.metric("Confidence", f"{report['confidence']:.1%}")
        urgency = report['urgency']
        c3.metric("Urgency", f"{URGENCY_COLORS.get(urgency, '')} {urgency}")

        st.markdown(f"**Next Action:** {report['next_action']}")

        if explanation:
            st.markdown(f"**Knowledge Base Explanation:** {explanation}")

        st.markdown("**Recommendations:**")
        for rec in report['recommendations']:
            st.markdown(f"- {rec}")

        with st.expander("Per-module breakdown"):
            for module_name, result in module_results.items():
                if isinstance(result, dict):
                    st.markdown(f"**[{module_name}]** {result.get('summary', '')}")


if __name__ == "__main__":
    main()
