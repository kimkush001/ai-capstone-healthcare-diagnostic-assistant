# ============================================================
# CAPSTONE MAIN APPLICATION
# Intelligent Healthcare Diagnostic Assistant
# Introduction to AI — 13-Week Capstone
# ============================================================

import sys
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

# Import all modules
from modules.agent          import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net   import SimpleBayesianDiagnostics
from modules.ml_classifier  import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner        import TreatmentPlanner

# ── ANSI Colors ────────────────────────────────────────────
class C:
    HEADER = '\033[95m'; BLUE   = '\033[94m'
    GREEN  = '\033[92m'; YELLOW = '\033[93m'
    RED    = '\033[91m'; BOLD   = '\033[1m'
    END    = '\033[0m'

def banner():
    print(f"""
{C.BOLD}{C.BLUE}
╔══════════════════════════════════════════════════════════╗
║        🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║
║         Introduction to AI — Capstone Project            ║
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy      ║
╚══════════════════════════════════════════════════════════╝
{C.END}""")

def section(title: str):
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")


def build_system() -> HealthcareDiagnosticAgent:
    """Instantiate and wire all AI modules"""
    section("🔧 Building AI System — Registering Modules")

    agent = HealthcareDiagnosticAgent()

    print("\n  Initializing modules...")
    modules = {
        'KnowledgeBase': MedicalKnowledgeBase(),
        'BayesianNet':   SimpleBayesianDiagnostics(),
        'MLClassifier':  MLDiagnosticClassifier(),
        'NeuralNetwork': NeuralDiagnosticModel(),
        'Fuzzy':         FuzzySeverityAssessor(),
        'Planner':       TreatmentPlanner()
    }

    # NOTE: ML and Neural Network models need to be trained before
    # they can predict. If your teammates' modules expose a .train()
    # method, call it here once, before registering. Uncomment as
    # each module becomes ready:
    #
    # print("  Training ML Classifier...")
    # modules['MLClassifier'].train()
    #
    # print("  Training Neural Network...")
    # modules['NeuralNetwork'].train(epochs=30)

    print("\n  Registering modules with agent...")
    for name, module in modules.items():
        agent.register_module(name, module)

    print(f"\n{C.GREEN}  ✅ System built — {len(modules)} modules registered.{C.END}")
    return agent


def get_test_patients():
    """A handful of hardcoded test patients covering different scenarios.
    Manual requires at least 5 test patients diagnosed by the full system."""
    return [
        PatientPercept(
            patient_id="P001",
            symptoms=["fever", "cough", "fatigue", "loss_of_smell"],
            age=34, temperature=38.9, heart_rate=98, blood_pressure="120/80"
        ),
        PatientPercept(
            patient_id="P002",
            symptoms=["fever", "rash", "joint_pain", "headache"],
            age=27, temperature=39.4, heart_rate=110, blood_pressure="118/76"
        ),
        PatientPercept(
            patient_id="P003",
            symptoms=["cough", "sore_throat", "runny_nose"],
            age=45, temperature=37.2, heart_rate=78, blood_pressure="122/80"
        ),
        PatientPercept(
            patient_id="P004",
            symptoms=["fever", "cough", "chest_pain", "shortness_of_breath"],
            age=61, temperature=40.1, heart_rate=124, blood_pressure="140/90"
        ),
        PatientPercept(
            patient_id="P005",
            symptoms=["fatigue", "headache"],
            age=22, temperature=36.8, heart_rate=70, blood_pressure="110/70"
        ),
    ]


def run_patient(agent: HealthcareDiagnosticAgent, patient: PatientPercept):
    section(f"🩺 Diagnosing Patient {patient.patient_id}")
    print(f"  Symptoms   : {', '.join(patient.symptoms)}")
    print(f"  Age        : {patient.age}")
    print(f"  Temp       : {patient.temperature}°C")
    print(f"  Heart Rate : {patient.heart_rate} bpm")

    report = agent.run(patient)

    color = {
        "CRITICAL": C.RED, "HIGH": C.YELLOW,
        "MEDIUM": C.BLUE, "LOW": C.GREEN
    }.get(report['urgency'], C.END)

    print(f"\n  {C.BOLD}Diagnosis :{C.END} {report['diagnosis']}")
    print(f"  {C.BOLD}Confidence:{C.END} {report['confidence']:.1%}")
    print(f"  {C.BOLD}Urgency   :{C.END} {color}{report['urgency']}{C.END}")
    print(f"  {C.BOLD}Next Step :{C.END} {report['next_action']}")
    print(f"\n  Recommendations:")
    for rec in report['recommendations']:
        print(f"    {rec}")

    return report


def run_all_test_patients(agent: HealthcareDiagnosticAgent):
    section("🧪 Running Full Test Suite — 5 Patients")
    reports = []
    for patient in get_test_patients():
        report = run_patient(agent, patient)
        reports.append(report)
    return reports


def save_reports(reports, path="reports/test_run_results.json"):
    import os
    os.makedirs("reports", exist_ok=True)
    with open(path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"\n{C.GREEN}  ✅ Saved {len(reports)} reports to {path}{C.END}")


def main():
    banner()
    agent = build_system()
    reports = run_all_test_patients(agent)
    save_reports(reports)

    section("📊 Run Summary")
    print(f"  Total patients diagnosed: {len(reports)}")
    print(agent.get_performance())


if __name__ == "__main__":
    main()