# ============================================================
# CAPSTONE MAIN APPLICATION
# Intelligent Healthcare Diagnostic Assistant
# Introduction to AI — 13-Week Capstone
# ============================================================

import sys
import json
import warnings
from typing import Optional
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

def build_system():
    """Instantiate and wire all AI modules"""
    section("🔧 Building AI System — Registering Modules")

    agent = HealthcareDiagnosticAgent()

    print("\n  Initializing modules...")
    kb = MedicalKnowledgeBase()
    modules = {
        'KnowledgeBase': kb,
        'BayesianNet':   SimpleBayesianDiagnostics(),
        'MLClassifier':  MLDiagnosticClassifier(),
        'NeuralNetwork': NeuralDiagnosticModel(),
        'Fuzzy':         FuzzySeverityAssessor(),
        'Planner':       TreatmentPlanner(),
    }

    print()
    for name, module in modules.items():
        agent.register_module(name, module)

    return agent, kb


def sample_patients():
    """A small set of sample patients covering different scenarios"""
    return [
        PatientPercept(
            patient_id="P-001",
            symptoms=["fever", "cough", "fatigue", "body_aches"],
            age=29,
            temperature=38.6,
            heart_rate=92,
            blood_pressure="118/76",
        ),
        PatientPercept(
            patient_id="P-002",
            symptoms=["chest_pain", "shortness_of_breath", "sweating", "fatigue"],
            age=61,
            temperature=37.2,
            heart_rate=128,
            blood_pressure="152/98",
        ),
        PatientPercept(
            patient_id="P-003",
            symptoms=["headache", "stiff_neck", "light_sensitivity", "fever"],
            age=34,
            temperature=39.7,
            heart_rate=110,
            blood_pressure="124/80",
        ),
    ]


def resolve_kb_explanation(kb: MedicalKnowledgeBase, diagnosis: str) -> Optional[str]:
    """
    The winning diagnosis after aggregation is a plain label (e.g. "flu"),
    but KB's own rule conclusions still use the "_confirmed"/"_suspected"
    suffixes. Try each candidate label against KB's actual rules and use
    whichever one really matches, so we don't silently fall through to
    the "is a base fact" default for a label that was never asserted.
    """
    for candidate in (f"{diagnosis}_confirmed", f"{diagnosis}_suspected", diagnosis):
        for conditions, conclusion, cf in kb.rules:
            if conclusion == candidate:
                return kb.get_explanation(candidate)
    return None


def print_report(report: dict, explanation: Optional[str] = None):
    """Pretty-print a single agent action report"""
    urgency_colors = {
        "CRITICAL": C.RED,
        "HIGH":     C.YELLOW,
        "MEDIUM":   C.BLUE,
        "LOW":      C.GREEN,
    }
    uc = urgency_colors.get(report["urgency"], C.END)

    print(f"\n  Patient ID     : {report['patient_id']}")
    print(f"  Symptoms       : {', '.join(report['symptoms'])}")
    print(f"  Diagnosis      : {C.BOLD}{report['diagnosis']}{C.END}")
    print(f"  Confidence     : {report['confidence']:.1%}")
    print(f"  Urgency        : {uc}{C.BOLD}{report['urgency']}{C.END}")
    print(f"  Next Action    : {report['next_action']}")
    if explanation:
        print(f"  KB Explanation : {explanation}")
    print(f"  Recommendations:")
    for rec in report["recommendations"]:
        print(f"     • {rec}")


def run_demo(agent: HealthcareDiagnosticAgent, kb: MedicalKnowledgeBase):
    """Run the full perceive → think → act cycle for each sample patient"""
    section("🩺 Running Diagnostic Cycles")

    for patient in sample_patients():
        print(f"\n{C.BOLD}{C.HEADER}{'─'*60}{C.END}")
        print(f"{C.BOLD}{C.HEADER}  New Patient: {patient.patient_id}{C.END}")
        print(f"{C.BOLD}{C.HEADER}{'─'*60}{C.END}")

        report = agent.run(patient)
        explanation = resolve_kb_explanation(kb, report['diagnosis'])
        print_report(report, explanation)

    section("📊 Agent Performance Summary")
    perf = agent.get_performance()
    for key, value in perf.items():
        print(f"  {key.replace('_', ' ').title():20s}: {value}")

    agent.print_log()


def main():
    banner()
    agent, kb = build_system()
    run_demo(agent, kb)

    section("✅ Capstone Demo Complete")
    print("\n  All modules ran successfully end-to-end.\n")


if __name__ == "__main__":
    main()