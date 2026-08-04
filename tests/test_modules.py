"""
Basic smoke tests for each diagnostic module.
Run with: python tests/test_modules.py
"""
import sys
import os

# Allow imports from the project root when running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner import TreatmentPlanner
from modules.agent import PatientPercept


def make_percept(symptoms, temp=38.0, hr=90):
    """Helper to build a test patient without repeating boilerplate."""
    return PatientPercept(
        patient_id="TEST",
        symptoms=symptoms,
        age=30,
        temperature=temp,
        heart_rate=hr,
        blood_pressure="120/80"
    )


def test_knowledge_base_runs():
    kb = MedicalKnowledgeBase()
    result = kb.analyze(make_percept(["fever", "cough", "fatigue"]))
    assert "diagnosis" in result, "KnowledgeBase.analyze() should return a diagnosis"


def test_bayesian_net_runs():
    bn = SimpleBayesianDiagnostics()
    result = bn.analyze(make_percept(["fever", "cough", "fatigue"]))
    assert "diagnosis" in result, "BayesianNet.analyze() should return a diagnosis"


def test_fuzzy_controller_runs():
    fz = FuzzySeverityAssessor()
    result = fz.analyze(make_percept(["fever"], temp=39.0, hr=105))
    assert "severity_score" in result, "Fuzzy.analyze() should return a severity_score"


def test_planner_finds_plan_for_flu():
    tp = TreatmentPlanner()
    result = tp.create_treatment_plan("flu", "MEDIUM")
    assert "error" not in result, "Planner should find a plan for flu (regression check for the fix we made)"
    assert result["steps"] > 0, "Planner should generate at least one treatment step"


def test_planner_handles_unknown_diagnosis_gracefully():
    """Even if a diagnosis has no matching plan, analyze() shouldn't crash."""
    tp = TreatmentPlanner()
    result = tp.analyze(make_percept(["fever"]))
    assert "summary" in result, "Planner.analyze() should always return a summary, even on failure"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            fn()
            print(f"✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {name} — {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {name} — unexpected error: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)