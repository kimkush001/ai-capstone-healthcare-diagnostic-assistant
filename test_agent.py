# ============================================================
# Standalone test for Module 1: HealthcareDiagnosticAgent
# Run this BEFORE any other modules are ready — it uses fake
# "dummy" modules so you can confirm agent.py works correctly
# on its own (perceive -> think -> act).
#
# Usage:
#   Save this file in your project ROOT folder (same level as app.py)
#   Then run:  python test_agent.py
# ============================================================

from modules.agent import HealthcareDiagnosticAgent, PatientPercept


# ----------------------------------------------------------
# Dummy module — stands in for a real module (e.g. KnowledgeBase)
# so we can test the agent's coordination logic in isolation.
# Any real module just needs an .analyze(patient) method that
# returns a dict with 'diagnosis', 'confidence', and 'summary'.
# ----------------------------------------------------------
class DummyDiagnosticModule:
    def __init__(self, name, fake_diagnosis, fake_confidence):
        self.name = name
        self.fake_diagnosis = fake_diagnosis
        self.fake_confidence = fake_confidence

    def analyze(self, patient: PatientPercept) -> dict:
        return {
            'diagnosis': self.fake_diagnosis,
            'confidence': self.fake_confidence,
            'summary': f"{self.name} thinks it's {self.fake_diagnosis} "
                       f"({self.fake_confidence:.0%} confident)"
        }


def main():
    print("=" * 60)
    print("TESTING MODULE 1: HealthcareDiagnosticAgent")
    print("=" * 60)

    # 1. Create the agent
    agent = HealthcareDiagnosticAgent()

    # 2. Register some FAKE modules (stand-ins for the real ones)
    agent.register_module(
        'DummyKB',
        DummyDiagnosticModule('DummyKB', 'covid19', 0.85)
    )
    agent.register_module(
        'DummyBayes',
        DummyDiagnosticModule('DummyBayes', 'covid19', 0.78)
    )
    agent.register_module(
        'DummyML',
        DummyDiagnosticModule('DummyML', 'flu', 0.60)
    )

    # 3. Build a fake patient (matches PatientPercept from agent.py)
    test_patient = PatientPercept(
        patient_id="P001",
        symptoms=["fever", "cough", "fatigue", "loss_of_smell"],
        age=34,
        temperature=38.9,
        heart_rate=98,
        blood_pressure="120/80"
    )

    # 4. Run the full Perceive -> Think -> Act cycle
    print("\nRunning agent.run() on test patient P001...\n")
    report = agent.run(test_patient)

    # 5. Print the final report
    print("-" * 60)
    print("FINAL REPORT")
    print("-" * 60)
    for key, value in report.items():
        print(f"{key:>16}: {value}")

    # 6. Print the full action log (shows perceive/think/act steps)
    agent.print_log()

    # 7. Print performance stats
    print("\nPerformance stats:", agent.get_performance())

    # 8. Basic sanity checks
    print("\n" + "=" * 60)
    assert report['diagnosis'] == 'covid19', "Expected covid19 to win (2 votes vs 1)"
    assert report['patient_id'] == 'P001'
    assert 0 <= report['confidence'] <= 1
    print("ALL CHECKS PASSED — Agent Core is working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
