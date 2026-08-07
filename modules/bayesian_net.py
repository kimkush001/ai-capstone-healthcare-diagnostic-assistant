# ============================================================
# MODULE 3 (v2): Bayesian Network — Probabilistic Diagnosis
# Built with pgmpy — a true graphical Bayesian Network
# ============================================================

from typing import Dict, List
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class SimpleBayesianDiagnostics:
    """
    Bayesian Network diagnostic model built with pgmpy.
    Structure: Disease -> Symptom (for each evidence symptom).
    This mirrors real medical reasoning: the disease causes
    the symptom, not the other way around.
    """

    DISEASES = ['flu', 'covid19', 'dengue', 'cardiac',
                'diabetes', 'common_cold', 'healthy']

    # Same priors as the original naive implementation
    PRIORS = {
        'flu': 0.15, 'covid19': 0.08, 'dengue': 0.05,
        'cardiac': 0.04, 'diabetes': 0.10,
        'common_cold': 0.30, 'healthy': 0.28,
    }

    # Evidence symptoms used as network nodes
    EVIDENCE_SYMPTOMS = [
        'fever', 'cough', 'fatigue', 'headache',
        'chest_pain', 'rash', 'joint_pain', 'loss_of_smell'
    ]

    # P(symptom=1 | disease) — same numbers as the original file
    LIKELIHOODS = {
        'flu':         {'fever':0.90,'cough':0.85,'fatigue':0.88,'headache':0.70,
                         'chest_pain':0.05,'rash':0.05,'joint_pain':0.40,'loss_of_smell':0.20},
        'covid19':     {'fever':0.88,'cough':0.80,'fatigue':0.90,'headache':0.65,
                         'chest_pain':0.20,'rash':0.05,'joint_pain':0.20,'loss_of_smell':0.85},
        'dengue':      {'fever':0.98,'cough':0.15,'fatigue':0.80,'headache':0.90,
                         'chest_pain':0.05,'rash':0.75,'joint_pain':0.85,'loss_of_smell':0.05},
        'cardiac':     {'fever':0.10,'cough':0.15,'fatigue':0.70,'headache':0.30,
                         'chest_pain':0.92,'rash':0.02,'joint_pain':0.10,'loss_of_smell':0.02},
        'diabetes':    {'fever':0.10,'cough':0.05,'fatigue':0.82,'headache':0.40,
                         'chest_pain':0.05,'rash':0.08,'joint_pain':0.20,'loss_of_smell':0.02},
        'common_cold': {'fever':0.50,'cough':0.90,'fatigue':0.55,'headache':0.60,
                         'chest_pain':0.05,'rash':0.02,'joint_pain':0.15,'loss_of_smell':0.30},
        'healthy':     {'fever':0.02,'cough':0.05,'fatigue':0.10,'headache':0.08,
                         'chest_pain':0.01,'rash':0.01,'joint_pain':0.05,'loss_of_smell':0.01},
    }

    def __init__(self):
        self.model = self._build_network()
        self.infer = VariableElimination(self.model)

    def _build_network(self) -> DiscreteBayesianNetwork:
        edges = [('Disease', s) for s in self.EVIDENCE_SYMPTOMS]
        model = DiscreteBayesianNetwork(edges)

        n_diseases = len(self.DISEASES)
        disease_probs = [[self.PRIORS[d]] for d in self.DISEASES]
        cpd_disease = TabularCPD('Disease', n_diseases, disease_probs,
                                  state_names={'Disease': self.DISEASES})

        cpds = [cpd_disease]
        for symptom in self.EVIDENCE_SYMPTOMS:
            # Column order must match self.DISEASES
            p_yes = [self.LIKELIHOODS[d][symptom] for d in self.DISEASES]
            p_no  = [1 - p for p in p_yes]
            cpd = TabularCPD(
                symptom, 2,
                [p_no, p_yes],
                evidence=['Disease'], evidence_card=[n_diseases],
                state_names={symptom: [0, 1], 'Disease': self.DISEASES}
            )
            cpds.append(cpd)

        model.add_cpds(*cpds)
        assert model.check_model(), "Bayesian network CPDs are inconsistent"
        return model

    def compute_posterior(self, symptoms: List[str]) -> Dict[str, float]:
        """Query P(Disease | observed symptoms) using variable elimination."""
        symptoms_clean = {s.lower().replace(' ', '_') for s in symptoms}
        evidence = {
            s: (1 if s in symptoms_clean else 0)
            for s in self.EVIDENCE_SYMPTOMS
        }

        result = self.infer.query(variables=['Disease'], evidence=evidence, show_progress=False)
        posterior = {
            state: round(float(prob), 4)
            for state, prob in zip(result.state_names['Disease'], result.values)
        }
        return posterior

    def analyze(self, percept) -> Dict:
        """Module interface for the agent — same shape as before."""
        posteriors = self.compute_posterior(percept.symptoms)
        top_disease = max(posteriors, key=posteriors.get)
        top_prob = posteriors[top_disease]
        sorted_dx = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)

        return {
            'summary':    f"Top: {top_disease} ({top_prob:.2%})",
            'diagnosis':  top_disease,
            'confidence': top_prob,
            'all_posteriors': posteriors,
            'ranked_diagnoses': sorted_dx[:5]
        }

    def explain(self, disease: str, symptoms: List[str]) -> str:
        symptoms_clean = [s.lower().replace(' ', '_') for s in symptoms]
        likelihoods = self.LIKELIHOODS.get(disease, {})
        evidence = [
            f"P({s}|{disease})={likelihoods.get(s, 0.01):.2f}"
            for s in symptoms_clean if s in self.EVIDENCE_SYMPTOMS
        ]
        return f"P({disease}) = {self.PRIORS[disease]} × " + " × ".join(evidence)