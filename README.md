# Healthcare Diagnostic Assistant — AI Capstone Project

An intelligent agent-based healthcare diagnostic system built for the Introduction to AI capstone. The system combines multiple classical and modern AI techniques — a rule-based knowledge base, a Bayesian network, a machine learning classifier, a deep neural network, and a fuzzy logic controller — into a single diagnostic agent that reasons about patient symptoms and generates a treatment plan.

## Overview

Given a patient's symptoms, vitals, and history, the agent runs six coordinated modules and aggregates their outputs into a final diagnosis, urgency level, and recommended next actions:

| Module | Technique | Role |
|---|---|---|
| Knowledge Base | Forward-chaining rule engine | Symbolic inference over symptom rules |
| Bayesian Net | Probabilistic graphical model (pgmpy) | P(Disease \| Symptoms) via variable elimination |
| ML Classifier | Gradient Boosting | Learned classification over symptom features |
| Neural Network | Deep feedforward network (TensorFlow/Keras) | Learned classification, higher-capacity model |
| Fuzzy | Fuzzy logic controller | Severity scoring from vitals and symptom counts |
| Planner | Classical STRIPS-style planner | Generates a treatment action sequence from the diagnosis |

The agent aggregates each module's diagnosis and confidence score, weighting by confidence, then hands the result to the Planner to produce a concrete treatment plan and urgency-based recommendations.

## Team

| Person | Module |
|---|---|
| Bridgett | Knowledge Base → Bayesian Network |
| Shekinah | ML Classifier |
| Christine | Neural Network |
| MoReen | Fuzzy Controller / Planner |
| Ken | Agent integration (Integration Lead) |

## Getting Started

### 1. Prerequisites

- Python 3.9+ (check "Add Python to PATH" during install)
- Git
- VS Code (recommended)

### 2. Clone the repository

```bash
cd Desktop
git clone https://github.com/<owner-username>/ai-capstone-healthcare-diagnostic-assistant.git
cd ai-capstone-healthcare-diagnostic-assistant
```

### 3. Set your Git identity (once, before your first commit)

```bash
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"
```

### 4. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

If PowerShell blocks the activation script:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Demo

```bash
python app.py
```

This builds the agent, registers all six modules, and runs three sample patients end-to-end — printing each patient's diagnosis, confidence, urgency, recommendations, and the full action log showing every module's individual output.

## Running Tests

```bash
python tests/test_modules.py
```

Runs unit tests covering the Knowledge Base, Bayesian Network, Fuzzy Controller, and Planner modules.

## Project Structure

```
ai-capstone-healthcare-diagnostic-assistant/
├── app.py                     # Entry point — builds system, runs demo patients
├── modules/
│   ├── knowledge_base.py      # Rule-based forward-chaining inference
│   ├── bayesian_net.py        # pgmpy Bayesian network
│   ├── ml_classifier.py       # Gradient Boosting classifier
│   ├── neural_network.py      # Deep neural network (TensorFlow/Keras)
│   ├── fuzzy_controller.py    # Fuzzy severity scoring
│   ├── planner.py             # STRIPS-style treatment planner
│   └── agent.py               # Agent orchestration and diagnosis aggregation
├── tests/
│   └── test_modules.py        # Unit tests
└── requirements.txt
```

## Branching Workflow

Branch names match each person's assigned module:

```bash
git checkout -b feature/yourname-module-name
```

Work in your assigned file inside `modules/`, commit, and push to your own branch — never directly to `main`:

```bash
git add .
git commit -m "Describe your change"
git push origin feature/yourname-module-name
```

Open a pull request on GitHub for review. Pull the latest `main` before starting new work each day to avoid merge conflicts:

```bash
git checkout main
git pull origin main
git checkout feature/your-branch
git merge main
```

## Known Limitations

**Synthetic-data circularity.** The ML Classifier and Neural Network modules are trained on synthetic patient data generated from the same hand-authored disease–symptom probability distributions (`LIKELIHOODS`) that the Bayesian Network module uses directly. This creates a degree of circularity: the high accuracy reported for the learned classifiers partly reflects their ability to recover known-in-advance probabilities rather than learning from independently observed clinical patterns. In a production setting, these models would need to be trained on real (or more realistically simulated) patient records to validate genuine predictive power.

**Bayesian Network coverage.** The Bayesian Network reasons over a fixed set of eight general symptom nodes and does not include disease-specific symptoms (e.g. `stiff_neck` and `light_sensitivity` for meningitis). As a result, it can be noticeably less precise than the ML/NN modules on diseases whose distinguishing symptoms fall outside that fixed symptom set — this is expected behavior, not a bug, and is one reason the agent aggregates across multiple modules rather than relying on any single one.

## License

Educational project — Introduction to AI capstone.
