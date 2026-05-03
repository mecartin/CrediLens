# 🛡️ CrediLens V4

### **Advanced AI Credit Risk Management & Recourse Engine**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CrediLens is a state-of-the-art credit risk platform that goes beyond simple "Approved/Denied" decisions. It integrates **Multi-Objective Optimization (NSGA-II)**, **Causal Recourse Graphing**, and **Adversarial Robustness** to provide actionable insights for both lenders and applicants.

---

## 🎯 Project Highlights
*   **Developed** "CrediLens V4", an AI-driven Credit Risk Management platform utilizing XGBoost and Scikit-Learn to automate financial risk assessment workflows across **4 core pillars**: Explainability, Causal Recourse, Portfolio Stability, and Fairness.
*   **Implemented** Explainable AI (XAI) and NSGA-II Multi-Objective Optimization to generate diverse, Pareto-optimal counterfactuals, while engineering **2** graph-based pathfinding algorithms (Dijkstra & Greedy Gradient) to provide actionable recourse paths for rejected applicants.
*   **Designed** a robust Python backend (FastAPI) and **2** simultaneous frontend UIs (Streamlit & Gradio) to serve predictive inferences, run stochastic stress tests, and automate bias detection across **2** key fairness metrics (Demographic Parity & Disparate Impact).

---

## 🚀 Detailed Feature Architecture

### 1. **Explainable AI (XAI) & Counterfactuals**
CrediLens demystifies black-box XGBoost models through a comprehensive explainability pipeline:
*   **NSGA-II Optimization:** Utilizes Non-dominated Sorting Genetic Algorithm II (`pymoo`) to discover diverse, Pareto-optimal counterfactual explanations. This allows the system to present rejected applicants with multiple viable pathways to approval (e.g., "increase income by $5,000" OR "reduce DTI by 2% and pay off 1 credit card").
*   **SHAP & LIME Wrappers:** Integrates both Global (SHAP summary plots, feature dependence) and Local (LIME surrogate models) explainability. This ensures stakeholders understand exactly which financial behaviors are driving approval probabilities across the entire dataset or for a single applicant.

### 2. **Causal Recourse Engine (`src/recourse`)**
A specialized sub-system designed to turn rejections into actionable financial plans:
*   **Graph-Based Modeling:** Financial features are modeled as a directed graph where nodes represent financial states (e.g., DTI, Credit Score) and edges represent actionable transitions.
*   **Pathfinding Algorithms:** Employs optimized Dijkstra's algorithm and a custom Greedy Gradient search to compute the "path of least resistance" or lowest-cost action sequence for an applicant to transition from a rejected state to an approved state.
*   **Cost Functions:** Incorporates customized, real-world cost models to ensure recommended changes are realistically achievable (e.g., penalizing massive sudden increases in income).

### 3. **Portfolio Analytics & Stability (`src/analytics`)**
Provides lenders with macroeconomic risk assessment tools:
*   **Stochastic Stress Testing (`portfolio.py`):** Simulates macroscopic economic shocks (e.g., inflation spikes causing +2% debt ratios, recessionary income drops) to evaluate how the aggregated portfolio's risk profile degrades under pressure.
*   **Decision Stability Analysis (`stability.py`):** Injects stochastic noise into individual applicant profiles to measure model "fragility." Decisions that flip from "Approved" to "Denied" under minimal noise are flagged for manual underwriter review.
*   **Automated Audit Reporting (`reporting.py`):** Automatically compiles performance metrics, stress test results, and fairness checks into comprehensive markdown/PDF audit reports for regulatory compliance.

### 4. **Governance & Fairness Auditing (`src/fairness`)**
Ensures ethical AI deployment by proactively combating algorithmic bias:
*   **Bias Detection (`bias_detector.py`):** Automated auditing against protected classes, calculating **Demographic Parity**, **Equal Opportunity**, and **Disparate Impact** ratios.
*   **Algorithmic Mitigation (`mitigation.py`):** Provides built-in post-processing mitigation techniques to adjust decision thresholds, ensuring fairer lending outcomes across demographic groups without heavily sacrificing model accuracy.

---

## 🛠️ Technology Stack

*   **Machine Learning Core:** Python 3.10+, XGBoost (Gradient Boosting), Scikit-Learn
*   **Optimization & Recourse:** Pymoo (NSGA-II Genetic Algorithms), NetworkX (Graph Traversal), Optuna (Hyperparameter Tuning)
*   **Explainability & Fairness:** SHAP, LIME, custom fairness modules
*   **Visualization & UI:** Streamlit (Technical Dashboard), Gradio (Consumer UI), PyVis (Interactive Network Graphs), Seaborn/Matplotlib
*   **Backend & API:** FastAPI, Uvicorn, Pydantic (Data Validation)
*   **Infrastructure:** Docker, Docker-compose

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- (Optional) CUDA-enabled GPU for accelerated XGBoost training

### Standard Setup
```bash
# Clone the repository
git clone https://github.com/mecartin/CrediLens.git
cd CrediLens

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚦 Usage & Interfaces

CrediLens is designed with a multi-interface approach catering to different stakeholders (Applicants vs. Underwriters/Data Scientists).

### 1. The Orchestrator (Local Dual-App Launch)
Launch both the **Simple Applicant UI (Gradio)** and the **Technical Underwriter Dashboard (Streamlit)** simultaneously:
```bash
python start.py
```
- **Simple UI:** `http://localhost:7860` (For applicant simulation)
- **Technical Dashboard:** `http://localhost:8501` (For fairness audits, stress testing, and portfolio analytics)

### 2. FastAPI Backend Server
Run the FastAPI production-ready server for microservice integration:
```bash
python src/interfaces/api_server.py
```
*API interactive documentation available at:* `http://localhost:8000/docs`

### 3. Model Training & Optimization
To re-train the XGBoost core or run hyperparameter tuning with Optuna:
```bash
python scripts/train_model.py
```

---

## 📂 Project Structure

```text
├── app/                  # UI Interfaces (Streamlit/Gradio)
│   ├── simple/           # Gradio applicant-facing interface
│   └── technical/        # Streamlit dashboard (credilens_dashboard.py)
├── config/               # YAML configurations for models & constraints
├── data/                 # Raw and processed datasets (Git-ignored)
├── models/               # Saved XGBoost model artifacts & preprocessors
├── scripts/              # Training, evaluation, and utility scripts
├── src/                  # Core Library Logic
│   ├── analytics/        # portfolio.py, stability.py, reporting.py
│   ├── core/             # Base classes and shared utilities
│   ├── counterfactuals/  # nsga2_problem.py (Optimization logic)
│   ├── evaluation/       # Model evaluation metrics
│   ├── explainability/   # SHAP/LIME wrappers
│   ├── fairness/         # bias_detector.py, mitigation.py
│   ├── interfaces/       # api_server.py (FastAPI implementation)
│   ├── models/           # trainer.py, xgboost_model.py, optimizer.py
│   └── recourse/         # graph_builder.py, action_space.py, path_finder.py
└── tests/                # Unit & Integration tests
```

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

---
**Developed by [mecartin](https://github.com/mecartin)**
