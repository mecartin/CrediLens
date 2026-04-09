# 🛡️ CrediLens V4

### **Advanced AI Credit Risk Management & Recourse Engine**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CrediLens is a state-of-the-art credit risk platform that goes beyond simple "Approved/Denied" decisions. It integrates **Multi-Objective Optimization (NSGA-II)**, **Causal Recourse Graphing**, and **Adversarial Robustness** to provide actionable insights for both lenders and applicants.

---

## 🚀 Key Features

### 1. **Explainable AI (XAI) & Counterfactuals**
*   **NSGA-II Discovery:** Generates diverse, Pareto-optimal counterfactuals to show applicants exactly what changes (e.g., "increase income by $5k and reduce DTI by 2%") would lead to approval.
*   **SHAP/LIME Integration:** Global and local feature importance mapping to ensure transparency in decision-making.

### 2. **Causal Recourse Engine**
*   **Pathfinding Algorithms:** Uses Dijkstra and Greedy Gradient search to find the "path of least resistance" for a rejected applicant to reach an approval state.
*   **Actionable Transitions:** Models realistic financial changes as edges in a directed graph, visualizing the journey to creditworthiness.

### 3. **Portfolio Analytics & Stability**
*   **Stochastic Stress Testing:** Simulate economic downturns (income shifts, interest rate spikes) to see how your portfolio risk profile changes.
*   **Decision Stability:** Uses stochastic noise injection to measure the "fragility" of individual credit decisions.

### 4. **Governance & Fairness**
*   **Bias Detection:** Automated monitoring for Demographic Parity and Disparate Impact across sensitive attributes.
*   **Audit Reporting:** One-click generation of technical audit reports for regulatory compliance.

---

## 🛠️ Technology Stack

*   **Core:** Python 3.10+, XGBoost, Scikit-Learn
*   **Optimization:** Pymoo (NSGA-II), Optuna
*   **Visualization:** Streamlit, Gradio, PyVis (NetworkGraphs), Seaborn
*   **API:** FastAPI, Uvicorn
*   **Infrastructure:** Docker, Docker-compose

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- (Optional) CUDA-enabled GPU for accelerated training

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

## 🚦 Usage

### 1. The Orchestrator (Easiest)
Launch both the **Simple UI (Gradio)** and the **Technical Dashboard (Streamlit)** simultaneously:
```bash
python start.py
```
- **Simple UI:** http://localhost:7860
- **Technical Dashboard:** http://localhost:8501

### 2. API Server
Run the FastAPI production-ready server:
```bash
python src/interfaces/api_server.py
```
*API docs available at:* http://localhost:8000/docs

### 3. Training & Optimization
To re-train the model or run hyperparameter tuning with Optuna:
```bash
python scripts/train_model.py
```

---

## 📂 Project Structure

```text
├── app/                  # UI Interfaces (Streamlit/Gradio)
├── config/               # YAML configurations for models & constraints
├── data/                 # Raw and processed datasets (Git-ignored)
├── models/               # Saved model artifacts & preprocessors
├── scripts/              # Training and utility scripts
├── src/                  # Core Library Logic
│   ├── analytics/        # Stress testing & Stability
│   ├── counterfactuals/  # NSGA-II Generation
│   ├── explainability/   # SHAP/LIME wrappers
│   ├── fairness/         # Bias detection
│   └── recourse/         # Graph-based pathfinding
└── tests/                # Unit & Integration tests
```

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

---
**Developed by [mecartin](https://github.com/mecartin)**
