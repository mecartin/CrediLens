import subprocess
import sys
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Project path configuration
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
VENV_STREAMLIT = PROJECT_ROOT / "venv" / "Scripts" / "streamlit.exe"

def run_gradio():
    """Launch the simple Gradio interface."""
    print("🚀 Starting Gradio (Simple Interface) on port 7860...")
    gradio_path = PROJECT_ROOT / "app" / "simple" / "credilens_simple.py"
    try:
        subprocess.run([str(VENV_PYTHON), str(gradio_path)], check=True)
    except Exception as e:
        print(f"❌ Gradio failed to start: {e}")

def run_streamlit():
    """Launch the technical Streamlit dashboard."""
    print("📊 Starting Streamlit (Technical Dashboard) on port 8501...")
    dashboard_path = PROJECT_ROOT / "app" / "technical" / "credilens_dashboard.py"
    try:
        subprocess.run([str(VENV_STREAMLIT), "run", str(dashboard_path)], check=True)
    except Exception as e:
        print(f"❌ Streamlit failed to start: {e}")

def main():
    print("="*60)
    print("🛡️  CrediLens Recourse Engine - Orchestrator 🛡️")
    print("="*60)
    print(f"Environment: {VENV_PYTHON}")
    print("Launching all modules...")
    
    # Check if necessary files exist
    if not (PROJECT_ROOT / "models" / "saved_models" / "xgb_model.pkl").exists():
        print("⚠️ Warning: Model artifacts not found. Please run training first if needed.")

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(run_gradio)
        # Delay to ensure Gradio starts up first or ports don't clash (though they shouldn't)
        time.sleep(2)
        executor.submit(run_streamlit)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping all services...")
        sys.exit(0)
