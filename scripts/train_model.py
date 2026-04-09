import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.logger import logger
from src.core.config import ConfigManager
from src.models.trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Train CrediLens V4 Model")
    parser.add_argument("--data", type=str, default="d:/credilens_ver2/accepted_2007_to_2018Q4.csv", help="Path to input dataset (CSV)")
    parser.add_argument("--sample", type=int, default=10000, help="Number of rows to sample (set to 0 for full dataset)")
    parser.add_argument("--save-dir", type=str, default="d:/credilens_ver2/credilens-v4-full/models/saved_models", help="Directory to save models")
    
    args = parser.parse_args()
    
    logger.info("Starting CrediLens V4 Model Training Script")
    
    config_manager = ConfigManager()
    trainer = ModelTrainer(config_manager)
    
    sample_size = args.sample if args.sample > 0 else None
    
    try:
        model, metrics = trainer.train_full_pipeline(
            filepath=args.data,
            sample_size=sample_size,
            save_path=args.save_dir
        )
        
        logger.info(f"Training completed successfully. Metrics: {metrics}")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
