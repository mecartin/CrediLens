import sys
from loguru import logger
from pathlib import Path

def setup_logger(log_dir: str = "logs", level: str = "INFO"):
    """
    Configure loguru logger.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level
    )
    
    # Add file handler
    logger.add(
        log_path / "credilens_{time:YYYY-MM-DD}.log",
        rotation="100 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    return logger

# Default logger instance
logger = setup_logger()
