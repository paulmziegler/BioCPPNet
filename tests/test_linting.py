import subprocess
import logging
from src.utils import setup_logger

logger = setup_logger("test_linting")

def test_codebase_linting():
    """
    Runs the project linter (via manage.py or direct ruff call) and fails if issues are found.
    Captures output to the session log.
    """
    logger.info("Starting Codebase Linting Check...")
    
    # Run lint command
    # using manage.py ensures we use the project standard
    cmd = ["python", "manage.py", "lint"]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=False # Don't raise immediately, we want to log output
        )
        
        # Log stdout/stderr
        if result.stdout:
            logger.info(f"Linter Output:\n{result.stdout}")
        if result.stderr:
            logger.error(f"Linter Errors:\n{result.stderr}")
            
        # Assert success
        assert result.returncode == 0, f"Linting failed with return code {result.returncode}"
        logger.info("Linting Passed.")
        
    except Exception as e:
        logger.exception("Failed to run linter process.")
        raise e
