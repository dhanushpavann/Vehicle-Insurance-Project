import logging 
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

# Constants for log configuration
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024 # 5 MB
BACKUP_COUNT = 3 # Number of backup files to keep

# Construct log file path
# log_dir_path = os.path.join(from_root(),LOG_DIR) #for windows from_root is suitable
log_dir_path = os.path.join(os.getcwd(), LOG_DIR) #for mac
os.makedirs(log_dir_path,exist_ok=True)
log_file_path=os.path.join(log_dir_path,LOG_FILE)

def configure_logger():
    """
    Configure logging with a rotating file handler and a console handler.
    """

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Define the formatter
    formatter = logging.Formatter("[%(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    

    # File Handler with rotation
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE,backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Add Handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Configure the logger
logger = configure_logger()

