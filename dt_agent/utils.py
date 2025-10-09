import logging

def setup_logging(log_file_path):
    """Update logging configuration to use the correct format"""
    root_logger = logging.getLogger()

    # Remove existing file handlers to avoid duplicate logging
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    # Add new file handler pointing to experiment directory
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(root_logger.level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    print(f"Logging recorded to: {log_file_path}")
