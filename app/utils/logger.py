import logging

def get_loger(name: str = "tryon") -> logging.Logger:
    """
    Returns a configured logger instance.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
