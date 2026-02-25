import logging
import sys
from app.config import APP_NAME, DEBUG


def init_logger():

    # Log level
    level = logging.DEBUG if DEBUG else logging.INFO

    # Custom formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    # Root logger (uvicorn included)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]

    # Stop verbose logger
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Logger app
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(level)

    return logger


# Instance
logger = init_logger()
