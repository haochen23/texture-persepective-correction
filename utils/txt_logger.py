import os
import logging
import time


def create_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    try:
        os.remove(log_dir + f'{name}.log')
    except:
        pass
    # file logger handler
    file_logger = logging.FileHandler(log_dir + f'{name}.log')
    file_logger.setLevel(logging.DEBUG)
    # console logger handler
    console_logger = logging.StreamHandler()
    console_logger.setLevel(logging.INFO)
    # create formatter and apply it to both handlers
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_logger.setFormatter(log_formatter)
    console_logger.setFormatter(log_formatter)
    # Add handlers to the logger
    logger.addHandler(file_logger)
    logger.addHandler(console_logger)

    return logger
