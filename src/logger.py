import logging
import colorlog

from myconfig import LOG_PATH, LOG_ERROR_PATH

# format logs
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
color_format = '%(log_color)s' + log_format
formatter_console = colorlog.ColoredFormatter(color_format)
formatter_file = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# all logs
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(LOG_PATH)
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter_file)
console_handler.setFormatter(formatter_console)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# error logs
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.ERROR)
file_handler_error = logging.FileHandler(LOG_ERROR_PATH)
console_handler_error = logging.StreamHandler()
console_handler_error.setLevel(logging.ERROR)
file_handler_error.setFormatter(formatter_file)
console_handler_error.setFormatter(formatter_console)
error_logger.addHandler(file_handler_error)
error_logger.addHandler(console_handler_error)


if __name__ == "__main__":
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
    error_logger.error('This is an error message for error_logger')
