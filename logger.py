"""Contains the class to manage logs."""


import logging
import os
import colorlog
import pytz
import pprint
from datetime import datetime
from logging.handlers import RotatingFileHandler


SAVE_LOGS = True
TIMEZONE = 'Europe/Athens'
LOGGING_PATH = "logs/"


def format_time_with_timezone(timestamp, timezone_str='Europe/Athens'):
    """Format timestamp according to the specified timezone."""
    tz = pytz.timezone(timezone_str)
    dt = datetime.fromtimestamp(timestamp)
    local_dt = dt.astimezone(tz)
    return local_dt.strftime('%Y-%m-%d %H:%M:%S %z')


class Logger:
    """Class to log messages with colors and manage the logs in general."""
    
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    COLOURED_LOG_FORMAT = f"%(log_color)s{LOG_FORMAT}"
    
    log_colors = {
        'DEBUG': 'white',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'magenta',
    }
    
    timezone: str
    
    def __init__(
        self,
        name,
        save_logs: bool = SAVE_LOGS,
        logs_filename: str = "project_logs",
        timezone: str = TIMEZONE
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatter with timezone support
        formatter = colorlog.ColoredFormatter(
            self.COLOURED_LOG_FORMAT,
            log_colors=self.log_colors,
            datefmt='%Y-%m-%d %H:%M:%S %z'
        )
        formatter.converter = lambda x: datetime.now(pytz.timezone(timezone)).timetuple()
        
        # Set up console handler with color formatter
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if save_logs:
            os.makedirs(LOGGING_PATH, exist_ok=True)
            file_handler = RotatingFileHandler(
                filename=os.path.join(LOGGING_PATH, f"{logs_filename}.log"),
                mode='a',
                maxBytes=15000000,
                backupCount=5
            )
            # Use the same timezone for file logs
            file_formatter = logging.Formatter(self.LOG_FORMAT)
            file_formatter.converter = lambda x: datetime.now(pytz.timezone(timezone)).timetuple()
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def pprint_format(self, obj_to_pprint: any) -> str:
        """Format message using pprint."""
        if isinstance(obj_to_pprint, dict):
            return pprint.pformat(obj_to_pprint, sort_dicts=False)
        return pprint.pformat(obj_to_pprint, indent=2)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
        
    def info_pprint(self, message: str, obj_to_pprint: any):
        """Log using pprint formatting.
        
        Example usage:
            config = {"key": "value"}
            logger.info_config("Using the following configuration:", config)
        """
        formatted_config = self.pprint_format(obj_to_pprint)
        self.logger.info(f"{message}\n{formatted_config}")
    
    def error_pprint(self, message: str, obj_to_pprint: any):
        """Log using pprint formatting."""
        formatted_config = self.pprint_format(obj_to_pprint)
        self.logger.error(f"An error occurred: {message}\n{formatted_config}")
        
    def warning_pprint(self, message: str, obj_to_pprint: any):
        """Log using pprint formatting."""
        formatted_config = self.pprint_format(obj_to_pprint)
        self.logger.warning(f"{message}\n{formatted_config}")
        
    def debug_pprint(self, message: str, obj_to_pprint: any):
        """Log using pprint formatting."""
        formatted_config = self.pprint_format(obj_to_pprint)
        self.logger.debug(f"{message}\n{formatted_config}")
        
    def critical_pprint(self, message: str, obj_to_pprint: any):
        """Log using pprint formatting."""
        formatted_config = self.pprint_format(obj_to_pprint)
        self.logger.critical(f"{message}\n{formatted_config}")