import logging
import sys
from datetime import datetime

class TerminalFormatter(logging.Formatter):
    """Custom logging formatter to meet specific format and color requirements."""
    
    # ANSI color codes
    RED = '\033[91m'
    YELLOW = '\033[93m'
    LIGHT_BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'

    def format(self, record):
        time_str = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        script_name = record.filename
        level_name = record.levelname.lower()
        msg = record.getMessage()
        
        if record.levelno >= logging.ERROR:
            color = self.RED
        elif record.levelno >= logging.WARNING:
            color = self.YELLOW
        elif record.levelno >= logging.INFO:
            color = self.LIGHT_BLUE
        elif record.levelno >= logging.DEBUG:
            color = self.MAGENTA
        else:
            color = self.RESET
            
        formatted_message = f"{time_str} | {level_name} | {script_name} | {msg}"
        return f"{color}{formatted_message}{self.RESET}"

def get_logger(name="robotics_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # The project path / messages may contain non-ASCII characters (e.g. a
        # Greek "Ρομποτικά" folder). The default Windows console encoding is
        # cp1252, which can't encode those and would raise UnicodeEncodeError
        # inside logging. Force UTF-8 with errors='replace' so logging never
        # crashes regardless of what characters appear in a path or message.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # stream doesn't support reconfigure (e.g. already wrapped)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(TerminalFormatter())
        logger.addHandler(console_handler)
        
    return logger

# Initialize the default logger
TerminalLogger = get_logger()
