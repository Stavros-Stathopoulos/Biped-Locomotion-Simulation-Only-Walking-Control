import json
import os
from datetime import datetime

class DataLogger:
    """Logs data (e.g., controller inputs) to a JSONL file."""
    
    def __init__(self, log_dir="logs", filename="test_run.jsonl"):
        # Get the absolute path to the root of the project to ensure correct logs directory
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.log_dir = os.path.join(base_path, log_dir)
        self.filename = filename
        self.filepath = os.path.join(self.log_dir, self.filename)
        
        # Ensure the log directory exists 
        os.makedirs(self.log_dir, exist_ok=True)
        
    def log_input(self, context: str, data:str):
        """
        Logs controller input data to the JSONL file.
        The data parameter should be a dictionary or something JSON-serializable.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "data": data
        }
        
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

# Initialize a default data logger
data_logger = DataLogger()
