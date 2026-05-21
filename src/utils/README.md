# Utils Module

Shared helper modules for logging and diagnostics. No simulation or control logic lives here.

---

## TerminalLogger

`src/utils/terminal_logger.py`

A colored, formatted terminal logger built on Python's standard `logging` module.

### Usage

```python
from src.utils.terminal_logger import TerminalLogger as logger

logger.debug("message")    # magenta
logger.info("message")     # light blue
logger.warning("message")  # yellow
logger.error("message")    # red
```

`TerminalLogger` is a module-level singleton created by `get_logger()`. All imports across the project share the same logger instance.

### Log Format

```
YYYY-MM-DD HH:MM:SS | level | filename.py | message
```

Example:

```
2025-09-01 14:32:05 | info | mujoco_env.py | Loading MJCF model from: .../scene.xml
```

### API

| Function | Description |
|----------|-------------|
| `get_logger(name="robotics_logger")` | Factory that returns (or creates) a named logger with `TerminalFormatter` attached |

---

## DataLogger

`src/utils/data_logger.py`

Appends structured telemetry to a JSONL file for offline analysis and debugging.

### Constructor

```python
DataLogger(log_dir: str = "logs", filename: str = "test_run.jsonl")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_dir` | `"logs"` | Directory relative to the project root where log files are written |
| `filename` | `"test_run.jsonl"` | Log file name |

The log directory is created automatically if it does not exist. The path is always resolved relative to the project root (two levels up from this file), regardless of the current working directory.

### Method: `log_input`

```python
log_input(context: str, data) -> None
```

Appends one JSON entry to the log file.

| Parameter | Description |
|-----------|-------------|
| `context` | Label for the log entry (e.g., `"PD Torques"`, `"Gravity/Coriolis Torques"`) |
| `data` | Any JSON-serializable value — typically a numpy array or dict |

### Output Format

One JSON object per line (JSONL):

```json
{"timestamp": "2025-09-01T14:32:05.123456", "context": "PD Torques", "data": [0.12, -0.05, ...]}
{"timestamp": "2025-09-01T14:32:05.124000", "context": "Gravity/Coriolis Torques", "data": [1.1, 0.3, ...]}
```

### Module-level Instance

A default instance is available at module level:

```python
from src.utils.data_logger import data_logger

data_logger.log_input(context="my context", data={"key": "value"})
```

This writes to `logs/test_run.jsonl` in the project root.
