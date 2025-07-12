import json
import time
import os
from threading import Lock

_log_lock = Lock()

def log_openai_call(request: dict, response) -> None:
    """
    Append each OpenAI request and response as a JSON object
    (one per line) into logs/openai_calls.jsonl.
    """
    os.makedirs('logs', exist_ok=True)
    record = {
        "timestamp": time.time(),
        "request": request,
        # response may be an OpenAI response object with to_dict()
        "response": response.to_dict() if hasattr(response, "to_dict") else dict(response)
    }
    path = os.path.join('logs', 'openai_calls.jsonl')
    with _log_lock, open(path, 'a') as f:
        f.write(json.dumps(record) + "\\n")
