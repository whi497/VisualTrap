from glob import glob
from pathlib import Path
import json, ujson
import os
import logging
import torch
import random
import contextlib
import numpy as np
import torch
import time
from functools import wraps
import concurrent.futures as futures
import signal
from functools import wraps
from contextlib import ContextDecorator
from pprint import pprint
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_jsonl(file_path):
    datas = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            datas.append(ujson.loads(line))
    return datas
    

def write_jsonl(datas, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        lines = []
        for data in datas:
            lines.append(ujson.dumps(data, ensure_ascii=False))
        f.write("\n".join(lines))


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return ujson.load(f)


def write_json(datas, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        ujson.dump(datas, f, ensure_ascii=False, indent=4)


def rd_js(file_path, etd=None):
    file_path = str(file_path)
    etd = etd or Path(str(file_path)).name.split(".")[-1]
    if etd == "json":
        return read_json(file_path)
    else:
        return read_jsonl(file_path)


def wr_js(data, file_path):
    file_path = str(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if str(file_path).endswith(".json"):
        write_json(data, file_path)
    else:
        write_jsonl(data, file_path)


@contextlib.contextmanager
def temp_seed(seed):
    random_state = random.getstate()
    state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
        random.setstate(random_state)


def derangement(s):
    d = s[:]
    while any([a == b for a, b in zip(d, s)]):
        random.shuffle(d)
    return d


class context_timeout(ContextDecorator):
    def __init__(self, seconds, *, timeout_exception=TimeoutError):
        self.seconds = seconds
        self.timeout_exception = timeout_exception
        self._original_handler = None

    def _handle_timeout(self, signum, frame):
        raise self.timeout_exception("Operation timed out")

    def __enter__(self):
        self._original_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        signal.alarm(0)  # Disable the alarm
        signal.signal(signal.SIGALRM, self._original_handler)  # Restore original handle


class TimeoutError(Exception):
    pass


def function_timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wrapper_args = args  # Store args for access in _handle_timeout
            wrapper_kwargs = kwargs  # Store kwargs for access in _handle_timeout

            def _handle_timeout(signum, frame):
                args_repr = [repr(a) for a in wrapper_args]  # Use stored args
                kwargs_repr = [
                    f"{k}={v!r}" for k, v in wrapper_kwargs.items()
                ]  # Use stored kwargs
                signature = ", ".join(args_repr + kwargs_repr)
                error_message = f"Function {func.__name__}({signature}) timed out"
                raise TimeoutError(error_message)

            # Set signal handler for SIGALRM
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)  # Schedule an alarm
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Clear the scheduled alarm
            return result

        return wrapper

    return decorator
