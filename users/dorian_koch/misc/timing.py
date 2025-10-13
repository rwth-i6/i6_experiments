import time
from functools import wraps
import inspect
import os


class Timing:
    def __init__(self, name: str):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.name = name
        print(f"> {name}", flush=True)

    def print(self, msg: str):
        if self.start_time == self.last_time:
            elapsed = time.time() - self.start_time
            print(f" - {self.name} [+{elapsed:.02f}s]: {msg}", flush=True)
        else:
            elapsed = time.time() - self.start_time
            elapsed_last = time.time() - self.last_time
            print(f" - {self.name} [+{elapsed_last:.02f}s, total {elapsed:.02f}]: {msg}", flush=True)
        self.last_time = time.time()

    def end(self):
        if self.start_time == self.last_time:
            elapsed = time.time() - self.start_time
            print(f"< END {self.name} [+{elapsed:.02f}s]", flush=True)
        else:
            elapsed = time.time() - self.start_time
            elapsed_last = time.time() - self.last_time
            print(f"< END {self.name} [+{elapsed_last:.02f}s, total {elapsed:.02f}]", flush=True)
        self.last_time = time.time()


def timing_decorator(func):
    # Inspect the function signature once
    sig = inspect.signature(func)
    accepts_timer = "timer" in sig.parameters

    # Get the filename where the function is defined
    full_path = func.__code__.co_filename
    filename = os.path.basename(full_path)  # just the filename

    @wraps(func)
    def wrapper(*args, **kwargs):
        timer = Timing(f"{filename}:{func.__name__}")
        try:
            if accepts_timer:
                kwargs["timer"] = timer
            return func(*args, **kwargs)
        finally:
            timer.end()

    return wrapper
