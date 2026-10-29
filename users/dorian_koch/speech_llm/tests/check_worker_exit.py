"""Guard: the podcast episode worker exits when its main loop raises, instead of hanging on its download threads.

``podcast_episode_main`` feeds a bounded queue from a ``ThreadPoolExecutor`` of downloaders. When anything outside the
per-episode ``try`` raised (the ASR server failing to start, a quota error writing progress), the producers stayed
blocked in ``q.put`` on the full queue and the executor's exit join kept the process, and its GPU, alive until walltime
(reproduced 2026-10-29). Checks: [1] a real ``_put`` on a full queue returns once ``_STOP`` is set, so the exit join
completes; [2] the worker's ``__main__`` sets ``_STOP`` on any exception, the ASR server is started before the pool,
and ``main`` never calls a blocking ``q.put``. Can fail: [1] with a plain ``q.put``.

Run from the setup root: ``CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_worker_exit.py``
"""

import ast
import importlib.util
import os
import queue
import threading
import time

PATH = os.path.join(os.getcwd(), "recipe/speech_llm/full_duplex/moshi_family/podcast_episode_main.py")
spec = importlib.util.spec_from_file_location("pem", PATH)
pem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pem)

q = queue.Queue(maxsize=1)
q.put("full")
t = threading.Thread(target=pem._put, args=(q, "blocked"))
t.start()
time.sleep(0.3)
pem._STOP.set()
t.join(timeout=5)
assert not t.is_alive(), "_put kept blocking on a full queue after _STOP"
pem._STOP.clear()
plain = threading.Thread(target=lambda: q.put("x", timeout=3), daemon=True)
plain.start()
plain.join(timeout=1.5)
assert plain.is_alive(), "the fixture cannot tell a blocking put from _put"
print("[1] _put on a full queue returns once _STOP is set (a plain put keeps blocking)")

src = open(PATH).read()
tree = ast.parse(src)
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
main_src = ast.get_source_segment(src, main)
assert "q.put(" not in main_src, "main() calls a blocking q.put"
assert main_src.index("AsrClient(") < main_src.index("ThreadPoolExecutor("), "the pool starts before the ASR server"
guard = src[src.index('if __name__ == "__main__":') :]
assert "_STOP.set()" in guard and "except BaseException" in guard, guard
print("[2] __main__ sets _STOP on an exception; ASR starts before the pool; no blocking put in main")
print("check_worker_exit: OK")
