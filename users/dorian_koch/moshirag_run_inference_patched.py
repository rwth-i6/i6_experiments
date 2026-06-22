# Our wrapper around the (third-party, venv-managed) moshi fork's batch inference entry point.
"""Run ``moshi.run_inference`` with the batched step loop instrumented to self-heal + fail loud.

**Confirmed deadlock (the 193/200 strand).** The fork steps the batch as a background task
``state._step_loop()`` (run_inference.py:91) which pulls per-slot inputs from each occupant's
``input_queue`` (server.py ``_gather_step_inputs``). Each job's ``_feed_loop`` only advances when
``step_index`` advances (``_wait_step_index_at_least``), and ``step_index`` is incremented **only**
inside ``InferenceJob._output_loop`` (inference_job.py:319) -- which is gated at the top of its loop
by ``while self._doing_retrieval: await asyncio.sleep(...)`` (inference_job.py:264-270). So while a
clip's RAG retrieval is in flight, that job stops incrementing ``step_index``. ``_doing_retrieval``
is cleared **only** when the retrieval background task (``RAGManager._background_task``) finishes and
calls ``_catch_reference_text`` (inference_job.py:182). If that background task ever exits WITHOUT
clearing it (cancellation via ``_cancel_and_await_pending``, or any edge path), ``_doing_retrieval``
stays ``True`` forever -> ``_output_loop`` spins -> ``step_index`` frozen -> every feeder blocks ->
all ``input_queue``s drain to empty -> ``_step_loop`` finds no input -> the batch wedges. The stalled
slots show ``in=0,out=0`` (queues empty, exactly what we observed). The inner LLM call IS timeout-
bounded (``asyncio.wait_for`` in llm_reference_generator), so a slow LLM alone can't hang it; the
freeze is the ``_doing_retrieval``-stuck-True condition.

This wrapper replaces ``ServerState._step_loop`` with a drop-in that:
  (a) **self-heals** the above deadlock: when no step input has progressed for
      ``MOSHIRAG_RETRIEVAL_STUCK_S`` while slots are occupied, any occupant still stuck with
      ``_doing_retrieval == True`` is force-cleared (degraded: that clip proceeds with an empty
      reference) so the batch makes progress instead of stranding. Logged loudly.
  (b) only **fails loud** (raises, cancelling siblings so the process exits with a real traceback)
      if the stall persists past ``MOSHIRAG_STEP_STALL_S`` with nothing left to clear -- a genuine
      starvation, not the known retrieval freeze.
No fork edits (it's a CreateVenv-managed site-packages, wiped on rebuild); we hand off to ``main()``.
"""

import asyncio
import logging
import os
import time

import moshi.server as _server

logger = logging.getLogger("moshirag.wrapper")

# After this many seconds of no step-input progress while slots are occupied, force-clear any
# occupant stuck in retrieval (the known deadlock) and keep going. Should be >> rag_timeout (10s).
_RETRIEVAL_STUCK_S = float(os.environ.get("MOSHIRAG_RETRIEVAL_STUCK_S", "60"))
# Hard backstop: if the loop is still starved this long with nothing left to self-heal, fail loud.
_STEP_STALL_S = float(os.environ.get("MOSHIRAG_STEP_STALL_S", "180"))


def _slot_debug(state) -> str:
    rows = []
    for i, occ in enumerate(state.slots):
        if occ is None:
            continue
        iq = getattr(getattr(occ, "input_queue", None), "qsize", lambda: "?")()
        oq = getattr(getattr(occ, "output_queue", None), "qsize", lambda: "?")()
        dr = getattr(occ, "_doing_retrieval", "?")
        rows.append(f"slot{i}={type(occ).__name__}(in={iq},out={oq},retr={dr})")
    return ", ".join(rows) or "<none occupied>"


def _unstick_retrieval(state) -> int:
    """Force-clear any occupant frozen with ``_doing_retrieval=True``. Returns how many were cleared."""
    cleared = 0
    for occ in state.slots:
        if occ is not None and getattr(occ, "_doing_retrieval", False):
            occ._doing_retrieval = False  # let _output_loop's spin exit -> step_index resumes
            cleared += 1
    return cleared


async def _guarded_step_loop(self):
    """Drop-in for ServerState._step_loop: self-heals the retrieval deadlock, else fails loud."""
    idle_since = None
    healed_at = None  # monotonic time of the last self-heal, for the failure message
    try:
        while True:
            ran = await self.run_one_step()
            if ran:
                idle_since = None
                await asyncio.sleep(0)
                continue
            if any(s is not None for s in self.slots):
                now = time.monotonic()
                if idle_since is None:
                    idle_since = now
                stalled_for = now - idle_since
                # (a) self-heal: clear stuck retrievals once we've been starved a bit.
                if stalled_for > _RETRIEVAL_STUCK_S:
                    cleared = _unstick_retrieval(self)
                    if cleared:
                        logger.error(
                            "[Step] step input starved %.0fs; force-cleared _doing_retrieval on %d "
                            "occupant(s) to break the retrieval deadlock [%s]",
                            stalled_for,
                            cleared,
                            _slot_debug(self),
                        )
                        idle_since = None  # give the unblocked output loops time to feed again
                        healed_at = now
                        await asyncio.sleep(0.005)
                        continue
                # (b) genuine starvation (nothing to heal): fail loud past the hard backstop.
                if stalled_for > _STEP_STALL_S:
                    raise RuntimeError(
                        f"step loop stalled: slot(s) occupied but no step input for "
                        f"{_STEP_STALL_S:.0f}s (no stuck retrieval to clear; "
                        f"last_heal={healed_at}) [{_slot_debug(self)}]"
                    )
            else:
                idle_since = None
            await asyncio.sleep(0.005)
    except asyncio.CancelledError:
        logger.info("[Step] step loop cancelled")
        raise
    except BaseException:
        logger.exception("[Step] step loop CRASHED [%s]; cancelling in-flight jobs to fail loudly", _slot_debug(self))
        current = asyncio.current_task()
        for task in asyncio.all_tasks():
            if task is not current:
                task.cancel()
        raise


_server.ServerState._step_loop = _guarded_step_loop

from moshi.run_inference import main  # noqa: E402  -- must import after the monkeypatch

if __name__ == "__main__":
    main()
