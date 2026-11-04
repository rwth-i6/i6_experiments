"""Wall-time budgets from measured audio durations of the pinned benchmark.

Durations are from completed native Granite runtime.json exports (same HF
datasets), not nominal corpus sizes. Resource-only changes preserve job ids.
"""

import math

AUDIO_SECONDS = {
    "gigaspeech-cleaned": 126515.196,
    "spgispeech": 360007.4995,
}


def apply_runtime_budget(job, dataset, *, max_rtf=1.5):
    if dataset not in AUDIO_SECONDS:
        return job
    # RTF <1.5, 10% margin and two hours of setup; SPGISpeech fits in
    # seven days. Never bypass the cluster limit, even for explicit overrides.
    hours = math.ceil(AUDIO_SECONDS[dataset] / 3600 * max_rtf * 1.1 + 2)
    job.rqmt["time"] = min(168, max(job.rqmt.get("time", 0), hours))
    job.rqmt.pop("runtime_cap_hours", None)
    if dataset == "spgispeech" and job.rqmt.get("gpu", 0):
        job.rqmt["gpu_mem"] = max(48, job.rqmt.get("gpu_mem", 0))
    return job
