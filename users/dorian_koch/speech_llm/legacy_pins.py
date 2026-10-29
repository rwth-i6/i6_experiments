"""Legacy pins: the jobs that deliberately keep a KNOWN-WRONG behaviour, by job id, so nothing new copies it.

A fix that changes a job's output keeps every finished job's hash by pinning the old value at the old call sites (a
hash-excluded constructor knob at its old value). Copying such a call site would silently give a NEW job the old bug.
So each pin lists the exact jobs allowed to carry it (frozen 2026-10-29, the graph after the code audit), and
:func:`check_graph` -- called at the end of the recipe config -- refuses any other job with that value: a new job
uses the default (the fixed behaviour). Never add an id here; a job that needs the old behaviour for a comparison
says so in a new knob with its own name.

Guard: ``tests/check_legacy_pins.py``.
"""

#: (job class, attribute, legacy value) -> (why the value is wrong, the job ids allowed to keep it)
LEGACY_PINS = {
    ("LLMPreprocess", "gold_from_source", False): (
        "the LLM echoes the gold: aliases sent joined by ', ' come back split ('Hitler, Adolph' -> two aliases); a new set takes answer/aliases from the source rows (the default)",
        frozenset(
            {
                "EF2tj6XtWeBW",
                "Ej7GQgQQIMv0",
                "RcwT9WILq9uu",
                "k4pJURHa6tXT",
                "utrsH3mfFshX",
            }
        ),
    ),
    ("PodcastDialogueSlice", "clamp_row_start", False): (
        "a word straddling a row start gets a negative start and is dropped from the text stream; a new slicing clamps it to 0 (the default)",
        frozenset(
            {
                "02iPYWv4hMsh",
                "0rsWVibPdGXO",
                "1XazfGDV6amk",
                "1zUKi3rdG8KB",
                "2IJnAkzguDlx",
                "2RZl1H5mtqI4",
                "2UZNHhQH8B63",
                "2hLJCtMoCcJD",
                "2ySiQxxCQhdm",
                "3L7gNmvVRH8A",
                "3fKxnHMbUm0S",
                "56OUHATxcyvM",
                "5IUOfo5IkbR6",
                "5VmOYklJiZus",
                "5ufpeGN91Nz7",
                "6DACrrnEZr4i",
                "6O8piLvMNmRq",
                "6l5bO1dX1VNf",
                "6zCVzivi0wzS",
                "7Vw1v0dfTFck",
                "8Y4LnXsnUqUs",
                "8nwwSYxOdVlX",
                "8sNBH1agRH2T",
                "9gBllZxkSxYL",
                "ALymYSWvVeoq",
                "B0IzvKeVaaPv",
                "BJEnzoGOGZ9F",
                "BqaWY2SmMXY2",
                "CBydnFPtHmD1",
                "CIgHVxSoHiVK",
                "Doc8w5qxfvlL",
                "DvJMyZNMH3uT",
                "E60rLO6pIlcV",
                "FzUiXvi3WsK7",
                "GKTio4kEW6R4",
                "Gf09zuYCkPUu",
                "HVqPb3AnTRxZ",
                "IAQNnN6Fu6lt",
                "JVRvjmqKxuBv",
                "JqHOhT6E4a6q",
                "JtW59OfayCJZ",
                "KNedNkKljXB0",
                "LB2U3XVYceJS",
                "LVhChnua18Uu",
                "MSzIjCi8PlbG",
                "MZ7bguDEvpxb",
                "NKRP17TBMGse",
                "NYkqG5P1COcC",
                "OCwNoFGQ48ft",
                "OhV88bYlRRVs",
                "PKNIXvm320WA",
                "PWgpVuJn6J0K",
                "QDuzju9rCTjn",
                "QNLTUR7jTwJ1",
                "Rgdzz84170nM",
                "SBgBhNS6IqYG",
                "SUgNOM9R80na",
                "SbGUH4p648k4",
                "SjY3ahmQDBKV",
                "Syn1QQMzcQkk",
                "TDTEQtq8AC2b",
                "TPNVWqMeyeum",
                "TaGy0bF0IRND",
                "TlIitSp45ipq",
                "UUhyfZf5ZpGr",
                "Ul4plKr1RQfY",
                "Up9StkbWz138",
                "VKibjD9BdHw5",
                "WAchHMtpJXP0",
                "WIQLLldILrWD",
                "WS6vIDFXz9R1",
                "WULNeSKfIWmU",
                "WcRKoDoFwJMP",
                "WdTq2Oc31DlQ",
                "WiPaNfaw1LjZ",
                "WxgwscC9sZ7J",
                "X7Y3zXZaxkMN",
                "X80lzViTS7Bg",
                "XOf5i1HrJKIO",
                "XcpTFUFSPHxB",
                "Yaku53zqLS3u",
                "YiFDy4EQEVJR",
                "YnKC7TxWWgRS",
                "YzA7bLxaxnud",
                "ZuXbIVHd8zVJ",
                "aTnypqiSrnhk",
                "bDfALZIiqxiS",
                "cUB83WuVsZxD",
                "dD7rJtIeGOsW",
                "dESOKrAhQi1Q",
                "dNP39I0q8Uqd",
                "daRcAyo1W1fq",
                "dlNI8qhFRJ2j",
                "fwpFOWLCSFqi",
                "gknVt0DYi5Bj",
                "glppLT0jCWwb",
                "h6tSMH7CkV1t",
                "hAbXlAB4agG3",
                "hGkWoIl5GA88",
                "hR2iiE48A0rj",
                "hSWMwKKN3DQa",
                "huW2SaO6hSEM",
                "i8E6Volt0hZx",
                "iMxgJgbzN96v",
                "jpA87724nS2M",
                "kYATIPXJq9cF",
                "kajBuREnooRC",
                "knUHMOVF8c27",
                "lEwcACBl6w5m",
                "lNW7zjnrCyIv",
                "lQJdmtkSRab1",
                "m1OciWrxK3DI",
                "m66D3OYNqluI",
                "mytikxr57nqS",
                "nH0zmNZrzw9l",
                "nIuNyCY9WCnz",
                "ne9vjSamVVSR",
                "ntk0e0D67Gtg",
                "nwWfWwCUojWa",
                "ooBM8ZcOAXYy",
                "oqU2aGyuQqlY",
                "oyF2vSEtAKEF",
                "ozOyrV6pnm5c",
                "pHpuvHjPVQ1M",
                "rTrrSuiPTfyW",
                "sLPDJPywmwLg",
                "sUifGhwcJC6R",
                "sY55BgpnmrIQ",
                "sYkwWkWDv295",
                "shQ8eoHceJrf",
                "t08FakEtCza4",
                "tGSuu7dgExEn",
                "tPvAAGi1IgPM",
                "tYTx52qK0w4d",
                "tezApx1q04Me",
                "twhDLYhXn7FW",
                "u0n2AML4Xpu4",
                "uL8KNEQqBlVL",
                "uXkaOiM5IH2N",
                "uylIQM37dgfh",
                "v8Fu2P8tgTOK",
                "vLDNsn92V6uf",
                "wwTie1zVrZQf",
                "x97b3XLB1t3e",
                "xNZwdzPOGAFI",
                "xSZUjENohzoL",
                "xYHoRN5zbBs9",
                "yQVk0mtKx8NA",
                "z0hIRS0vb2GY",
                "zksiJJoDymw0",
            }
        ),
    ),
    ("FisherToMoshiTrainData", "whole_conversations", False): (
        "keeps a window only if it fits, i.e. 2.9% of the calls, not one row per call; a new corpus uses the default (True)",
        frozenset(
            {
                "98IFVP8879x4",
                "E6HnLS4wyaba",
                "JJi8ACF3GodT",
            }
        ),
    ),
    ("MoshiRolloutToDialogue", "filters_version", 1): (
        "v1 cuts decimals and abbreviations into false sentence ends and flags plural possessives as glitches; a new corpus uses the default (2)",
        frozenset(
            {
                "I9uskTK6Bl42",
            }
        ),
    ),
    ("KnowledgeUserWordsCTC", "drop_rows_with_failed_turns", False): (
        "a row with a failed CTC turn trains PAD over speech; a new sidecar drops such rows (the default)",
        frozenset(
            {
                "1vE3btcUzFCA",
            }
        ),
    ),
    ("MTRDuplexEval", "redetect_pause", False): (
        "scores the conversational-features pause cell with the old detector's (often empty) pause windows; a new "
        "evaluation re-detects (mtr_benchmark_py passes True)",
        frozenset(
            {
                "4I4XanPdIDIt",
                "4aJgrZa7DoJj",
                "57Kt4i299qiB",
                "98lt8J9e77C8",
                "EnKRbmMFslBS",
                "NM91cT8TpIzW",
                "RhawaakAFNUk",
                "svX1CQJ9AVVr",
                "wfL2U0eRC2O5",
            }
        ),
    ),
    ("MTRDuplexInference", "engine_version", 1): (
        "PersonaPlex on the pre-port engine (ring-KV off-by-one, no reset) / MoshiRAG retrieving once per dialogue "
        "with no turn history; a new cell runs the fixed engine (mtr_benchmark_py passes 2)",
        frozenset(
            {
                "1iKc684D7PO8",
                "3FiOMMAgAAs7",
                "7A2NgHtn55xG",
                "UvZ6ih2zWSXX",
                "mImrydfA0OkZ",
                "qgeC9LoTWg2g",
                "wQldmM6WH33A",
                "mKtYiFWMJWfn",
            }
        ),
    ),
}


#: Pins that concern only some jobs of a class (the value is right elsewhere): (class, attribute, value) -> predicate.
APPLIES = {
    ("MTRDuplexEval", "redetect_pause", False): lambda j: j.dimension == "conv_features" and j.scenario == "pause",
    ("MTRDuplexInference", "engine_version", 1): lambda j: j.mtr_backend in MTR_FIXED_ENGINE_BACKENDS,
}

#: MTR driver backends whose engine was fixed after their first cells ran (engine_version 2 = the fixed engine).
MTR_FIXED_ENGINE_BACKENDS = ("personaplex", "moshirag")


def pinned_or_default(job_cls, attr: str, legacy_value, new_value=None, **kwargs):
    """``job_cls(**kwargs, attr=legacy_value)`` if that is one of the pinned jobs (a job built before the fix, same
    hash), else ``job_cls(**kwargs)`` with the default (the fixed behaviour). The one way a call site that once pinned
    the legacy value builds its job, so a NEW job from the same code never inherits it. ``new_value``: the value a new
    job gets when the class default is still the legacy one (None = the class default)."""
    legacy = job_cls(**kwargs, **{attr: legacy_value})
    _why, ids = LEGACY_PINS[(job_cls.__name__, attr, legacy_value)]
    if legacy._sis_id().rsplit("/", 1)[-1].split(".", 1)[1] in ids:
        return legacy
    return job_cls(**kwargs) if new_value is None else job_cls(**kwargs, **{attr: new_value})


def _value(job, attr):
    return getattr(job, attr, None)


def violations(jobs) -> list:
    """[(job id, class, attribute, value, why)] for every job that carries a legacy value without being pinned."""
    out = []
    for job in jobs:
        cls = type(job).__name__
        for (pcls, attr, val), (why, ids) in LEGACY_PINS.items():
            if cls != pcls or _value(job, attr) != val or not APPLIES.get((pcls, attr, val), lambda j: True)(job):
                continue
            jid = job._sis_id().rsplit("/", 1)[-1].split(".", 1)[1]
            if jid not in ids:
                out.append((job._sis_id(), cls, attr, val, why))
    return out


def check_graph() -> None:
    """Raise if any job in the Sisyphus graph copies a legacy pin (see the module docstring)."""
    from sisyphus import graph

    bad = violations(graph.graph.jobs())
    if bad:
        msg = "; ".join(f"{j}: {a}={v!r} is a legacy pin ({w})" for j, _c, a, v, w in bad[:5])
        raise RuntimeError(f"{len(bad)} new job(s) copy a legacy pin -- use the default instead. {msg}")
