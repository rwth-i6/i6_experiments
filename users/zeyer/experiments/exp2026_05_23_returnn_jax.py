"""
Loquacious AED+CTC on RETURNN's JAX backend: the backend comparison to ``base-small-v2``.

Companion recipe with its OWN sis manager, started with the torch-2.12 env
(``/home/az668407/work/py-envs/py3.12-torch2.12/bin/python``), because JAX is not installed in the
production env and cannot be added there without upgrading ``nvidia-cudnn-cu12`` underneath
torch 2.7.1 (measured with ``pip --dry-run``: 9.5.1.17 -> 9.24.0.43), which the other sessions
train from. The manager's interpreter IS the jobs' interpreter
(``RETURNN_PYTHON_EXE = sys.executable`` in ``settings.py``), so nothing else has to change --
same setup dir, same settings, same RETURNN checkout (which already carries ``returnn/jax/``).
Same pattern as the grad-align setup's ``*_p212`` companion.

Start it with (its own tmux window, `--inspect` first to see what would be submitted)::

    ./hpc-sis-m.py --inspect --target 0:<win> \
        --setup-dir /rwthfs/rz/cluster/home/az668407/setups/2026-05-23-returnn-paper \
        --py /home/az668407/work/py-envs/py3.12-torch2.12/bin/python \
        recipe/i6_experiments/users/zeyer/experiments/exp2026_05_23_returnn_jax.py

Only the backend differs from ``base-small-v2`` -- same model, data, LR schedule -- so the
training curves are directly comparable. The model config comes from
:func:`small_model_overrides` rather than a copy, so it cannot drift from the baseline.
"""

from __future__ import annotations

import subprocess
from typing import Optional, Tuple

from sisyphus import Job, Task, tk

from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output

from .exp2026_05_23_returnn import loq_train, small_model_overrides, _loq_batch_size_factor, TrainStepBenchmarkJob


# Declared input shapes for the compiled step. The JAX step is compiled per input signature and a
# compile costs ~200 s, so the shapes are DECLARED rather than discovered: every batch is padded
# into the first bucket it fits, all buckets are compiled at startup, and nothing compiles during
# training (a batch fitting none is an error, not a compile).
#
# Derived from the PT baseline's own training log (job ReturnnTrainingJob.je6PefFx3gz2, log.run.1,
# ep 100: 3541 batches, 2722 distinct shapes -- so per-shape compilation is hopeless). Grouping
# those by padded length into 8 buckets costs 1.21x the batcher's own padding. Every bucket is
# ~one batch's worth of work (16-23M samples, vs batch_size 16M): a bucket must be, since
# max_seqs x max_seq_length would be 64M and OOMs (measured: 44.7 GiB allocation, fp32).
# Audio levels are MULTIPLES OF time_multiple (16000) and text of 8, because the engine rounds a
# batch up to those before matching a bucket -- buckets derived from raw observed lengths sit just
# below the rounded shapes and cover nothing (measured: batch (200, 96000) fit no bucket, run died).
#
# Audio every 16k samples, so a batch is padded at most 1 s beyond its own longest sequence.
# The sequence count per cell is FITTED to a replay of 114910 real batches (the first run's log),
# with a 15% margin, floored by what the batcher could produce at that length
# (n = min(max_seqs, 20M / audio)) so that cells the replay never visited are still covered.
#
# The previous grid derived n from a 1.5x budget headroom at 7 audio levels. Measured against the
# same replay, that computed 1.41x the volume PyTorch does (which pads only to the batch's own max),
# and the run was 1.40x slower -- the headroom WAS the slowdown. This grid computes 1.19x.
#
# 60 buckets is 60 compiles at ~35 s. That is ~35 min, and it would be paid again on every
# resubmission (the 11.9h limit splits the run), which is why jax_compilation_cache_dir is set
# below: a resubmitted run reloads the programs instead of rebuilding them.
_AUDIO_LEVELS = list(range(16_000, 320_001, 16_000))

# The text axis is derived, one cap per audio level, not a flat set of levels.
#
# It used to be the cross product of the audio levels with [128, 384, 768],
# where 768 was headroom over a run that died at text 192.
# But labels are bounded by the audio that carries them,
# so (200 seqs, 16000 audio, 768 text) declares 200 one-second utterances of 768 labels each,
# against a corpus max of 13.3 labels/sec.
# The AED logits are batch_dim x text x vocab,
# so that corner cost 5.86 GB per buffer and OOMed the ~500M model at 42.64 GiB.
# PT never pays it: it pads to the batch's own max text.
#
# The cap is max{labels : duration <= audio} over the train corpus,
# joined per sequence by tag from the shards' `duration` column and ExtractSeqLensJob
# (9,487,873 seqs), rounded up to the text time_multiple of 8.
# It reaches 246 labels at 18-20 s,
# independently reproducing the documented spm10k max of 246 under the 19.5 s filter,
# and it matches the per-level maxima of a 118k-batch replay at every level the replay covers.
# Two levels per audio level, the cap and half of it,
# so batches with short transcripts do not pad all the way up.
#
# The seq count stays as it was:
# batch_size bounds the summed content frames, not the padded product
# (a real batch: 171 seqs, max audio 96000, sum 14.05M within the 16M limit, 16.4M padded),
# so it cannot be derived as batch_size // audio -- it comes from the replay, +15%.
_TEXT_CAPS = {
    16_000: 8,
    32_000: 24,
    48_000: 112,
    64_000: 112,
    80_000: 144,
    96_000: 144,
    112_000: 200,
    128_000: 200,
    144_000: 200,
    160_000: 200,
    176_000: 200,
    192_000: 200,
    208_000: 200,
    224_000: 200,
    240_000: 208,
    256_000: 208,
    272_000: 208,
    288_000: 248,
    304_000: 248,
    320_000: 248,
}


# Per audio level, the seq count the replayed batches actually needed, +15%,
# and never below what the batcher could produce at that length.
# Reproduced from a 118k-batch replay.
_FITTED_SEQ_COUNTS = {
    112_000: 191,
    128_000: 164,
    144_000: 143,
    160_000: 127,
    176_000: 114,
    192_000: 104,
    208_000: 96,
    224_000: 88,
    240_000: 82,
    256_000: 76,
    272_000: 72,
    288_000: 67,
    304_000: 64,
    320_000: 60,
}
_MAX_SEQS, _SEQ_BUDGET = 200, 20_000_000


def _text_levels(audio: int):
    """
    :param audio: declared audio extent
    :return: the text extents to compile for it, ascending
    """
    cap = _TEXT_CAPS[audio]
    half = max(8, -(-(cap // 2) // 8) * 8)
    return sorted({half, cap})


# audio-major, text ascending: _bucket_for takes the first fit, so a batch lands in the smallest
# text bucket that holds it
_BUCKETS = [
    {
        "batch_dim": min(_MAX_SEQS, max(_SEQ_BUDGET // audio, _FITTED_SEQ_COUNTS.get(audio, 0))),
        "audio": audio,
        "text": text,
    }
    for audio in _AUDIO_LEVELS
    for text in _text_levels(audio)
]


def py():
    """Sisyphus entry point."""
    small_jax_exp, _, _ = loq_train(
        "base-small-v2-jax",
        {},
        config_overrides={
            "model.behavior_version": 29,
            **small_model_overrides(),
            "train.backend": "jax",
            # bf16 compute, f32 parameters and optimizer (returnn.frontend.amp), the same split
            # torch_amp gives. Measured on an H100 at this batching: 1.4-1.7x faster than f32.
            "train.jax_amp": "bfloat16",
            "train.jax_jit": {
                # the padded time extent is rounded up per key, in that key's own unit
                # (samples for audio, labels for text), which bounds how many shapes exist at all
                "time_multiple": {"audio": 16_000, "text": 8},
                "buckets": _BUCKETS,
            },
            # 60 buckets is ~35 min of compiles at startup, and the 11.9h SLURM limit splits the
            # run into several submissions. Cached, a resubmitted run reloads them instead.
        },
        # torch-only, no JAX counterpart. The post-config torch_* entries are dropped by train()
        # itself; only the hashed train config needs an explicit delete, as for the TF variant.
        # The JAX engine REJECTS any option it would otherwise ignore silently, so if the first
        # run names further ones, they belong here.
        config_deletes=["train.torch_amp"],
    )

    # The real 500M AED+CTC with packed tensors, mirroring the PT packed setup:
    # same option names, same bounds, same behavior version. No small_model_overrides.
    # Packed needs no bucket grid -- one bound-shaped program covers every batch,
    # so jax_jit only keeps the per-key rounding,
    # and jax_static_shapes carries what the buckets carried.
    # The training itself is not scheduled yet, only the step benchmark below:
    # the packed step is complete (no fallbacks, every attention on a native packed kernel)
    # and matches eager step for step on the small model,
    # but its speed vs the PT packed setup is what has to come first.
    # JAX row of the backend table.
    # The torch and TF rows are the same cell on the same small-v2 config,
    # in exp2026_05_23_returnn.py.
    # Mode as_is adds no override text, so each arm is measured as it trains:
    # the arms carry options with no cross-backend counterpart.
    job = TrainStepBenchmarkJob(
        returnn_config=small_jax_exp.get_training_job().returnn_config, mode="as_is", num_steps=300
    )
    tk.register_output("returnn/backend-bench-small-v2-jax.json", job.out_results)

    _packed_jax_anchor()
    bench_packed()


def _packed_jax_anchor():
    """
    :return: (exp, align, audio_bound) of the 500M packed-JAX config

    One definition for the training anchor and the benchmark,
    so the benchmark cannot drift from the config it is supposed to measure.
    """
    align = 960
    audio_bound = 100_000 * _loq_batch_size_factor() + 200 * align
    audio_bound = -(-audio_bound // align) * align
    with disable_register_output():
        exp, _, _ = loq_train(
            "base-v2-jax-packed",
            {},
            config_overrides={
                "model.behavior_version": 29,
                "train.backend": "jax",
                "train.jax_amp": "bfloat16",
                "train.packed_tensors": {
                    "per_key": {"audio": {"gap": 0, "align": align}, "text": {"gap": 0, "align": 1}},
                },
                "train.jax_jit": {"time_multiple": {"audio": 16_000, "text": 8}},
                "train.jax_static_shapes": {
                    "batch_size_bound": 200,  # == max_seqs
                    # text 256 = the conditional max label count under the 19.5s audio filter
                    # plus headroom, as measured for the PT setup
                    "dim_capacity": {"audio": 312_960, "text": 256},
                    "packed_total_bound": {"audio": audio_bound, "text": 18_000},
                },
            },
            config_deletes=["train.torch_amp"],
        )
    return exp, align, audio_bound


# The producer benchmark below lives here rather than in its own recipe: it belongs to this
# experiment, and `--inspect` already guarantees what a manager run would submit.
_BENCH_SCRIPT = r'''
"""Runs inside the job: the arms on one engine, results appended to the output file."""

import os

os.environ["JAX_PLATFORMS"] = "cuda"

import sys
import time
import tempfile

RETURNN = {returnn_root!r}
CFG = {config_file!r}
RESULT = {result_path!r}
WARM_STEPS, N_STEPS, PASSES = {warm_steps}, {num_steps}, {passes}
PROD_S_PER_STEP, PROD_MS_PER_MSAMPLE = 0.2949, 17.858
PT_S_PER_STEP, PT_MS_PER_MSAMPLE = 0.2366, 14.980
VOLUME_KEY = "audio"

ARMS = [("A old", False), ("B fixed", True)]
SELECT = {arms!r}
if SELECT:
    ARMS = [arm for arm in ARMS if arm[0].split()[0] in SELECT]


def _emit(line):
    """:param line: result line, appended to RESULT and echoed"""
    print(line, flush=True)
    with open(RESULT, "a") as f:
        f.write(line + "\n")


def _old_to_jax():
    """
    :return: `_to_jax` as it was before the fix

    It routes through `jnp.asarray`, which traces per input shape: 1.7 ms on a repeated shape,
    29.9 ms on a fresh one, and the shapes here are fresh almost every batch.
    """
    import numpy
    import jax
    import jax.numpy as jnp

    # noinspection PyProtectedMember
    from returnn.jax.frontend._backend import _device_from_str

    def _to_jax(value, *, dtype, device):
        """:return: the value as a JAX array, the slow way"""
        raw = jnp.asarray(numpy.asarray(value, dtype=dtype))
        if device:
            raw = jax.device_put(raw, _device_from_str(device))
        return raw

    return _to_jax


def _main():
    """build the engine once, then run every arm on it, PASSES times in rotated order"""
    sys.path.insert(0, RETURNN)
    import jax
    from returnn.config import Config, set_global_config
    from returnn.util.basic import BackendEngine, BehaviorVersion
    from returnn.datasets import init_dataset
    from returnn.jax import data as jax_data

    config = Config()
    config.load_file(CFG)
    config.typed_dict["backend"] = "jax"
    config.typed_dict["device"] = "gpu"
    for key in ["eval_datasets", "dev", "eval", "cleanup_old_models", "use_train_proc_manager",
                "watch_memory", "use_lovely_tensors", "startup_callback"]:
        config.typed_dict.pop(key, None)
        config.dict.pop(key, None)
    config.typed_dict["model"] = f"{{tempfile.mkdtemp(prefix='bench-')}}/model"
    config.typed_dict["learning_rate_file"] = f"{{tempfile.mkdtemp(prefix='bench-lr-')}}/lr"
    set_global_config(config)
    BackendEngine.select_engine(config=config)
    BehaviorVersion.set(config.int("behavior_version", 0) or None)

    from returnn.jax.engine import Engine

    fixed_to_jax = jax_data._to_jax          # the checkout's current, fixed one
    old_to_jax = _old_to_jax()

    _emit(f"\n=== JAX producer benchmark: {{N_STEPS}} steps/arm after a {{WARM_STEPS}}-step warm epoch,"
          f" {{PASSES}} passes in rotated order")
    _emit(f"  control: production JAX {{PROD_S_PER_STEP:.4f}} s/step, {{PROD_MS_PER_MSAMPLE:.3f}} ms/Msample")
    _emit(f"  target:  PT baseline    {{PT_S_PER_STEP:.4f}} s/step, {{PT_MS_PER_MSAMPLE:.3f}} ms/Msample")

    t0 = time.time()
    engine = Engine(config=config)
    engine.init_train_from_config(
        config=config,
        train_data=init_dataset(config.typed_value("train"), default_kwargs={{"name": "train"}}),
    )
    _emit(f"  engine built (bucket precompile included) in {{time.time() - t0:.0f}} s")

    def _run_arm(fixed):
        """
        :param fixed: whether to use the fixed `_to_jax`
        :return: (sec/step, ms/Msample) over the trimmed measured window
        """
        jax.clear_caches()  # the tracing cache is process-wide; arms must not inherit each other's
        jax_data._to_jax = fixed_to_jax if fixed else old_to_jax

        steps = []
        orig_iter = engine._iter_batches
        limit = [WARM_STEPS]
        measuring = [False]

        def _limited(*args, **kwargs):
            """the real batch iterator, cut off after `limit` batches, timing each step"""
            t_prev = time.time()
            for i, item in enumerate(orig_iter(*args, **kwargs)):
                if i >= limit[0]:
                    return
                raws = item[0] if isinstance(item, tuple) else item
                data = raws.get(VOLUME_KEY)
                vol = int(data.shape[0]) * int(data.shape[1]) if data is not None and data.ndim >= 2 else 0
                yield item
                now = time.time()
                if measuring[0]:
                    steps.append((now - t_prev, vol))
                t_prev = now

        engine._iter_batches = _limited
        engine.epoch += 1
        engine.train_epoch()   # warm
        limit[0] = N_STEPS
        measuring[0] = True
        engine.epoch += 1
        engine.train_epoch()
        engine._iter_batches = orig_iter
        # each arm abandons its epoch partway, and DistributeFilesDataset starts its workers per
        # epoch: without this the next arm's worker dies with "buffer too small"
        engine.train_dataset.finish_epoch(free_resources=True)

        durs = sorted(d for d, _ in steps)
        median = durs[len(durs) // 2]
        kept = [(d, v) for d, v in steps if d <= 3 * median]  # a compile that slipped into the window
        elapsed = sum(d for d, _ in kept)
        volume = sum(v for _, v in kept)
        return elapsed / max(len(kept), 1), elapsed * 1000 / max(volume, 1) * 1e6

    # One pass measured every arm ONCE, in one fixed order, and produced an incoherent result: D
    # (both fixes) came out slower than either fix alone, which cannot be true of the mechanism and
    # put the noise at ~8%. Rotating the order across passes keeps a drift or a warm-up from landing
    # on the same arm every time, and the per-arm conclusion is the MEDIAN over passes.
    results = {{name: [] for name, _, _ in ARMS}}
    for p in range(PASSES):
        for name, fixed in ARMS[p % len(ARMS):] + ARMS[: p % len(ARMS)]:
            sec, ms_vol = _run_arm(fixed)
            results[name].append((sec, ms_vol))
            _emit(f"  pass {{p + 1}} {{name:15s}} {{sec:.4f}} s/step  {{ms_vol:6.3f}} ms/Msample  "
                  f"vs PT {{sec / PT_S_PER_STEP:.3f}}x step, {{ms_vol / PT_MS_PER_MSAMPLE:.3f}}x volume")

    _emit("  --- median over passes ---")
    for name, _ in ARMS:
        secs = sorted(s for s, _ in results[name])
        vols = sorted(v for _, v in results[name])
        sec = secs[len(secs) // 2]
        ms_vol = vols[len(vols) // 2]
        _emit(f"  {{name:15s}} {{sec:.4f}} s/step  {{ms_vol:6.3f}} ms/Msample  "
              f"vs PT {{sec / PT_S_PER_STEP:.3f}}x step, {{ms_vol / PT_MS_PER_MSAMPLE:.3f}}x volume  "
              f"(passes {{[f'{{s:.4f}}' for s in secs]}})")


if __name__ == "__main__":
    _main()
'''


class JaxProducerBenchmarkJob(Job):
    """
    Run the real JAX train step for a bounded number of steps, per `_to_jax` variant.

    One process for all arms: the 60-shape bucket precompile takes ~5 min warm (25-30 min cold) and
    would otherwise be paid per arm, and paying it once also makes the compiled step identical
    across arms by construction.
    """

    __sis_hash_exclude__ = {"arms": None}

    def __init__(
        self,
        *,
        returnn_config_file: tk.Path,
        returnn_root: tk.Path,
        python_exe: str,
        num_steps: int = 120,
        warm_steps: int = 30,
        passes: int = 3,
        arms: Optional[Tuple[str, ...]] = None,
        version: int = 1,
    ):
        """
        :param returnn_config_file: serialized config of the JAX training job to benchmark
        :param returnn_root: RETURNN checkout to import
        :param python_exe: interpreter with the jax/cuda env
        :param num_steps: measured steps per arm
        :param warm_steps: steps run before measuring, so first-touch work is not charged to the arm
        :param passes: how often to measure every arm, each pass rotating the order
        :param arms: which arms to run, by letter, e.g. ``("B", "D")``.
            Fewer arms buys more passes at the same cost,
            which is what separating two close configurations needs.
        :param version: bump to force a re-run
        """
        self.returnn_config_file = returnn_config_file
        self.returnn_root = returnn_root
        self.python_exe = python_exe
        self.num_steps = num_steps
        self.warm_steps = warm_steps
        self.passes = passes
        self.arms = arms
        self.version = version
        # 24 CPUs because the dataset's MultiProcDataset workers need them and a starved producer is
        # exactly what this measures. 2 h covers the precompile plus every pass.
        self.rqmt = {"gpu": 1, "cpu": 24, "mem": 100, "time": 2}
        self.out_results = self.output_path("results.txt")

    def tasks(self):
        """tasks"""
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        """write the benchmark script into the job dir and run it"""
        with open("bench.py", "w") as f:
            f.write(
                _BENCH_SCRIPT.format(
                    returnn_root=self.returnn_root.get_path(),
                    config_file=self.returnn_config_file.get_path(),
                    result_path=self.out_results.get_path(),
                    warm_steps=self.warm_steps,
                    num_steps=self.num_steps,
                    passes=self.passes,
                    arms=self.arms,
                )
            )
        subprocess.check_call([self.python_exe, "-u", "bench.py"])


def bench():
    """
    Sisyphus entry point for the producer benchmark alone.

    Separate from :func:`py` so a manager run can be pointed at the benchmark without building the
    training graph.
    """
    job = JaxProducerBenchmarkJob(
        returnn_config_file=tk.Path(
            "/rwthfs/rz/cluster/home/az668407/setups/2026-05-23-returnn-paper/work/"
            "i6_core/returnn/training/ReturnnTrainingJob.sTkB9L1B974a/output/returnn.config"
        ),
        returnn_root=tk.Path("/home/az668407/setups/combined/2021-05-31/tools/returnn"),
        python_exe="/home/az668407/work/py-envs/py3.12-torch2.12/bin/python",
    )
    job.add_alias("jax-producer-benchmark")
    tk.register_output("jax_producer_benchmark.txt", job.out_results)


# Bucket-grid benchmark: the text axis is where the grid's padding sits.
# Over the real epoch the grid rounds text to 3.06x the batch's own max and 5.44x its content,
# because the smallest declared level is 128 while typical batches carry max_size:text 32.
# Adding levels 32/64/256 drops that to 1.64x / 2.91x, leaves audio untouched (1.197x either way),
# and costs 120 buckets instead of 60.
_GRID_SCRIPT = r'''
"""Runs inside the job: one engine per grid, alternating, results appended to the output file."""

import os

os.environ["JAX_PLATFORMS"] = "cuda"

import sys
import time
import tempfile

RETURNN = {returnn_root!r}
CFG = {config_file!r}
RESULT = {result_path!r}
WARM_STEPS, N_STEPS, PASSES = {warm_steps}, {num_steps}, {passes}
EXTRA_TEXT_LEVELS = {extra_text_levels!r}
PT_S_PER_STEP, PT_MS_PER_MSAMPLE = 0.2366, 14.980
VOLUME_KEY = "audio"


def _emit(line):
    """:param line: result line, appended to RESULT and echoed"""
    print(line, flush=True)
    with open(RESULT, "a") as f:
        f.write(line + "\n")


def _build(config, buckets, init_dataset):
    """
    :param config: the global config, already selected as the JAX backend
    :param buckets: the grid this engine compiles for
    :param init_dataset: the dataset factory
    :return: an engine holding the compiled programs for that grid
    """
    from returnn.jax.engine import Engine

    config.typed_dict["jax_jit"] = dict(config.typed_value("jax_jit"), buckets=buckets)
    engine = Engine(config=config)
    engine.init_train_from_config(
        config=config,
        train_data=init_dataset(config.typed_value("train"), default_kwargs={{"name": "train"}}),
    )
    return engine


def _main():
    """build one engine per grid, then alternate them"""
    sys.path.insert(0, RETURNN)
    import jax
    from returnn.config import Config, set_global_config
    from returnn.util.basic import BackendEngine, BehaviorVersion
    from returnn.datasets import init_dataset

    config = Config()
    config.load_file(CFG)
    config.typed_dict["backend"] = "jax"
    config.typed_dict["device"] = "gpu"
    for key in ["eval_datasets", "dev", "eval", "cleanup_old_models", "use_train_proc_manager",
                "watch_memory", "use_lovely_tensors", "startup_callback"]:
        config.typed_dict.pop(key, None)
        config.dict.pop(key, None)
    config.typed_dict["model"] = f"{{tempfile.mkdtemp(prefix='grid-')}}/model"
    config.typed_dict["learning_rate_file"] = f"{{tempfile.mkdtemp(prefix='grid-lr-')}}/lr"
    set_global_config(config)
    BackendEngine.select_engine(config=config)
    BehaviorVersion.set(config.int("behavior_version", 0) or None)

    base_buckets = list(config.typed_value("jax_jit")["buckets"])
    audio_levels = sorted({{b["audio"] for b in base_buckets}})
    text_levels = sorted({{b["text"] for b in base_buckets}})
    seq_counts = {{b["audio"]: b["batch_dim"] for b in base_buckets}}
    fine_buckets = [
        {{"batch_dim": seq_counts[audio], "audio": audio, "text": text}}
        for audio in audio_levels
        for text in sorted(set(text_levels) | set(EXTRA_TEXT_LEVELS))
    ]

    _emit(f"\n=== bucket grid: {{len(base_buckets)}} buckets vs {{len(fine_buckets)}}"
          f" (text levels {{text_levels}} vs {{sorted(set(text_levels) | set(EXTRA_TEXT_LEVELS))}}),"
          f" {{N_STEPS}} steps/arm, {{PASSES}} passes")
    _emit(f"  target: PT baseline {{PT_S_PER_STEP:.4f}} s/step, {{PT_MS_PER_MSAMPLE:.3f}} ms/Msample")

    engines = {{}}
    for name, buckets in [("base", base_buckets), ("fine", fine_buckets)]:
        t0 = time.time()
        engines[name] = _build(config, buckets, init_dataset)
        _emit(f"  engine '{{name}}' built ({{len(buckets)}} buckets) in {{time.time() - t0:.0f}} s")

    def _run_arm(engine):
        """
        :param engine: the engine for this arm's grid
        :return: (sec/step, ms/Msample) over the trimmed measured window

        Volume is the batch's own padded extent, which the grid does NOT change,
        so a grid that computes less padding shows up as fewer ms per the same Msample.
        """
        steps = []
        orig_iter = engine._iter_batches
        limit = [WARM_STEPS]
        measuring = [False]

        def _limited(*args, **kwargs):
            """the real batch iterator, cut off after `limit` batches, timing each step"""
            t_prev = time.time()
            for i, item in enumerate(orig_iter(*args, **kwargs)):
                if i >= limit[0]:
                    return
                raws = item[0] if isinstance(item, tuple) else item
                data = raws.get(VOLUME_KEY)
                vol = int(data.shape[0]) * int(data.shape[1]) if data is not None and data.ndim >= 2 else 0
                yield item
                now = time.time()
                if measuring[0]:
                    steps.append((now - t_prev, vol))
                t_prev = now

        engine._iter_batches = _limited
        engine.epoch += 1
        engine.train_epoch()   # warm
        limit[0] = N_STEPS
        measuring[0] = True
        engine.epoch += 1
        engine.train_epoch()
        engine._iter_batches = orig_iter
        engine.train_dataset.finish_epoch(free_resources=True)

        durs = sorted(d for d, _ in steps)
        median = durs[len(durs) // 2]
        kept = [(d, v) for d, v in steps if d <= 3 * median]
        elapsed = sum(d for d, _ in kept)
        volume = sum(v for _, v in kept)
        return elapsed / max(len(kept), 1), elapsed * 1000 / max(volume, 1) * 1e6

    results = {{"base": [], "fine": []}}
    order = ["base", "fine"]
    for p in range(PASSES):
        for name in (order if p % 2 == 0 else order[::-1]):
            sec, ms_vol = _run_arm(engines[name])
            results[name].append((sec, ms_vol))
            _emit(f"  pass {{p + 1}} {{name}}: {{sec:.4f}} s/step  {{ms_vol:6.3f}} ms/Msample  "
                  f"vs PT {{sec / PT_S_PER_STEP:.3f}}x step, {{ms_vol / PT_MS_PER_MSAMPLE:.3f}}x volume")

    _emit("  --- median over passes ---")
    for name in order:
        secs = sorted(s for s, _ in results[name])
        vols = sorted(v for _, v in results[name])
        _emit(f"  {{name}}: {{secs[len(secs) // 2]:.4f}} s/step  {{vols[len(vols) // 2]:6.3f}} ms/Msample  "
              f"vs PT {{secs[len(secs) // 2] / PT_S_PER_STEP:.3f}}x step  "
              f"(passes {{[f'{{s:.4f}}' for s in secs]}})")


if __name__ == "__main__":
    _main()
'''


class JaxBucketGridBenchmarkJob(Job):
    """
    Compare the production bucket grid against one with finer text levels, on the real train step.

    Two engines in ONE process, alternating: a cross-job comparison would put the two grids on
    different nodes, and this node's run-to-run spread is larger than the effect.
    """

    def __init__(
        self,
        *,
        returnn_config_file: tk.Path,
        returnn_root: tk.Path,
        python_exe: str,
        extra_text_levels: Tuple[int, ...] = (32, 64, 256),
        num_steps: int = 120,
        warm_steps: int = 30,
        passes: int = 3,
        version: int = 1,
    ):
        """
        :param returnn_config_file: serialized config of the JAX training job to benchmark
        :param returnn_root: RETURNN checkout to import
        :param python_exe: interpreter with the jax/cuda env
        :param extra_text_levels: text extents to add to the config's own
        :param num_steps: measured steps per arm
        :param warm_steps: steps run before measuring
        :param passes: how often to measure each grid, alternating which goes first
        :param version: bump to force a re-run
        """
        self.returnn_config_file = returnn_config_file
        self.returnn_root = returnn_root
        self.python_exe = python_exe
        self.extra_text_levels = extra_text_levels
        self.num_steps = num_steps
        self.warm_steps = warm_steps
        self.passes = passes
        self.version = version
        # 3 h: two precompiles (60 and 120 buckets) plus every pass.
        self.rqmt = {"gpu": 1, "cpu": 24, "mem": 100, "time": 3}
        self.out_results = self.output_path("results.txt")

    def tasks(self):
        """tasks"""
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        """write the benchmark script into the job dir and run it"""
        with open("grid_bench.py", "w") as f:
            f.write(
                _GRID_SCRIPT.format(
                    returnn_root=self.returnn_root.get_path(),
                    config_file=self.returnn_config_file.get_path(),
                    result_path=self.out_results.get_path(),
                    warm_steps=self.warm_steps,
                    num_steps=self.num_steps,
                    passes=self.passes,
                    extra_text_levels=tuple(self.extra_text_levels),
                )
            )
        subprocess.check_call([self.python_exe, "-u", "grid_bench.py"])


def bench_grid():
    """Sisyphus entry point for the bucket-grid benchmark."""
    job = JaxBucketGridBenchmarkJob(
        returnn_config_file=tk.Path(
            "/rwthfs/rz/cluster/home/az668407/setups/2026-05-23-returnn-paper/work/"
            "i6_core/returnn/training/ReturnnTrainingJob.sTkB9L1B974a/output/returnn.config"
        ),
        returnn_root=tk.Path("/home/az668407/setups/combined/2021-05-31/tools/returnn"),
        python_exe="/home/az668407/work/py-envs/py3.12-torch2.12/bin/python",
    )
    job.add_alias("jax-bucket-grid-benchmark")
    tk.register_output("jax_bucket_grid_benchmark.txt", job.out_results)


# JAX vs PT, measured side by side.
# Every JAX/PT comparison so far paired a benchmark-harness number against PRODUCTION's 0.2366,
# i.e. different nodes, different epochs, different measurement code. This runs both backends on the
# same node, in the same job, with the same step counting, alternating so drift cannot favour one.
# Separate SUBPROCESSES per arm, run one after another: JAX preallocates GPU memory and both
# frameworks hold CUDA handles, so co-hosting them would distort exactly what is being measured.
_VS_ARM_SCRIPT = r'''
"""One arm: run a real training config for a bounded number of steps, report sec/step."""

import sys
import time
import tempfile

# JAX keeps its default preallocation here. Disabling it OOMed the bucket precompile at 42.39 GiB:
# the largest bucket program needs one big contiguous arena, which the on-demand allocator cannot
# assemble. The arms are separate processes run one after another, so nothing shares the GPU anyway.

RETURNN = {returnn_root!r}
RESULT = {result_path!r}
WARM_STEPS, N_STEPS = {warm_steps}, {num_steps}
VOLUME_KEY = "audio"


def _emit(line):
    """:param line: result line, appended to RESULT and echoed"""
    print(line, flush=True)
    with open(RESULT, "a") as f:
        f.write(line + "\n")


def _config(path, backend):
    """
    :param path: serialized RETURNN config of the training job
    :param backend: "jax" or "torch"
    :return: the config, stripped of everything that would evaluate or write
    """
    from returnn.config import Config, set_global_config
    from returnn.util.basic import BackendEngine, BehaviorVersion

    config = Config()
    config.load_file(path)
    config.typed_dict["backend"] = backend
    config.typed_dict["device"] = "gpu" if backend == "jax" else "cuda"
    for key in ["eval_datasets", "dev", "eval", "cleanup_old_models", "use_train_proc_manager",
                "watch_memory", "use_lovely_tensors", "startup_callback"]:
        config.typed_dict.pop(key, None)
        config.dict.pop(key, None)
    config.typed_dict["model"] = f"{{tempfile.mkdtemp(prefix='vs-')}}/model"
    config.typed_dict["learning_rate_file"] = f"{{tempfile.mkdtemp(prefix='vs-lr-')}}/lr"
    set_global_config(config)
    BackendEngine.select_engine(config=config)
    BehaviorVersion.set(config.int("behavior_version", 0) or None)
    return config


def _padded_volume(item):
    """
    :param item: one batch, as either engine hands it out
        (JAX yields ``(raws, complete_frac)``, torch a dict)
    :return: padded audio samples in it, i.e. batch x padded time -- the work the step computes on
    """
    raws = item[0] if isinstance(item, tuple) else item
    data = raws.get(VOLUME_KEY) if hasattr(raws, "get") else None
    if data is None or getattr(data, "ndim", 0) < 2:
        return 0
    return int(data.shape[0]) * int(data.shape[1])


class _Limiter:
    """Wraps the batch source, cutting it off and timing each step the loop takes."""

    def __init__(self, inner, steps):
        """
        :param inner: the real iterable
        :param steps: list the (duration, volume) pairs land in
        """
        self.inner = inner
        self.steps = steps
        self.limit = WARM_STEPS
        self.measuring = False

    def __iter__(self):
        """:return: the inner items, up to the limit"""
        t_prev = time.time()
        for i, item in enumerate(self.inner):
            if i >= self.limit:
                return
            vol = _padded_volume(item)
            yield item
            now = time.time()
            if self.measuring:
                self.steps.append((now - t_prev, vol))
            t_prev = now


def _run_jax(config_path):
    """
    :param config_path: the JAX training config
    :return: per-step durations
    """
    config = _config(config_path, "jax")
    from returnn.datasets import init_dataset
    from returnn.jax.engine import Engine

    engine = Engine(config=config)
    engine.init_train_from_config(
        config=config,
        train_data=init_dataset(config.typed_value("train"), default_kwargs={{"name": "train"}}),
    )
    steps = []
    orig = engine._iter_batches
    limiter = [None]

    def _wrapped(*args, **kwargs):
        """the real iterator, wrapped so the arm can stop and time it"""
        limiter[0] = _Limiter(orig(*args, **kwargs), steps)
        limiter[0].limit = state["limit"]
        limiter[0].measuring = state["measuring"]
        return iter(limiter[0])

    state = {{"limit": WARM_STEPS, "measuring": False}}
    engine._iter_batches = _wrapped
    engine.epoch += 1
    engine.train_epoch()
    state.update(limit=N_STEPS, measuring=True)
    engine.epoch += 1
    engine.train_epoch()
    return steps


def _run_torch(config_path):
    """
    :param config_path: the PyTorch training config
    :return: per-step durations
    """
    config = _config(config_path, "torch")
    from returnn.datasets import init_dataset
    from returnn.torch.engine import Engine

    engine = Engine(config=config)
    engine.init_train_from_config(
        config=config,
        train_data=init_dataset(config.typed_value("train"), default_kwargs={{"name": "train"}}),
    )
    steps = []
    # train_epoch does iter(self._train_dataloader), so wrapping the object is enough
    inner = engine._train_dataloader
    limiter = _Limiter(inner, steps)
    engine._train_dataloader = limiter

    def _epoch():
        """advance one epoch the way `train()` does

        Bumping `engine.epoch` directly leaves the LR control without an entry for it,
        and train_epoch ends on `learning_rate_control.epoch_data[self.epoch]` -- KeyError.
        """
        engine.set_epoch(engine.epoch + 1)
        engine.init_train_epoch()

    _epoch()
    engine.train_epoch()
    limiter.inner = inner
    limiter.limit, limiter.measuring = N_STEPS, True
    _epoch()
    engine.train_epoch()
    return steps


def _main():
    """run one arm, named by argv"""
    sys.path.insert(0, RETURNN)
    backend, config_path, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    steps = _run_jax(config_path) if backend == "jax" else _run_torch(config_path)
    if not steps:
        _emit(f"  {{tag}} {{backend}}: NO STEPS MEASURED")
        return
    ordered = sorted(d for d, _ in steps)
    median = ordered[len(ordered) // 2]
    kept = [(d, v) for d, v in steps if d <= 3 * median]
    elapsed = sum(d for d, _ in kept)
    volume = sum(v for _, v in kept)
    ms_per_msample = elapsed * 1000 / volume * 1e6 if volume else float("nan")
    _emit(f"  {{tag}} {{backend:5s}}: {{elapsed / len(kept):.4f}} s/step  "
          f"median {{median:.4f}}  p10 {{ordered[len(ordered) // 10]:.4f}}  "
          f"{{ms_per_msample:6.3f}} ms/Msample  [{{len(kept)}} kept of {{len(steps)}}, "
          f"{{volume / 1e6:.0f}} Msample]")


if __name__ == "__main__":
    _main()
'''


class JaxVsTorchBenchmarkJob(Job):
    """
    Measure the JAX and PyTorch engines side by side, same node, same step counting.

    One SUBPROCESS per arm, alternating: the two frameworks would otherwise share a GPU and a
    process, and JAX's allocator plus both CUDA runtimes would distort what is being compared.
    """

    def __init__(
        self,
        *,
        jax_config_file: tk.Path,
        torch_config_file: tk.Path,
        returnn_root: tk.Path,
        python_exe: str,
        num_steps: int = 120,
        warm_steps: int = 30,
        passes: int = 3,
        version: int = 1,
    ):
        """
        :param jax_config_file: serialized config of the JAX training job
        :param torch_config_file: serialized config of the PyTorch baseline
        :param returnn_root: RETURNN checkout to import
        :param python_exe: interpreter with both frameworks
        :param num_steps: measured steps per arm
        :param warm_steps: steps run before measuring
        :param passes: how often to run each backend, alternating which goes first
        :param version: bump to force a re-run
        """
        self.jax_config_file = jax_config_file
        self.torch_config_file = torch_config_file
        self.returnn_root = returnn_root
        self.python_exe = python_exe
        self.num_steps = num_steps
        self.warm_steps = warm_steps
        self.passes = passes
        self.version = version
        self.rqmt = {"gpu": 1, "cpu": 24, "mem": 100, "time": 4}
        self.out_results = self.output_path("results.txt")

    def tasks(self):
        """tasks"""
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        """write the arm script, then alternate backends across passes"""
        with open("vs_arm.py", "w") as f:
            f.write(
                _VS_ARM_SCRIPT.format(
                    returnn_root=self.returnn_root.get_path(),
                    result_path=self.out_results.get_path(),
                    warm_steps=self.warm_steps,
                    num_steps=self.num_steps,
                )
            )
        with open(self.out_results.get_path(), "a") as f:
            f.write(f"\n=== JAX vs PyTorch, same node, {self.num_steps} steps/arm, {self.passes} passes, alternating\n")
        arms = [("jax", self.jax_config_file.get_path()), ("torch", self.torch_config_file.get_path())]
        for p in range(self.passes):
            for backend, cfg in arms if p % 2 == 0 else arms[::-1]:
                subprocess.check_call([self.python_exe, "-u", "vs_arm.py", backend, cfg, f"pass {p + 1}"])


def bench_vs():
    """Sisyphus entry point for the JAX-vs-PyTorch comparison."""
    work = "/rwthfs/rz/cluster/home/az668407/setups/2026-05-23-returnn-paper/work/i6_core/returnn/training"
    job = JaxVsTorchBenchmarkJob(
        jax_config_file=tk.Path(f"{work}/ReturnnTrainingJob.sTkB9L1B974a/output/returnn.config"),
        torch_config_file=tk.Path(f"{work}/ReturnnTrainingJob.je6PefFx3gz2/output/returnn.config"),
        returnn_root=tk.Path("/home/az668407/setups/combined/2021-05-31/tools/returnn"),
        python_exe="/home/az668407/work/py-envs/py3.12-torch2.12/bin/python",
        version=2,  # v2 also records padded volume, so both engines report ms/Msample
    )
    job.add_alias("jax-vs-torch-benchmark")
    tk.register_output("jax_vs_torch_benchmark.txt", job.out_results)


# JAX vs PT on the FULL model (~500M: 16L Conformer 1024d + 6L Transformer dec 1024d).
# Everything before this ran base-small-v2 (256d, 4L/2L), where the step is so cheap that both
# backends idle on the input pipeline and the comparison measures the dataset, not the engines.
# ONE config drives both arms -- the PT base-v2 training config -- with the JAX arm deriving from it
# by swapping the backend and adding jax_amp/jax_jit, so model, data, batching and behavior version
# are identical by construction.
_VS_FULL_SCRIPT = r'''
"""One arm on the full-size model: real dataset, production prefetch, warmup steps dropped."""

import sys
import time
import tempfile

RETURNN = {returnn_root!r}
RESULT = {result_path!r}
WARM_STEPS, N_STEPS = {warm_steps}, {num_steps}
BUCKETS = {buckets!r}
TIME_MULTIPLE = {time_multiple!r}
VOLUME_KEY = "audio"


def _emit(line):
    """:param line: result line, appended to RESULT and echoed"""
    print(line, flush=True)
    with open(RESULT, "a") as f:
        f.write(line + "\n")


def _config(path, backend):
    """
    :param path: the PT training config both arms derive from
    :param backend: "jax" or "torch"
    :return: the config for this arm

    The JAX engine REJECTS options it would otherwise ignore, so every torch_* key goes; jax_amp
    mirrors torch_amp, and jax_jit carries the declared bucket grid.
    """
    from returnn.config import Config, set_global_config
    from returnn.util.basic import BackendEngine, BehaviorVersion

    config = Config()
    config.load_file(path)
    for key in ["eval_datasets", "dev", "eval", "cleanup_old_models", "use_train_proc_manager",
                "watch_memory", "use_lovely_tensors", "startup_callback"]:
        config.typed_dict.pop(key, None)
        config.dict.pop(key, None)
    config.typed_dict["model"] = f"{{tempfile.mkdtemp(prefix='full-')}}/model"
    config.typed_dict["learning_rate_file"] = f"{{tempfile.mkdtemp(prefix='full-lr-')}}/lr"

    if backend == "jax":
        for key in [k for k in list(config.typed_dict) + list(config.dict) if k.startswith("torch_")]:
            config.typed_dict.pop(key, None)
            config.dict.pop(key, None)
        config.typed_dict["backend"] = "jax"
        config.typed_dict["device"] = "gpu"
        config.typed_dict["jax_amp"] = "bfloat16"
        config.typed_dict["jax_jit"] = {{"time_multiple": TIME_MULTIPLE, "buckets": BUCKETS}}
        # the engine reads RETURNN_JAX_COMPILATION_CACHE_DIR itself; no path here
        pass
        # the per-shape timing pass EXECUTES every bucket and copies params + optimizer state each
        # time; at 500M that is real memory and minutes of startup, and it is only a diagnostic
        config.typed_dict["jax_time_buckets"] = 0
    else:
        config.typed_dict["backend"] = "torch"
        config.typed_dict["device"] = "cuda"

    set_global_config(config)
    BackendEngine.select_engine(config=config)
    BehaviorVersion.set(config.int("behavior_version", 0) or None)
    return config


def _padded_volume(item):
    """
    :param item: one batch (JAX yields ``(raws, complete_frac)``, torch a dict)
    :return: padded audio samples, i.e. batch x padded time
    """
    raws = item[0] if isinstance(item, tuple) else item
    data = raws.get(VOLUME_KEY) if hasattr(raws, "get") else None
    if data is None or getattr(data, "ndim", 0) < 2:
        return 0
    return int(data.shape[0]) * int(data.shape[1])


class _Limiter:
    """Wraps the batch source, cutting it off and timing each step the loop takes."""

    def __init__(self, inner, steps):
        """
        :param inner: the real iterable
        :param steps: list the (duration, volume) pairs land in
        """
        self.inner = inner
        self.steps = steps
        self.limit = WARM_STEPS
        self.measuring = False

    def __iter__(self):
        """:return: the inner items, up to the limit"""
        t_prev = time.time()
        for i, item in enumerate(self.inner):
            if i >= self.limit:
                return
            vol = _padded_volume(item)
            yield item
            now = time.time()
            if self.measuring:
                self.steps.append((now - t_prev, vol))
            t_prev = now


def _run_jax(config_path):
    """
    :param config_path: the shared training config
    :return: per-step (duration, volume)
    """
    config = _config(config_path, "jax")
    from returnn.datasets import init_dataset
    from returnn.jax.engine import Engine

    t0 = time.time()
    engine = Engine(config=config)
    engine.init_train_from_config(
        config=config,
        train_data=init_dataset(config.typed_value("train"), default_kwargs={{"name": "train"}}),
    )
    _emit(f"    jax engine built ({{len(BUCKETS)}} buckets compiled) in {{time.time() - t0:.0f}} s")

    steps = []
    orig = engine._iter_batches
    state = {{"limit": WARM_STEPS, "measuring": False}}

    def _wrapped(*args, **kwargs):
        """the real iterator, wrapped so the arm can stop and time it"""
        limiter = _Limiter(orig(*args, **kwargs), steps)
        limiter.limit, limiter.measuring = state["limit"], state["measuring"]
        return iter(limiter)

    engine._iter_batches = _wrapped
    engine.epoch += 1
    engine.train_epoch()   # warm: prefetch spin-up and first-touch land here
    state.update(limit=N_STEPS, measuring=True)
    engine.epoch += 1
    engine.train_epoch()
    return steps


def _run_torch(config_path):
    """
    :param config_path: the shared training config
    :return: per-step (duration, volume)
    """
    config = _config(config_path, "torch")
    from returnn.datasets import init_dataset
    from returnn.torch.engine import Engine

    t0 = time.time()
    engine = Engine(config=config)
    engine.init_train_from_config(
        config=config,
        train_data=init_dataset(config.typed_value("train"), default_kwargs={{"name": "train"}}),
    )
    _emit(f"    torch engine built in {{time.time() - t0:.0f}} s")

    steps = []
    inner = engine._train_dataloader
    limiter = _Limiter(inner, steps)
    engine._train_dataloader = limiter

    def _epoch():
        """advance one epoch the way `train()` does, so the LR control has this epoch's entry"""
        engine.set_epoch(engine.epoch + 1)
        engine.init_train_epoch()

    _epoch()
    engine.train_epoch()   # warm
    limiter.inner = inner
    limiter.limit, limiter.measuring = N_STEPS, True
    _epoch()
    engine.train_epoch()
    return steps


def _main():
    """run one arm, named by argv"""
    sys.path.insert(0, RETURNN)
    backend, config_path, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    steps = _run_jax(config_path) if backend == "jax" else _run_torch(config_path)
    if not steps:
        _emit(f"  {{tag}} {{backend}}: NO STEPS MEASURED")
        return
    ordered = sorted(d for d, _ in steps)
    median = ordered[len(ordered) // 2]
    kept = [(d, v) for d, v in steps if d <= 3 * median]
    elapsed = sum(d for d, _ in kept)
    volume = sum(v for _, v in kept)
    ms_per_msample = elapsed * 1000 / volume * 1e6 if volume else float("nan")
    _emit(f"  {{tag}} {{backend:5s}}: {{elapsed / len(kept):.4f}} s/step  "
          f"median {{median:.4f}}  p10 {{ordered[len(ordered) // 10]:.4f}}  "
          f"{{ms_per_msample:6.3f}} ms/Msample  [{{len(kept)}} kept of {{len(steps)}}, "
          f"{{volume / 1e6:.0f}} Msample]")


if __name__ == "__main__":
    _main()
'''


class JaxVsTorchFullBenchmarkJob(Job):
    """
    JAX vs PyTorch on the full ~500M model, real dataset, production prefetch.

    One subprocess per arm, alternating: JAX preallocates GPU memory and both frameworks hold CUDA
    runtimes, so co-hosting them would distort what is measured.
    """

    def __init__(
        self,
        *,
        returnn_config_file: tk.Path,
        returnn_root: tk.Path,
        python_exe: str,
        buckets,
        time_multiple,
        num_steps: int = 100,
        warm_steps: int = 30,
        passes: int = 3,
        version: int = 1,
    ):
        """
        :param returnn_config_file: the PT training config both arms derive from
        :param returnn_root: RETURNN checkout to import
        :param python_exe: interpreter with both frameworks
        :param buckets: declared input shapes for the JAX arm's compiled step
        :param time_multiple: per-key rounding of the padded time extent
        :param num_steps: measured steps per arm
        :param warm_steps: steps run before measuring, so prefetch spin-up does not count
        :param passes: how often to run each backend, alternating which goes first
        :param version: bump to force a re-run
        """
        self.returnn_config_file = returnn_config_file
        self.returnn_root = returnn_root
        self.python_exe = python_exe
        self.buckets = buckets
        self.time_multiple = time_multiple
        self.num_steps = num_steps
        self.warm_steps = warm_steps
        self.passes = passes
        self.version = version
        # 8 h: the 60-bucket precompile is ~30 min at 256d and unmeasured at 1024d, paid per JAX arm
        # process, plus the arms themselves.
        self.rqmt = {"gpu": 1, "cpu": 24, "mem": 200, "time": 8}
        self.out_results = self.output_path("results.txt")

    def tasks(self):
        """tasks"""
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        """write the arm script, then alternate backends across passes"""
        with open("vs_full_arm.py", "w") as f:
            f.write(
                _VS_FULL_SCRIPT.format(
                    returnn_root=self.returnn_root.get_path(),
                    result_path=self.out_results.get_path(),
                    warm_steps=self.warm_steps,
                    num_steps=self.num_steps,
                    buckets=self.buckets,
                    time_multiple=self.time_multiple,
                )
            )
        with open(self.out_results.get_path(), "a") as f:
            f.write(
                f"\n=== JAX vs PyTorch, full model, same node, {self.num_steps} steps/arm,"
                f" {self.passes} passes, alternating\n"
            )
        cfg = self.returnn_config_file.get_path()
        for p in range(self.passes):
            arms = ["jax", "torch"] if p % 2 == 0 else ["torch", "jax"]
            for backend in arms:
                subprocess.check_call([self.python_exe, "-u", "vs_full_arm.py", backend, cfg, f"pass {p + 1}"])


def bench_packed():
    """
    Sisyphus entry point for the packed-JAX speed benchmark.

    Uses the same job as the PT packed benches, so the numbers are comparable:
    same harness, same log parsing, same step counting.
    The packed/jit options ride in the config from :func:`_packed_jax_anchor`;
    mode ``packed_jax`` adds no override text of its own.
    """
    exp, _, _ = _packed_jax_anchor()
    job = TrainStepBenchmarkJob(
        returnn_config=exp.get_training_job().returnn_config,
        mode="packed_jax",
        num_steps=31,
    )
    tk.register_output("returnn/jax-packed-bench.json", job.out_results)
    # same arm under nsys, for the kernel mix.
    # Separate job: profiling must not perturb the timing the bench above reports.
    # More steps, so the profile window covers steady state after the compile.
    prof = TrainStepBenchmarkJob(
        returnn_config=exp.get_training_job().returnn_config,
        mode="packed_jax",
        num_steps=80,
        nsys="185,40",
    )
    tk.register_output("returnn/jax-packed-bench-nsys.json", prof.out_results)


def bench_vs_full():
    """Sisyphus entry point: JAX vs PyTorch on the full-size model."""
    job = JaxVsTorchFullBenchmarkJob(
        # base-v2: 16L Conformer 1024d + 6L Transformer dec 1024d, padded, eager, bhv 29
        returnn_config_file=tk.Path(
            "/rwthfs/rz/cluster/home/az668407/setups/2026-05-23-returnn-paper/work/"
            "i6_core/returnn/training/ReturnnTrainingJob.LySQsb8NNT9g/output/returnn.config"
        ),
        returnn_root=tk.Path("/home/az668407/setups/combined/2021-05-31/tools/returnn"),
        python_exe="/home/az668407/work/py-envs/py3.12-torch2.12/bin/python",
        buckets=_BUCKETS,
        time_multiple={"audio": 16_000, "text": 8},
    )
    job.add_alias("jax-vs-torch-full")
    tk.register_output("jax_vs_torch_full.txt", job.out_results)
