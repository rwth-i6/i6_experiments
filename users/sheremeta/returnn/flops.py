"""
The FLOPs of a training step, counted by torch's FlopCounterMode on the first batches of the training's own data run
eagerly on one GPU, and the FLOPS and the utilization the training reached with them.
"""

import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

from sisyphus import Job, Task, tk

from i6_core.returnn.config import ReturnnConfig

from .throughput import _rank_zero_lines, summarize_log

__all__ = [
    "PEAK_TFLOPS",
    "FlopCountingTrainStep",
    "TrainingFlopsJob",
    "TrainingUtilizationJob",
    "probe_config",
    "summarize_flops",
    "utilization",
]

# dense BF16 tensor core peak of one H100, the GPU of the GH200, after NVIDIA's data sheet
PEAK_TFLOPS = 990.0
STEPS_FILE = "flop_steps.json"
# the distributed run, the graph capture with its compile and the evaluation of a training, which the probe drops
_DROPPED_KEYS = ("torch_distributed", "torch_cuda_graph", "torch_compile", "eval_datasets", "dev")
# the relative position attention, whose packed fast paths are kernels the counter does not see, runs padded
_PADDED_ATTENTION_OPS = ("rel_pos_self_attention", "chunked_rel_pos_self_attention")


def _counts(counter: Any, phase: str) -> Dict[str, Any]:
    """
    :param counter: a FlopCounterMode after its pass
    :param phase: forward or backward
    :return: its total and its FLOPs by op
    """
    by_op = {str(op): int(flops) for op, flops in counter.get_flop_counts().get("Global", {}).items()}
    return {f"{phase}_flops": int(counter.get_total_flops()), f"{phase}_by_op": by_op}


def _batch_size(extern_data: Any, *, config: Any, sample_rate: int, frame_rate: float) -> Dict[str, Any]:
    """
    :param extern_data: the batch of a step
    :param config: the RETURNN config of the run
    :param sample_rate: audio samples per second
    :param frame_rate: encoder frames per second
    :return: its sequences, its audio, its encoder frames, padded too, and their squares, which the attention over the
        frames grows with
    """
    audio = extern_data[config.typed_value("default_input")]
    lens = audio.dims[1].get_size_tensor().raw_tensor.detach().cpu().long()
    frames = [math.ceil(int(n) * frame_rate / sample_rate) for n in lens]
    size = {
        "num_seqs": len(frames),
        "audio_seconds": float(lens.sum()) / sample_rate,
        "frames": sum(frames),
        "padded_frames": len(frames) * max(frames),
        "frames_squared": sum(n * n for n in frames),
        "padded_frames_squared": len(frames) * max(frames) ** 2,
    }
    target = config.typed_value("target", None)
    if target in extern_data.data:
        size["target_frames"] = int(extern_data[target].dims[1].get_size_tensor().raw_tensor.sum())
    return size


class FlopCountingTrainStep:
    """
    Wraps the train step of a RETURNN config. It counts the FLOPs of the forward pass of each of the first steps and
    of the backward pass and the update RETURNN runs after the step returned, writes them with the size of every
    batch and ends the run, as RETURNN does after its profiler.
    """

    def __init__(
        self, train_step: Any, *, num_steps: int, sample_rate: int, frame_rate: float, out_file: str = STEPS_FILE
    ):
        """
        :param train_step: the train step of the config
        :param num_steps: the steps counted
        :param sample_rate: audio samples per second
        :param frame_rate: encoder frames per second
        :param out_file: the file the counts are written to
        """
        self.train_step = train_step
        self.num_steps = num_steps
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.out_file = out_file
        self._steps: List[Dict[str, Any]] = []
        self._backward: Any = None

    def __call__(self, *, model: Any, extern_data: Any, **kwargs: Any) -> Any:
        """
        :param model: the model
        :param extern_data: the batch
        :param kwargs: what RETURNN passes besides
        :return: what the train step returns
        """
        from returnn.config import get_global_config
        from torch.utils.flop_counter import FlopCounterMode

        # the backward pass and the update of the step before ran between its return and this call
        if self._backward is not None:
            self._backward.__exit__(None, None, None)
            self._steps[-1].update(_counts(self._backward, "backward"))
            self._backward = None
            if len(self._steps) == self.num_steps:
                self._finish()

        config = get_global_config()
        step = _batch_size(extern_data, config=config, sample_rate=self.sample_rate, frame_rate=self.frame_rate)
        forward = FlopCounterMode(display=False)
        with forward:
            result = self.train_step(model=model, extern_data=extern_data, **kwargs)
        step.update(_counts(forward, "forward"))
        self._steps.append(step)

        self._backward = FlopCounterMode(display=False)
        self._backward.__enter__()
        return result

    def _finish(self) -> None:
        """
        Writes the counts with the attention paths the packed backend took and ends the run.
        """
        from returnn.frontend import _packed_backend as packed

        with open(self.out_file, "wt") as f:
            json.dump(
                {
                    "steps": self._steps,
                    "attention_paths": dict(packed.attention_path_counts),
                    "fallbacks": sorted(packed._warned_fallback_ops),
                },
                f,
                indent=1,
            )
        print(f"Exiting after counting the FLOPs of {len(self._steps)} train steps.")
        sys.exit(0)


def probe_config(
    returnn_config: ReturnnConfig, *, num_steps: int, sample_rate: int, frame_rate: float, model_dir: str
) -> ReturnnConfig:
    """
    Turns the config of a training into the config of its FLOP probe: one epoch on one GPU, eager without the graph
    capture and without evaluation, the relative position attention on its padded path, the train step counted.

    :param returnn_config: the config of the training
    :param num_steps: the steps counted
    :param sample_rate: audio samples per second
    :param frame_rate: encoder frames per second
    :param model_dir: the folder the run would save its checkpoints in
    :return: the config of the probe, the training's left as it is
    """
    from i6_core.returnn.training import ReturnnTrainingJob

    probe = ReturnnTrainingJob.create_returnn_config(
        returnn_config,
        log_verbosity=5,
        device="gpu",
        num_epochs=1,
        save_interval=1,
        keep_epochs=None,
        horovod_num_processes=None,
    )
    for key in _DROPPED_KEYS:
        probe.config.pop(key, None)
        probe.post_config.pop(key, None)

    # the packed fast paths of scaled dot product attention call flash kernels, which the counter has formulas for
    probe.config["rf_packed_att_fast_paths"] = ["scaled_dot_product_attention"]
    allowed = probe.config.get("packed_fallback_allowed") or []
    if allowed is not True:
        probe.config["packed_fallback_allowed"] = sorted(set(allowed) | set(_PADDED_ATTENTION_OPS))
    probe.post_config["model"] = os.path.join(model_dir, "epoch")

    epilog = probe.python_epilog
    epilog = list(epilog) if isinstance(epilog, (list, tuple)) else [epilog] if epilog else []
    epilog.append(
        "from i6_experiments.users.sheremeta.returnn.flops import FlopCountingTrainStep\n"
        f"train_step = FlopCountingTrainStep(train_step, num_steps={num_steps}, sample_rate={sample_rate},"
        f" frame_rate={frame_rate!r})\n"
    )
    probe.python_epilog = epilog
    return probe


def summarize_flops(counted: Dict[str, Any]) -> Dict[str, Any]:
    """
    :param counted: the counts a probe wrote, see :class:`FlopCountingTrainStep`
    :return: the FLOPs per step of one GPU and per second of audio, by op, with the padding of the attention over the
        frames, which runs padded in the probe
    """
    steps = counted["steps"]
    forward = [step["forward_flops"] for step in steps]
    total = [step["forward_flops"] + step["backward_flops"] for step in steps]
    audio = sum(step["audio_seconds"] for step in steps)
    by_op: Dict[str, float] = defaultdict(float)
    for step in steps:
        for phase in ("forward_by_op", "backward_by_op"):
            for op, flops in step[phase].items():
                by_op[op] += flops
    return {
        "steps": len(steps),
        "mean_forward_flops_per_step": statistics.fmean(forward),
        "mean_flops_per_step": statistics.fmean(total),
        "flops_per_audio_second": sum(total) / audio,
        "forward_flops_per_audio_second": sum(forward) / audio,
        "mean_audio_seconds_per_step": audio / len(steps),
        "mean_num_seqs": statistics.fmean(step["num_seqs"] for step in steps),
        "attention_padding": sum(s["padded_frames_squared"] for s in steps) / sum(s["frames_squared"] for s in steps),
        "flops_by_op": dict(sorted(by_op.items(), key=lambda item: -item[1])),
        "attention_paths": counted["attention_paths"],
        "fallbacks": counted["fallbacks"],
        "per_step": steps,
    }


def utilization(
    flops: Dict[str, Any], log: Dict[str, Any], *, num_gpus: int, peak_tflops: float = PEAK_TFLOPS
) -> Dict[str, Any]:
    """
    :param flops: the summary of the probe, see :func:`summarize_flops`
    :param log: the summary of the training log, see :func:`summarize_log`
    :param num_gpus: GPUs the training ran on, each on a batch of its own per step
    :param peak_tflops: the peak of one GPU in TFLOPS
    :return: the TFLOPS per GPU at the median logged step time and over the epochs, their share of the peak and the
        FLOPs of the whole training
    """
    per_step = flops["mean_flops_per_step"]
    steps = sum(epoch["steps"] for epoch in log["epochs"])
    seconds = sum(epoch["seconds"] for epoch in log["epochs"])
    tflops = per_step / log["median_sec_per_step"] / 1e12
    mean_tflops = per_step / (seconds / steps) / 1e12
    return {
        "flops_per_step_per_gpu": per_step,
        "median_sec_per_step": log["median_sec_per_step"],
        "tflops_per_gpu": tflops,
        "utilization": tflops / peak_tflops,
        "mean_sec_per_step": seconds / steps,
        "mean_tflops_per_gpu": mean_tflops,
        "mean_utilization": mean_tflops / peak_tflops,
        "train_steps": steps,
        "train_hours": seconds / 3600,
        "training_flops": per_step * num_gpus * steps,
        "num_gpus": num_gpus,
        "peak_tflops": peak_tflops,
    }


class TrainingFlopsJob(Job):
    """
    Counts the FLOPs of the first steps of a training on its own batches, eagerly on one GPU, see :func:`probe_config`.
    """

    def __init__(
        self,
        *,
        returnn_config: ReturnnConfig,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        sample_rate: int,
        frame_rate: float,
        num_steps: int = 30,
        cpu_rqmt: int = 24,
        mem_rqmt: float = 120,
        time_rqmt: float = 1,
    ):
        """
        :param returnn_config: the config of the training, before the training job completes it
        :param returnn_python_exe: the python of the run
        :param returnn_root: the RETURNN checkout of the run
        :param sample_rate: audio samples per second
        :param frame_rate: encoder frames per second
        :param num_steps: the steps counted
        :param cpu_rqmt: CPUs of the run, for the dataset workers of the training
        :param mem_rqmt: memory of the run in GB
        :param time_rqmt: hours of the run
        """
        self.returnn_config = returnn_config
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.num_steps = num_steps
        self.out_returnn_config = self.output_path("returnn.config")
        self.out_models = self.output_path("models", directory=True)
        self.out_flops = self.output_path("flops.json")
        self.out_report = self.output_path("flops.txt")
        self.rqmt = {"gpu": 1, "cpu": cpu_rqmt, "mem": mem_rqmt, "time": time_rqmt}

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        """
        :param parsed_args: the arguments of the job
        :return: its hash, without its resources
        """
        resources = ("cpu_rqmt", "mem_rqmt", "time_rqmt")
        return super().hash({key: value for key, value in parsed_args.items() if key not in resources})

    def tasks(self):
        """
        :return: the tasks
        """
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def create_files(self):
        """
        Writes the config of the probe.
        """
        probe_config(
            self.returnn_config,
            num_steps=self.num_steps,
            sample_rate=self.sample_rate,
            frame_rate=self.frame_rate,
            model_dir=self.out_models.get_path(),
        ).write(self.out_returnn_config.get_path())

    def run(self):
        """
        Runs the probe and writes the summary of its counts.
        """
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.rqmt["cpu"])
        env["MKL_NUM_THREADS"] = str(self.rqmt["cpu"])
        rnn = os.path.join(self.returnn_root.get_path(), "rnn.py")
        subprocess.check_call([self.returnn_python_exe.get_path(), rnn, self.out_returnn_config.get_path()], env=env)

        with open(STEPS_FILE, "rt") as f:
            summary = summarize_flops(json.load(f))
        with open(self.out_flops.get_path(), "wt") as f:
            json.dump(summary, f, indent=1)
            f.write("\n")
        with open(self.out_report.get_path(), "wt") as f:
            f.write(_flops_report(summary))


def _flops_report(summary: Dict[str, Any]) -> str:
    """
    :param summary: the summary of a probe
    :return: its report
    """
    lines = [
        f"{summary['steps']} steps of one GPU, {summary['mean_audio_seconds_per_step']:.1f} s of audio and"
        f" {summary['mean_num_seqs']:.1f} sequences each on average",
        f"{summary['mean_flops_per_step'] / 1e12:.2f} TFLOPs per step, of them"
        f" {summary['mean_forward_flops_per_step'] / 1e12:.2f} in the forward pass",
        f"{summary['flops_per_audio_second'] / 1e9:.1f} GFLOPs per second of audio, of them"
        f" {summary['forward_flops_per_audio_second'] / 1e9:.1f} in the forward pass",
        f"attention over the frames padded to {summary['attention_padding']:.3f} times its frames squared",
        "attention paths " + ", ".join(f"{path} {count}" for path, count in sorted(summary["attention_paths"].items())),
        "fallbacks " + (", ".join(summary["fallbacks"]) or "none"),
        "TFLOPs by op over all steps",
    ]
    lines += [f"{flops / 1e12:12.2f}  {op}" for op, flops in summary["flops_by_op"].items()]
    return "\n".join(lines) + "\n"


class TrainingUtilizationJob(Job):
    """
    Divides the FLOPs a probe counted per step by the step times the log of the training holds.
    """

    def __init__(self, *, flops: tk.Path, learning_rates: tk.Path, num_gpus: int, peak_tflops: Optional[float] = None):
        """
        :param flops: the summary of the probe of the training
        :param learning_rates: the learning rates file of the training, two levels below its job directory
        :param num_gpus: GPUs the training ran on
        :param peak_tflops: the peak of one GPU in TFLOPS, None for :data:`PEAK_TFLOPS`
        """
        self.flops = flops
        self.learning_rates = learning_rates
        self.num_gpus = num_gpus
        self.peak_tflops = peak_tflops
        self.out_summary = self.output_path("utilization.json")
        self.out_report = self.output_path("utilization.txt")

    def tasks(self):
        """
        :return: the tasks
        """
        yield Task("run", mini_task=True)

    def run(self):
        """
        Writes the FLOPS and the utilization.
        """
        with open(self.flops.get_path(), "rt") as f:
            flops = json.load(f)
        log = summarize_log(_rank_zero_lines(os.path.dirname(os.path.dirname(self.learning_rates.get_path()))))
        result = utilization(flops, log, num_gpus=self.num_gpus, peak_tflops=self.peak_tflops or PEAK_TFLOPS)
        with open(self.out_summary.get_path(), "wt") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
        with open(self.out_report.get_path(), "wt") as f:
            f.write(
                f"{result['tflops_per_gpu']:.1f} TFLOPS per GPU at the median step of"
                f" {result['median_sec_per_step']:.3f} s, {100 * result['utilization']:.1f} percent of"
                f" {result['peak_tflops']:.0f}\n"
                f"{result['mean_tflops_per_gpu']:.1f} TFLOPS per GPU over the epochs at"
                f" {result['mean_sec_per_step']:.3f} s per step, {100 * result['mean_utilization']:.1f} percent\n"
                f"{result['training_flops'] / 1e18:.1f} EFLOPs in {result['train_steps']} steps on"
                f" {result['num_gpus']} GPUs, {result['train_hours']:.2f} h\n"
            )
