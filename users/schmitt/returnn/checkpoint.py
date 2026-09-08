"""
Conversion and averaging of RETURNN PyTorch checkpoints.
"""

import os
import re
import shutil
import tarfile
from typing import Any, Dict, List, Optional, Sequence, Union

from sisyphus import tk, Job, Task

from i6_core.returnn.training import PtCheckpoint

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep


def hf_nemotron_key_to_nemo_key(name: str) -> str:
    """
    Map a parameter name of the HuggingFace `NemotronAsrStreamingForRNNT` (or `ParakeetForRNNT`,
    same structure) to the corresponding name in NeMo's `EncDecRNNTBPEModel`.

    The two implementations only differ in the parameter names, the tensors themselves are
    identical (verified against the released `nvidia/nemotron-speech-streaming-en-0.6b` checkpoints:
    all 651 tensors of `model.safetensors` are bit-identical to the ones in the `.nemo` archive).
    The only NeMo params without a HF counterpart are the non-trained
    `preprocessor.featurizer.{window,fb}` buffers.

    :param name: HF parameter name, e.g. "encoder.layers.0.self_attn.q_proj.weight"
    :return: NeMo parameter name, e.g. "encoder.layers.0.self_attn.linear_q.weight"
    """
    # conv sub-sampling: NeMo has one flat `conv` sequential (conv, act, conv, conv, act, ...)
    name = re.sub(r"^encoder\.subsampling\.linear\.", "encoder.pre_encode.out.", name)
    name = re.sub(r"^encoder\.subsampling\.conv_in\.", "encoder.pre_encode.conv.0.", name)
    match = re.match(r"^encoder\.subsampling\.layers\.(\d+)\.(depthwise|pointwise)_conv\.(.*)$", name)
    if match:
        layer_idx, conv_type, suffix = int(match.group(1)), match.group(2), match.group(3)
        conv_idx = 2 + 3 * layer_idx + (0 if conv_type == "depthwise" else 1)
        name = f"encoder.pre_encode.conv.{conv_idx}.{suffix}"

    # conformer layers: only the attention and the conv norm are renamed
    for hf_proj, nemo_proj in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "out"), ("relative_k", "pos")):
        name = re.sub(rf"(self_attn)\.{hf_proj}_proj\.", rf"\1.linear_{nemo_proj}.", name)
    name = re.sub(r"(self_attn)\.bias_([uv])$", r"\1.pos_bias_\2", name)
    name = re.sub(r"\.conv\.norm\.", ".conv.batch_norm.", name)  # is a LayerNorm despite the name

    # prediction network + joint. NeMo keeps both joint projections in the joint module, HF keeps
    # them next to the module whose output they project.
    name = re.sub(r"^decoder\.embedding\.", "decoder.prediction.embed.", name)
    name = re.sub(r"^decoder\.lstm\.", "decoder.prediction.dec_rnn.lstm.", name)
    name = re.sub(r"^decoder\.decoder_projector\.", "joint.pred.", name)
    name = re.sub(r"^encoder_projector\.", "joint.enc.", name)
    name = re.sub(r"^joint\.head\.", "joint.joint_net.2.", name)

    return name


class ReturnnPtCheckpointToNemoJob(Job):
    """
    Convert a RETURNN PyTorch checkpoint of a HuggingFace Nemotron/Parakeet RNN-T model into a
    `.nemo` archive, so that the fine-tuned model can be used with NeMo (`ASRModel.restore_from`).

    A `.nemo` file is an uncompressed tar of `model_config.yaml`, `model_weights.ckpt` (a plain
    `torch.save` of the state dict) and the tokenizer files. Everything except the weights is taken
    over unchanged from `reference_nemo_file`, i.e. from the checkpoint the training started from,
    so the architecture/tokenizer description stays exactly the one NVIDIA shipped.
    """

    def __init__(
        self,
        *,
        checkpoint: Union[tk.Path, PtCheckpoint],
        reference_nemo_file: tk.Path,
        checkpoint_key: Optional[str] = "model",
        param_prefix: str = "model.",
        config_updates: Optional[Dict[str, Any]] = None,
    ):
        """
        :param checkpoint: RETURNN checkpoint, e.g. `train_job.out_checkpoints[epoch]`
        :param reference_nemo_file: the `.nemo` the training was initialized from. Provides
            `model_config.yaml`, the tokenizer files and the non-trained preprocessor buffers.
        :param checkpoint_key: sub-dict of the RETURNN checkpoint holding the state dict
        :param param_prefix: prefix of the HF model params inside the RETURNN model, i.e. the
            attribute the `NemotronAsrStreamingForRNNT` is stored under in the model wrapper
        :param config_updates: updates for `model_config.yaml` (dotted keys, cf. `dict_update_deep`),
            e.g. `{"encoder.att_context_size": [[70, 13]]}` if the set of trained right contexts
            changed. `None` copies the config over unchanged.
        """
        super().__init__()
        self.checkpoint = checkpoint
        self.reference_nemo_file = reference_nemo_file
        self.checkpoint_key = checkpoint_key
        self.param_prefix = param_prefix
        self.config_updates = config_updates

        self.rqmt = {"time": 1, "cpu": 1, "mem": 16}
        self.out_nemo_file = self.output_path("model.nemo")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import torch

        stage_dir = "nemo_archive"
        os.makedirs(stage_dir, exist_ok=True)

        # everything but the weights is taken over 1:1 (the tokenizer file names are referenced from
        # `model_config.yaml` as "nemo:<file name>", so they have to keep their names)
        with tarfile.open(self.reference_nemo_file.get_path(), "r:") as tar:
            ref_state_dict = torch.load(
                tar.extractfile("./model_weights.ckpt"), map_location="cpu", weights_only=True
            )
            for member in tar.getmembers():
                if member.isfile() and os.path.basename(member.name) != "model_weights.ckpt":
                    tar.extract(member, path=stage_dir)

        if self.config_updates:
            import yaml

            config_file = os.path.join(stage_dir, "model_config.yaml")
            with open(config_file) as f:
                config = yaml.safe_load(f)
            config = dict_update_deep(config, self.config_updates)
            with open(config_file, "w") as f:
                yaml.safe_dump(config, f, sort_keys=False)

        checkpoint = self.checkpoint.path if isinstance(self.checkpoint, PtCheckpoint) else self.checkpoint
        state_dict = torch.load(checkpoint.get_path(), map_location="cpu", weights_only=True)
        if self.checkpoint_key is not None:
            state_dict = state_dict[self.checkpoint_key]

        out_state_dict = {}
        for name, param in state_dict.items():
            assert name.startswith(self.param_prefix), f"unexpected param {name!r} outside {self.param_prefix!r}"
            out_state_dict[hf_nemotron_key_to_nemo_key(name[len(self.param_prefix) :])] = param

        # the preprocessor buffers (mel filterbank + STFT window) are deterministic and thus not
        # part of the HF model
        for name, param in ref_state_dict.items():
            if name not in out_state_dict:
                print(f"Taking {name} from the reference checkpoint")
                out_state_dict[name] = param

        # the mapping has to yield exactly the params of the reference checkpoint
        assert set(out_state_dict) == set(ref_state_dict), (
            f"params not in the reference checkpoint: {sorted(set(out_state_dict) - set(ref_state_dict))}, "
            f"missing params: {sorted(set(ref_state_dict) - set(out_state_dict))}"
        )
        for name, param in out_state_dict.items():
            ref_param = ref_state_dict[name]
            assert param.shape == ref_param.shape, f"{name}: shape {param.shape} != {ref_param.shape}"
            out_state_dict[name] = param.to(ref_param.dtype)

        torch.save(out_state_dict, os.path.join(stage_dir, "model_weights.ckpt"))

        with tarfile.open(self.out_nemo_file.get_path(), "w") as tar:
            tar.add(stage_dir, arcname=".")

        shutil.rmtree(stage_dir)


class AverageReturnnPtCheckpointsJob(Job):
    """
    Average the parameters of several RETURNN PyTorch checkpoints into one checkpoint.

    This is the standard remedy for a training whose *decoded* quality wobbles from checkpoint to
    checkpoint even though the training loss is smooth: the wobble is a movement of the model
    around a minimum, and the average of several such points sits closer to it than any of them.
    For the Nemotron RNN-T fine-tunings of this setup the wobble is almost entirely a shift of the
    blank/emit operating point (the deletion rate swings by ~2 points peak-to-peak while
    substitutions and insertions stay within ~0.3), which is exactly the kind of single-direction
    movement averaging cancels.

    The output has the same layout as a RETURNN checkpoint (the parameters under `"model"`), so it
    can be fed to `ReturnnPtCheckpointToNemoJob` -- or preloaded into a further training -- like
    any epoch checkpoint. `epoch`/`step` are the maxima over the inputs and `merged_epochs` records
    what went in, mirroring RETURNN's own `tools/torch_avg_checkpoints.py`.

    Averaging happens in float64 and is cast back to the dtype of the first checkpoint, so a
    bfloat16-stored parameter does not lose the average in its own rounding. Note that averaging
    only makes sense over checkpoints of *one* training run: the parameters of two runs are not in
    the same basin and their average is not a model.
    """

    def __init__(
        self,
        *,
        checkpoints: Sequence[Union[tk.Path, PtCheckpoint]],
        checkpoint_key: Optional[str] = "model",
    ):
        """
        :param checkpoints: the checkpoints to average, e.g.
            `[train_job.out_checkpoints[epoch] for epoch in epochs]`. Averaging a single checkpoint
            is rejected: that is a copy, and the caller almost certainly meant something else.
        :param checkpoint_key: sub-dict of the RETURNN checkpoint holding the state dict. `None`
            treats the whole checkpoint as the state dict, and then the output is a bare state dict
            as well.
        """
        super().__init__()
        assert len(checkpoints) > 1, f"averaging needs at least two checkpoints, got {len(checkpoints)}"
        self.checkpoints = [ckpt.path if isinstance(ckpt, PtCheckpoint) else ckpt for ckpt in checkpoints]
        self.checkpoint_key = checkpoint_key

        # 0.6B fp32 parameters are 2.4G per checkpoint; the fp64 accumulator doubles that, and only
        # one input is held open at a time (mmap'ed, so it stays in the page cache rather than RSS)
        self.rqmt = {"time": 2, "cpu": 1, "mem": 24}
        self.out_checkpoint = PtCheckpoint(self.output_path("model/average.pt"))

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import torch

        num_checkpoints = len(self.checkpoints)
        accumulator: Dict[str, torch.Tensor] = {}
        dtypes: Dict[str, torch.dtype] = {}
        merged_epochs: List[Optional[int]] = []
        merged_steps: List[Optional[int]] = []
        out_state: Dict[str, Any] = {}

        for checkpoint in self.checkpoints:
            print(f"reading {checkpoint.get_path()}")
            # `mmap` keeps only the tensors that are actually touched resident, i.e. the peak is the
            # accumulator plus one parameter at a time rather than the whole checkpoint
            state = torch.load(checkpoint.get_path(), map_location="cpu", weights_only=True, mmap=True)
            state_dict = state if self.checkpoint_key is None else state[self.checkpoint_key]

            if not accumulator:
                for name, param in state_dict.items():
                    assert param.dtype.is_floating_point, (
                        f"{name} is {param.dtype}, which cannot be averaged. Non-float parameters "
                        "(quantized weights, integer buffers) need a decision the caller has to make."
                    )
                    dtypes[name] = param.dtype
                    accumulator[name] = param.to(torch.float64)
            else:
                assert set(state_dict) == set(accumulator), (
                    f"{checkpoint.get_path()} has other parameters than the first checkpoint: "
                    f"extra {sorted(set(state_dict) - set(accumulator))}, "
                    f"missing {sorted(set(accumulator) - set(state_dict))}"
                )
                for name, param in state_dict.items():
                    assert param.shape == accumulator[name].shape, (
                        f"{name}: shape {tuple(param.shape)} != {tuple(accumulator[name].shape)}"
                    )
                    accumulator[name] += param.to(torch.float64)

            # only a checkpoint with a `checkpoint_key` has metadata around the state dict; without
            # one, `state` *is* the state dict and `state["epoch"]` would be a parameter name
            if self.checkpoint_key is not None:
                merged_epochs.append(state.get("epoch"))
                merged_steps.append(state.get("step"))

                for name, value in state.items():
                    if name == self.checkpoint_key:
                        continue
                    if name in ("epoch", "step"):
                        # the average is "as trained as" the latest checkpoint that went into it
                        out_state[name] = max(value, out_state[name]) if name in out_state else value
                    elif name not in out_state:
                        # everything else (`returnn_version`, ...) is taken from the first checkpoint
                        out_state[name] = value

        averaged = {name: (param / num_checkpoints).to(dtypes[name]) for name, param in accumulator.items()}
        print(f"averaged {len(averaged)} parameters over {num_checkpoints} checkpoints, epochs {merged_epochs}")

        if self.checkpoint_key is None:
            out_state = averaged
        else:
            out_state[self.checkpoint_key] = averaged
            out_state["merged_epochs"] = merged_epochs
            out_state["merged_steps"] = merged_steps

        os.makedirs(os.path.dirname(self.out_checkpoint.path.get_path()), exist_ok=True)
        torch.save(out_state, self.out_checkpoint.path.get_path())
