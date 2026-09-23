"""Guard: self-play's shared base load (``selfplay.load_models``) == one load per side.

Until 2026-09-24 ``selfplay`` loaded the 16.7 GB PersonaPlex base twice (once per side, ~3.5 min each on
/hpcwork). ``load_models`` loads it once, copies it on its device BEFORE any side's finetune, and each
side applies its own LoRA/overlay to its own copy.

  1. SAME MODELS. A tiny random PersonaPlex LM is saved as a safetensors checkpoint, with two different
     LoRA adapters (side A, side B). The real ``engine.load_base_lm`` (prefetch + vendored
     ``get_moshi_lm``) and ``engine.apply_finetune`` build both sides once through ``load_models``
     and once with one load per side (the old path). Every tensor of every side is bit-identical,
     the two sides share no storage, and the shared path read the checkpoint once (old: twice).
  2. SAME CONVERSATIONS. ``converse_batched`` (sampled, seeded) on both pairs gives identical codes
     and text for every row.
  3. CAN FAIL. (a) B built with A's adapter gives different conversations, so check 2 sees a side
     carrying the wrong finetune; (b) with the copy removed (both sides on one module), check 1 or 2
     fails.

moshi_family venv, CPU, seconds (no checkpoint download):
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_selfplay_shared_base.py'
"""

import contextlib
import copy
import io
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import moshi_family.personaplex.selfplay as sp  # noqa: E402
from moshi_family.build import LoraConfig, apply_lora_to  # noqa: E402
from moshi_family.personaplex import engine  # noqa: E402
from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMGen, LMModel  # noqa: E402
from moshi_family.train_loop import is_lora_param  # noqa: E402

TINY = dict(
    dim=64,
    num_heads=4,
    num_layers=2,
    hidden_scale=2.0,
    text_card=100,
    context=256,
    depformer_dim=32,
    depformer_dim_feedforward=64,
    depformer_num_heads=2,
    depformer_num_layers=1,
    dep_q=16,
)
SIL, CTX, NF, RANK = 6, 200, 20, 4


class _NoMimi:
    _streaming_state = None

    def reset_streaming(self):
        pass

    def streaming_forever(self, _b):
        pass

    def _stop_streaming(self):
        pass


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def save_adapter(base: LMModel, seed: int, path: str) -> None:
    lm = copy.deepcopy(base)
    quiet(apply_lora_to, lm, LoraConfig(rank=RANK, scaling=2.0))
    g = torch.Generator().manual_seed(seed)
    sd = {
        n: (0.05 * torch.randn(p.shape, generator=g)).to(p.dtype) for n, p in lm.named_parameters() if is_lora_param(n)
    }
    save_file(sd, path)


def rows(seed):
    r = np.random.default_rng(seed)
    return [
        (
            r.integers(0, 2048, size=(8, v)),
            [int(x) for x in r.integers(4, 100, size=t)],
            r.integers(0, 2048, size=(8, v)),
            [int(x) for x in r.integers(4, 100, size=t)],
        )
        for v, t in ((50, 5), (44, 9), (47, 2))
    ]


def main():
    torch.manual_seed(0)
    loaders._lm_kwargs = dict(loaders._lm_kwargs, **TINY)  # get_moshi_lm builds from this dict
    base = LMModel(device="cpu", dtype=torch.float32, **loaders._lm_kwargs).eval()
    with tempfile.TemporaryDirectory() as d:
        base_path = os.path.join(d, "model.safetensors")
        save_file({k: v.contiguous().clone() for k, v in base.state_dict().items()}, base_path)
        ad = {s: os.path.join(d, f"lora_{s}.safetensors") for s in "ab"}
        save_adapter(base, 1, ad["a"])
        save_adapter(base, 2, ad["b"])
        specs = [ad["a"], ad["b"]]
        loads = []

        def load_base():
            loads.append(1)
            return engine.load_base_lm(None, weight=base_path, device="cpu", context=CTX, dtype=torch.float32)

        def build(lm, overlay):
            engine.apply_finetune(lm, lora=LoraConfig(rank=RANK, scaling=2.0), overlay=overlay, device="cpu")
            g = LMGen(
                lm, device="cpu", use_sampling=True, audio_silence_frame_cnt=SIL, sample_rate=24000, frame_rate=12.5
            )
            g.streaming_forever(1)
            return types.SimpleNamespace(lm_gen=g, mimi=_NoMimi(), other_mimi=_NoMimi())

        def per_side(specs_):
            return [build(load_base(), s) for s in specs_]

        def talk(a, b):
            rs = rows(7)
            torch.manual_seed(1234)
            ((ca, ta), (cb, tb)), _ = sp.converse_batched(
                a,
                b,
                voices_a=[x[0] for x in rs],
                texts_a=[x[1] for x in rs],
                voices_b=[x[2] for x in rs],
                texts_b=[x[3] for x in rs],
                n_frames=NF,
                silence_frames=SIL,
            )
            return [np.asarray(x) for x in (ca, ta, cb, tb)]

        def sd(m):
            return m.lm_gen.lm_model.state_dict()

        def models_equal(xs, ys) -> bool:
            for x, y in zip(xs, ys):
                a, b = sd(x), sd(y)
                if a.keys() != b.keys() or not all(torch.equal(a[k], b[k]) for k in a):
                    return False
            return True

        # 1
        shared = quiet(sp.load_models, specs, load_base=load_base, build=build)
        n_shared, loads[:] = len(loads), []
        ref = quiet(per_side, specs)
        n_ref = len(loads)
        assert (n_shared, n_ref) == (1, 2), (n_shared, n_ref)
        assert models_equal(shared, ref), "shared-base models differ from one load per side"
        ptrs = [{t.data_ptr() for t in sd(m).values() if t.numel()} for m in shared]
        assert not ptrs[0] & ptrs[1], "the two sides share tensor storage"
        assert not torch.equal(sd(shared[0])[_first_lora(shared[0])], sd(shared[1])[_first_lora(shared[1])])
        print(
            f"[ok] 1 shared base: {n_shared} checkpoint load vs {n_ref}; all {len(sd(shared[0]))} tensors per side "
            "bit-identical to one load per side; no shared storage; A and B carry different adapters"
        )

        # 2
        got, want = quiet(talk, *shared), quiet(talk, *ref)
        assert all(np.array_equal(x, y) for x, y in zip(got, want)), "shared-base conversations differ"
        print(f"[ok] 2 converse_batched: 3 rows x {NF} frames, codes + text of both sides identical")

        # 3a
        wrong = quiet(talk, *quiet(per_side, [ad["a"], ad["a"]]))
        assert not all(np.array_equal(x, y) for x, y in zip(wrong, want)), "check 2 cannot see B carrying A's adapter"
        # 3b
        real_copy = sp.copy
        sp.copy = types.SimpleNamespace(deepcopy=lambda x: x)
        try:
            try:
                broken = quiet(sp.load_models, specs, load_base=load_base, build=build)
                caught = not models_equal(broken, ref) or not all(
                    np.array_equal(x, y) for x, y in zip(quiet(talk, *broken), want)
                )
                how = "differs"
            except Exception as e:  # e.g. the second LoRA wrap / streaming scope on the same module
                caught, how = True, f"raises {type(e).__name__}"
        finally:
            sp.copy = real_copy
        assert caught, "check 1/2 cannot fail: sharing one module between the sides passed"
        print(f"[ok] 3 B with A's adapter changes the conversations; without the copy the shared path {how}")
    print("\nALL CHECKS PASSED")


def _first_lora(m):
    return next(k for k in sd_keys(m) if is_lora_param(k))


def sd_keys(m):
    return list(m.lm_gen.lm_model.state_dict().keys())


if __name__ == "__main__":
    main()
