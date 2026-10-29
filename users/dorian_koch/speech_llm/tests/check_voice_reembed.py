"""Guard: a ``.pt`` voice prompt is replayed with the finetune's OWN input embeddings, never the base model's.

A ``.pt`` voice stores the saving (base) model's summed input embeddings per frame; a full finetune trains ``emb.*`` /
``text_emb``, so replaying the file primed it with stale embeddings (audit PPX-04). ``moshi_family.personaplex.
voice_reembed`` recovers the frame tokens with the base tables (snapshotted by ``build.overlay_state_dict``) and embeds
them with the current ones; ``LMGen.load_voice_prompt_embeddings`` calls it.

Unit mode (moshi_family venv, CPU, ~1 min; reads only the base embedding tables):
  [1] all 18 released voices decompose into the base tables bit for bit;
  [2] a simulated full finetune (perturbed tables): the re-embedded prompt equals the perturbed model's embeddings of
      the recovered tokens. CAN FAIL: without the base snapshot nothing is re-embedded (the stale prompt stays);
  [3] the real ``LMGen.load_voice_prompt_embeddings`` re-embeds (driven on a stub generator);
  [4] ``overlay_state_dict`` snapshots the base tables exactly when the overlay replaces them.
Graph mode (``--graph``, setup .venv, ~2 min): [5] every ``SpeechInference`` downstream of a PersonaPlex
FULL-finetune ``SpeechFinetune`` runs at ``code_version`` >= 2 (``track_persona.PPX_FULL_FT_CODE_VERSION``), i.e.
after the fix.

    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'PYTHONPATH=$PWD/recipe/speech_llm/full_duplex \\
        python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_voice_reembed.py'
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_voice_reembed.py --graph
"""

import glob
import os
import sys
import tempfile

FAILURES = []
HF = "/hpcwork/p0023999/common_hf_home/hub/models--nvidia--personaplex-7b-v1/snapshots"


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


def unit():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from moshi_family.build import BASE_TABLES_ATTR, overlay_state_dict
    from moshi_family.personaplex import voice_reembed as vr
    from moshi_family.personaplex.models.lm import LMGen, LMModel, ScaledEmbedding

    snap = sorted(glob.glob(f"{HF}/*/"))[-1]
    with safe_open(os.path.join(snap, "model.safetensors"), "pt") as f:
        base = {k: f.get_tensor(k) for k in ["text_emb.weight", *[f"emb.{i}.weight" for i in range(16)]]}
    tables = [base["text_emb.weight"], *[base[f"emb.{i}.weight"] for i in range(16)]]

    print("[1] released voices decompose exactly")
    voices = sorted(glob.glob(os.path.join(snap, "voices", "*.pt")))
    check(len(voices) == 18, f"18 released voices ({len(voices)})")
    toks = {}
    for p in voices:
        e = torch.load(p, map_location="cpu")["embeddings"].reshape(-1, 4096)
        try:
            tok = vr.recover_tokens(e, tables)
            ok = bool((vr.bf16_sum(tables, tok) == e).all())
        except RuntimeError as err:
            ok, tok = False, None
            print("   ", err)
        toks[os.path.basename(p)] = tok
        check(ok, f"{os.path.basename(p)}: {e.shape[0]} frames bit-exact")

    class FakeLM(torch.nn.Module):
        num_audio_codebooks, audio_offset, num_codebooks = 16, 1, 17
        embed_codes = LMModel.embed_codes
        device = torch.device("cpu")

        def __init__(self, t):
            super().__init__()
            self.text_emb = ScaledEmbedding(*t["text_emb.weight"].shape, zero_idx=-1, dtype=torch.bfloat16)
            self.emb = torch.nn.ModuleList(
                ScaledEmbedding(*t[f"emb.{i}.weight"].shape, zero_idx=-1, dtype=torch.bfloat16) for i in range(16)
            )
            self.load_state_dict(t)

    print("[2] a simulated full finetune gets its own embeddings")
    g = torch.Generator().manual_seed(0)
    tuned = {k: (v.float() + 0.02 * torch.randn(v.shape, generator=g)).to(torch.bfloat16) for k, v in base.items()}
    lm = FakeLM(tuned)
    stored = torch.load(voices[0], map_location="cpu")["embeddings"]

    class Gen:
        pass

    gen = Gen()
    gen.lm_model, gen.voice_prompt_embeddings = lm, stored.clone()
    check(not vr.reembed_voice_prompt(gen), "CAN FAIL: without a base snapshot nothing is re-embedded")
    check(torch.equal(gen.voice_prompt_embeddings, stored), "... and the stale base prompt would be replayed")
    setattr(lm, BASE_TABLES_ATTR, {k: v.clone() for k, v in base.items()})
    check(vr.reembed_voice_prompt(gen), "with the snapshot the prompt is re-embedded")
    want = lm.embed_codes(toks[os.path.basename(voices[0])][:, :, None])
    got = gen.voice_prompt_embeddings
    check(got.shape == stored.shape and torch.equal(got.reshape(want.shape), want), "= the finetune's own embed_codes")
    check(not torch.equal(got, stored), "and differs from the stored base embeddings")

    print("[3] LMGen.load_voice_prompt_embeddings re-embeds")
    stub = Gen()
    stub.lm_model = lm
    LMGen.load_voice_prompt_embeddings(stub, voices[0])
    check(torch.equal(stub.voice_prompt_embeddings, got), "the real loader replays the finetune's embeddings")

    print("[4] overlay_state_dict snapshots the replaced tables")
    m = FakeLM(base)
    with tempfile.TemporaryDirectory() as d:
        save_file({"emb.3.weight": tuned["emb.3.weight"]}, os.path.join(d, "o.safetensors"))
        overlay_state_dict(m, os.path.join(d, "o.safetensors"), tag="guard")
        snapd = getattr(m, BASE_TABLES_ATTR, None) or {}
        check(set(snapd) == {"emb.3.weight"} and torch.equal(snapd["emb.3.weight"], base["emb.3.weight"]),
              "an overlay with emb.3 snapshots the BASE emb.3")  # fmt: skip
        nonemb = torch.nn.Linear(2, 2)
        save_file({"weight": nonemb.weight.detach().clone()}, os.path.join(d, "q.safetensors"))
        overlay_state_dict(nonemb, os.path.join(d, "q.safetensors"), tag="guard")
        check(getattr(nonemb, BASE_TABLES_ATTR, None) is None, "an overlay without input tables snapshots nothing")


def graph():
    import logging

    setup = next(p for p in (os.getcwd(), *map(str, __import__("pathlib").Path(__file__).absolute().parents))
                 if os.path.isdir(os.path.join(p, "recipe")))  # fmt: skip
    os.chdir(setup)
    sys.path[:0] = ["recipe", "recipe/sisyphus"]
    os.environ.setdefault("CUDA_HOME", "/usr")
    logging.disable(logging.WARNING)
    from sisyphus import graph as sgraph
    from sisyphus.loader import config_manager

    config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])
    jobs = list(sgraph.graph.jobs())
    # PersonaPlex only: base-Moshi inference has no voice prompt
    full = [j for j in jobs if type(j).__name__ == "SpeechFinetune" and j.adapter.overlay_kind == "full"
            and "personaplex" in str(j.adapter.base_model)]  # fmt: skip
    rev = {}
    for j in jobs:
        for p in j._sis_inputs:
            if p.creator is not None:
                rev.setdefault(id(p.creator), []).append(j)
    seen, stack, infer = set(), list(full), []
    while stack:
        for d in rev.get(id(stack.pop()), []):
            if id(d) not in seen:
                seen.add(id(d))
                stack.append(d)
                if type(d).__name__ == "SpeechInference":
                    infer.append(d)
    print(f"[5] {len(full)} PersonaPlex full-finetune runs, {len(infer)} SpeechInference jobs downstream")
    check(all("personaplex" in str(j.offline_module) for j in infer), "every one runs the PersonaPlex engine")
    old = [j._sis_id() for j in infer if int(getattr(j, "code_version", 1)) < 2]
    check(full and infer and not old, f"every one at code_version >= 2 ({len(old)} not: {old[:3]})")


if __name__ == "__main__":
    graph() if "--graph" in sys.argv else unit()
    print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
    sys.exit(1 if FAILURES else 0)
