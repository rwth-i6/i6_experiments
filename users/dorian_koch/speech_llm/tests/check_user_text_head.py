"""Guard: the PersonaPlex user-text head (2026-10-12; ``user_text_loss_weight``).

WHY. An auxiliary head predicts the USER's words (delayed) from the temporal transformer's output. Four ways it could
silently go wrong: the user's words leak into the ASSISTANT text row (``interleave_text`` ignores the speaker field, so
the separation is only by column -- a leak would teach the model to echo), the targets come from the wrong side on a
swapped read, the head's logits get fed back / its gradient never reaches the backbone, or the knob changes something
when it is off. Real code throughout: the real ``persona_item_builder`` / ``build_knowledge_mix_loader`` on real corpora,
the real ``collate_codes``, a tiny real PersonaPlex ``LMModel`` wrapped by the launcher's own ``apply_lora_to``, and the
shared ``make_user_text_head`` / ``TextHeadInputCapture`` / ``add_user_text_loss`` the launcher calls.
  1. ITEMS (p4's corpus, swap + stored user augmentation, 160 seeded rows, the run's text knobs): with vs without
     ``user_text_delay``: identical codes and loss masks (the assistant text row is built from the assistant's words
     only); one more tensor, long ``[T]``; IGNORE on the prompt; on the dialogue = ``interleave_text`` of THIS read's
     user words (``alignments_other``, or the stored ``alignments`` when swapped) delayed 6 frames; IGNORE throughout
     when ``other_ok`` is false. CAN FAIL: the same expectation from the WRONG side differs on the swapped reads, and
     the undelayed user row differs from the assistant text row (a leak would be visible).
  2. COLLATE: arity 2 / 3 batches equal a verbatim copy of the pre-2026-10-12 ``collate_codes``; the user-text row pads
     with IGNORE (a 4-tuple and a 3-tuple with a long last tensor).
  3. MIX (p4 + v6 knowledge at 0.3, the real loader with item_meta): codes / masks / (source, row, swap) per slot equal
     the run without the knob; knowledge slots are all IGNORE, primary slots carry targets.
  4. LOSS PARITY: ``user_text_loss`` = ``duplex_ce_loss``'s text half fed the same logits with the targets as the text
     row (count and weight normalisation, EPAD weight 1.0, IGNORE frames = masked frames). CAN FAIL: PAD weight 0.5.
  5. MODEL (tiny LMModel + LoRA, the launcher's order): the capture hook changes no output; the captured tensor reproduces
     ``text_logits`` through text_linear (it IS its input), the head is fp32 and zero-init; knob on: the head and the LoRA get gradient
     from the user-text CE (LoRA grads differ from the knob-off ones); perturbing the head changes no model output
     (never fed back); the adapter checkpoint (``save_trained_params(is_lora_param)``) has the same keys with the head
     as without and reloads onto an engine-style model.
LP phase (``detach``): the head learns while the backbone's gradient is bit-identical to the knob-off one.
  6. MID LAYER (``user_text_layer``, 2026-10-13; tiny 4-layer LMModel + LoRA, gradient checkpointing ON and the training
     fast path as the launcher applies them, the real ``forward_train``): the capture = ``lm.transformer.layers[k]``'s
     output (an independent hook's copy), untouched by the backward (full recompute); the head = a fresh
     norm of out_norm's type (fp32, identity scale) + the zero-init fp32 Linear; the user-text CE alone reaches LoRA
     params of layers <= k and NOT of layers > k (nor the depformer); LP (``detach``): no backbone gradient at all;
     the default (``"last"``) keeps the plain Linear head on text_linear's input.
  7. LR SCHEDULE (``lr_schedule``, 2026-10-13; the REAL ``run_training`` on a tiny model, 3 param groups, warmup 3 of
     10 steps): "cosine" = ``cosine_lr_scale`` exactly (every group, every step; the pre-change schedule); "constant" =
     the same warmup, then every group at its base LR to the last step.
  8. CTC LOSS (``user_text_loss_kind: "ctc"``, 2026-10-13, ablation ab2_usertext_kn_w02_ctc; the real
     ``user_text_ctc_spans`` / ``user_text_ctc_loss`` / ``add_user_text_loss``): (a) spans = an independent
     frame-by-frame rule (word frames +- the delay, merged, valid frames only, EPAD never a label) on random loader-format
     targets + a hand case; (b) the loss = an independent numpy CTC forward algorithm per span (blank = PAD) + the
     outside frames' weighted CE, over the CE's own denominator (both normalisations), its two logged parts sum to it;
     (c) without user words it IS the framewise CE; (d) a perfect peaky head scores ~0 at lag 0 / +1 / +2 / -2 and when
     it holds each token 2 frames (stutter), while the framewise CE of the +1 head and the CTC of wrong words are large
     (CAN FAIL); (e) through ``add_user_text_loss`` on a tiny LMModel + LoRA with a bf16 hidden state: finite, the head
     and the LoRA get gradient, LP detaches (backbone grad = knob-off), kind "ce" = the default call.
``--selftest``: 17 mutants -- CTC: blank = EPAD, span margin 1, spans ignore loss_mask, EPAD as a label, outside PAD CE
dropped, normalised per label token, CTC on bf16 logits, kind ignored by add_user_text_loss; and the nine before (delay 0, hook feeding back, detached capture, PAD weight 1.0 in the head loss, LP phase
not detached, collate padding the user row with 0; mid: the wrong layer, LP not detached; constant decaying like
cosine) must each fail. (The capture also ignores hook calls inside backward, in case a checkpoint recompute re-ran the
layer's hooks; measured on torch 2.12: it does not, even with early stop off, so no mutant can exercise it -- 6a checks
the capture is untouched after backward.)

moshi_family venv, CPU, ~3 min (setup root):
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && HF_HUB_OFFLINE=1 python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_user_text_head.py [--selftest]'
"""

import copy
import os
import sys
import tempfile
from pathlib import Path

SETUP = os.getcwd()
sys.path.insert(0, os.path.join(SETUP, "recipe", "speech_llm", "full_duplex"))
sys.path.insert(0, os.path.join(SETUP, "recipe"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import moshi_family.train_data_common as tdc  # noqa: E402
from moshi_family.personaplex import train_data as td  # noqa: E402

P4, V6 = "output/persona_v4/p1_data", "output/persona_knowledge/v6_codes"
D = 6
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def old_collate(samples, *, pad_text=tdc.TEXT_PADDING_ID):
    """VERBATIM copy of collate_codes before 2026-10-12 (arity 2 / 3)."""
    arity = len(samples[0])
    assert arity in (2, 3)
    num_codebooks = samples[0][0].shape[0]
    max_frames = max(codes.shape[1] for codes, *_ in samples)
    batch = len(samples)
    out_codes = torch.zeros(batch, num_codebooks, max_frames, dtype=torch.long)
    out_codes[:, tdc.TEXT_ROW] = pad_text
    out_mask = torch.zeros(batch, max_frames, dtype=torch.bool)
    out_extra = torch.zeros(batch, max_frames, samples[0][2].shape[1]) if arity == 3 else None
    for i, sample in enumerate(samples):
        codes, mask = sample[0], sample[1]
        frames = codes.shape[1]
        out_codes[i, :, :frames] = codes
        out_mask[i, :frames] = mask
        if out_extra is not None:
            out_extra[i, :frames] = sample[2]
    return (out_codes, out_mask) if out_extra is None else (out_codes, out_mask, out_extra)


_TOK = []


def tokenizer():
    if not _TOK:
        _TOK.append(
            td.PersonaPlexTokenizer(
                "nvidia/personaplex-7b-v1",
                device="cpu",
                cfg=td.PersonaPlexDataConfig(
                    duration_sec=60.0, silence_frames=6, emit_epad=True, onset_floor=True, word_overlap="drop"
                ),
            )
        )
    return _TOK[0]


def expected_user_row(tok, words, n):
    row = tok.interleave_text(tdc.normalize_alignments(words), n, **tok._text_kwargs())[0]
    out = torch.full((n,), tdc.TEXT_PADDING_ID, dtype=torch.long)
    out[D:] = row[: n - D]
    return out, row


def part_items():
    tok = tokenizer()
    table = tdc.read_arrow_table(P4)
    kw = dict(system_prompt_key="context", voice_codes_key="voice_codes", swap_channels=True, stored_user_augment=True)
    metas = []
    off = td.persona_item_builder(table, tok, **kw)
    on = td.persona_item_builder(table, tok, user_text_delay=D, on_item=metas.append, **kw)
    rows = np.sort(np.random.default_rng(0).choice(table.num_rows, 160, replace=False))
    other_ok = table.column("other_ok").to_pylist()
    bad_rows = [i for i, ok in enumerate(other_ok) if not ok][:5]
    same = shape_ok = prompt_ok = tgt_ok = True
    n_swap = wrong_side_differs = leak_visible = n_ignore_rows = n_ok_false = 0
    for r in [*rows.tolist(), *bad_rows]:
        a = off(int(r), np.random.default_rng([7, int(r)]))
        b = on(int(r), np.random.default_rng([7, int(r)]))
        swap = metas[-1]["swap"]
        same &= len(b) == len(a) + 1 and all(torch.equal(x, y) for x, y in zip(a, b[: len(a)]))
        tgt = b[-1]
        codes, mask = b[0], b[1]
        shape_ok &= tgt.dtype == torch.long and tgt.shape == (codes.shape[1],)
        n = int(mask.sum())
        P = codes.shape[1] - n
        prompt_ok &= bool((tgt[:P] == tdc.USER_TEXT_IGNORE).all())
        if not other_ok[r] and not swap:
            n_ok_false += 1
            tgt_ok &= bool((tgt == tdc.USER_TEXT_IGNORE).all())
            n_ignore_rows += 1
            continue
        stored, other = table.column("alignments")[r].as_py(), table.column("alignments_other")[r].as_py()
        user_words, asst_words = (stored, other) if swap else (other, stored)
        exp, undelayed = expected_user_row(tok, user_words, n)
        tgt_ok &= torch.equal(tgt[P:], exp)
        if swap:
            n_swap += 1
            wrong, _ = expected_user_row(tok, asst_words, n)
            wrong_side_differs += int(not torch.equal(tgt[P:], wrong))
        # the assistant text row is the ASSISTANT's words, and the user's row is visibly different
        leak_visible += int(not torch.equal(codes[0, P:].cpu(), undelayed.cpu()))
    check("1a items: codes / masks / weights unchanged, one more tensor", same)
    check("1b user-text tensor is long [T]", shape_ok)
    check("1c IGNORE on the prompt", prompt_ok)
    check(
        "1d targets = THIS read's user words delayed 6 frames (other_ok false -> IGNORE)",
        tgt_ok,
        f"(other_ok false {n_ok_false})",
    )
    check(
        "1e CAN FAIL: the wrong side's expectation differs on swapped reads",
        n_swap > 10 and wrong_side_differs == n_swap,
        f"({wrong_side_differs}/{n_swap})",
    )
    check("1f CAN FAIL: the user row differs from the assistant text row", leak_visible >= 150, f"({leak_visible})")


def part_collate():
    g = torch.Generator().manual_seed(0)
    items = []
    for t in (7, 11, 5):
        c = torch.randint(0, 100, (17, t), generator=g)
        m = torch.rand(t, generator=g) > 0.3
        items.append((c, m, torch.rand(t, 3, generator=g), torch.randint(0, 50, (t,), generator=g)))
    two = [it[:2] for it in items]
    three = [it[:3] for it in items]
    ok2 = all(torch.equal(x, y) for x, y in zip(tdc.collate_codes(two), old_collate(two)))
    ok3 = all(torch.equal(x, y) for x, y in zip(tdc.collate_codes(three), old_collate(three)))
    b4 = tdc.collate_codes(items)
    b3l = tdc.collate_codes([(it[0], it[1], it[3]) for it in items])
    pad_ok = (
        len(b4) == 4 and bool((b4[3][2, 5:] == tdc.USER_TEXT_IGNORE).all()) and torch.equal(b4[3][0, :7], items[0][3])
    )
    pad_ok &= len(b3l) == 3 and b3l[2].dtype == torch.long and bool((b3l[2][2, 5:] == tdc.USER_TEXT_IGNORE).all())
    pad_ok &= all(torch.equal(x, y) for x, y in zip(b4[:3], old_collate(three)))
    check("2a collate: arity 2 / 3 = the old collate", ok2 and ok3)
    check("2b collate: the user row pads with IGNORE (4-tuple, long 3-tuple)", pad_ok)


def part_mix():
    tok = tokenizer()

    def batches(**kw):
        meta = []
        it = td.build_knowledge_mix_loader(
            P4,
            V6,
            tok,
            batch_size=8,
            share=0.3,
            seed=0,
            system_prompt_key="context",
            voice_codes_key="voice_codes",
            swap_channels=True,
            stored_user_augment=True,
            item_meta=meta,
            **kw,
        )
        out = [next(it) for _ in range(4)]
        return out, meta

    a, ma = batches()
    b, mb = batches(user_text_delay=D)
    same = all(torch.equal(x, y) for ba, bb in zip(a, b) for x, y in zip(ba, bb[:2])) and all(len(bb) == 3 for bb in b)
    slots = [(m["source"], m["row"], m["swap"]) for m in ma] == [(m["source"], m["row"], m["swap"]) for m in mb]
    know_ignore = prim_has = True
    for k, bb in enumerate(b):
        for i in range(8):
            src = mb[k * 8 + i]["source"]
            t = bb[2][i]
            if src == "knowledge":
                know_ignore &= bool((t == tdc.USER_TEXT_IGNORE).all())
            else:
                prim_has &= bool((t != tdc.USER_TEXT_IGNORE).any())
    check("3a mix: codes / masks / slots unchanged by the knob", same and slots)
    check("3b mix: knowledge slots all IGNORE, primary slots carry targets", know_ignore and prim_has)


def part_loss():
    g = torch.Generator().manual_seed(1)
    B, T, V = 3, 40, 100
    logits = torch.randn(B, T, V, generator=g)
    tgt = torch.randint(4, V, (B, T), generator=g)
    tgt[torch.rand(B, T, generator=g) < 0.5] = tdc.TEXT_PADDING_ID
    tgt[torch.rand(B, T, generator=g) < 0.1] = tdc.TEXT_EPAD_ID
    ign = torch.rand(B, T, generator=g) < 0.2
    tgt_i = tgt.clone()
    tgt_i[ign] = tdc.USER_TEXT_IGNORE
    mask = torch.rand(B, T, generator=g) > 0.1

    class Out:  # duplex_ce_loss's LMOutput fields, audio zero-sized
        pass

    o = Out()
    o.text_logits = logits[:, None]
    o.text_mask = torch.ones(B, 1, T, dtype=torch.bool)
    o.logits = torch.zeros(B, 1, T, 8)
    o.mask = torch.zeros(B, 1, T, dtype=torch.bool)
    codes = torch.zeros(B, 2, T, dtype=torch.long)
    codes[:, 0] = tgt
    ok = True
    for wm in (False, True):
        _, ref = tdc.duplex_ce_loss(
            o,
            codes,
            mask & ~ign,
            dep_q=1,
            other_audio_weight=0.02,
            text_pad_weight=0.3,
            text_epad_id=tdc.TEXT_EPAD_ID,
            text_epad_weight=1.0,
            weighted_mean=wm,
            return_components=True,
        )
        got, _ = tdc.user_text_loss(
            logits,
            tgt_i,
            mask,
            text_pad_weight=0.3,
            text_epad_id=tdc.TEXT_EPAD_ID,
            text_epad_weight=1.0,
            weighted_mean=wm,
        )
        ok &= abs(float(got) - float(ref["text_loss"])) < 1e-5
        wrong, _ = tdc.user_text_loss(
            logits,
            tgt_i,
            mask,
            text_pad_weight=0.5,
            text_epad_id=tdc.TEXT_EPAD_ID,
            text_epad_weight=1.0,
            weighted_mean=wm,
        )
        ok &= abs(float(wrong) - float(ref["text_loss"])) > 1e-3
    check("4 loss parity with duplex_ce_loss's text half (CAN FAIL: PAD 0.5 differs)", ok)


def tiny_lm():
    from moshi_family.personaplex.models import loaders
    from moshi_family.personaplex.models.lm import LMModel

    torch.manual_seed(0)
    kw = dict(loaders._lm_kwargs)
    kw.update(
        dim=64,
        num_heads=4,
        num_layers=2,
        hidden_scale=2.0,
        text_card=100,
        context=64,
        depformer_dim=32,
        depformer_dim_feedforward=64,
        depformer_num_heads=2,
        depformer_num_layers=1,
        dep_q=16,
    )
    return LMModel(device="cpu", dtype=torch.float32, **kw)


def part_model():
    from moshi_family.build import LoraConfig, apply_lora_to, overlay_state_dict
    from moshi_family.train_loop import is_lora_param, save_trained_params

    base = tiny_lm()
    T = 12
    g = torch.Generator().manual_seed(2)
    codes = torch.randint(0, 50, (2, 17, T), generator=g)
    codes[:, 0] = torch.randint(4, 100, (2, T), generator=g)
    mask = torch.ones(2, T, dtype=torch.bool)
    mask[:, :3] = False
    tgt = torch.randint(0, 100, (2, T), generator=g)
    tgt[:, :3] = tdc.USER_TEXT_IGNORE

    # ONE LoRA'd model (lora_A is drawn from the global RNG), copied for the knob-off and knob-on runs
    lora_base = copy.deepcopy(base)
    apply_lora_to(lora_base, LoraConfig(rank=4, scaling=2.0), dtype=torch.float32)
    for n, p in lora_base.named_parameters():
        p.requires_grad = is_lora_param(n)
    with torch.no_grad():
        for n, p in lora_base.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02, generator=torch.Generator().manual_seed(3))

    def build(with_head, trained=True):
        m = copy.deepcopy(lora_base)
        # shaped from the text_linear the launcher sees (a LoRA wrapper here: the helper reads its frozen_W)
        head = tdc.make_user_text_head(m) if with_head else None
        zero_init = head is None or all(float(p.abs().sum()) == 0 for p in head.parameters())
        if head is not None and trained:  # a zero head passes no gradient to the backbone: stand in for a trained one
            with torch.no_grad():
                head.weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(4))
        cap = tdc.TextHeadInputCapture(m.text_linear) if with_head else None
        return m, head, cap, zero_init

    def loss_kw():
        return dict(text_pad_weight=0.3, text_epad_id=tdc.TEXT_EPAD_ID, text_epad_weight=1.0, weighted_mean=True)

    def main_loss(m):
        out = m.forward_train(codes)
        return out, td.personaplex_loss(
            out,
            codes,
            mask,
            dep_q=16,
            weighted_mean=True,
            text_epad_id=tdc.TEXT_EPAD_ID,
            return_components=True,
            text_epad_weight=1.0,
        )

    m0, _, _, _ = build(False)
    out0, (l0, _) = main_loss(m0)
    l0.backward()
    g0 = {n: p.grad.clone() for n, p in m0.named_parameters() if p.requires_grad}
    m1, head, cap, zero_init = build(True)
    out1, main1 = main_loss(m1)
    same_out = torch.equal(torch.nan_to_num(out0.text_logits), torch.nan_to_num(out1.text_logits)) and torch.equal(
        torch.nan_to_num(out0.logits), torch.nan_to_num(out1.logits)
    )
    check("5a the capture hook changes no output", same_out)
    h = cap._hidden
    reprod = torch.allclose(m1.text_linear(h), torch.nan_to_num(out1.text_logits[:, 0]), atol=1e-5)
    reprod &= head.weight.dtype == torch.float32 and zero_init
    check("5b the captured tensor is text_linear's input; the head is fp32 and zero-initialised", reprod)
    total, comps = tdc.add_user_text_loss(
        main1, head=head, capture=cap, targets=tgt, loss_mask=mask, weight=1.0, **loss_kw()
    )
    total.backward()
    hg = head.weight.grad is not None and float(head.weight.grad.abs().sum()) > 0
    g1 = {n: p.grad for n, p in m1.named_parameters() if p.requires_grad}
    differs = any(not torch.allclose(g0[n], g1[n]) for n in g0)
    check(
        "5c knob on: the head and the backbone (LoRA) get gradient from the user-text CE",
        hg and differs,
        f"{sorted(comps)[:3]}",
    )
    check(
        "5d knob off: the duplex loss is bit-identical (the head adds a term only)",
        torch.equal(l0.detach(), main1[0].detach()),
    )
    # the linear-probing phase (detach=True): the head learns, the backbone's gradient is the duplex loss's exactly
    m3, head3, cap3, _ = build(True)
    _, main3 = main_loss(m3)
    total3, comps3 = tdc.add_user_text_loss(
        main3, head=head3, capture=cap3, targets=tgt, loss_mask=mask, weight=1.0, detach=True, **loss_kw()
    )
    total3.backward()
    g3 = {n: p.grad for n, p in m3.named_parameters() if p.requires_grad}
    lp_ok = all(torch.equal(g0[n], g3[n]) for n in g0) and float(head3.weight.grad.abs().sum()) > 0
    lp_ok &= comps3["user_text_lp"] == 1.0 and comps["user_text_lp"] == 0.0
    check("5g LP phase: the head gets gradient, the backbone's gradient is bit-identical to the knob-off one", lp_ok)
    with torch.no_grad():
        head.weight.add_(1.0)
        out2 = m1.forward_train(codes)
    check(
        "5e the head is never fed back (perturbing it changes no output)",
        torch.equal(torch.nan_to_num(out2.text_logits), torch.nan_to_num(out1.text_logits)),
    )
    tmp = Path(tempfile.mkdtemp(prefix="check_uthead_"))
    s0 = save_trained_params(tmp / "a.safetensors", m0, predicate=is_lora_param)
    s1 = save_trained_params(tmp / "b.safetensors", m1, predicate=is_lora_param)
    fresh = copy.deepcopy(base)
    apply_lora_to(fresh, LoraConfig(rank=4, scaling=2.0), dtype=torch.float32)
    overlay_state_dict(fresh, str(tmp / "b.safetensors"), tag="check", device="cpu")
    keys_ok = set(s0) == set(s1) and not any("user_text" in k for k in m1.state_dict())
    check("5f the adapter checkpoint has the same keys with the head and reloads", keys_ok)


def tiny_lm4():
    from moshi_family.personaplex.models import loaders
    from moshi_family.personaplex.models.lm import LMModel

    torch.manual_seed(0)
    kw = dict(loaders._lm_kwargs)
    kw.update(
        dim=64,
        num_heads=4,
        num_layers=4,
        hidden_scale=2.0,
        text_card=100,
        context=64,
        depformer_dim=32,
        depformer_dim_feedforward=64,
        depformer_num_heads=2,
        depformer_num_layers=1,
        dep_q=16,
    )
    return LMModel(device="cpu", dtype=torch.float32, **kw)


MID_K = 1  # tiny model: the head on the output of layer 1 of 4 (the run: 15 of 32)


def part_mid():
    import re

    from moshi_family.build import LoraConfig, apply_lora_to
    from moshi_family.train_loop import is_lora_param

    m = tiny_lm4()
    apply_lora_to(m, LoraConfig(rank=4, scaling=2.0), dtype=torch.float32)
    for n, p in m.named_parameters():
        p.requires_grad = is_lora_param(n)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02, generator=torch.Generator().manual_seed(3))
    m.transformer.checkpointing = True  # the launcher's gradient checkpointing (non-reentrant)
    try:
        from moshi_family.train_perf import apply_train_perf

        perf = apply_train_perf(m, contiguous_step_inputs=True, fused_causal_attention=True, fused_lora_add=True)
    except Exception as e:  # noqa: BLE001  (reported; the rest still runs)
        perf = f"not applicable on CPU: {e!r}"[:80]
    print(f"  part 6: train_perf {perf}", flush=True)
    m.train()
    T = 12
    g = torch.Generator().manual_seed(2)
    codes = torch.randint(0, 50, (2, 17, T), generator=g)
    codes[:, 0] = torch.randint(4, 100, (2, T), generator=g)
    mask = torch.ones(2, T, dtype=torch.bool)
    mask[:, :3] = False
    tgt = torch.randint(0, 100, (2, T), generator=g)
    tgt[:, :3] = tdc.USER_TEXT_IGNORE
    kw = dict(text_pad_weight=0.3, text_epad_id=tdc.TEXT_EPAD_ID, text_epad_weight=1.0, weighted_mean=True)

    def run(detach):
        for p in m.parameters():
            p.grad = None
        ref = {}

        def ref_hook(_m, _i, o):  # returns None: a returned value would REPLACE the layer output
            ref.setdefault("x", (o[0] if isinstance(o, tuple) else o).detach().clone())

        h_ref = m.transformer.layers[MID_K].register_forward_hook(ref_hook)
        head = tdc.make_user_text_head(m, layer=MID_K)
        with torch.no_grad():
            head[1].weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(4))
        cap = tdc.make_user_text_capture(m, MID_K)
        m.forward_train(codes)
        h_ref.remove()
        same = cap._hidden is not None and torch.equal(cap._hidden.detach(), ref["x"])
        zero = torch.zeros((), requires_grad=False)
        total, _ = tdc.add_user_text_loss(
            (zero, {}), head=head, capture=cap, targets=tgt, loss_mask=mask, weight=1.0, detach=detach, **kw
        )
        # worst case: recompute the WHOLE checkpointed layer in backward (torch stops early by default, before the
        # layer returns and its hooks run), so a capture hook that ignored backward would be overwritten here
        from torch.utils.checkpoint import set_checkpoint_early_stop

        with set_checkpoint_early_stop(False):
            total.backward()
        no_overwrite = cap._hidden is None
        cap.close()
        layer_grad = {}
        other = 0.0
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            gsum = 0.0 if p.grad is None else float(p.grad.abs().sum())
            mm = re.match(r"transformer\.layers\.(\d+)\.", n)
            if mm:
                layer_grad[int(mm.group(1))] = layer_grad.get(int(mm.group(1)), 0.0) + gsum
            else:
                other += gsum
        head_g = sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None)
        return same, no_overwrite, layer_grad, other, head_g, head

    same, no_ow, lg, other, hg, head = run(detach=False)
    check(
        "6a mid capture = the output of transformer.layers[k] (checkpointing + fast path); the recompute does not "
        "overwrite it",
        same and no_ow,
        f"(same {same}, untouched after backward {no_ow})",
    )
    norm = head[0]
    norm_ok = (
        type(norm) is type(m.out_norm)
        and all(p.dtype == torch.float32 for p in head.parameters())
        and all(float(p.detach().abs().sum()) > 0 for n, p in norm.named_parameters() if not n.endswith("bias"))
        and isinstance(head[1], torch.nn.Linear)
    )
    check("6b mid head = fresh norm of out_norm's type (fp32) + Linear", norm_ok, f"({type(norm).__name__})")
    below = all(lg.get(i, 0.0) > 0 for i in range(MID_K + 1))
    above = all(lg.get(i, 0.0) == 0 for i in range(MID_K + 1, len(m.transformer.layers)))
    check(
        "6c the user-text CE reaches LoRA of layers <= k only (none above k, none outside the stack); the head learns",
        below and above and other == 0 and hg > 0,
        f"(per-layer |g| {[round(lg.get(i, 0.0), 6) for i in range(len(m.transformer.layers))]}, other {other:.3g})",
    )
    _, _, lg3, other3, hg3, _ = run(detach=True)
    check(
        "6d LP phase on the mid layer: no backbone gradient at all, the head learns",
        all(v == 0 for v in lg3.values()) and other3 == 0 and hg3 > 0,
        f"({lg3})",
    )
    d_head = tdc.make_user_text_head(m)
    d_cap = tdc.make_user_text_capture(m, "last")
    check(
        "6e default ('last'): a plain Linear head on text_linear's input (TextHeadInputCapture)",
        isinstance(d_head, torch.nn.Linear)
        and type(d_cap) is tdc.TextHeadInputCapture
        and tdc.user_text_layer_index(m, None) is None,
    )
    d_cap.close()


def part_lr():
    import math

    from moshi_family import train_loop as tl

    def lrs_of(schedule):
        torch.manual_seed(0)
        mods = [torch.nn.Linear(3, 1) for _ in range(3)]
        base = [1e-3, 2e-3, 5e-4]
        opt = torch.optim.AdamW([{"params": list(mm.parameters()), "lr": b} for mm, b in zip(mods, base)])
        seen = []

        def loss_step(_batch):
            seen.append([gr["lr"] for gr in opt.param_groups])
            x = torch.ones(1, 3)
            return sum(mm(x).sum() for mm in mods), {}

        def batches():
            while True:
                yield None

        out = tempfile.mkdtemp(prefix="check_uthead_lr_")
        tl.run_training(
            optimizer=opt,
            base_lrs=base,
            group_names=["a", "b", "c"],
            trainable_params=[p for mm in mods for p in mm.parameters()],
            batch_iter=batches(),
            loss_step=loss_step,
            save_fn=lambda *_a, **_k: None,
            cfg=tl.TrainConfig(
                max_steps=10,
                grad_accum=1,
                warmup_steps=3,
                save_every=1000,
                log_every=1000,
                track_weight_delta=False,
                lr_schedule=schedule,
            ),
            out_dir=out,
            log=lambda *_a: None,
        )
        return base, seen

    base, cos = lrs_of("cosine")
    exp_cos = [[b * tl.cosine_lr_scale(s, 3, 10) for b in base] for s in range(1, 11)]
    check("7a lr_schedule cosine = cosine_lr_scale exactly (every group, every step)", cos == exp_cos, f"({cos[-1]})")
    base, const = lrs_of("constant")
    exp_c = [[b * (s / 3 if s < 3 else 1.0) for b in base] for s in range(1, 11)]
    check(
        "7b lr_schedule constant: the same warmup, then every group at its base LR to the end",
        len(const) == 10
        and all(math.isclose(x, y, rel_tol=0, abs_tol=0) for r, e in zip(const, exp_c) for x, y in zip(r, e)),
        f"({const[2]} .. {const[-1]})",
    )


def ref_ctc_nll(lp, labels, blank):
    """Independent reference: the CTC forward algorithm (Graves et al. 2006) in log space, numpy, one sequence.
    ``lp [T, V]`` log-probs. Returns -log p(labels | lp); inf when infeasible."""
    ext = [blank]
    for lab in labels:
        ext += [lab, blank]
    S, T = len(ext), lp.shape[0]
    alpha = np.full((T, S), -np.inf)
    alpha[0, 0] = lp[0, blank]
    if S > 1:
        alpha[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            a = [alpha[t - 1, s]]
            if s >= 1:
                a.append(alpha[t - 1, s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                a.append(alpha[t - 1, s - 2])
            alpha[t, s] = np.logaddexp.reduce(a) + lp[t, ext[s]]
    end = alpha[T - 1, S - 1] if S == 1 else np.logaddexp(alpha[T - 1, S - 1], alpha[T - 1, S - 2])
    return -float(end)


def ref_spans(tgt, mask, margin, epad):
    """Independent reference of the span rule, frame by frame: a valid frame is inside iff some valid word frame lies
    within +-margin; spans = maximal runs; labels = the word frames' targets in order."""
    out = []
    B, T = tgt.shape
    for b in range(B):
        valid = [bool(mask[b, f]) and int(tgt[b, f]) != tdc.USER_TEXT_IGNORE for f in range(T)]
        word = [valid[f] and int(tgt[b, f]) not in (tdc.TEXT_PADDING_ID, epad) for f in range(T)]
        inside = [
            valid[f] and any(word[g] for g in range(max(0, f - margin), min(T, f + margin + 1))) for f in range(T)
        ]
        f = 0
        while f < T:
            if inside[f]:
                e = f
                while e < T and inside[e]:
                    e += 1
                out.append((b, f, e, [int(tgt[b, g]) for g in range(f, e) if word[g]]))
                f = e
            else:
                f += 1
    return out


def ctc_case(seed=5, B=3, T=120, V=40):
    """Random user-text targets in the loader's format: IGNORE prompt, PAD, EPAD before each word after a pause, word
    tokens (ids >= 4), a short loss-mask tail cut, one row without targets."""
    g = np.random.default_rng(seed)
    tgt = torch.full((B, T), tdc.TEXT_PADDING_ID, dtype=torch.long)
    tgt[:, :4] = tdc.USER_TEXT_IGNORE
    for b in range(B - 1):
        f = 6
        prev = -1
        while f < T - 12:  # no word in the last frames: the late / early shifts of 8d stay inside every span
            if g.random() < 0.35:
                n = int(g.integers(1, 4))
                tgt[b, f - 1] = tdc.TEXT_EPAD_ID if tgt[b, f - 2] == tdc.TEXT_PADDING_ID else tgt[b, f - 1]
                for k in range(n):
                    y = int(g.integers(4, V))
                    while y == prev:  # no two equal labels in a row (CTC would need a blank between them)
                        y = int(g.integers(4, V))
                    tgt[b, f + k], prev = y, y
                f += n + int(g.integers(0, 14))
            else:
                f += 1
    tgt[B - 1] = tdc.USER_TEXT_IGNORE  # a row without user words
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, :4] = False
    mask[0, T - 3 :] = False  # right padding of a shorter row
    logits = torch.from_numpy(g.normal(0, 2.0, (B, T, V))).float()
    return logits, tgt, mask


CTC_KW = dict(text_pad_weight=0.3, text_epad_id=tdc.TEXT_EPAD_ID, text_epad_weight=1.0)


def part_ctc():
    import torch.nn.functional as F

    logits, tgt, mask = ctc_case()
    m = D
    # 8a spans = the independent frame-by-frame rule (margin, merge, clipped to valid frames, EPAD never a label)
    got = tdc.user_text_ctc_spans(tgt, mask, margin=m, text_epad_id=tdc.TEXT_EPAD_ID)
    exp = ref_spans(tgt, mask, m, tdc.TEXT_EPAD_ID)
    hand_t = torch.full((1, 30), tdc.TEXT_PADDING_ID, dtype=torch.long)
    hand_t[0, :3] = tdc.USER_TEXT_IGNORE
    hand_t[0, 4], hand_t[0, 5], hand_t[0, 6], hand_t[0, 11], hand_t[0, 25] = 0, 7, 8, 9, 10
    hand_m = torch.ones(1, 30, dtype=torch.bool)
    hand_m[0, 27:] = False
    hand = tdc.user_text_ctc_spans(hand_t, hand_m, margin=2, text_epad_id=tdc.TEXT_EPAD_ID)
    hand_exp = [(0, 3, 14, [7, 8, 9]), (0, 23, 27, [10])]  # 5-2=3 (IGNORE before), 11+2 -> 13; 25 +-2 cut at 27
    check(
        "8a CTC spans = the frame-by-frame reference (margin = delay, merged, valid frames only, no EPAD label)",
        got == exp and hand == hand_exp and len(got) >= 4,
        f"{len(got)} spans; hand {hand}",
    )
    # 8b the loss = (sum of span CTC NLLs by the numpy forward algorithm + outside weighted CE) / the CE's denominator
    ok = True
    for wm in (True, False):
        loss, comps = tdc.user_text_ctc_loss(logits, tgt, mask, margin=m, weighted_mean=wm, **CTC_KW)
        lp = F.log_softmax(logits.double(), -1).numpy()
        ctc_num = sum(ref_ctc_nll(lp[b, s:e], lab, tdc.TEXT_PADDING_ID) for b, s, e, lab in exp)
        inside = np.zeros(tgt.shape, bool)
        for b, s, e, _ in exp:
            inside[b, s:e] = True
        valid = ((tgt != tdc.USER_TEXT_IGNORE) & mask).numpy()
        t_np = tgt.numpy()
        w = np.where(t_np == tdc.TEXT_PADDING_ID, 0.3, 1.0)
        w = np.where(t_np == tdc.TEXT_EPAD_ID, 1.0, w) * valid
        ce = -np.take_along_axis(lp, np.clip(t_np, 0, None)[..., None], -1)[..., 0]
        pad_num = float((ce * w * ~inside).sum())
        den = float(w.sum()) if wm else float(valid.sum())
        ref = (ctc_num + pad_num) / den
        ok &= abs(float(loss) - ref) < 1e-4 * max(1.0, abs(ref))
        ok &= abs(comps["user_text_ctc"] + comps["user_text_pad_ce"] - float(loss)) < 1e-4
        ok &= abs(comps["user_text_ctc"] - ctc_num / den) < 1e-4 and comps["user_text_ctc_inf"] == 0
    check(
        "8b CTC loss = numpy forward-algorithm CTC per span + outside PAD CE, over the CE's denominator",
        ok,
        f"{ref:.4f}",
    )
    # 8c no word in the batch -> exactly the framewise CE (the outside part IS user_text_loss's)
    t0 = tgt.clone()
    t0[(t0 != tdc.USER_TEXT_IGNORE)] = tdc.TEXT_PADDING_ID
    a, _ = tdc.user_text_ctc_loss(logits, t0, mask, margin=m, weighted_mean=True, **CTC_KW)
    b_, _ = tdc.user_text_loss(logits, t0, mask, weighted_mean=True, **CTC_KW)
    check("8c no user words: CTC variant = the framewise CE", torch.allclose(a, b_, atol=1e-5), f"{float(a):.4f}")

    # 8d alignment-free: a head emitting every label 0..2 frames late or 2 early, or holding a word for 2 frames
    # (stutter), scores ~0 and the same; the framewise CE of the same logits does not (CAN FAIL)
    def peaky(shift, hold=1, wrong=False):
        lg = torch.full(logits.shape, -30.0)
        lg[..., tdc.TEXT_PADDING_ID] = 30.0
        for b, s, e, lab in exp:
            frames = [f for f in range(s, e) if int(tgt[b, f]) not in (tdc.TEXT_PADDING_ID, tdc.TEXT_EPAD_ID)]
            for f, y in zip(frames, lab):
                for h in range(hold):
                    g_ = min(max(f + shift + h, s), e - 1)
                    lg[b, g_] = -30.0
                    lg[b, g_, (y + 1 - 4) % (lg.shape[-1] - 4) + 4 if wrong else y] = 30.0
        return lg

    vals = {}
    for name, lg in {
        "aligned": peaky(0),
        "late1": peaky(1),
        "late2": peaky(2),
        "early2": peaky(-2),
        "stutter": peaky(0, hold=2),
    }.items():
        vals[name] = float(tdc.user_text_ctc_loss(lg, tgt, mask, margin=m, weighted_mean=True, **CTC_KW)[0])
    ce_late = float(tdc.user_text_loss(peaky(1), tgt, mask, weighted_mean=True, **CTC_KW)[0])
    bad = float(tdc.user_text_ctc_loss(peaky(0, wrong=True), tgt, mask, margin=m, weighted_mean=True, **CTC_KW)[0])
    ok = max(vals.values()) < 1e-3 and ce_late > 5.0 and bad > 5.0
    check(
        "8d CTC ~0 for a perfect prediction at any lag in the span and for stutter; wrong words / framewise CE of "
        "the late one are large (CAN FAIL)",
        ok,
        f"{ {k: round(v, 5) for k, v in vals.items()} } ce_late {ce_late:.2f} wrong {bad:.2f}",
    )

    # 8e through the launcher's composition on a tiny LMModel + LoRA (bf16 hidden): finite, the head and the LoRA get
    # gradient, kind "ce" (the default) = the pre-change call, LP -> backbone grad = knob-off grad
    from moshi_family.build import LoraConfig, apply_lora_to
    from moshi_family.train_loop import is_lora_param

    base = tiny_lm()
    apply_lora_to(base, LoraConfig(rank=4, scaling=2.0), dtype=torch.float32)
    for n, p in base.named_parameters():
        p.requires_grad = is_lora_param(n)
    with torch.no_grad():
        for n, p in base.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02, generator=torch.Generator().manual_seed(3))
    T = 24
    g = torch.Generator().manual_seed(2)
    codes = torch.randint(0, 50, (2, 17, T), generator=g)
    codes[:, 0] = torch.randint(4, 100, (2, T), generator=g)
    lmask = torch.ones(2, T, dtype=torch.bool)
    lmask[:, :3] = False
    utgt = torch.full((2, T), tdc.TEXT_PADDING_ID, dtype=torch.long)
    utgt[:, :3] = tdc.USER_TEXT_IGNORE
    utgt[0, 9:12] = torch.tensor([11, 12, 13])
    utgt[1, 15:17] = torch.tensor([21, 22])

    def run(mode, detach=False, bf16=False, **kw):
        mm = copy.deepcopy(base)
        head = tdc.make_user_text_head(mm)
        with torch.no_grad():
            head.weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(4))
        cap = tdc.TextHeadInputCapture(mm.text_linear)
        out = mm.forward_train(codes)
        main = td.personaplex_loss(
            out,
            codes,
            lmask,
            dep_q=16,
            weighted_mean=True,
            text_epad_id=tdc.TEXT_EPAD_ID,
            return_components=True,
            text_epad_weight=1.0,
        )
        if bf16:
            orig_pop = cap.pop
            cap.pop = lambda b, t: orig_pop(b, t).to(torch.bfloat16)
        if mode is None:
            total, comps = main
        else:
            total, comps = tdc.add_user_text_loss(
                main,
                head=head,
                capture=cap,
                targets=utgt,
                loss_mask=lmask,
                weight=1.0,
                detach=detach,
                weighted_mean=True,
                **CTC_KW,
                **kw,
            )
        total.backward()
        gr = {n: p.grad.clone() for n, p in mm.named_parameters() if p.requires_grad}
        return total.detach(), comps, gr, head

    l_off, _, g_off, _ = run(None)
    l_ce, c_ce, _, _ = run("ce")
    l_ce2, _, _, _ = run("ce", kind="ce")
    l_ctc, c_ctc, g_ctc, h_ctc = run("ctc", kind="ctc", ctc_margin=D, bf16=True)
    _, c_lp, g_lp, h_lp = run("ctc", detach=True, kind="ctc", ctc_margin=D)
    ok = bool(torch.isfinite(l_ctc)) and "user_text_ctc" in c_ctc and "user_text_ctc" not in c_ce
    ok &= torch.equal(l_ce, l_ce2)
    ok &= h_ctc.weight.grad is not None and bool(torch.isfinite(h_ctc.weight.grad).all())
    ok &= float(h_ctc.weight.grad.abs().sum()) > 0
    ok &= any(not torch.allclose(g_off[n], g_ctc[n]) for n in g_off)
    ok &= all(torch.equal(g_off[n], g_lp[n]) for n in g_off) and float(h_lp.weight.grad.abs().sum()) > 0
    ok &= abs(float(l_ctc - l_off) - c_ctc["user_text_loss"]) < 1e-3 * max(1.0, c_ctc["user_text_loss"])
    check(
        "8e add_user_text_loss kind ctc (bf16 hidden): finite, head + LoRA gradient, LP detaches, kind ce = default",
        ok,
        f"ctc parts {c_ctc.get('user_text_ctc', 0):.3f} + {c_ctc.get('user_text_pad_ce', 0):.3f}",
    )


def run_all():
    FAILS.clear()
    part_collate()
    part_loss()
    part_model()
    part_items()
    part_mix()
    part_lr()
    part_mid()
    part_ctc()
    return list(FAILS)


def selftest():
    """Each mutant must make at least one check fail."""
    results = {}
    orig_targets, orig_loss, orig_hook, orig_collate_long = (
        td.user_text_targets,
        tdc.user_text_loss,
        tdc.TextHeadInputCapture._hook,
        tdc._collate_with_long_extra,
    )

    def m_delay0(tok, words, n, *, delay_frames, text_kwargs=None):
        return orig_targets(tok, words, n, delay_frames=0, text_kwargs=text_kwargs)

    def m_feedback(self, module, inputs, output):
        self._hidden = inputs[0]
        return output + 0.01 * torch.randn_like(output)

    def m_detach(self, module, inputs, output):
        self._hidden = inputs[0].detach()

    orig_add = tdc.add_user_text_loss

    def m_lp_ignored(*a, **k):
        k["detach"] = False
        return orig_add(*a, **k)

    def m_pad1(*a, **k):
        k["text_pad_weight"] = 1.0
        return orig_loss(*a, **k)

    def m_pad0(samples, out_codes, out_mask):
        out = orig_collate_long(samples, out_codes, out_mask)
        u = out[-1]
        u[u == tdc.USER_TEXT_IGNORE] = 0
        return out

    orig_capture, orig_layer_hook = tdc.make_user_text_capture, tdc.LayerOutputCapture._hook
    from moshi_family import train_loop as _tl

    orig_lr_scale = _tl.lr_scale

    def m_wrong_layer(lm, layer=None):
        return orig_capture(lm, None if layer in (None, "last") else int(layer) + 1)

    def m_const_decays(step, warmup, max_steps, schedule="cosine"):
        return _tl.cosine_lr_scale(step, warmup, max_steps)

    orig_spans, orig_ctc, orig_blank = tdc.user_text_ctc_spans, tdc.user_text_ctc_loss, tdc.USER_TEXT_CTC_BLANK

    def m_spans_margin1(targets, loss_mask, *, margin, text_epad_id=None):
        return orig_spans(targets, loss_mask, margin=1, text_epad_id=text_epad_id)

    def m_spans_nomask(targets, loss_mask, *, margin, text_epad_id=None):
        return orig_spans(targets, torch.ones_like(loss_mask), margin=margin, text_epad_id=text_epad_id)

    def m_spans_epad_label(targets, loss_mask, *, margin, text_epad_id=None):
        return orig_spans(targets, loss_mask, margin=margin, text_epad_id=None)

    def m_ctc_no_pad(*a, **k):
        loss, comps = orig_ctc(*a, **k)
        return loss - comps["user_text_pad_ce"], comps

    def m_ctc_per_token(*a, **k):
        loss, comps = orig_ctc(*a, **k)
        return loss * 0 + comps["ctc_user_text_per_token"] + comps["user_text_pad_ce"], comps

    def m_ctc_bf16(logits, *a, **k):
        return orig_ctc(logits.to(torch.bfloat16), *a, **k)

    def m_kind_ignored(*a, **k):
        k.pop("kind", None), k.pop("ctc_margin", None)
        return orig_add(*a, **k)

    mutants = {
        "ctc blank = EPAD": (lambda: setattr(tdc, "USER_TEXT_CTC_BLANK", tdc.TEXT_EPAD_ID), part_ctc),
        "ctc spans margin 1": (lambda: setattr(tdc, "user_text_ctc_spans", m_spans_margin1), part_ctc),
        "ctc spans ignore loss_mask": (lambda: setattr(tdc, "user_text_ctc_spans", m_spans_nomask), part_ctc),
        "ctc EPAD as a label": (lambda: setattr(tdc, "user_text_ctc_spans", m_spans_epad_label), part_ctc),
        "ctc outside PAD CE dropped": (lambda: setattr(tdc, "user_text_ctc_loss", m_ctc_no_pad), part_ctc),
        "ctc normalised per label token": (lambda: setattr(tdc, "user_text_ctc_loss", m_ctc_per_token), part_ctc),
        "ctc on bf16 logits": (lambda: setattr(tdc, "user_text_ctc_loss", m_ctc_bf16), part_ctc),
        "loss kind ignored": (lambda: setattr(tdc, "add_user_text_loss", m_kind_ignored), part_ctc),
        "mid: wrong layer": (lambda: setattr(tdc, "make_user_text_capture", m_wrong_layer), part_mid),
        "mid: LP phase not detached": (lambda: setattr(tdc, "add_user_text_loss", m_lp_ignored), part_mid),
        "constant lr decays": (lambda: setattr(_tl, "lr_scale", m_const_decays), part_lr),
        "delay 0": (lambda: setattr(td, "user_text_targets", m_delay0), part_items),
        "hook feeds back": (lambda: setattr(tdc.TextHeadInputCapture, "_hook", m_feedback), part_model),
        "detached capture": (lambda: setattr(tdc.TextHeadInputCapture, "_hook", m_detach), part_model),
        "head loss PAD 1.0": (lambda: setattr(tdc, "user_text_loss", m_pad1), part_loss),
        "LP phase not detached": (lambda: setattr(tdc, "add_user_text_loss", m_lp_ignored), part_model),
        "user row padded with 0": (lambda: setattr(tdc, "_collate_with_long_extra", m_pad0), part_collate),
    }
    for name, (apply, part) in mutants.items():
        FAILS.clear()
        apply()
        try:
            part()
        except Exception as e:  # a crash counts as caught
            FAILS.append(f"raised {e!r}"[:80])
        finally:
            td.user_text_targets, tdc.user_text_loss, tdc.add_user_text_loss = orig_targets, orig_loss, orig_add
            tdc.TextHeadInputCapture._hook, tdc._collate_with_long_extra = orig_hook, orig_collate_long
            tdc.make_user_text_capture, tdc.LayerOutputCapture._hook = orig_capture, orig_layer_hook
            _tl.lr_scale = orig_lr_scale
            tdc.user_text_ctc_spans, tdc.user_text_ctc_loss = orig_spans, orig_ctc
            tdc.USER_TEXT_CTC_BLANK = orig_blank
        results[name] = bool(FAILS)
        print(f"MUTANT {name}: {'caught' if FAILS else 'NOT CAUGHT'} {FAILS[:2]}", flush=True)
    return results


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        res = selftest()
        print(f"selftest: {sum(res.values())}/{len(res)} mutants caught")
        sys.exit(0 if all(res.values()) else 1)
    fails = run_all()
    print("OK" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)
