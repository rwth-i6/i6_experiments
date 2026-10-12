"""Guard: the PersonaPlex user-text head (2026-10-12; ``user_text_loss_weight``; rl_alignment.md "User-text probe and head").

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
``--selftest``: six mutants (delay 0, hook feeding back, detached capture, PAD weight 1.0 in the head loss, LP phase
not detached, collate padding the user row with 0) must each fail.

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


def run_all():
    FAILS.clear()
    part_collate()
    part_loss()
    part_model()
    part_items()
    part_mix()
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

    mutants = {
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
