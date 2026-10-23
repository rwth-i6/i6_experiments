"""
Text generation evaluation: sampling from a trained RETURNN LM (AR, MDLM, USDM, Flex-DiffLM) and its metrics
(gen PPL, continuation PPL, entropy, MAUVE, mean length), written for the LM1B comparison.
"""

import gzip
import json
import math
from typing import Any, Dict, List, Optional

from sisyphus import Job, Task, tk


class Lm1bEvalTextsJob(Job):
    # LM1B test sentences in model tokens: MAUVE reference, a disjoint real set, continuation prompts (first half)
    def __init__(self, hf_dir: tk.Path, tokenizer_dir: tk.Path, *, n: int = 1500, min_len: int = 4, max_len: int = 128):
        super().__init__()
        self.hf_dir, self.tokenizer_dir, self.n, self.min_len, self.max_len = hf_dir, tokenizer_dir, n, min_len, max_len
        self.out_reference = self.output_path("reference.jsonl.gz")
        self.out_real = self.output_path("real.jsonl.gz")
        self.out_prompts = self.output_path("prompts.jsonl.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import datasets
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.tokenizer_dir.get_path())
        test = datasets.load_from_disk(self.hf_dir.get_path())["test"]
        rows = []
        for text in test["text"]:
            ids = tok.encode(text, add_special_tokens=False)
            if self.min_len <= len(ids) <= self.max_len:
                rows.append(ids)
            if len(rows) == 3 * self.n:
                break
        assert len(rows) == 3 * self.n
        ref, real, prompts = rows[: self.n], rows[self.n : 2 * self.n], rows[2 * self.n :]
        _write(self.out_reference, [dict(ids=x, text=tok.decode(x)) for x in ref])
        _write(self.out_real, [dict(ids=x, text=tok.decode(x)) for x in real])
        # a prompt row is also the real-text continuation row: prefix + its true second half as ids / text
        _write(self.out_prompts, [dict(prefix=x[: len(x) // 2], prefix_text=tok.decode(x[: len(x) // 2]),
                                       ids=x[len(x) // 2 :], text=tok.decode(x[len(x) // 2 :])) for x in prompts])


class SampleTextJob(Job):
    # samples of one model: sampler in ("ar", "mdlm", "usdm", "flex"); length = number of generated tokens (None:
    # variable, AR stops at EOS, Flex inserts); prompts given -> continuations of their prefixes, else unconditional
    def __init__(self, *, config: tk.Path, checkpoint: tk.Path, sampler: str, tokenizer_dir: tk.Path, n: int = 1500,
                 length: Optional[int] = 32, steps: int = 96, temperature: float = 1.0, top_p: float = 0.9,
                 psi: float = 0.05, prompts: Optional[tk.Path] = None, batch: int = 100, seed: int = 20261023,
                 bos: int = 101, eos: int = 102, mask: int = 103, max_len: int = 128, ar_cap: int = 256):
        super().__init__()
        self.config, self.checkpoint, self.sampler, self.tokenizer_dir = config, checkpoint, sampler, tokenizer_dir
        self.opts = dict(n=n, length=length, steps=steps, temperature=temperature, top_p=top_p, psi=psi, batch=batch,
                         seed=seed, bos=bos, eos=eos, mask=mask, max_len=max_len, ar_cap=ar_cap)
        self.prompts = prompts
        self.out_samples = self.output_path("samples.jsonl.gz")
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import torch
        from transformers import AutoTokenizer

        o = self.opts
        tok = AutoTokenizer.from_pretrained(self.tokenizer_dir.get_path())
        model = _load(self.config.get_path(), self.checkpoint.get_path())
        prompts = _read(self.prompts.get_path()) if self.prompts else [None] * o["n"]
        assert len(prompts) == o["n"]
        fn = {"ar": _sample_ar, "mdlm": _sample_mdlm, "usdm": _sample_usdm, "flex": _sample_flex}[self.sampler]
        out = []
        for i in range(0, o["n"], o["batch"]):
            torch.manual_seed(o["seed"] + i)
            pre = [p["prefix"] if p else [] for p in prompts[i : i + o["batch"]]]
            with torch.inference_mode():
                gen = fn(model, pre, o)
            for p, ids in zip(pre, gen):
                out.append(dict(prefix=p, prefix_text=tok.decode(p), ids=ids, text=tok.decode(ids)))
            print(i, out[-1]["text"], flush=True)
        _write(self.out_samples, out)


class TextGenMetricsJob(Job):
    # gen PPL (unconditional: EOS-joined 1024-token GPT-2 blocks) or continuation PPL (prefix + space + continuation),
    # mean entropy of the model-token unigram distribution, mean length, MAUVE against the reference (unconditional)
    def __init__(self, samples: tk.Path, *, hub_dir: tk.Path, evaluator: str = "gpt2-large",
                 reference: Optional[tk.Path] = None, continuation: bool = False, detok: bool = False):
        super().__init__()
        self.samples, self.hub_dir, self.evaluator = samples, hub_dir, evaluator
        self.reference, self.continuation, self.detok = reference, continuation, detok
        self.out_metrics = self.output_path("metrics.json")
        self.out_value = self.output_var("metrics")

    def tasks(self):
        yield Task("run", rqmt={"gpu": 1, "cpu": 4, "mem": 32, "time": 2})

    def run(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        rows = _read(self.samples.get_path())
        if self.detok:  # the evaluator reads normally spaced text; every row gets the same treatment
            rows = [dict(r, text=_detok(r["text"]), **({"prefix_text": _detok(r["prefix_text"])} if "prefix_text" in r else {}))
                    for r in rows]
        kw = dict(cache_dir=self.hub_dir.get_path(), local_files_only=True)
        tok = AutoTokenizer.from_pretrained(self.evaluator, **kw)
        lm = AutoModelForCausalLM.from_pretrained(self.evaluator, dtype=torch.float32, **kw).cuda().eval()
        m = dict(n=len(rows), empty=sum(not r["ids"] for r in rows), mean_len=sum(len(r["ids"]) for r in rows) / len(rows),
                 entropy=sum(_entropy(r["ids"]) for r in rows) / len(rows))
        if self.continuation:
            m["cont_ppl"] = _cont_ppl(lm, tok, [(r["prefix_text"], r["text"]) for r in rows])
        else:
            m["gen_ppl"] = _gen_ppl(lm, tok, [r["text"] for r in rows])
            if self.reference:
                ref = [_detok(r["text"]) if self.detok else r["text"] for r in _read(self.reference.get_path())]
                m["mauve"] = _mauve(lm, tok, [r["text"] for r in rows if r["ids"]], ref)
        json.dump(m, open(self.out_metrics.get_path(), "w"), indent=2)
        self.out_value.set(m)


class TextGenTableJob(Job):
    # the table: one row per (model, steps, length) with gen PPL, mean length, entropy, MAUVE, cont PPL
    def __init__(self, rows: List[Dict[str, Any]]):
        super().__init__()
        self.rows = rows
        self.out_table = self.output_path("table.txt")
        self.out_data = self.output_path("data.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        data, lines = [], ["| model | steps | length | gen PPL | mean length | entropy | MAUVE | cont PPL |",
                           "|---|---|---|---|---|---|---|---|"]
        for r in self.rows:
            u = r["uncond"].get() if r.get("uncond") is not None else {}
            c = r["cont"].get() if r.get("cont") is not None else {}
            d = dict(model=r["model"], steps=r["steps"], length=r["length"], gen_ppl=u.get("gen_ppl"),
                     mean_len=u.get("mean_len"), entropy=u.get("entropy"), mauve=u.get("mauve"), cont_ppl=c.get("cont_ppl"))
            data.append(d)
            f = lambda v, p=1: "pending" if v is None else f"{v:.{p}f}"
            lines.append(f"| {d['model']} | {d['steps']} | {d['length']} | {f(d['gen_ppl'])} | {f(d['mean_len'])} | "
                         f"{f(d['entropy'], 2)} | {f(d['mauve'], 2)} | {f(d['cont_ppl'])} |")
        open(self.out_table.get_path(), "w").write("\n".join(lines) + "\n")
        json.dump(data, open(self.out_data.get_path(), "w"), indent=2)


class TokenUnigramJob(Job):
    # token unigram law of a text (spm pieces, first max_lines lines, add-one): X0 of the Uniform-X0 Edit Flow
    def __init__(self, text: tk.Path, spm_model: tk.Path, *, max_lines: int = 2_000_000):
        super().__init__()
        self.text, self.spm_model, self.max_lines = text, spm_model, max_lines
        self.out_probs = self.output_path("unigram.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 8, "time": 2})

    def run(self):
        import numpy
        import sentencepiece

        sp = sentencepiece.SentencePieceProcessor(model_file=self.spm_model.get_path())
        counts = numpy.zeros(sp.get_piece_size(), dtype=numpy.int64)
        with (gzip.open if self.text.get_path().endswith(".gz") else open)(self.text.get_path(), "rt") as f:
            for i, line in enumerate(f):
                if i == self.max_lines:
                    break
                numpy.add.at(counts, sp.encode(line.strip()), 1)
        numpy.savetxt(self.out_probs.get_path(), (counts + 1) / (counts + 1).sum())


def _detok(t):
    # BERT-uncased decode -> normal spacing: clitics, decimals / thousands, hyphens, abbreviations, punctuation, quotes
    import re

    t = re.sub(r" ' (s|t|re|ve|m|ll|d)\b", r"'\1", t)
    t = re.sub(r"(\d)\s*([.,])\s+(\d)", r"\1\2\3", t)
    t = re.sub(r"(\w) - (\w)", r"\1-\2", t)
    t = re.sub(r"\b([a-z])\. ([a-z])\.", r"\1.\2.", t)
    t = re.sub(r" ([,.;:!?%)])", r"\1", t)
    t = re.sub(r"([($]) ", r"\1", t)
    return re.sub(r'" (.*?) "', r'"\1"', t)


def _write(path, rows):
    with gzip.open(path.get_path() if hasattr(path, "get_path") else path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


# ---- model loading and calls ----


def _load(config_file, checkpoint):
    # the training's config (model dict) and checkpoint; the vocab size is read from the embedding
    import torch
    import returnn.frontend as rf
    from returnn.config import Config, set_global_config
    from returnn.tensor import Dim
    from returnn.torch.frontend.bridge import rf_module_to_pt_module
    from returnn.util.basic import BehaviorVersion

    cfg = Config()
    cfg.load_file(config_file)
    set_global_config(cfg)
    BehaviorVersion._behavior_version = None  # the training's version, as RETURNN sets it at startup
    BehaviorVersion.set(cfg.int("behavior_version", None))
    rf.select_backend_torch()
    rf.set_default_device("cuda")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    (key,) = [k for k in state if k.endswith("input_embedding.weight")]
    model = rf.build_from_dict(cfg.typed_value("_model_def_dict"), vocab_dim=Dim(state[key].shape[0], name="vocab"))
    pt = rf_module_to_pt_module(model)
    pt.load_state_dict(state, strict=True)
    pt.cuda().eval()
    rf.init_train_step_run_ctx(train_flag=False, step=0)
    return model


def _dit(model, x, lens, sigma, sigma_len=None):
    # logits [B, T, V] (fp32) of a DIT on padded ids x [B, T] with lengths lens [B], noise levels [B]
    import torch
    import returnn.frontend as rf
    from returnn.tensor import Dim

    bd = Dim(x.shape[0], name="batch")
    td = Dim(rf.convert_to_tensor(lens.to(torch.int32).cpu(), dims=[bd]), name="time")
    xx = rf.convert_to_tensor(x.to(torch.int32), dims=[bd, td], sparse_dim=model.vocab_dim)
    kw = {"sigma_len": rf.convert_to_tensor(sigma_len.float(), dims=[bd])} if sigma_len is not None else {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(xx, spatial_dim=td, sigma=rf.convert_to_tensor(sigma.float(), dims=[bd]), **kw)
    return y.copy_transpose([bd, td, model.vocab_dim]).raw_tensor.float()


def _nucleus_sample(logits, temperature, top_p):
    # one draw per row from the top-p nucleus of softmax(logits / T); logits [..., V]
    import torch

    p = (logits / temperature).softmax(-1)
    vals, idx = p.sort(-1, descending=True)
    vals = vals * ((vals.cumsum(-1) - vals) < top_p)
    s = torch.multinomial(vals.reshape(-1, vals.shape[-1]), 1).reshape(vals.shape[:-1])
    return idx.gather(-1, s[..., None]).squeeze(-1)


def _nucleus_probs(logits, temperature, top_p):
    import torch

    p = (logits / temperature).softmax(-1)
    vals, idx = p.sort(-1, descending=True)
    vals = vals * ((vals.cumsum(-1) - vals) < top_p)
    out = torch.zeros_like(p).scatter(-1, idx, vals)
    return out / out.sum(-1, keepdim=True)


# ---- samplers: each takes the prefixes (lists of ids, [] = unconditional) and returns the generated ids ----


def _sample_ar(model, prefixes, o):
    # left to right from [BOS] + prefix; fixed length: EOS suppressed, stop after `length`; variable: stop at EOS
    import torch
    import returnn.frontend as rf
    from returnn.tensor import Dim, single_step_dim

    b = len(prefixes)
    bd = Dim(b, name="batch")
    state = model.default_initial_state(batch_dims=[bd])
    n_gen = o["length"] or o["ar_cap"]
    toks = [[o["bos"]] + p for p in prefixes]  # fed so far + to feed; rows advance in lockstep
    out, done, pos = [[] for _ in range(b)], [False] * b, 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while not all(done):
            cur = torch.tensor([t[pos] if pos < len(t) else o["eos"] for t in toks], dtype=torch.int32, device="cuda")
            logits, state = model(rf.convert_to_tensor(cur, dims=[bd], sparse_dim=model.vocab_dim),
                                  spatial_dim=single_step_dim, state=state, encoder=None)
            lg = logits.copy_transpose([bd, model.vocab_dim]).raw_tensor.float()
            if o["length"]:
                lg[:, o["eos"]] = -1e9
            nxt = _nucleus_sample(lg, o["temperature"], o["top_p"]).tolist()
            for r in range(b):
                if done[r] or pos != len(toks[r]) - 1:  # still inside the prefix, or finished
                    continue
                if not o["length"] and nxt[r] == o["eos"]:
                    done[r] = True
                    continue
                toks[r].append(nxt[r])
                out[r].append(nxt[r])
                done[r] = len(out[r]) >= n_gen
            pos += 1
    return out


def _rows(prefixes, L, o, fill):
    # [BOS] prefix fill*L [EOS] per row, padded at the end (EOS) to the widest row; lengths and the editable mask
    import torch

    seqs = [[o["bos"]] + p + [fill] * L + [o["eos"]] for p in prefixes]
    width = max(map(len, seqs))
    x = torch.tensor([s + [o["eos"]] * (width - len(s)) for s in seqs], device="cuda")
    edit = torch.zeros_like(x, dtype=torch.bool)
    for r, p in enumerate(prefixes):
        edit[r, 1 + len(p) : 1 + len(p) + L] = True
    return x, torch.tensor([len(s) for s in seqs], device="cuda"), edit


def _sample_mdlm(model, prefixes, o):
    # ancestral MDLM (log-linear, no time conditioning): [BOS] prefix [MASK]*L [EOS]; a mask unmasks from t to s with
    # probability (t - s) / t, its token from the nucleus of x_theta; the last step takes the argmax
    import torch

    L, n, mask = o["length"], o["steps"], o["mask"]
    x, lens, edit = _rows(prefixes, L, o, mask)
    starts = [1 + len(p) for p in prefixes]
    for i in range(n):
        t, s = 1 - i / n, 1 - (i + 1) / n
        lg = _dit(model, x, lens, torch.zeros(len(prefixes), device="cuda"))
        lg[..., mask] = -1e9
        masked = (x == mask) & edit
        if s == 0:
            new = lg.argmax(-1)
            x = torch.where(masked, new, x)
            break
        new = _nucleus_sample(lg, o["temperature"], o["top_p"])
        unmask = masked & (torch.rand(x.shape, device="cuda") < (t - s) / t)
        x = torch.where(unmask, new, x)
    return [x[r, starts[r] : starts[r] + L].tolist() for r in range(len(prefixes))]


def _usdm_step(xt, p, at, as_, V, psi, last):
    # uniform-state posterior q(x_s | x_t, x_theta) with the psi correction; xt [B, T], p [B, T, V] nucleus x_theta
    import torch

    if last:
        return p.argmax(-1)
    r = at / as_  # alpha_{t|s}
    th = p.gather(-1, xt[..., None]).squeeze(-1)
    w = (1 - r) / V * (as_ * p + (1 - as_) / V)
    w.scatter_(-1, xt[..., None], ((r + (1 - r) / V) * (as_ * th + (1 - as_) / V))[..., None])
    w = w / w.sum(-1, keepdim=True)
    if psi > 0:
        # psi correction: mix in the forward-corrector (re-noise x_theta from alpha_t to alpha_s)
        sig = psi * min(1.0, (1 - as_) / at)
        kappa = min(1.0, max(0.0, 1 - sig / (1 - as_)))
        w0 = (1 - at) / V * p
        w0.scatter_(-1, xt[..., None], ((at + (1 - at) / V) * th)[..., None])
        w = kappa * w + (1 - kappa) * (as_ * w0 / w0.sum(-1, keepdim=True) + (1 - as_) / V)
    return torch.multinomial(w.reshape(-1, V), 1).reshape(xt.shape)


def _sample_usdm(model, prefixes, o):
    # ancestral USDM (DUO uniform state, log-linear alpha_t = 1 - (1 - eps) t, sigma = -log alpha_t): [BOS] prefix
    # noise*L [EOS], BOS / EOS / prefix kept clean
    import torch

    L, n, eps = o["length"], o["steps"], 1e-3
    V = model.vocab_dim.dimension
    b = len(prefixes)
    x, lens, edit = _rows(prefixes, L, o, 0)
    x = torch.where(edit, torch.randint(0, V, x.shape, device="cuda"), x)
    for i in range(n):
        t, s = 1 - i / n, 1 - (i + 1) / n
        at, as_ = 1 - (1 - eps) * t, 1 - (1 - eps) * s
        lg = _dit(model, x, lens, torch.full((b,), -math.log(at), device="cuda"))
        p = _nucleus_probs(lg, o["temperature"], o["top_p"])
        x = torch.where(edit, _usdm_step(x, p, at, as_, V, o["psi"], s == 0), x)
    return [x[r, 1 + len(p) : 1 + len(p) + L].tolist() for r, p in enumerate(prefixes)]


def _canvas(tokens, blank):
    c = [blank]
    for x in tokens:
        c += [x, blank]
    return c


def _sample_flex(model, prefixes, o):
    # Flex-DiffLM on the blank-interleaved canvas (no BOS / EOS); sigma = tau_sub, sigma_len = tau_len.
    # fixed length: tau_len = 0, L uniform tokens, `steps` substitution updates (USDM posterior over non-blank).
    # variable: from an empty middle, 2 init updates insert one token into every editable gap, then steps / 3 times
    # [length update, 2 substitution updates] with tau_len 1 -> 0 and tau_sub 1 -> 0 linear; a gap inserts with
    # probability (1 - p_blank) (t - s) / t (one dead token per gap), a token is deleted with p_blank times the
    # junk factor 1 - s(1 - s) / (t(1 - t)) (t <= .5)
    import torch

    blank = model.vocab_dim.dimension - 1
    V = blank
    b = len(prefixes)
    mids = [[int(v) for v in torch.randint(0, V, (o["length"],))] for _ in range(b)] if o["length"] else [[] for _ in range(b)]

    def call(tl, ts):
        cans = [_canvas(p + m, blank) for p, m in zip(prefixes, mids)]
        width = max(map(len, cans))
        x = torch.tensor([c + [blank] * (width - len(c)) for c in cans], device="cuda")
        lens = torch.tensor([len(c) for c in cans], device="cuda")
        return _dit(model, x, lens, torch.full((b,), ts, device="cuda"), torch.full((b,), tl, device="cuda"))

    def substitute(tl, t, s):
        lg = call(tl, t)
        at, as_ = max(1 - t, 1e-6), max(1 - s, 1e-6)
        for r, (p, m) in enumerate(zip(prefixes, mids)):
            if not m:
                continue
            rows = lg[r, [2 * (len(p) + j) + 1 for j in range(len(m))], :V]
            pr = _nucleus_probs(rows, o["temperature"], o["top_p"])
            xt = torch.tensor(m, device="cuda")
            mids[r] = _usdm_step(xt[None], pr[None], at, as_, V, o["psi"], s == 0)[0].tolist()

    def insert_everywhere(tl, ts):
        lg = call(tl, ts)
        for r, (p, m) in enumerate(zip(prefixes, mids)):
            gaps = [2 * len(p) + 2 * j for j in range(len(m) + 1)]
            tok = _nucleus_sample(lg[r, gaps, :V], o["temperature"], o["top_p"]).tolist()
            new = []
            for j in range(len(m) + 1):
                new.append(tok[j])
                if j < len(m):
                    new.append(m[j])
            mids[r] = new[: o["max_len"] - len(p)]

    def length_step(t, s, ts):
        lg = call(t, ts)
        for r, (p, m) in enumerate(zip(prefixes, mids)):
            gaps = [2 * len(p) + 2 * j for j in range(len(m) + 1)]
            pg = (lg[r, gaps] / o["temperature"]).softmax(-1)
            ins = torch.rand(len(gaps), device="cuda") < (1 - pg[:, blank]) * ((t - s) / t)
            tok = _nucleus_sample(lg[r, gaps, :V], o["temperature"], o["top_p"]).tolist()
            dfac = max(0.0, 1 - s * (1 - s) / (t * (1 - t))) if 0 < t <= 0.5 else 0.0
            keep = [True] * len(m)
            if m and dfac > 0:
                pt = (lg[r, [g + 1 for g in gaps[:-1]]] / o["temperature"]).softmax(-1)[:, blank]
                keep = (torch.rand(len(m), device="cuda") >= pt * dfac).tolist()
            new = []
            for j in range(len(m) + 1):
                if ins[j] and len(new) + len(p) < o["max_len"]:
                    new.append(tok[j])
                if j < len(m) and keep[j]:
                    new.append(m[j])
            mids[r] = new

    n = o["steps"]
    if o["length"]:
        for i in range(n):
            substitute(0.0, 1 - i / n, 1 - (i + 1) / n)
    else:
        insert_everywhere(1.0, 1.0)
        insert_everywhere(1.0, 1.0)
        rounds = n // 3
        for k in range(rounds):
            t, s = 1 - k / rounds, 1 - (k + 1) / rounds
            ts = 1 - 2 * k / (2 * rounds)
            length_step(t, s, ts)
            for j in range(2):
                a, c = 1 - (2 * k + j) / (2 * rounds), 1 - (2 * k + j + 1) / (2 * rounds)
                substitute(s, a, c)
    return mids


# ---- metrics ----


def _entropy(ids):
    if not ids:
        return 0.0
    n = len(ids)
    cnt = {}
    for x in ids:
        cnt[x] = cnt.get(x, 0) + 1
    return -sum(c / n * math.log(c / n) for c in cnt.values())


def _gen_ppl(lm, tok, texts, block=1024):
    # samples joined in order with EOS separators, cut into 1024-token blocks; separator targets and the first token
    # of each block are not scored; token NLLs summed before the exponential
    import torch

    ids, scored = [], []
    for t in texts:
        piece = tok.encode(t) if t else []
        ids += piece + [tok.eos_token_id]
        scored += [True] * len(piece) + [False]
    nll, cnt = 0.0, 0
    for i in range(0, len(ids), block):
        x = ids[i : i + block]
        if len(x) < 2:
            continue
        m = torch.tensor(scored[i : i + block][1:], device="cuda")
        with torch.inference_mode():
            lg = lm(torch.tensor([x], device="cuda")).logits[0, :-1].float()
        lp = lg.log_softmax(-1).gather(-1, torch.tensor(x[1:], device="cuda")[:, None]).squeeze(-1)
        nll -= float((lp * m).sum())
        cnt += int(m.sum())
    return math.exp(nll / max(cnt, 1))


def _cont_ppl(lm, tok, pairs):
    # prefix and continuation tokenized together (space separated); only continuation tokens scored, per example
    import torch

    nll, cnt = 0.0, 0
    for pre, cont in pairs:
        if not cont:
            continue
        a = tok.encode(pre)
        x = tok.encode(pre + " " + cont)
        k = len(a)
        if len(x) <= k:
            continue
        with torch.inference_mode():
            lg = lm(torch.tensor([x], device="cuda")).logits[0, :-1].float()
        lp = lg.log_softmax(-1).gather(-1, torch.tensor(x[1:], device="cuda")[:, None]).squeeze(-1)
        nll -= float(lp[k - 1 :].sum())
        cnt += len(x) - k
    return math.exp(nll / max(cnt, 1))


def _features(lm, tok, texts):
    # GPT-2 last-token hidden state of the final layer (after ln_f)
    import torch

    out = []
    for t in texts:
        x = tok.encode(t)[:1024] or [tok.eos_token_id]
        with torch.inference_mode():
            h = lm(torch.tensor([x], device="cuda"), output_hidden_states=True).hidden_states[-1][0, -1].float()
        out.append(h.cpu().numpy())
    import numpy

    return numpy.stack(out)


def _mauve(lm, tok, gen, ref, c=5.0, seed=0):
    # MAUVE: k-means over both feature sets, K = max(2, round(min(N_gen, N_ref) / 10)); divergence curve over lambda,
    # area under (exp(-c KL(ref || mix)), exp(-c KL(gen || mix)))
    import numpy
    from sklearn.cluster import KMeans

    if not gen:
        return 0.0
    fg, fr = _features(lm, tok, gen), _features(lm, tok, ref)
    k = max(2, round(min(len(fg), len(fr)) / 10))
    lab = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(numpy.concatenate([fg, fr]))
    pg = numpy.bincount(lab[: len(fg)], minlength=k) / len(fg)
    pr = numpy.bincount(lab[len(fg) :], minlength=k) / len(fr)

    def kl(p, q):
        m = p > 0
        return float((p[m] * numpy.log(p[m] / q[m])).sum())

    xs, ys = [1.0], [0.0]
    for lam in numpy.linspace(1e-6, 1 - 1e-6, 25):
        mix = lam * pg + (1 - lam) * pr
        xs.append(math.exp(-c * kl(pr, mix)))
        ys.append(math.exp(-c * kl(pg, mix)))
    xs.append(0.0)
    ys.append(1.0)
    pts = sorted(zip(xs, ys))
    return float(sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(pts, pts[1:])))
