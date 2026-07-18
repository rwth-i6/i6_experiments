"""SAE §0a-style target-quality audit on wav2vec2-Large L15 features (in-process, conda `speech_llm`).

Called directly from `W2v2ReprAuditJob.run()` -- not a subprocess: the job already runs under the
speech_llm python, so there is no cross-env reason to shell out. On the SAME 500 dev-clean utts and
with the SAME metric primitives as real_repr_probe.py (§0a BEST-RQ),

  audit  hard-kmeans oracle-map PER vs K  -> the *unsupervised* clusterability ceiling
  probe  supervised linear / MLP PER      -> the *linear-decodability* ceiling

so the wav2vec2 numbers land next to BEST-RQ's 0.63 / 0.145. The audit/preprocess/_decode_stats logic
is imported from real_repr_probe so the methodology is byte-identical; only the encoder (HF wav2vec2,
hidden_states[15], 1024-d @ 50 Hz) and a dim-general probe are new. 50 Hz has 2x BEST-RQ's
frames/phone, inflating hard-unit over-segmentation, so a pool=2 (-> 25 Hz) rate-matched variant is
also reported.

This module is imported only inside the job's run(), so its torch/transformers imports never reach
the graph-building manager (which runs under sis_env, without them).
"""

import io
import time
from contextlib import redirect_stdout

import numpy as np
import torch

from i6_experiments.users.wu.experiments.ssl.analysis import real_repr_probe as RP
from i6_experiments.users.wu.experiments.unsupervised_asr import vad_port

RA = RP.RA


def extract(model_dir, layer, fps, n_utts, split, device):
    """[(feat[T,1024], gold[T], sil[T] bool)] for `n_utts` of `split`, frames on the fps grid."""
    from rVADfast import rVADfast
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    model = Wav2Vec2Model.from_pretrained(model_dir).to(device).eval()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    vad = rVADfast(vad_threshold=0.4)
    subframes = int(round(100.0 / fps))
    recs = vad_port._load_gilkeyio(split, limit=n_utts)
    utts = []
    for wav, phon in recs:
        inp = fe(wav, sampling_rate=16000, return_tensors="pt")
        am = inp.get("attention_mask")
        with torch.no_grad():
            out = model(inp.input_values.to(device),
                        attention_mask=am.to(device) if am is not None else None,
                        output_hidden_states=True)
        e = out.hidden_states[layer][0].float().cpu().numpy()  # [T,1024]
        g = RA.frame_phone_labels(phon, e.shape[0], frame_rate_hz=fps)
        sil = vad_port.rvad_silence(wav, vad=vad, subframes=subframes)
        sil = sil[:e.shape[0]] if len(sil) >= e.shape[0] else np.concatenate(
            [sil, np.ones(e.shape[0] - len(sil), bool)])
        utts.append((e, g, sil))
    return utts


def probe(utts, hiddens=(0, 256)):
    """Dim-general mirror of real_repr_probe.probe (which hardcodes d=512); same protocol otherwise."""
    train_i, test_i = RP._split(len(utts))
    Xtr = np.concatenate([utts[i][0] for i in train_i]).astype(np.float32)
    ytr = np.concatenate([utts[i][1] for i in train_i]).astype(np.int64)
    d0 = Xtr.shape[1]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    Xtr = (Xtr - mu) / sd
    torch.manual_seed(0)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    print(f"[probe] train frames={len(ytr)} test utts={len(test_i)} dim={d0}", flush=True)
    for hidden in hiddens:
        layers, d = [], d0
        if hidden:
            layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU()]
            d = hidden
        layers += [torch.nn.Linear(d, RA.NUM_PHONES)]
        net = torch.nn.Sequential(*layers)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = torch.nn.CrossEntropyLoss()
        for _ in range(25):
            perm = torch.randperm(len(yt))
            for b in range(0, len(yt), 4096):
                j = perm[b:b + 4096]
                opt.zero_grad()
                lossf(net(Xt[j]), yt[j]).backward()
                opt.step()
        net.eval()
        pred, gold, corr, tot = {}, {}, 0, 0
        with torch.no_grad():
            for i in test_i:
                p = net(torch.tensor((utts[i][0].astype(np.float32) - mu) / sd)).argmax(1).numpy()
                g = utts[i][1]
                pred[i], gold[i] = p, g
                n = min(len(p), len(g))
                corr += int((p[:n] == g[:n]).sum())
                tot += n
        st = RP._decode_stats(pred, gold)
        tag = "linear" if not hidden else f"MLP-{hidden}"
        print(f"  {tag:10s} frame_acc={corr/tot:.3f}  PER={st['PER']:.3f}  "
              f"sub={st['sub']:.3f} ins={st['ins']:.3f} del={st['dele']:.3f}", flush=True)


def run_audit(model_dir, *, layer=15, fps=50.0, n_utts=500, split="validation.clean",
              Ks=(500, 1000, 2000, 4000)) -> str:
    """Extract features on GPU, run the §0a audit + probe over three rate variants, return the report."""
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable -- the repr-audit forward needs a GPU (fail fast, resubmit).")
    t0 = time.time()
    utts = extract(model_dir, layer, fps, n_utts, split, "cuda")
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"== wav2vec2 L{layer} @ {fps:g} Hz | {len(utts)} utts | {split} ==")
        print("anchors (BEST-RQ §0a, SAE_0.md): k-means oracle-PER 0.632 | linear probe 0.145")
        print(f"\n[RAW {fps:g} Hz]  audit + probe")
        RP.audit(utts, list(Ks))
        probe(utts)
        print(f"\n[POOL=2 -> {fps / 2:g} Hz | rate-matched to BEST-RQ 25 Hz]")
        p2 = RP._preprocess(utts, do_vad=False, pool_w=2)
        RP.audit(p2, list(Ks))
        probe(p2)
        print(f"\n[VAD-trimmed | RAW {fps:g} Hz]")
        pv = RP._preprocess(utts, do_vad=True, pool_w=1)
        RP.audit(pv, list(Ks))
        probe(pv)
    return f"# repr-audit wav2vec2 L{layer} ({time.time() - t0:.0f}s)\n{buf.getvalue()}"
