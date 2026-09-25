import sys, numpy as np
sys.path.insert(0, "analysis")
import long_sil_after_vad as L
pairs = {"dev-other": ("work/i6_core/returnn/oggzip/BlissToOggZipJob.6RIPlWd7awp7/output/out.ogg.zip", "work/i6_core/returnn/forward/ReturnnForwardJobV2.QO3M1P9dOc2o/output/feats.hdf"),
         "dev-clean": ("work/i6_core/returnn/oggzip/BlissToOggZipJob.sohGDj24P4Qm/output/out.ogg.zip", "work/i6_core/returnn/forward/ReturnnForwardJobV2.5z8bUupJ1f8j/output/feats.hdf")}
for s, (z, h) in pairs.items():
    job = L.vad_masks("alias/sae/4a/data/vad", s)
    rec = L.recompute_masks(z, h, max_utts=10**9)
    common = set(job) & set(rec)
    diff = [u for u in common if job[u][1] != rec[u][1] or not np.array_equal(job[u][0], rec[u][0])]
    fd = sum(len(np.setxor1d(job[u][0], rec[u][0])) for u in diff)
    print(s, "job", len(job), "rec", len(rec), "common", len(common), "differ", len(diff), f"{100*len(diff)/len(common):.2f}%", "frames_xor", fd, flush=True)
