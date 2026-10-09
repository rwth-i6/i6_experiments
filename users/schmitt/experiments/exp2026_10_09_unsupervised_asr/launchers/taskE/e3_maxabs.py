import numpy as np, sys
sys.path.insert(0, "/u/schmitt/experiments/2026_04_09_unsupervised_asr/recipe/i6_experiments/users/schmitt/experiments/exp2026_10_09_unsupervised_asr/scripts")
from calc_emission_ladder import FeatureStore
s = FeatureStore("pooled")
z = np.load("/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskE/e3/pooled_seg1.00/features.npz")
for name in ("audio", "eval", "ceil"):
    tags = [str(t) for t in z["%s_tags" % name]]
    mx = np.array([np.abs(s.get(t)).max() for t in tags])
    print(name, len(tags), "max|x| quantiles 50/99/99.9/max: %.1f %.1f %.1f %.3g" % tuple(np.quantile(mx, [.5, .99, .999, 1.0])),
          " >65504: %d  >1e5: %d  >1e3: %d" % ((mx > 65504).sum(), (mx > 1e5).sum(), (mx > 1e3).sum()))
