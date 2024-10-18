# Having this here triggers this package to be used by our code as the unhashed_package_root,
# also to define this as the root for experiment prefixes,
# and the specific value is added to the hash of corresponding experiments.
__setup_base_name__ = "dorian.koch--2024-10-11--denoising-lm"

import sys
# fix import path for i6_experiments

# find cur path
cur = __file__
# go up until we find i6_experiments
while not cur.endswith("i6_experiments") and len(cur) > 1:
    cur = "/".join(cur.split("/")[:-1])
assert cur.endswith("i6_experiments"), f"Could not find i6_experiments in path of {__file__}"
cur = "/".join(cur.split("/")[:-1]) # remove the dir
sys.path.append(cur)
print(f"Added {cur} to path")