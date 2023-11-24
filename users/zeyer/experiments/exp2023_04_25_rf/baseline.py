"""
Baseline, also intended to setup sis dir for import
"""

from .conformer_import_moh_att_2023_06_30 import train_exp, config_24gb_v4


# run directly via `sis m ...`
def py():
    train_exp("base-24gb-v4", config_24gb_v4)

    from sisyphus import tk
    from i6_core.audio.encoding import BlissChangeEncodingJob

    for target in list(tk.sis_graph.targets):
        job = target._sis_path.creator
        if isinstance(job, BlissChangeEncodingJob):
            tk.sis_graph.targets.remove(target)
            tk.sis_graph.active_targets.remove(target)
