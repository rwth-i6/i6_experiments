from sisyphus import tk

from i6_experiments.users.berger.helpers import rasr_lm_config


def get_lm(name: str) -> rasr_lm_config.LMData:
    if name == "zoltan_4gram":
        return rasr_lm_config.ArpaLMData(10, tk.Path("/work/asr4/berger/dependencies/switchboard/lm/zoltan_4gram.gz"))
    elif name == "fisher_4gram":
        return rasr_lm_config.ArpaLMData(
            10, tk.Path("/home/tuske/work/ASR/switchboard/corpus/lm/data/mylm/swb.fsh.4gr.voc30k.LM.gz")
        )
    raise ValueError
