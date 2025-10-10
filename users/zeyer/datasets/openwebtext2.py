"""
OpenWebText2

https://openwebtext2.readthedocs.io/
https://huggingface.co/datasets/Geralt-Targaryen/openwebtext2
"""


def _load():
    from datasets import load_dataset

    ds = load_dataset("Geralt-Targaryen/openwebtext2", split="train")
    # Dataset({
    #     features: ['title', 'text', 'reddit_scores'],
    #     num_rows: 13071217
    # })
    return ds
