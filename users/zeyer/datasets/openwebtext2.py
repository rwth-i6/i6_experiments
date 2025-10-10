"""
OpenWebText2

https://openwebtext2.readthedocs.io/
https://huggingface.co/datasets/Geralt-Targaryen/openwebtext2
"""


def _load_cleaned_en():
    from datasets import load_dataset

    ds = load_dataset("Geralt-Targaryen/openwebtext2", split="train")
    # Dataset({
    #     features: ['title', 'text', 'reddit_scores'],
    #     num_rows: 13071217
    # })
    # Not sure how accurate this estimate of number of words is:
    # sum(len(x["text"].split()) for x in ds) = 9_638_326_931
    return ds
