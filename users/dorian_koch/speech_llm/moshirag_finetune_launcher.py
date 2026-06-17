"""moshirag_finetune_launcher.py -- SCAFFOLD entry point for MoshiRAG finetuning.

Mirrors ``moshi_finetune_launcher.py``: it would import the MoshiRAG training fork's
``train`` module, swap in our arrow-capable data loader (``moshi_arrow_dataset``), install
the ``ArrowDataConfig`` sidecar, and additionally wire the RAG-specific seams the *released*
(inference-only) moshi-rag fork does not provide:

  * the **ARC-Encoder reference conditioner** forward/collate (encode the reference doc,
    project through the trainable linear layer, add into the temporal-transformer input
    embeddings);
  * the **<ret> retrieval-trigger token** supervision, **reference dropout (0.2)**, and the
    **retrieval-delay simulation** (paper Eq. 3) in the collator.

Until those are built (see ``projects/2026-01-speech-llm/moshirag.md`` for the D3 effort
estimate -- ~1.5-3 weeks for checkpoint-init LoRA, dominated by the conditioner train-loop
seam), this raises so a misconfigured ``SpeechFinetune(adapter=MOSHIRAG_ADAPTER)`` run fails
loudly instead of silently training a non-RAG model.

To finalize: point the import at the real training loop, init from
``kyutai/moshika-rag-pytorch-bf16`` (conditioner + <ret> already trained), implement the two
seams above, then delete the NotImplementedError.
"""

import sys


def main():
    raise NotImplementedError(
        "MoshiRAG finetuning is scaffolded but not implemented: the released moshi-rag fork is "
        "inference-only, so the ARC-Encoder reference-conditioner forward/collate and the "
        "<ret>/reference-dropout/retrieval-delay-sim seams are not wired yet. See "
        "projects/2026-01-speech-llm/moshirag.md. "
        f"(invoked with argv={sys.argv[1:]})"
    )


if __name__ == "__main__":
    main()
