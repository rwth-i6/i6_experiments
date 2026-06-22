"""moshirag_finetune_launcher.py -- SCAFFOLD entry point for MoshiRAG finetuning.

NOTE (2026-06-18 investigation): the released moshi-rag fork ships NO training loop, and
``moshi_finetune`` is not installed in the moshirag venv (and models plain Moshi -- no ARC
conditioner), so this does NOT mirror ``moshi_finetune_launcher.py``. The decided path is to
build a minimal LoRA training loop directly on the fork's own model:
``moshi.models.loaders.get_moshi_lm`` (load kyutai/moshika-rag-pytorch-bf16) +
``moshi.modules.lora.replace_all_linear_with_lora`` (freeze base, train LoRA), loss on
``LMModel.forward(codes, condition_tensors).logits``. The conditioner + LoRA already exist in
the fork; only the training plumbing + the RAG-specific seams below are missing:

  * the **ARC-Encoder reference conditioner** forward/collate (encode the reference doc,
    project through the trainable linear layer, add into the temporal-transformer input
    embeddings);
  * the **<ret> retrieval-trigger token** supervision, **reference dropout (0.2)**, and the
    **retrieval-delay simulation** (paper Eq. 3) in the collator.

Until those are built (see ``projects/2026-01-speech-llm/moshirag.md`` for the D3 effort
estimate -- ~1.5-3 weeks for checkpoint-init LoRA, dominated by the conditioner train-loop
seam), this raises so a misconfigured ``SpeechFinetune(adapter=MOSHIRAG_ADAPTER)`` run fails
loudly instead of silently training a non-RAG model.

To finalize: build the LoRA loop on the fork's ``LMModel`` (above), init from
``kyutai/moshika-rag-pytorch-bf16`` (conditioner + <ret> already trained), implement the two
seams above plus a collator that encodes references into ``condition_tensors``, then delete the
NotImplementedError. See projects/2026-01-speech-llm/moshirag.md ("Training base -- decided").
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
