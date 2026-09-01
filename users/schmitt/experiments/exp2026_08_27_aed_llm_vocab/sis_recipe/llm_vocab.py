"""
LLM (Qwen2) vocab on the Loquacious transcripts: tokenizer handles, lowercasing, usage stats.

Background: the AED here is meant to be an encoder donor for a speech LLM, so it should be trained
on the LLM's own token inventory rather than a dedicated ASR SPM. The full Qwen2 vocab is 151646
entries, which as an AED decoder embedding + output projection would be ~155M parameters each --
more than half of the 561M model -- while the Loquacious transcripts only ever produce a small
fraction of it.

LOWERCASING IS NOT COSMETIC. The Loquacious transcripts are stored ALL-CAPS without punctuation:

    AND WHAT ABOUT INTEROPERABILITY IN THE RAIL SECTOR ARE NATIONAL BARRIERS PREVENTING PROGRESS

Qwen2's BPE was trained on ordinary mixed-case text, so all-caps input falls off the merge table
and shatters into short pieces. Measured here on 300k Loquacious train transcripts (9.34M words)
with the real Qwen2 tokenizer:

    UPPERCASE (as stored)    distinct ids = 4_731    1.591 tokens/word
    lowercased               distinct ids = 30_839   1.071 tokens/word

So lowercasing is a ~33% reduction in target sequence length -- which is also a direct compute
saving in the decoder and in the packed text budget -- at the price of a larger (but still ~5x
smaller than full) vocab. The all-caps run's small id count is not a good thing: it is the
tokenizer failing, spending 1.59 tokens per word on byte-level fallbacks.

This mirrors ``use_lowercase=True`` in
``speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/loquacious/configs/
config_loquacious_25k_albert_qwen_lowercase_v1.py``, which reaches it via a
``PostprocessingDataset`` whose ``map_seq`` is
``speech_llm.prefix_lm.model.util.vocab.LowerCaseTextAndApplyVocab`` -- the text stream is read as
raw UTF-8 bytes (``Utf8ByteTargets``) and lowercased+tokenized in the post-processor.

Here we use RETURNN's own ``HuggingFaceTokenizer(text_preprocessing=...)`` hook instead
(``returnn/datasets/util/vocabulary.py:815``, applied inside ``get_seq``). It is equivalent for
our purposes and considerably simpler: no byte-stream detour, no PostprocessingDataset, and the
vocab stays a real ``Vocabulary`` object -- which matters because the recog/scoring path needs
``id_to_label`` and ``get_seq_labels``.
"""

from __future__ import annotations

from functools import cache
from typing import Any, Dict, Optional

from sisyphus import tk

from i6_experiments.users.schmitt.external_models.huggingface import DownloadHuggingFaceRepoJobV2

# Full Qwen2 tokenizer size, i.e. the length of the usage arrays.
# NB: this is len(tokenizer) (incl. the added special tokens), not tokenizer.vocab_size.
# It is also the hard-coded constant in returnn/tools/vocab_usage.py.
QWEN2_VOCAB_SIZE = 151646

_DEFAULT_MODEL_ID = "Qwen/Qwen2-0.5B"


def lowercase(text: str) -> str:
    """
    ``text_preprocessing`` for :class:`HuggingFaceTokenizer`.

    Module-level (not a lambda) so it is importable from the generated RETURNN config and stable
    under Sisyphus hashing.
    """
    return text.lower()


@cache
def get_qwen_tokenizer_repo(model_id: str = _DEFAULT_MODEL_ID) -> tk.Path:
    """
    :return: directory holding ``tokenizer.json`` + ``tokenizer_config.json``.

    Same job and same ``file_list`` as ``prefix_lm/sis_recipe/data/hf_tokenizer.get_hf_datastream``,
    so this resolves to the existing job rather than downloading again.
    """
    job = DownloadHuggingFaceRepoJobV2(model_id=model_id, file_list=["tokenizer_config.json", "tokenizer.json"])
    job.add_alias(f"tokenizers/{model_id}")
    return job.out_content_dir


def get_qwen_vocab_opts(*, model_id: str = _DEFAULT_MODEL_ID, use_lowercase: bool = True) -> Dict[str, Any]:
    """
    :return: RETURNN vocab opts (the dict consumed by ``Vocabulary.create_vocab``).
    """
    opts: Dict[str, Any] = {
        "class": "HuggingFaceTokenizer",
        "huggingface_repo_dir": get_qwen_tokenizer_repo(model_id),
    }
    if use_lowercase:
        opts["text_preprocessing"] = lowercase
    return opts


def get_loquacious_vocab_usage(
    *,
    subset_name: str = "large",
    split: str = "train",
    use_lowercase: bool = True,
    model_id: str = _DEFAULT_MODEL_ID,
    use_audio_filter: bool = False,
    max_audio_len_secs: Optional[float] = 19.5,
    multi_proc: int = 8,
    train_partition_epoch: int = 25,
    alias_prefix: Optional[str] = None,
):
    """
    Vocab usage of one Loquacious split under the Qwen2 tokenizer.

    Reproduces the reference ``dump_vocab_usage.sh`` run (see :mod:`.vocab_usage`), with
    lowercasing added.

    :param use_audio_filter: if True, read the real (audio) corpus and condition the counts on
        ``max_audio_len_secs``, exactly as the reference did. Costs a full decode of the 25k-hour
        corpus. Default False: read the audio-free corpus variant instead and count over all
        sequences, which yields a superset of the filtered id set (see the branch below).
    :param max_audio_len_secs: the audio filter, as ``max_seq_length``. The reference config used
        ``{"audio": 312000.0}`` = 19.5s at 16kHz, the trainings' ``max_seq_length_default_input``.
        Only has an effect together with ``use_audio_filter``.
    :return: the :class:`ExtractVocabUsageJob`
    """
    from i6_experiments.users.zeyer.datasets.loquacious import (
        get_hf_text_only,
        get_loquacious_hf_ogg,
        _make_hf_dataset,
        _make_hf_dataset_text_only,
    )
    from i6_experiments.users.zeyer.tools_paths import get_returnn_root

    from .vocab_usage import ExtractVocabUsageJob

    vocab_opts = get_qwen_vocab_opts(model_id=model_id, use_lowercase=use_lowercase)

    # A thin stand-in for a VocabConfig: _make_hf_dataset only calls .get_opts() on it.
    class _VocabCfg:
        @staticmethod
        def get_opts():
            return vocab_opts

    is_train = split == "train"

    if use_audio_filter:
        # Faithful to the reference: read the real audio dataset so max_seq_length can filter on
        # the decoded audio length. EXPENSIVE -- it decodes the whole 25k-hour corpus (396 GB over
        # 848 shards) purely to learn each utterance's length.
        hf_data_dir = get_loquacious_hf_ogg(name=subset_name)
        ds = _make_hf_dataset(
            hf_data_dir=hf_data_dir,
            split=split,
            vocab=_VocabCfg(),
            # "default" = corpus order: we read every sequence exactly once, so the order is
            # irrelevant, and the default "sorted_reverse" would force a full length pass first.
            seq_ordering="default",
            use_distrib_files=is_train,
            # The train split is 396 GB over 848 shards. DistributeFilesDataset stages the shards
            # of the CURRENT sub-epoch to local disk, so WITHOUT a partition_epoch all 848 land in
            # one sub-epoch -- which is what OOM-killed the first attempt. 25 is what the reference
            # vocab-usage config used, i.e. ~16 GB per sub-epoch.
            partition_epoch=train_partition_epoch if is_train else None,
            multi_proc_dataset={"num_workers": multi_proc} if multi_proc >= 2 else None,
        )
    else:
        # Default. The audio is only ever touched to obtain a length for the filter, so use the
        # audio-free variant of the corpus (``_hf_dataset_remove_audio``): ~100 MB instead of
        # 396 GB, no ogg decoding, no FFmpeg, minutes instead of a day.
        # The filter can only REMOVE sequences, so the resulting id set is a SUPERSET of the
        # filtered one -- the safe direction for restricting a vocab (an extra id costs ~5k
        # parameters; a missing id is unrecoverable). Measured offline on the extracted
        # transcripts: 39,558 distinct lowercase ids unfiltered, vs 3,564 / 3,531 on dev
        # unfiltered/filtered, i.e. the filter moves the count by well under 1%.
        hf_data_dir = get_hf_text_only(name=subset_name)
        ds = _make_hf_dataset_text_only(
            hf_data_dir=hf_data_dir,
            split=split,
            vocab=_VocabCfg(),
            seq_ordering="default",
            multi_proc_dataset={"num_workers": multi_proc} if multi_proc >= 2 else None,
        )

    main_dataset = ds.main_dataset
    if use_audio_filter and is_train:
        # partition_epoch alone would only cover 1/25 of the data in our single pass.
        # MultiEpochDataset with multi_epoch == partition_epoch makes one outer epoch iterate all
        # inner sub-epochs, i.e. the full train set exactly once while only 1/25 of the shards are
        # resident at a time. This is precisely the case its docstring describes, and what the
        # reference config did.
        main_dataset = {
            "class": "MultiEpochDataset",
            "dataset": main_dataset,
            "multi_epoch": train_partition_epoch,
        }

    extra_config: Dict[str, Any] = {}
    if use_audio_filter and max_audio_len_secs is not None:
        extra_config["max_seq_length"] = {"audio": max_audio_len_secs * 16_000}

    job = ExtractVocabUsageJob(
        dataset=main_dataset,
        vocab_size=QWEN2_VOCAB_SIZE,
        key="text",
        extra_config=extra_config or None,
        returnn_root=get_returnn_root(),
        # train: 9.5M seqs, every one ogg-decoded to get its audio length for the filter, with
        # `multi_proc` worker processes each holding a shard reader -> the first attempt was
        # OOM-killed at the 8 GB default. Not hashed.
        rqmt=(
            # audio path: 9.5M seqs, every one ogg-decoded just for its length -> the first
            # attempt was OOM-killed at the 8 GB default. Not hashed.
            {"cpu": max(multi_proc, 2) + 2, "mem": 64, "time": 48}
            if (use_audio_filter and is_train)
            else ({"cpu": max(multi_proc, 2) + 2, "mem": 16, "time": 8} if is_train else None)
        ),
    )
    name = f"{subset_name}-{split}{'-lower' if use_lowercase else '-upper'}"
    if use_audio_filter:
        name += "-audiofilt"
    job.add_alias(f"vocab_usage/qwen2/{name}")
    if alias_prefix:
        tk.register_output(f"{alias_prefix}/vocab_usage/qwen2/{name}/stats.txt", job.out_stats)
        tk.register_output(f"{alias_prefix}/vocab_usage/qwen2/{name}/used_ids.txt", job.out_used_ids)
        tk.register_output(f"{alias_prefix}/vocab_usage/qwen2/{name}/num_used", job.out_num_used)
    return job


def py():
    """Register the vocab-usage measurements."""
    from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

    prefix = get_setup_prefix_for_module(__name__)

    # The number that decides the AED vocab size. Lowercase is the configuration we intend to
    # train with; uppercase is registered as the control that makes the lowercase gain legible.
    for use_lowercase in (True, False):
        get_loquacious_vocab_usage(
            subset_name="large", split="train", use_lowercase=use_lowercase, alias_prefix=prefix
        )
    # dev/test: needed to check coverage, i.e. that restricting the vocab to train-used ids does
    # not make some eval reference unreachable.
    for split in ("dev", "test"):
        get_loquacious_vocab_usage(subset_name="large", split=split, use_lowercase=True, alias_prefix=prefix)
