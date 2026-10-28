"""A phoneme 5-gram whose silence matches the VAD frames: the original LM text, re-silenced.

``docs/free_segmentation/findings_and_directions.md`` §6.12. The LM every run so far
used (``phonetic_lm_dict[5]``) was trained on text with ``<SIL>`` inserted at random:
10% of word boundaries and 10% at each sentence end
(``SilenceModel(between_word=0.1, sentence_bounds=0.1)``). On the ls-100h VAD frames
the GMM alignment has silence at the start of 95.2% of the utterances, at the end of
64.6%, and at 15.8% of the word boundaries (149,946 internal pauses over 951,197
boundaries; 2026-10-27). In ``vq_gmm_tau_silence``'s branch arms ``<SIL>`` settled at
~57% of the data's silence segments, about the LM's rate relative to the data's.

**The same pipeline, one change.** The colleague's finished inputs are reused as they
are - the concatenated LibriSpeech LM text and ls-100h transcripts, the merged Bliss
lexicon, the G2P pronunciations of the OOVs, and the KenLM build - and only the
phonemization (``setup/phoneme_lm.TextToPhonemeJob``, restored from the ``haotian_lm``
pipeline) and ``lmplz`` run again, with the silence model fitted to the data. With the
original silence model the restored job reproduces the original LM text line for line
(checked on the first 300k input lines, 2026-10-27).

* The G2P output (``ApplyG2PModelJob.Zsy4prYCQc6Q``) is readable only by its owner -
  ``mkstemp`` made it - but its 30 parts in ``work/`` are not, and that job's output is
  their concatenation filtered to four-field lines, which the lexicon loader skips
  anyway. A ``ConcatenateJob`` makes a readable copy.
* ``lmplz`` exactly as before: order 5, no pruning, interpolated unigrams, discount
  fallback 0.5 / 1.0 / 1.5. ``-S`` (the memory rqmt) only sizes its sort buffer.

Matched per position, not per phoneme: LM text lines are single sentences, shorter
than LibriSpeech utterances, so the same rates give more silence per 100 phonemes
(~6.2 against the data's 5.6); what the 5-gram learns is silence after ``<s>``,
before ``</s>`` and after a word end, and those are what match.

Outputs: ``phon_lm/vad_silence/phon_count_5gram.arpa.gz``, the inserted silence
(``silence_statistics.json``), and both LMs' per-phoneme cost and candidate pull
(``PhonemeLMCostJob``, which also loads the new ARPA through the op).
"""

from sisyphus import tk

from i6_core.lm.kenlm import CompileKenLMJob, KenLMplzJob
from i6_core.text.processing import ConcatenateJob
from i6_core.tools import CloneGitRepositoryJob

from i6_experiments.example_setups.guided_kmeans.setup.emission_balance import PhonemeLMCostJob
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_fb_lexicon,
    phonetic_lm_dict,
)
from i6_experiments.example_setups.guided_kmeans.setup.phoneme_lm import (
    SilenceModel,
    TextToPhonemeJob,
)
from i6_experiments.example_setups.guided_kmeans import tools

exp_dir = "vq_gmm_tau_silence_lm"

#: The colleague's finished jobs behind ``phonetic_lm_dict``'s count LMs.
_COLLEAGUE_WORK = "/u/lkleppel/experiments/20260520_unsupervised_asr/work"
#: LibriSpeech LM normalized text + ls-100h transcripts (``ConcatenateJob``).
LM_TEXT = tk.Path(f"{_COLLEAGUE_WORK}/i6_core/text/processing/ConcatenateJob.93h5n4R89dxr/output/out.gz")
#: The Bliss lexicon the phonemization reads first pronunciations from.
LM_LEXICON = tk.Path(
    f"{_COLLEAGUE_WORK}/i6_core/lexicon/modification/MergeLexiconJob.qKaOAPqURCkK/output/lexicon.xml.gz")
#: The G2P pronunciations of the text's OOVs, as the 30 parts of
#: ``ApplyG2PModelJob(concurrent=30)`` (its merged output is unreadable, see above).
LM_G2P_PARTS = [
    tk.Path(f"{_COLLEAGUE_WORK}/i6_core/g2p/apply/ApplyG2PModelJob.Zsy4prYCQc6Q/work/g2p.lexicon.{i}")
    for i in range(1, 31)
]

#: Fitted to the GMM alignment on the ls-100h VAD frames (module docstring). The
#: seed is the original's.
VAD_SILENCE = SilenceModel(
    silence_token="<SIL>",
    between_word=0.158,
    sentence_begin=0.952,
    sentence_end=0.646,
    random_seed=1307,
)
LM_ORDERS = [5]


def vad_silence_lm(order: int = 5) -> tk.Path:
    """The re-silenced count LM of ``order`` (ARPA, gzipped)."""
    g2p_lexicon = ConcatenateJob(LM_G2P_PARTS, zip_out=False, out_name="g2p.lexicon").out
    phonemized = TextToPhonemeJob(
        text_file=LM_TEXT,
        bliss_lexicon=LM_LEXICON,
        g2p_lexicon=g2p_lexicon,
        gzip_output=True,
        silence_model=VAD_SILENCE,
    )
    phonemized.add_alias(f"{exp_dir}/phonemize_lm_train")
    tk.register_output("phon_lm/vad_silence/silence_statistics.json",
                       phonemized.out_silence_statistics)
    kenlm = CompileKenLMJob(
        repository=CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository
    ).out_binaries
    lm = KenLMplzJob(
        text=[phonemized.out_text],
        order=order,
        interpolate_unigrams=True,
        pruning=None,
        vocabulary=None,
        discount_fallback=[0.5, 1.0, 1.5],
        kenlm_binary_folder=kenlm,
        mem=32,
        time=8,
    )
    lm.add_alias(f"{exp_dir}/kenlm_lmplz_o{order}")
    return lm.out_lm


def run():
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)
    for order in LM_ORDERS:
        arpa = vad_silence_lm(order)
        tk.register_output(f"phon_lm/vad_silence/phon_count_{order}gram.arpa.gz", arpa)
        for name, path in (("vad_silence", arpa), ("original", phonetic_lm_dict[order])):
            cost = PhonemeLMCostJob(arpa_path=path, lexicon=fb_lexicon,
                                    backoff_fb_root=tools.BACKOFF_FB)
            tk.register_output(f"phon_lm/vad_silence/lm_cost_{order}gram_{name}.json",
                               cost.out_summary)


def py():
    run()
