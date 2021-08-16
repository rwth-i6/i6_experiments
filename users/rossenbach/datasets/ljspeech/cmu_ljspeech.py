from i6_core.corpus.convert import CorpusToTxtJob

from i6_experiments.common.datasets.ljspeech import get_uppercase_cmu_g2p, get_uppercase_lexicon



def apply_uppercase_cmu_corpus_processing(bliss_corpus, tts_toolchain, path_prefix):

    bliss_text = CorpusToTxtJob(bliss_corpus).out
    tokenizer = "/u/rossenbach/src/tts-toolchain/tokenizer/tokenizer.perl"
    tokenized_text = Pipeline(bliss_text, [str(tokenizer) + " -l en -no-escape", "tr '[:lower:]' '[:upper:]'"]).out
    phoneme_text = ApplySequitur(tokenized_text, cmu_lexicon_uppercase, sequitur_cmu_uppercase_v1).out
    bliss_corpus = BlissMergeText(bliss_corpus, phoneme_text).out
    # text processing
    process_text_job = ProcessBlissText(bliss_corpus, [('uppercase', {}),
                                                       ('end_token', {'token': ' ~'})],
                                        vocabulary=self.vocabulary,
                                        vocabulary_mode="word")
    self.job_alias(process_text_job, "text_preprocessing_%s" % name)
    bliss_corpus = process_text_job.out






