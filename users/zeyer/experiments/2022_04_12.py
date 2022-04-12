
from sisyphus import tk

from i6_experiments.common.datasets import librispeech

for name, path in librispeech.get_ogg_zip_dict().items():
  tk.register_output(f"librispeech/dataset/{name}", path)

# Get Bliss corpus. Same audio format as in ogg_zip, so already there anyway due to how we created the ogg_zip.
bliss_corpus_dict = librispeech.get_bliss_corpus_dict(audio_format="ogg")
bliss_train_corpus = bliss_corpus_dict["train-other-960"]
assert isinstance(bliss_train_corpus, tk.Path)

from i6_core.corpus.convert import CorpusToTxtJob
train_corpus_text = CorpusToTxtJob(bliss_train_corpus, gzip=False).out_txt

from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
spm_train_job = TrainSentencePieceJob(
  training_text=train_corpus_text,
  vocab_size=2000,
  model_type=SentencePieceType.UNIGRAM,
  additional_options={
    "unk_id": 2,  # default is 0
    "bos_id": 1,  # default is 1
    "eos_id": 0  # default is 2
  })
spm_5k = spm_train_job.out_model
tk.register_output("librispeech/sentencepiece-5k.model", spm_5k)

