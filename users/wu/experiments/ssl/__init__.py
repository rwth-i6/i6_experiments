"""
Self-supervised learning (SSL) experiments.

Root package used as the hashing anchor for everything under ``pytorch_networks``
(mirrors ``posterior_hmm``). Contains:

* BEST-RQ self-supervised pretraining on LibriSpeech 960h (raw audio).
* CTC finetuning on train-clean-100 (BPE labels, torch ``nn.functional.ctc_loss``,
  greedy decoding -> sclite WER).

See ``pytorch_networks/best_rq`` (SSL) and ``pytorch_networks/ctc`` (downstream).
"""

PACKAGE = __package__
