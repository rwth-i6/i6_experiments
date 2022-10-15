The starting point (`start`) of the reproduction of the transducer pipeline (still only the first step, teh full sum training) is far off the original config.

Here we will do experiments, both from this start point and also from the original (`orig`) config, towards each other, to compare and find out the relevant differences.

See `sis_config_main` for the main Sis entry point.
It will automatically collect all configs in this directory and run them.
Every config is just a separate Python file where `sis_run_with_prefix` is defined.

---

Differences/Observations:

- native vs pure TF
- pretrain: growing, repeat, etc.
- adam vs nadam
- accum grad vs larger batch size
- lr schedule?
- lr schedule error key?
- batching "random"?
- model really the same?
- L2, dropout, etc?

- batch_size was 5000 (and grad accum 3) in the first 20 subepochs in the original config before it was edited...

- joint CTC? -> both no.
- correct dataset? normalized? kazuki version?
  -> both use `/work/asr3/irie/data/switchboard/corpora`, same BPE

...
