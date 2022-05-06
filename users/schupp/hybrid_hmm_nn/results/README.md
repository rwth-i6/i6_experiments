# Results of Conformer experiments


### Baseline comparison:

`./baseline_comparison_vertical.html` See the table:

[View rendered](https://htmlpreview.github.io/?https://github.com/rwth-i6/i6_experiments/blob/main/users/schupp/hybrid_hmm_nn/results/baseline_comparison_vertical.html)


### Setup 1 ( until 25.04.22 )

- Returnn ( always neewest version ) + `behavior_version = 12`
- Rasr + tf2.3
- Based on Lueschers hybrid hmm pipeline ( see `pipeline/librispeech_hybrid_system.py` )

RESUTLS: `./results_conformer_tf2.3_rasr+tf2.3_rt-bhv\=12.md` ( or as csv `[--"--].csv`)

Table also links the configs, they can only be acessed from the i6 file system.

### Setup 2 ( after 25.04.22 )

- Returnn ( always neewest version ) + `behavior_version = 12`
- Rasr + tf2.3
- Refactor old hybrid pipeline, and setup ( see `pipeline/hybrid_job_dispatcher.py` + `pipeline/librispeech_hybrid_tim_refactor.py` )


> (WIP) results comming soon

( RESUTLS: `./setup2_results_conformer_tf2.3_rasr+tf2.3_rt-bhv\=12.md` )