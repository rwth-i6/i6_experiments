Rerun of `conformer_half_orig` performs much worse in my setup:

Config 1 ( in my pipeline ): `/u/schupp/conformer_tf23_rasrV2/output/conformer/batchnorm_ping/conformer_half_orig/returnn.config`
sis config: `/u/schupp/conformer_tf23_rasrV2/config/conformer_half_batchnorm.py` 1zu1 config ( nur pipeline wurde angepasst )

WER 13.2 dev-other ( sepoch 600 ) ( with optimizes am, lm we can get to 13.07 )

---- VS

Config 2 ( in pings pipeline ): `/work/asr3/luescher/hiwis/pzheng/librispeech/transformer_conformer_21_11_10/output/conformer/batchnorm/conformer_half_orig/crnn.config`
Sis config: `/work/asr3/luescher/hiwis/pzheng/librispeech/transformer_conformer_21_11_10/config/conformer_half_batchnorm.py`

WER 10.8 dev-other ( sepoch 600 )

### Differences setup 

mine:
- tf2.3 for training 
- rasr+tf2.3 for decoding

ping:
- tf1.15 for training
- rasr+tf1.15 for decoding

### Only differences in returnn configs ( diff ) `[< me, > ping]`

```
diff output/conformer/batchnorm_ping/conformer_half_orig/returnn.config /work/asr3/luescher/hiwis/pzheng/librispeech/transformer_conformer_21_11_10/output/conformer/batchnorm/conformer_half_orig/crnn.config
```

On sprint dataset:

The reason for this diff is, that I use i6_core, and ping used: CrnnSprintTrainJob from:
/work/asr3/luescher/hiwis/pzheng/librispeech/transformer_conformer_21_11_10/recipe/crnn/sprint_training.py ( which does have the following as defautls )

```
>  '--*.segment-order-sort-by-time-length=true --*.segment-order-sort-by-time-length-chunk-size=-1',  --*.corpus.segment-order-shuffle=true
```


In transform ( the second version is from updated specaugment code )

```
>  max_reps_time = tf.maximum(tf.shape(x)[1] // 100, 1)
< max_reps_time = tf.maximum(tf.shape(x)[1] // (max_len_time or 20), 1)
```

Same here this is part of updated specaugment code

```
<  num = tf.cond(tf.less(tf.shape(x)[axis], max_num), lambda: tf.random.uniform(shape=(n_batch,), minval=min_num, maxval=tf.shape(x)[axis] + 1, dtype=tf.int32), lambda: num)
```

# Other stuff

This rerun doesn't use `behavior_version=12` so that can't be an issue.
I've also checked the `rasr.dev.config` and `rasr.train.config` they also match.

Also checked the recognition.config for the tree search they do match asual. So it can really be an issue with recognition.

e.g.:
`/work/asr3/zeyer/schupp/setups-data/ping_setup_refactor_tf23_work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.WKTa75gcj3SS/work/recognition.config`

