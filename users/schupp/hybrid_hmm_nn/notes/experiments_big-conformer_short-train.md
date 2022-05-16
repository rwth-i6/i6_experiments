# List of experiments for the 'big-short' conformer

This model is based on pings best conformer: Here adapted to behavior_version = 12 and i6_core reciepes.

Stats of that model are:

- TODO


## List of experiments


- [x] `baseline_03_big_short`


#### Ablation study


- [x] `baseline_03_big_short+no-aux`
- [x] `baseline_03_big_short+no-frame-stacking`


#### Non determinism 

- [ ] `baseline_03_big_short+pretrain-same-seed=42` x 5
- [ ] `baseline_03_big_short+pretrain-diff-seed=X` x 5


#### Learning rate


- [x] `baseline_03_big_short+shorter-lr-const=10-warmup-10-decay=0.98`
- [x] `baseline_03_big_short+newbob-multi-epoch`


#### Norms


- [x] `baseline_03_big_short+batchnorm`
- [x] `baseline_03_big_short+batchnorm-old-defaults`
- [x] `baseline_03_big_short+groupnorm`
- [ ] `baseline_03_big_short+tfa-groupnorm-g=32`


#### Activations:

- [x] `baseline_03_big_short+conv-act=gelu`
- [x] `baseline_03_big_short+conv-act=relu`
- [x] `baseline_03_big_short+conv-act=relu`


#### Sequence/Chunk order


- [x] `baseline_03_big_short+no-seq-order+no-shuffle`
- [x] `baseline_03_big_short+only-shuffle`
- [x] `baseline_03_big_short+shuffle+order-chunks=1000`


#### Stochastic Depth 


- [x] `baseline_03_big_short+stoch-depth-v2.0-ff-mod`
- [x] `baseline_03_big_short+stoch-depth-v2.0-conv-mod`
- [x] `baseline_03_big_short+stoch-depth-v2.0-att-mod`

- [ ] `baseline_03_big_short+stoch-depth-v2.1-ff-mod`
- [ ] `baseline_03_big_short+stoch-depth-v2.1-conv-mod`
- [ ] `baseline_03_big_short+stoch-depth-v2.1-att-mod`

- [x] `baseline_03_big_short+stoch-depth-v2.0-ff-mod+linear-scale-survival-1.0-0.5`
- [x] `baseline_03_big_short+stoch-depth-v2.0-ff-mod+depth-scale-survival-prob-v1-p=0.2`


#### Squeeze and Exication


- [x] `baseline_03_big_short+se-block-v1.0-ff-mod` 
- [x] `baseline_03_big_short+se-block-v1.0-conv-mod` 
- [x] `baseline_03_big_short+se-block-v1.0-att-mod` 


#### Huge model:


- [x] `baseline_03_big_short+XL`
- [x] `baseline_03_big_short+16blocks+2aux-6-12`

#### Misc

- [x] `baseline_03_big_short+switch-att-conv-mod`


## Other

- [x] Group Norm ( see above )
- [x] Conv activation, Relu, Gelu
- [ ] Stage ration, ( no ff mods in early layers maybe )


## Legend

| Name | Description |
| :----: | :---: |
| `+no-aux` | No auxilary loss, per default there is auxilary CE after block 6 | 
| `+no-frame-stacking` | w/o frame stacking, uses downsampling strides instead |
| `+batchnorm` | Uses batch norm in the convolution module ( default is layernorm ) |
| `+batchnorm-old-defaults` | Uses old batchnorm defaults from `behavior_version = 0` |
| `+stoch-depth-v2.0` | Implementation of stochastic depth that uses shared params, found here. <br/> In eval this multiplies by survival-prob |
| `+stoch-depth-v2.1` | In eval just use as is ( not multiply with survival-prob ) |
| `+linear-scale-survival` | Linearly scales survial-prob from first to last conformer block |
| `+se-block-v1.0` | Se block as in ConvNets, implementation see here TODO |
| `+se-block[...]-ff-mod` | SE in ff mod TODO specify how |
| `+se-block[...]-conv-mod` | SE block after second convolution in conv-mod |
| `+se-block[...]-att-mod` | TODO |
