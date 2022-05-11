# List of experiments for the 'big-short' conformer

This model is based on pings best conformer: Here adapted to behavior_version = 12 and i6_core reciepes.

Stats of that model are:

- TODO


## List of experiments

- [x] `baseline_03_big_short`

#### Ablation study


- [x] `baseline_03_big_short+no-aux`
- [ ] `baseline_03_big_short+no-frame-stacking`

#### Learning rate

- [ ] `baseline_03_big_short+lr-shorter`
- [ ] `baseline_03_big_short+newbob-lr`

#### Batch Norm 

- [x] `baseline_03_big_short+batchnorm`
- [x] `baseline_03_big_short+batchnorm-old-defaults`

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

- [x] `baseline_03_big_short+stoch-depth-v2.0-ff-mod+linear-scale-survival-0.1-0.6`


### Squeeze and Exication TODO

- [ ] `baseline_03_big_short+se-block-v1.0-ff-mod` 
- [ ] `baseline_03_big_short+se-block-v1.0-conv-mod` 
- [ ] `baseline_03_big_short+se-block-v1.0-att-mod` 


## Legend

| Name | Description |
| :----: | :---: |
| `+batchnorm` | Uses batch norm in the convolution module ( default is layernorm ) |
| `+batchnorm-old-defaults` | Uses old batchnorm defaults from `behavior_version = 12` |
| `+stoch-depth-v2.0` | Implementation of stochastic depth that uses shared params, found here. <br/> In eval this multiplies by survival-prob |
| `+stoch-depth-v2.1` | In eval just use as is ( not multiply with survival-prob ) |
| `+linear-scale-survival` | Linearly scales survial-prob from first to last conformer block |
| `+se-block-v1.0` | Se block as in ConvNets, implementation see here ... |
| `+se-block[...]-ff-mod` | SE in ff mod TODO specify how |
| `+se-block[...]-conv-mod` | SE block after second convolution in conv-mod |
| `+se-block[...]-att-mod` | TODO |
