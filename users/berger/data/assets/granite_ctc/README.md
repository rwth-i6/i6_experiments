# Frozen public Granite CTC prior

`librispeech_train_clean_100_prior.json` is the exact selected public prior,
including label order, blank index, source revision and the 512 sample IDs.
It contains **16,384** log-prior values. The original bytes and SHA256 are
preserved; this is not a newly estimated approximation.

Model: `ibm-granite/granite-speech-5.0-470m-turboctc-nc`, full revision
`1ea89b77a944f4a5ea26e5fcf36273e008594436`.
Source: `openslr/librispeech_asr`, `clean/train.100`, revision
`71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`.
Recipe: 512 utterances; frame-weighted probabilities; 83,686 frames and
6,708.3101875 seconds. Use prior scale 0.1 for the selected transfer condition.
The adjacent metadata records the checksum and inventory size.

The absolute public paths in the JSON are provenance, not runtime requirements.
Load this file from the company checkout and check its `pieces`/`blank` against
the acoustic output inventory. Do not overwrite it with a company-estimated
prior while calling the condition an exact transfer of the public prior.
