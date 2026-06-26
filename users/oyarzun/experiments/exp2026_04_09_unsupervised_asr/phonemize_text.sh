#!/usr/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

lg=$1
text_path=$2
target_dir=$3
min_phones=$4
phonemizer=$5
lid_path=$6
sil_prob=$7

if [ -z "$lid_path" ]; then
    lid_path="lid.187.bin"
fi

ph_lg=${lg:l}
if test "$lg" = 'fr'; then
    ph_lg='fr-fr'
elif test "$lg" = 'en'; then
    ph_lg='en-us'
elif test "$lg" = 'pt'; then
    ph_lg='pt-br'
fi

ESPEAK_PATH=''

echo $lg
echo $ph_lg
echo $text_path
echo $target_dir
echo "min phone seen threshold is $min_phones"

mkdir -p $target_dir
echo $target_dir

python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/normalize_and_filter_text.py --lang $lg --fasttext-model $lid_path < $text_path | grep -v '\-\-\-' >! $target_dir/lm.upper.lid.txt
python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/lm.upper.lid.txt --only-source --destdir $target_dir  --padding-factor 1 --dict-only  #--thresholdsrc 2
cut -f1 -d' ' $target_dir/dict.txt | grep -v -x '[[:punct:]]*' | grep -Pv '\d\d\d\d\d+' >! $target_dir/words.txt

if [ -z "$ESPEAK_PATH" ]; then
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/g2p_wrd_to_phn.py --compact < $target_dir/words.txt > $target_dir/phones.txt
else
    # echoing 1 into corpus will prevent the mismatch lines between lexicon and phones in case the phonemizer fails
    one=$(echo "1" | PHONEMIZER_ESPEAK_PATH=$ESPEAK_PATH phonemize -p ' ' -w '' -l $ph_lg --language-switch remove-flags)
    sed 's/$/ 1/' $target_dir/words.txt | PHONEMIZER_ESPEAK_PATH=$ESPEAK_PATH phonemize -o $target_dir/phones.txt -p ' ' -w '' -l $ph_lg -j 70 --language-switch remove-flags
    echo "one is ${one}"
    sed -i "s/${one}$//" $target_dir/phones.txt
fi

paste $target_dir/words.txt $target_dir/phones.txt >! $target_dir/lexicon.lst
python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones.txt --only-source --destdir $target_dir/phones --thresholdsrc $min_phones --padding-factor 1 --dict-only
python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/filter_lexicon.py -d $target_dir/phones/dict.txt < $target_dir/lexicon.lst >! $target_dir/lexicon_filtered.lst
