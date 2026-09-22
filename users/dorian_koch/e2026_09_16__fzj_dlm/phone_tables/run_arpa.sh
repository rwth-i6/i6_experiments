#!/usr/bin/env bash
#SBATCH --job-name=mfa_arpa
#SBATCH --partition=c23ms
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/hpcwork/tt201262/mfa/fix1h/run_arpa.%j.log
# German 1 h and 9 h aligned with english_us_arpa (trained on LibriSpeech only = the English ASR's own corpus),
# German phones mapped to ARPAbet automatically by PanPhon feature distance (de2arpa.json). Same procedure for
# both budgets, so the 1 h vs 9 h tables differ only in the amount of German audio.
set -uo pipefail
cd /hpcwork/tt201262/mfa
F=/hpcwork/tt201262/mfa/fix1h
export PHONE_MAP=$F/de2arpa.json
python3 $F/remap.py make_dict german_ext.dict $F/german_ext_arpa.dict
python3 $F/remap.py make_dict german_ext9.dict $F/german_ext9_arpa.dict
EM=/work/fix1h/root_models/pretrained_models/acoustic/english_us_arpa.zip
for b in 1:corpus_1_hours:german_ext_arpa.dict:german_ext.dict 9:corpus_9_hours:german_ext9_arpa.dict:german_ext9.dict; do
  IFS=: read h corpus adict gdict <<< "$b"
  echo "=== arpa${h}h start $(date -Is)"
  apptainer exec --bind /hpcwork/tt201262/mfa:/work --env MFA_ROOT_DIR=/work/fix1h/root_arpa$h --env NUMBA_CACHE_DIR=/work/fix1h/numba_cache mfa.sif \
    bash -lc "cd /work && mfa align $corpus fix1h/$adict $EM fix1h/align_arpa${h}h_enlabels --clean -j 16" > $F/arpa${h}h.log 2>&1
  echo "=== arpa${h}h exit=$? $(date -Is) textgrids=$(find $F/align_arpa${h}h_enlabels -name '*.TextGrid' | wc -l)"
  python3 $F/remap.py recover $gdict $F/align_arpa${h}h_enlabels $F/align_arpa${h}h
done
echo "all done $(date -Is)"
