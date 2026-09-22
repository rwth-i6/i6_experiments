# German phone tables ("enarpa" prior) — how to rebuild

Reproduces `/e/project1/spell/koch13/tables_2026-09-22/de_{1h_arpa1h,9h_arpa9h}*` (german_xling `_DE_TABLES`).
No audio beyond the pipeline's own: the aligner is MFA `english_us_arpa` (trained on all of LibriSpeech,
292,367 utts / 982 h — the English ASR's corpus); German audio = MLS-de `1_hours` / `9_hours` only.

1. `map_arpa.py "<German phone inventory>" de2arpa.json` — German IPA -> ARPAbet, nearest by PanPhon
   weighted feature edit distance (CMUdict ARPAbet->IPA convention; vowels stress 1 except AH0=ə, ER0=ɚ;
   ties alphabetical). Needs `pip install panphon`. Committed output: `de2arpa.json`.
2. On an x86 host with the MFA 3.4 container (RZ: `/hpcwork/tt201262/mfa/mfa.sif`; FZJ is aarch64 and has
   no native MFA): `run_arpa.sh` — rewrites the German dicts to ARPAbet (`remap.py make_dict`), runs
   `mfa align <corpus> <dict_arpa> english_us_arpa`, restores German labels by position (`remap.py recover`).
   Set `NUMBA_CACHE_DIR` to a writable dir (the container is read-only).
3. On FZJ: `build_de_table_fzj.py <textgrids> <parquet> <german dict> <out_prefix>` — per-German-phone mean
   log-mel (RETURNN `log_mel_filterbank_from_raw`, identical to the English table) + median durations.

Ablations kept for the paper (RZ `/hpcwork/tt201262/mfa/fix1h/`): MFA trained from zero on 1 h (default
topology collapses 21.8% of phones to 10 ms: `min_states` defaults to 1), smaller/no-SAT, monophone,
3-state-minimum topology (`topo3.yaml`), and single-Gaussian EM bootstrap (`table_em.py`).
