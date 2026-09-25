DONE

1. A17 (i): verdict "BASIN SUFFICIENT". PER at ep1/2/4/8 (class_ep8):
   gold-EM: 0.28447327598364125/0.25463263291496263/0.2527993230856015/0.20043435340572557 -> LIFT
   r30-EM: 0.3232999576928501/0.2828486814271612/0.22377097729516288/0.20074460583838669 -> LIFT
   r70-EM: 0.4342405866591454/0.43231138062332536/0.4122070229868848/0.3619630517557467 -> LIFT
   r100-EM: 0.8688421943308419/0.8588520659991539/0.8358764631222677/0.8388943731490622 -> NO LIFT
   No paired rows reported beyond above.
   Phi generative PER (direct/Hungarian) ep8: gold 0.2316941731222634/0.2316941731222634; r30 0.239003031323678/0.239003031323678; r70 0.39262377905018525/0.43152576625126304; r100 0.8584035028629168/0.8583361401145166.

2. A17 (ii): verdict "OBJECTIVE DRIFT". Gold arm generative PER (direct=Hungarian, equal) at sub-epoch 0/4/8/12: 0.19326372515998652 / 0.26641966992253285 / 0.283496126641967 / 0.2949141124957898. Drift PER(12)-PER0(0.193 registered) = 0.1019141124957898.
   Gold S: s_260 at 0/4/8/12: 3.4735117504199167 / 3.264450210138765 / 3.249045239770329 / 3.2408409762482893; s_285: 3.4702351223115904 / 3.2675754296570485 / 3.2523031602076204 / 3.24443676382925.
   r70 arm generative PER (direct/Hungarian) at 0/4/8/12: 0.6082519366790166/0.6281576288312564; 0.3852475581003705/0.4134051869316268; 0.3821488716739643/0.4118558437184237; 0.3925564163017851/0.42142135399124286.
   r70 S s_260 at 0/4/8/12: 4.411324798804284 / 3.3995440872339695 / 3.3043480787906163 / 3.265532943557374; s_285: 4.40843515707539 / 3.4058704572359764 / 3.310582713710992 / 3.271540947873758.
   A14(ii) matched-sub-epoch gold values printed (per "a14" field): direct/Hungarian PER at 0/4/8/12: 0.19326372515998652 / 0.2998989558773998 / 0.31694173122263386 / 0.3277871337150556; a14_drift_last=0.13478713371505557. A14(ii) r70: PER at 0/4/8/12: 0.6082519366790166 direct(0.6281576288312564 Hung.) / 0.4770966655439542 (both direct&Hung.) / 0.47847760188615696 / 0.4822499157965645.

3. Warnings/NaN: none found in read job outputs read. Checkpoints complete: A17(i) all 4 arms x ep1/2/4/8 present (lift json populated); A17(ii) gold and r70 through sub-epoch 12 present (drift json populated, both jobs' `finished`/`finished.run.1` or archived `finished.tar.gz` markers present, no CANNOT_TELL in verdict).

4. Read jobs & paths:
   A17(i): A17BasinLiftReadJob.q0C0J9o54cwW -> /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_a17_jobs/A17BasinLiftReadJob.q0C0J9o54cwW/output/a17_lift.json (registered at output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_em/a17/lift/read/a17_lift.json under project-relocated tree)
   A17(ii): A17AnnealDriftReadJob.jcIFDGis10WJ -> /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_a17_jobs/A17AnnealDriftReadJob.jcIFDGis10WJ/output/a17_drift.json (registered at output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_em/a17/drift/read/a17_drift.json under project-relocated tree)
   Note: the setup dir's own `work`/`output`/`alias` (/e/project1/spell/wu24/2026-07-13_unsupervised) resolve into /e/scratch/spell/wu24/2026-07-13_unsupervised/work, but the a17-specific alias/output tree lives under a second scratch tree, /e/scratch/spell/wu24/project-relocated/2026-07-13_unsupervised/{alias,output}, whose symlinks in turn point into the same /e/scratch/spell/wu24/2026-07-13_unsupervised/work job dirs above.
