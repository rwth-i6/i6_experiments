The starting point (`start`) of the reproduction of the transducer pipeline (still only the first step, teh full sum training) is far off the original config.

Here we will do experiments, both from this start point and also from the original (`orig`) config, towards each other, to compare and find out the relevant differences.

See `sis_config_main` for the main Sis entry point.
It will automatically collect all configs in this directory and run them.
Every config is just a separate Python file where `sis_run_with_prefix` is defined.

---

Observations:

...
