# RWTH I6 Experiments

This repository contains all public [Sisyphus](https://github.com/rwth-i6/sisyphus) recipes/jobs as well as
Sisyphus pipeline code that is shared among i6 assistants and with other interested users.
The repository contains two parts: `common` and `users`:

`Users`
-------

Here every i6 assistant can upload setup pipelines and custom jobs within his personal folder.
There are no direct restrictions on the structure, but of course it is helpful to keep it organsized
so that other can easily make use of the code.

Please be aware that all code under a user folder should be treated as **NOT STABLE**.

If pipeline code turns out to be more generic,
it should be moved to `common` with a corresponding pull request.
If this pipeline code makes use of additional jobs,
they should be submitted (via PR) to [`i6_core`](https://github.com/rwth-i6/i6_core).


`Common`
--------

This part is for submitting generic pipelines that are shared among different users.
Any code changed here should go through a PR, or be only about changes that do not change
the existing API or the resulting pipeline.

As this is still under construction, what kind of pipelines are grouped here is still not decided.
So far there is:
 - `datasets` which will contain pipelines related to specific corpora.
   - This code is protected with hash-checks and thus can be treated as safe and stable 
 - `setups` which contain corpus independed pipeline helpers for specific systems (e.g. Hybrid-ASR)
   - This code is still under developement. While the resulting graph should be stable, the API definitely is not. 

For i6 users: if pipelines have a related `export` function,
pre-computed jobs may be found under `/work/common/asr` and can be imported via the console via `tk.import_work_directory()`.
For further details on the exact location please look into the export function docstring.






