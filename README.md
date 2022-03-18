# RWTH I6 Experiments

This repository contains all public [Sisyphus](https://github.com/rwth-i6/sisyphus) recipes/jobs as well as
Sisyphus pipeline code that is shared among i6 assistants and with other interested users.

This depends on jobs defined in [`i6_core`](https://github.com/rwth-i6/i6_core).
Many jobs make use of
[RETURNN](https://github.com/rwth-i6/returnn),
[returnn-common](https://github.com/rwth-i6/returnn_common)
and [RASR](https://github.com/rwth-i6/rasr).

The repository contains two parts: `common` and `users`:


## `Users`

Here every i6 assistant can upload setup pipelines and custom jobs within his personal folder.
There are no direct restrictions on the structure, but of course it is helpful to keep it organsized
so that other can easily make use of the code.

Please be aware that all code under a user folder should be treated as **NOT STABLE**.

If pipeline code turns out to be more generic,
it should be moved to `common` with a corresponding pull request.
If this pipeline code makes use of additional jobs,
they should be submitted (via PR) to [`i6_core`](https://github.com/rwth-i6/i6_core).


## `Common`

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


# Setup

1. Create the new setup folder in your user directory, typically like `~/setups/<setup_name>` or `~/experiments/<setup_name>`. This will be your new setup root directory.
```
mkdir ~/experiments/<setup_name>
cd ~/experiments/<setup_name>
```

2. Create a new work folder under a "work" file system such as `asr3` and link this as `work` into the Sisyphus setup root (`~/experiments/<setup_name>`).
```
mkdir /work/asr3/<username>/sisyphus_work_dirs/<setup_name>
ln -s /work/asr3/<username>/sisyphus_work_dirs/<setup_name> work
```

`<username>` might be replaced `<assistant_username>/<your_username>` if you are a Hiwi.

3. Create a recipe folder in the Sisyphus setup root (`~/experiments/<setup_name>`) and clone the necessary recipe repositories:
```
mkdir recipe
cd recipe
git clone git@github.com:rwth-i6/i6_core.git
git clone git@github.com:rwth-i6/i6_experiments.git
git clone git@github.com:rwth-i6/returnn_common.git
```

If the access is denied for the Github repositories, you need to add your i6 public ssh key (usually `~/.ssh/id_rsa.pub`) to your github account.
This can be done by pasting the content (displayed with `cat ~/.ssh/id_rsa.pub`) into your Github key settings.

More information on adding keys to a Github account can be found [here](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

5. Create a `config` folder and add a default init file
```
mkdir config
touch config/__init__.py
```

6. Add a main function in the `__init__.py` which will be the root of the full graph:
```
def main():
    print("Starting main Graph")
    # call experiments (sub-graphs) from here
```
 
## PyCharm setup

For setting up a PyCharm correctly, please have a look [here](https://github.com/rwth-i6/i6_core/wiki/Sisyphus-PyCharm-Setup).




