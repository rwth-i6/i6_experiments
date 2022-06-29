# RWTH I6 Experiments

This repository contains all public [Sisyphus](https://github.com/rwth-i6/sisyphus) recipes/jobs as well as
Sisyphus pipeline code that is shared among i6 assistants and with other interested users.

This depends on jobs defined in [`i6_core`](https://github.com/rwth-i6/i6_core).
Many jobs make use of
[RETURNN](https://github.com/rwth-i6/returnn),
[returnn-common](https://github.com/rwth-i6/returnn_common)
and [RASR](https://github.com/rwth-i6/rasr).

You can combine i6_core, i6_experiments, returnn_common, etc
in one Sisyphus setup to run whole pipelines
and do new experiments.
See [setup](#setup) below.

This `i6_experiments` repository contains two parts: `common` and `users`:


## `Users`

Here every i6 assistant can upload setup pipelines and custom jobs within his personal folder.
There are no direct restrictions on the structure, but of course it is helpful to keep it organsized
so that other can easily make use of the code.

Please be aware that all code under a user folder should be treated as **NOT STABLE**.

If pipeline code turns out to be more generic,
it should be moved to [`common`](#common) with a corresponding pull request.
If this pipeline code makes use of additional jobs,
they should be submitted (via PR) to [`i6_core`](https://github.com/rwth-i6/i6_core).


## `Common`

This part is for submitting generic pipelines that are shared among different users.
This should be stable in the API and also in hashes.
Any code changed here should go through a PR, or be only about changes that do not change
the existing API or the resulting pipeline.

There can be exceptions for completely novel sub packages
which are still under development.
Those sub packages are clearly marked as *work-in-progress*
in a corresponding Readme.

As this is still under construction, what kind of pipelines are grouped here is still not decided.
So far there is:
 - `datasets` which will contain pipelines related to specific corpora.
   - This code is protected with hash-checks and thus can be treated as safe and stable 
 - `setups` which contain corpus independed pipeline helpers for specific systems (e.g. Hybrid-ASR)
   - This code is still under developement. While the resulting graph should be stable, the API definitely is not. 
 - `helpers`, additional code which does not directly fit into the categories above. This code is considered stable.

For i6 users: if pipelines have a related `export` function,
pre-computed jobs may be found under `/work/common/asr/work` and can be imported via the console via `tk.import_work_directory()`.
For further details on the exact location please look into the export function docstring.


# Setup

First you follow 

## Directory setup

1. Create the new setup folder in your user directory, typically like `~/setups/<setup_name>` or `~/experiments/<setup_name>`. This will be your new setup root directory.
```
mkdir ~/experiments/<setup_name>
cd ~/experiments/<setup_name>
```

Optional: The setup directory should be a Git repo itself,
to keep track of the changes. You can do now:

```
git init .
edit README.md  # write a short description about your setup
git add README.md
git commit . -m initial

# some initial content for gitignore
cat << EOF > .gitignore
/output
/alias
.*.swp
*.pyc
__pycache__
.idea
*.history*
.directory
EOF
git add .gitignore
git commit .gitignore -m gitignore
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

If the access is denied for the Github repositories, you need to add your i6 public ssh key (usually `~/.ssh/id_rsa.pub`) to your Github account.
This can be done by pasting the content (displayed with `cat ~/.ssh/id_rsa.pub`) into your Github key settings.
More information on adding keys to a Github account can be found [here](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

4. Create a `config` folder and add a default init file
```
mkdir config
touch config/__init__.py
```

5. Add a main function in the `config/__init__.py` which will be the root of the full graph:
```
def main():
    print("Starting main Graph")
    # call experiments (sub-graphs) from here
```

You can also check out the [Sisyphus structure page](https://sisyphus-workflow-manager.readthedocs.io/en/latest/structure.html) for more information.

6. Add a `settings.py` with Sisyphus settings. For example:

```python
VERBOSE_TRACEBACK_TYPE = 'better_exchook'
USE_SIGNAL_HANDLERS = True
```

This file is loaded via sisyphus.global_settings.py,
update_global_settings_from_file specifically.
See [Sisyphus documentation](https://sisyphus-workflow-manager.readthedocs.io/).

8. Optional: Setup Sisyphus and maybe other setup-wide tools.

```shell
mkdir -p tools
cd tools
```

You might have installed Sisyphus already elsewhere.
Otherwise, you might want to clone it here as well:

```shell
git clone git@github.com:rwth-i6/sisyphus.git
ln -s tools/sisyphus/sis ../sis
```

For tools like RETURNN, RASR, and others,
you can choose to have them setup-wide, and not be part of the hash
(under the assumption that different versions should not change the outcome).
In that case, for example:
```shell
git clone git@github.com:rwth-i6/returnn.git
```
In the `settings.py` file, you then can add sth like:
```python
import os
import sys

_root_dir = os.path.dirname(os.path.abspath(__file__))

RETURNN_PYTHON_EXE = sys.executable
RETURNN_ROOT = _root_dir + "/tools/returnn"
sys.path.insert(0, RETURNN_ROOT)
```

Alternatively, you can also use `CloneGitRepositoryJob` to have that explicit as part of the recipe pipeline, to clone RETURNN and maybe other tools.
 
## PyCharm setup

For setting up PyCharm correctly, please have a look [here](https://github.com/rwth-i6/i6_core/wiki/Sisyphus-PyCharm-Setup).
