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

2. Create a new work folder under a "work" file system such as `asr4` and link this as `work` into the Sisyphus setup root (`~/experiments/<setup_name>`).
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

def file_caching(path: str) -> str:
  """file caching"""
  return f'`cf {path}`'

def engine():
  ...

...
```
You might want to copy and adapt this file from someone working in the same environment. 

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
