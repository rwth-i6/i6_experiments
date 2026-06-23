# Important Setup Notes

## G2P
Newer versions of G2P may produce a file `g2p` (without `.py` suffix) inside the python binary directory. However, `i6_core` assumes a py file. To fix this
``` (Bash)
cp [Path_to_python_binaries]/g2p{,.py}
```
Before the g2p import within that file add the following lines:
```
import sys
import os

# --- FIX: Remove this directory from sys.path to prevent shadowing ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != script_dir and p != '']
# ---------------------------------------------------------------------
```
These remove the current directory from the python path, solving an import error that would otherwise happen.