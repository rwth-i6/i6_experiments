"""
Paths to the tokenization scripts with their respective sha256 checksum
"""
from sisyphus import setup_path

# local relative paths
Path = setup_path(__package__)

tokenizer_tts = Path("tokenizer_tts.perl", hash_overwrite="a6a4b68cc9176c85539757bd08f3cda93d26a2b107d36fcd067b5dba98d0e4ee")