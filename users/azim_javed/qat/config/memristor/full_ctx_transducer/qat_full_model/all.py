from sisyphus import tk
from . import w4_a8, w8_a8


async def py():
    await tk.async_gather(w4_a8.py(), w8_a8.py())
