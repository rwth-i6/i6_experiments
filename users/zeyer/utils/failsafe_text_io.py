"""
Failsafe text I/O wrapper
"""

from typing import TextIO
from io import TextIOBase


class FailsafeTextOutput(TextIOBase):
    def __init__(self, io: TextIO, fallback: TextIO):
        self.io = io
        self.fallback = fallback

    def readable(self):
        return self.io.readable()

    def writable(self):
        return self.io.writable()

    def seekable(self):
        return self.io.seekable()

    def read(self, size=-1):
        return self.io.read(size)

    def write(self, s):
        try:
            return self.io.write(s)
        except OSError:
            self.fallback.write(s)

    def seek(self, offset, whence=0):
        return self.io.seek(offset, whence)

    def tell(self):
        return self.io.tell()

    def flush(self):
        return self.io.flush()

    def close(self):
        try:
            self.io.close()
        except OSError:
            pass  # nothing we can do here
