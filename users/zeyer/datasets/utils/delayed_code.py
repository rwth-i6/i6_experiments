"""
delayed code utils
"""

from __future__ import annotations

from i6_core.returnn import CodeWrapper
from sisyphus.delayed_ops import DelayedFormat


class DelayedCodeFormat(DelayedFormat):
    """Delayed code"""

    def get(self) -> CodeWrapper:
        """get"""
        return CodeWrapper(super(DelayedCodeFormat, self).get())
