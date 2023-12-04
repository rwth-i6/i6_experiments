from sisyphus.delayed_ops import DelayedBase


class GetWrapper(DelayedBase):
    def get(self):
        return self.a
