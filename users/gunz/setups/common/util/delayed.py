from sisyphus.delayed_ops import Delayed


class GetWrapper(Delayed):
    def get(self):
        return self.a
