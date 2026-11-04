"""Model-combination experiment entrypoints, with shared data/runtime elsewhere."""


def py():
    # Historical module identity also preserves monkeypatching and old launchers.
    from i6_experiments.users.berger.configs.baselines import main as run

    return run()


def main():
    return py()
