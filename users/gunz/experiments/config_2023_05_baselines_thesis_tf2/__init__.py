from functools import lru_cache

from sisyphus import tk


cache = lru_cache(maxsize=None)


def _clone_returnn() -> tk.Path:
    import i6_core.tools as tools

    clone_r_job = tools.CloneGitRepositoryJob(
        url="https://github.com/rwth-i6/returnn.git",
        # commit="8c26cf6c626d1e10f1f1edda647b921b85ac311e",
        commit="eb6dcbe9d7b2e05d7a013fe77ff6d2ff45f8dc43",  # old commit from ok thesis-baselines
        checkout_folder_name="returnn",
    )
    return clone_r_job.out_repository


@cache
@tk.block("fullsum")
def get_ffnn_a():
    from i6_experiments.users.gunz.experiments.config_2023_05_baselines_thesis_tf2 import (
        config_00_monophone_linear_fullsum_10ms,
    )

    returnn_root = _clone_returnn()
    configs = config_00_monophone_linear_fullsum_10ms.run(returnn_root=returnn_root)

    sys = next((v for v in configs.values()))
    return sys.experiments["fh"]["alignment_job"].out_alignment_bundle


@tk.block("mono")
def mono():
    from i6_experiments.users.gunz.experiments.config_2023_05_baselines_thesis_tf2 import (
        config_10_monophone,
    )

    returnn_root = _clone_returnn()
    config_10_monophone.run(returnn_root=returnn_root, additional_alignments=[(get_ffnn_a(), "10ms-FF-v8")])


@tk.block("di")
def di():
    from i6_experiments.users.gunz.experiments.config_2023_05_baselines_thesis_tf2 import (
        config_11_diphone,
    )

    returnn_root = _clone_returnn()
    config_11_diphone.run(returnn_root=returnn_root, additional_alignments=[(get_ffnn_a(), "10ms-FF-v8")])


@tk.block("di_joint")
def di_joint():
    from i6_experiments.users.gunz.experiments.config_2023_05_baselines_thesis_tf2 import (
        config_12_diphone_joint_softmax,
    )

    returnn_root = _clone_returnn()
    config_12_diphone_joint_softmax.run(returnn_root=returnn_root)


@tk.block("tri")
def tri():
    from i6_experiments.users.gunz.experiments.config_2023_05_baselines_thesis_tf2 import (
        config_13_triphone,
    )

    returnn_root = _clone_returnn()
    config_13_triphone.run(returnn_root=returnn_root, additional_alignments=[(get_ffnn_a(), "10ms-FF-v8")])


@tk.block("blstm_mono")
def blstm_mono():
    from i6_experiments.users.gunz.experiments.config_2023_05_baselines_thesis_tf2 import (
        config_30_blstm_monophone,
    )

    returnn_root = _clone_returnn()
    config_30_blstm_monophone.run(returnn_root=returnn_root)


@tk.block("blstm_di")
def blstm_di():
    from i6_experiments.users.gunz.experiments.config_2023_05_baselines_thesis_tf2 import (
        config_31_blstm_diphone,
    )

    returnn_root = _clone_returnn()
    config_31_blstm_diphone.run(returnn_root=returnn_root)


@tk.block("blstm_tri")
def blstm_tri():
    from i6_experiments.users.gunz.experiments.config_2023_05_baselines_thesis_tf2 import (
        config_32_blstm_triphone,
    )

    returnn_root = _clone_returnn()
    config_32_blstm_triphone.run(returnn_root=returnn_root)


@tk.block("cart")
def cart():
    from i6_experiments.users.gunz.experiments.config_2023_05_baselines_thesis_tf2 import (
        config_20_blstm_cart,
        config_21_conf_cart,
    )

    returnn_root = _clone_returnn()
    config_20_blstm_cart.run(returnn_root=returnn_root)
    config_21_conf_cart.run(returnn_root=returnn_root)


@tk.block("plot alignments")
def plot_a():
    from i6_experiments.users.gunz.experiments.config_2023_05_baselines_thesis_tf2 import (
        config_90_alignment_plots,
    )

    config_90_alignment_plots.run()


def main():
    mono()
    di()
    di_joint()
    tri()

    blstm_mono()
    blstm_di()
    blstm_tri()

    cart()

    plot_a()
