from functools import cache
from sisyphus import tk
from typing import Any, Dict, Optional


def _clone_returnn_safe() -> tk.Path:
    import i6_core.tools as tools

    clone_r_job = tools.CloneGitRepositoryJob(
        url="https://github.com/rwth-i6/returnn.git",
        commit="eb6dcbe9d7b2e05d7a013fe77ff6d2ff45f8dc43",
        checkout_folder_name="returnn",
    )
    return clone_r_job.out_repository


@cache
def run_linear_a():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_00_monophone_linear_fullsum,
    )

    returnn_root = _clone_returnn_safe()
    exps = config_00_monophone_linear_fullsum.run(returnn_root=returnn_root)
    return exps


def get_n_linear_a(t_step, t_scale=0.3):
    exps = run_linear_a()
    target_s = next((s for e, s in exps.items() if e.output_time_step == t_step and e.bw_transition_scale == t_scale))
    return target_s.experiments["fh"]["alignment_job"].out_alignment_bundle


@tk.block("mono linear fullsum 40ms")
def get_30ms_linear_a(t_scale=0.3):
    return get_n_linear_a(t_step=0.03, t_scale=t_scale)


@tk.block("mono linear fullsum 40ms")
def get_40ms_linear_a(t_scale=0.3):
    return get_n_linear_a(t_step=0.04, t_scale=t_scale)


@tk.block("mono linear fullsum 60ms")
def get_60ms_linear_a(t_scale=0.3):
    return get_n_linear_a(t_step=0.06, t_scale=t_scale)


@cache
def run_tdnn_a():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_01_monophone_tdnn_fullsum,
    )

    returnn_root = _clone_returnn_safe()
    exps = config_01_monophone_tdnn_fullsum.run(returnn_root=returnn_root)
    return exps


def get_n_tdnn_a(t_step):
    exps = run_tdnn_a()
    target_s = next((s for e, s in exps.items() if e.output_time_step == t_step))
    return target_s.experiments["fh"]["alignment_job"].out_alignment_bundle


@tk.block("mono tdnn fullsum")
def get_30ms_tdnn_a():
    return get_n_tdnn_a(t_step=0.03)


@tk.block("mono tdnn fullsum")
def get_40ms_tdnn_a():
    return get_n_tdnn_a(t_step=0.04)


@tk.block("forced 40ms triphonized")
def get_40ms_wei_a():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_02_wei_align_40ms,
    )

    return config_02_wei_align_40ms.run()


@cache
def align_tinas_models():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_03b_monophone_blstm_fullsum_tina,
    )

    returnn_root = _clone_returnn_safe()
    exps = config_03b_monophone_blstm_fullsum_tina.run(returnn_root=returnn_root)
    return exps


@cache
def run_blstm_a():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_03_monophone_blstm_fullsum,
    )

    returnn_root = _clone_returnn_safe()
    exps = config_03_monophone_blstm_fullsum.run(returnn_root=returnn_root)
    return exps


def get_n_blstm_a(
    feature_stacking: bool,
    t_step: float,
    transition_scale: float,
    adapted_tdps: Optional[bool],
):
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new.config import (
        ALIGN_30MS_BLSTM_MP,
        ALIGN_40MS_BLSTM_MP,
    )
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new.config_03_monophone_blstm_fullsum import (
        Experiment,
    )
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new.config_03b_monophone_blstm_fullsum_tina import (
        Experiment as TinaE,
    )

    via_dict = {
        (False, 30 / 1000, 0.3, False): tk.Path(ALIGN_30MS_BLSTM_MP, cached=True),
        (False, 30 / 1000, 0.3, None): tk.Path(ALIGN_30MS_BLSTM_MP, cached=True),
        (False, 40 / 1000, 0.3, False): tk.Path(ALIGN_40MS_BLSTM_MP, cached=True),
        (False, 40 / 1000, 0.3, None): tk.Path(ALIGN_40MS_BLSTM_MP, cached=True),
    }

    result = via_dict.get((feature_stacking, t_step, transition_scale, adapted_tdps), None)
    if result is not None:
        return result

    exps_tina: Dict[TinaE, Any] = align_tinas_models()
    if t_step == 30 / 1000 and feature_stacking and (adapted_tdps is None or adapted_tdps == False):
        target_s = next(iter(exps_tina.values()))
        return target_s.experiments["fh"]["alignment_job"].out_alignment_bundle

    exps: Dict[Experiment, Any] = run_blstm_a()
    target_s = next(
        (
            s
            for e, s in exps.items()
            if e.output_time_step == t_step
            and e.bw_transition_scale == transition_scale
            and (adapted_tdps is None or e.adapt_transition_model_to_ss == adapted_tdps)
            and (not feature_stacking or "fs:" in e.subsampling_approach)
        )
    )
    return target_s.experiments["fh"]["alignment_job"].out_alignment_bundle


@tk.block("viterbi_30ms")
def viterbi_30ms():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_10_monophone_mpc1x3_30ms,
        config_11_diphone_mpc1x3_30ms,
        config_12_triphone_mpc1x3_30ms,
    )

    returnn_root = _clone_returnn_safe()
    lin_a = get_30ms_linear_a()
    tdnn_a = get_30ms_tdnn_a()
    alignments = [(lin_a, "30ms-FF-v8"), (tdnn_a, "30ms-TD-v8")]
    config_10_monophone_mpc1x3_30ms.run(returnn_root=returnn_root, alignments=alignments)
    config_11_diphone_mpc1x3_30ms.run(returnn_root=returnn_root, alignments=alignments)
    config_12_triphone_mpc1x3_30ms.run(
        returnn_root=returnn_root,
        alignments=[(*a, run) for a, run in zip(alignments, [True, False])],
    )


@tk.block("viterbi_40ms")
def e21():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_21e_diphone_pp_mpc1x4_40ms,
    )

    returnn_root = _clone_returnn_safe()
    config_21e_diphone_pp_mpc1x4_40ms.run(returnn_root=returnn_root)


@tk.block("viterbi_40ms")
def viterbi_40ms():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config,
        config_20_monophone_mpc1x4_40ms,
        config_20b_monophone_zhou_a_mpc1x4_40ms,
        config_21_diphone_mpc1x4_40ms,
        config_21b_diphone_multi_mpc1x4_40ms,
        config_21c_diphone_zhou_40ms,
        config_21d_diphone_fs_mpc1x4_40ms,
        config_21e_diphone_pp_mpc1x4_40ms,
        config_21f_diphone_zhou_compare_mpc1x4_40ms,
        config_21g_diphone_zhou_a_mpc1x4_40ms,
        config_22_triphone_mpc1x4_40ms,
        config_22b_triphone_multi_mpc1x4_40ms,
        config_22c_triphone_zhou_a_mpc1x4_40ms,
    )

    returnn_root = _clone_returnn_safe()

    lin_a = get_40ms_linear_a()
    # tdnn_a = get_40ms_tdnn_a()
    for a, a_name, run_tri_lrs in [
        (lin_a, "40ms-FF-v8", True),
        # (tdnn_a, "40ms-TD-v8", False),
    ]:
        exp, sys = config_20_monophone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)
        config_21_diphone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)
        config_21b_diphone_multi_mpc1x4_40ms.run(
            returnn_root=returnn_root,
            alignment=a,
            a_name=a_name,
            init_from_system=sys,
        )
        config_21c_diphone_zhou_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)
        config_21d_diphone_fs_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)
        config_21f_diphone_zhou_compare_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)
        config_22_triphone_mpc1x4_40ms.run(
            returnn_root=returnn_root,
            alignment=a,
            a_name=a_name,
            run_additional_lrs=run_tri_lrs,
        )
        config_22b_triphone_multi_mpc1x4_40ms.run(
            returnn_root=returnn_root,
            alignment=a,
            a_name=a_name,
            init_from_system=sys,
        )

    a = tk.Path(config.ALIGN_BLSTM_40MS)
    a_name = "40ms-B-v6"
    config_20_monophone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)
    config_21_diphone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)
    config_22_triphone_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=a,
        a_name=a_name,
        run_additional_lrs=False,
    )

    config_21e_diphone_pp_mpc1x4_40ms.run(returnn_root=returnn_root)

    # au_wei_a = get_40ms_wei_a()
    # wei_a_name = "40ms-WZ"
    # config_20b_monophone_zhou_a_mpc1x4_40ms.run(
    #     returnn_root=returnn_root, alignment=au_wei_a, a_name=wei_a_name
    # )
    # config_21g_diphone_zhou_a_mpc1x4_40ms.run(
    #     returnn_root=returnn_root, alignment=au_wei_a, a_name="40ms-WZ"
    # )
    # config_22c_triphone_zhou_a_mpc1x4_40ms.run(
    #     returnn_root=returnn_root, alignment=au_wei_a, a_name=wei_a_name
    # )

    lin_a_phmms = get_40ms_linear_a(t_scale=0.0)
    config_20_monophone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=lin_a_phmms, a_name="40ms-FFs-v8")
    config_21_diphone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=lin_a_phmms, a_name="40ms-FFs-v8")
    config_22_triphone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=lin_a_phmms, a_name="40ms-FFs-v8")


@tk.block("viterbi_60ms")
def viterbi_60ms():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_30_monophone_mpc2x3_60ms,
    )

    returnn_root = _clone_returnn_safe()

    a = get_60ms_linear_a()
    a_name = "60ms-FF-v8"
    config_30_monophone_mpc2x3_60ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)


@tk.block("alignment eval")
def alignment_eval():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_91_alignment_eval,
    )

    config_91_alignment_eval.run()


def the_plan():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_03_monophone_blstm_fullsum,
        config_11_diphone_mpc1x3_30ms,
        config_11b_diphone_fs1x3_30ms,
        config_21_diphone_mpc1x4_40ms,
        config_21h_diphone_fs1x4_40ms,
        config_21i_diphone_ss_variations_40ms,
    )

    returnn_root = _clone_returnn_safe()

    # FS vs. MP

    phmm_30ms_mp_a = get_n_blstm_a(feature_stacking=False, transition_scale=0.3, adapted_tdps=False, t_step=30 / 1000)
    phmm_30ms_fs_a = get_n_blstm_a(feature_stacking=True, transition_scale=0.3, adapted_tdps=False, t_step=30 / 1000)

    config_11_diphone_mpc1x3_30ms.run(
        returnn_root=returnn_root, alignments=[(phmm_30ms_mp_a, "30ms-Bmp-v8"), (phmm_30ms_fs_a, "30ms-Bfs-v8")]
    )
    config_11b_diphone_fs1x3_30ms.run(
        returnn_root=returnn_root, alignments=[(phmm_30ms_mp_a, "30ms-Bmp-v8"), (phmm_30ms_fs_a, "30ms-Bfs-v8")]
    )

    phmm_40ms_mp_a = get_n_blstm_a(feature_stacking=False, transition_scale=0.3, adapted_tdps=False, t_step=40 / 1000)
    phmm_40ms_fs_a = get_n_blstm_a(feature_stacking=True, transition_scale=0.3, adapted_tdps=False, t_step=40 / 1000)

    config_21_diphone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=phmm_40ms_mp_a, a_name="40ms-Bmp-v8")
    config_21_diphone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=phmm_40ms_fs_a, a_name="40ms-Bfs-v8")
    config_21h_diphone_fs1x4_40ms.run(returnn_root=returnn_root, alignment=phmm_40ms_mp_a, a_name="40ms-Bmp-v8")
    config_21h_diphone_fs1x4_40ms.run(returnn_root=returnn_root, alignment=phmm_40ms_fs_a, a_name="40ms-Bfs-v8")

    # P-HMM ADAPTED TDPs

    phmm_40ms_mp_adapted_a = get_n_blstm_a(
        feature_stacking=False, transition_scale=0.3, adapted_tdps=True, t_step=40 / 1000
    )
    config_21_diphone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=phmm_40ms_mp_adapted_a, a_name="40ms-Ba-v8")

    # P-HMM-S

    phmms_30ms_mp_a = get_n_blstm_a(feature_stacking=False, transition_scale=0.0, adapted_tdps=None, t_step=30 / 1000)
    phmms_40ms_mp_a = get_n_blstm_a(feature_stacking=False, transition_scale=0.0, adapted_tdps=None, t_step=40 / 1000)

    config_11_diphone_mpc1x3_30ms.run(returnn_root=returnn_root, alignments=[(phmms_30ms_mp_a, "30ms-Bs-v8")])
    config_21_diphone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=phmms_40ms_mp_a, a_name="40ms-Bs-v8")

    # P-HMM-S FF-NN

    phmms_30ms_ffnn_a = get_30ms_linear_a(t_scale=0.0)
    phmms_40ms_ffnn_a = get_40ms_linear_a(t_scale=0.0)

    config_11_diphone_mpc1x3_30ms.run(returnn_root=returnn_root, alignments=[(phmms_30ms_ffnn_a, "30ms-FFs-v8")])
    config_21_diphone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=phmms_40ms_ffnn_a, a_name="40ms-FFs-v8")
    config_21i_diphone_ss_variations_40ms.run(
        returnn_root=returnn_root, alignment=phmms_40ms_ffnn_a, a_name="40ms-FFs-v8"
    )

    # P-HMM FF-NN

    phmm_40ms_ffnn_a = get_40ms_linear_a()
    config_21i_diphone_ss_variations_40ms.run(
        returnn_root=returnn_root, alignment=phmm_40ms_ffnn_a, a_name="40ms-FF-v8"
    )


def main():
    the_plan()

    viterbi_30ms()
    viterbi_40ms()
    viterbi_60ms()

    alignment_eval()
