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


@tk.block("mono linear fullsum 80ms")
def get_80ms_linear_a(t_scale=0.3):
    return get_n_linear_a(t_step=0.08, t_scale=t_scale)


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
def align_previous_blstm_models():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_03b_monophone_blstm_fullsum_previous,
    )

    returnn_root = _clone_returnn_safe()
    exps = config_03b_monophone_blstm_fullsum_previous.run(returnn_root=returnn_root)
    return exps


@cache
def run_blstm_a():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_03_monophone_blstm_fullsum,
    )

    returnn_root = _clone_returnn_safe()
    exps = config_03_monophone_blstm_fullsum.run(returnn_root=returnn_root)
    return exps


def run_all_blstm_a():
    run_blstm_a()
    align_previous_blstm_models()


def get_n_blstm_a(
    feature_stacking: bool,
    t_step: float,
    transition_scale: float,
    prior_scale: Optional[float],
    adapted_tdps: Optional[bool],
):
    from i6_core.mm import AlignmentJob
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new.config_03_monophone_blstm_fullsum import (
        Experiment,
    )
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new.config_03b_monophone_blstm_fullsum_previous import (
        Experiment as PreviousE,
    )

    previous_exps: Dict[PreviousE, Any] = align_previous_blstm_models()
    previous_as_dict = {
        (
            "fs" in exp.name,
            exp.t_step,
            0.3,
            exp.p_c,
            False,
        ): s
        for exp, s in previous_exps.items()
    }
    current_exps: Dict[Experiment, Any] = run_blstm_a()
    current_as_dict = {
        (
            "fs" in exp.subsampling_approach,
            exp.output_time_step,
            exp.bw_transition_scale,
            exp.alignment_prior_scale_center,
            exp.adapt_transition_model_to_ss,
        ): s
        for exp, s in current_exps.items()
    }
    all_exps = {**current_as_dict, **previous_as_dict}

    exp = next(
        (
            s
            for (fs, t_step_, bw_t, p_c, adapt), s in all_exps.items()
            if fs == feature_stacking
            and t_step_ == t_step
            and bw_t == transition_scale
            and (prior_scale is None or p_c == prior_scale)
            and (adapted_tdps is None or adapt == adapted_tdps)
        )
    )
    a_job: AlignmentJob = exp.experiments["fh"]["alignment_job"]
    bundle = a_job.out_alignment_bundle
    print(
        f"({feature_stacking}, {t_step}, {transition_scale}, {prior_scale}, {adapted_tdps}) -> {bundle}, {a_job.get_aliases()}"
    )
    return bundle


@tk.block("viterbi_30ms")
def viterbi_30ms():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_10_monophone_mpc1x3_30ms,
        config_11_diphone_mpc1x3_30ms,
        config_12_triphone_mpc1x3_30ms,
    )

    returnn_root = _clone_returnn_safe()
    lin_a = get_30ms_linear_a()
    # tdnn_a = get_30ms_tdnn_a()
    # alignments = [(lin_a, "30ms-FF-v8"), (tdnn_a, "30ms-TD-v8")]
    alignments = [(lin_a, "30ms-FF-v8")]
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
        config_21c_diphone_zhou_40ms,
        config_21d_diphone_fs_mpc1x4_40ms,
        config_21e_diphone_pp_mpc1x4_40ms,
        config_21f_diphone_zhou_compare_mpc1x4_40ms,
        config_21g_diphone_zhou_a_mpc1x4_40ms,
        config_22_triphone_mpc1x4_40ms,
        config_22c_triphone_zhou_a_mpc1x4_40ms,
    )

    returnn_root = _clone_returnn_safe()

    lin_a = get_40ms_linear_a()
    # tdnn_a = get_40ms_tdnn_a()
    for a, a_name, run_tri_lrs in [
        (lin_a, "40ms-FF-v8", True),
        # (tdnn_a, "40ms-TD-v8", False),
    ]:
        config_20_monophone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)
        config_21_diphone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)
        config_21c_diphone_zhou_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)
        config_21d_diphone_fs_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)
        config_21f_diphone_zhou_compare_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)
        config_22_triphone_mpc1x4_40ms.run(
            returnn_root=returnn_root,
            alignment=a,
            a_name=a_name,
            run_additional_lrs=run_tri_lrs,
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
        config_20_monophone_mpc1x4_40ms,
        config_21_diphone_mpc1x4_40ms,
        config_21b_diphone_multi_mpc1x4_40ms,
        config_21h_diphone_fs1x4_40ms,
        config_21i_diphone_ss_variations_40ms,
        config_21j_diphone_realign_mpc1x4_40ms,
        config_22b_triphone_multi_mpc1x4_40ms,
        config_31_diphone_mpc2x3_60ms,
        config_41_diphone_mpc2x4_80ms,
    )

    returnn_root = _clone_returnn_safe()

    # FS vs. MP

    phmm_30ms_mp_p0_3_a = get_n_blstm_a(
        feature_stacking=False,
        transition_scale=0.3,
        adapted_tdps=False,
        t_step=30 / 1000,
        prior_scale=0.3,
    )
    phmm_30ms_mp_p0_6_a = get_n_blstm_a(
        feature_stacking=False,
        transition_scale=0.3,
        adapted_tdps=False,
        t_step=30 / 1000,
        prior_scale=0.6,
    )
    phmm_30ms_fs_p0_3_a = get_n_blstm_a(
        feature_stacking=True,
        transition_scale=0.3,
        adapted_tdps=False,
        t_step=30 / 1000,
        prior_scale=0.3,
    )
    phmm_30ms_fs_p0_6_a = get_n_blstm_a(
        feature_stacking=True,
        transition_scale=0.3,
        adapted_tdps=False,
        t_step=30 / 1000,
        prior_scale=0.6,
    )

    config_11_diphone_mpc1x3_30ms.run(
        returnn_root=returnn_root,
        alignments=[
            (phmm_30ms_mp_p0_3_a, "30ms-Bmp-pC0.3"),
            (phmm_30ms_mp_p0_6_a, "30ms-Bmp-pC0.6"),
            (phmm_30ms_fs_p0_3_a, "30ms-Bfs-pC0.3"),
            (phmm_30ms_fs_p0_6_a, "30ms-Bfs-pC0.6"),
        ],
    )
    config_11b_diphone_fs1x3_30ms.run(
        returnn_root=returnn_root,
        alignments=[
            (phmm_30ms_mp_p0_3_a, "30ms-Bmp-pC0.3"),
            (phmm_30ms_mp_p0_6_a, "30ms-Bmp-pC0.6"),
            (phmm_30ms_fs_p0_3_a, "30ms-Bfs-pC0.3"),
            (phmm_30ms_fs_p0_6_a, "30ms-Bfs-pC0.6"),
        ],
    )

    phmm_40ms_mp_p0_0_a = get_n_blstm_a(
        feature_stacking=False,
        transition_scale=0.3,
        adapted_tdps=False,
        t_step=40 / 1000,
        prior_scale=0.0,
    )
    phmm_40ms_mp_p0_3_a = get_n_blstm_a(
        feature_stacking=False,
        transition_scale=0.3,
        adapted_tdps=False,
        t_step=40 / 1000,
        prior_scale=0.3,
    )
    phmm_40ms_mp_p0_6_a = get_n_blstm_a(
        feature_stacking=False,
        transition_scale=0.3,
        adapted_tdps=False,
        t_step=40 / 1000,
        prior_scale=0.6,
    )
    phmm_40ms_ffnn_a = get_40ms_linear_a()

    config_21_diphone_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=phmm_40ms_mp_p0_0_a,
        a_name="40ms-Bmp-pC0.0",
    )
    config_21_diphone_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=phmm_40ms_mp_p0_3_a,
        a_name="40ms-Bmp-pC0.3",
    )
    config_21_diphone_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=phmm_40ms_mp_p0_6_a,
        a_name="40ms-Bmp-pC0.6",
    )
    di_exps = config_21_diphone_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=phmm_40ms_ffnn_a,
        a_name="40ms-FF-v8",
    )
    config_22b_triphone_multi_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=phmm_40ms_ffnn_a,
        a_name="40ms-FF-v8",
        init_from_system=next(iter(di_exps.values())),
    )
    config_21j_diphone_realign_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=phmm_40ms_ffnn_a,
        a_name="40ms-FF-v8",
    )

    # phmm_40ms_fs_a = get_n_blstm_a(
    #     feature_stacking=True, transition_scale=0.3, adapted_tdps=False, t_step=40 / 1000, prior_scale=0.0
    # )
    # config_21_diphone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=phmm_40ms_fs_a, a_name="40ms-Bfs-pC0.0")
    # config_21h_diphone_fs1x4_40ms.run(returnn_root=returnn_root, alignment=phmm_40ms_fs_a, a_name="40ms-Bfs-pC0.0")

    # P-HMM ADAPTED TDPs

    phmm_40ms_mp_adapted_a = get_n_blstm_a(
        feature_stacking=False,
        transition_scale=0.3,
        adapted_tdps=True,
        t_step=40 / 1000,
        prior_scale=None,
    )
    config_21_diphone_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=phmm_40ms_mp_adapted_a,
        a_name="40ms-Ba-v8",
    )

    # P-HMM-S

    phmms_30ms_mp_a = get_n_blstm_a(
        feature_stacking=False,
        transition_scale=0.0,
        adapted_tdps=None,
        t_step=30 / 1000,
        prior_scale=0.6,
    )
    phmms_40ms_mp_a_very_silency = get_n_blstm_a(
        feature_stacking=False,
        transition_scale=0.0,
        adapted_tdps=None,
        t_step=40 / 1000,
        prior_scale=0.0,
    )
    phmms_40ms_mp_a_silency = get_n_blstm_a(
        feature_stacking=False,
        transition_scale=0.0,
        adapted_tdps=None,
        t_step=40 / 1000,
        prior_scale=0.3,
    )
    phmms_40ms_mp_a = get_n_blstm_a(
        feature_stacking=False,
        transition_scale=0.0,
        adapted_tdps=None,
        t_step=40 / 1000,
        prior_scale=0.6,
    )

    config_11_diphone_mpc1x3_30ms.run(
        returnn_root=returnn_root,
        alignments=[(phmms_30ms_mp_a, "30ms-Bs-pC0.6")],
    )
    config_21_diphone_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=phmms_40ms_mp_a_very_silency,
        a_name="40ms-Bs-pC0.0",
    )
    config_21_diphone_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=phmms_40ms_mp_a_silency,
        a_name="40ms-Bs-pC0.3",
    )
    config_21_diphone_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=phmms_40ms_mp_a,
        a_name="40ms-Bs-pC0.6",
    )

    # P-HMM-S FF-NN

    phmms_30ms_ffnn_a = get_30ms_linear_a(t_scale=0.0)
    phmms_40ms_ffnn_a = get_40ms_linear_a(t_scale=0.0)
    phmms_60ms_ffnn_a = get_60ms_linear_a(t_scale=0.0)
    # phmms_80ms_ffnn_a = get_80ms_linear_a(t_scale=0.0)

    config_11_diphone_mpc1x3_30ms.run(
        returnn_root=returnn_root,
        alignments=[(phmms_30ms_ffnn_a, "30ms-FFs-v8")],
    )
    config_21_diphone_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=phmms_40ms_ffnn_a,
        a_name="40ms-FFs-v8",
    )
    config_21i_diphone_ss_variations_40ms.run(
        returnn_root=returnn_root,
        alignment=phmms_40ms_ffnn_a,
        a_name="40ms-FFs-v8",
    )
    config_21j_diphone_realign_mpc1x4_40ms.run(
        returnn_root=returnn_root,
        alignment=phmms_40ms_ffnn_a,
        a_name="40ms-FFs-v8",
    )

    for a, a_name in [(phmm_40ms_ffnn_a, "40ms-FF-v8"), (phmms_40ms_ffnn_a, "40ms-FFs-v8")]:
        _, mono_sys = config_20_monophone_mpc1x4_40ms.run(
            returnn_root=returnn_root,
            alignment=a,
            a_name=a_name,
        )
        _, di_sys = config_21b_diphone_multi_mpc1x4_40ms.run(
            returnn_root=returnn_root,
            alignment=a,
            a_name=a_name,
            init_from_system=mono_sys,
        )
        config_22b_triphone_multi_mpc1x4_40ms.run(
            returnn_root=returnn_root,
            alignment=a,
            a_name=a_name,
            init_from_system=di_sys,
        )
    config_31_diphone_mpc2x3_60ms.run(
        returnn_root=returnn_root,
        alignment=phmms_60ms_ffnn_a,
        a_name="60ms-FFs-v8",
    )
    # not converged
    # config_41_diphone_mpc2x4_80ms.run(
    #     returnn_root=returnn_root,
    #     alignment=phmms_80ms_ffnn_a,
    #     a_name="80ms-FFs-v8",
    # )

    # P-HMM FF-NN

    config_21i_diphone_ss_variations_40ms.run(
        returnn_root=returnn_root,
        alignment=phmm_40ms_ffnn_a,
        a_name="40ms-FF-v8",
    )


def quant_data_util():
    from i6_experiments.users.gunz.experiments.config_2023_08_subsampling_new import (
        config_00b_monophone_linear_fullsum_all_data,
        config_11_diphone_mpc1x3_30ms,
        config_21_diphone_mpc1x4_40ms,
    )

    returnn_root = _clone_returnn_safe()
    alignment_exps = config_00b_monophone_linear_fullsum_all_data.run(returnn_root=returnn_root)

    alignments_30ms = [
        (alignment_exps.phmm_30ms[1].experiments["fh"]["alignment_job"].out_alignment_bundle, "30ms-FFall-v8"),
        # (alignment_exps.phmms_30ms[1].experiments["fh"]["alignment_job"].out_alignment_bundle, "30ms-FFsall-v8"),
    ]
    config_11_diphone_mpc1x3_30ms.run(returnn_root=returnn_root, alignments=alignments_30ms)

    alignments_40ms = [
        (alignment_exps.phmm_40ms[1].experiments["fh"]["alignment_job"].out_alignment_bundle, "40ms-FFall-v8"),
        # (alignment_exps.phmms_40ms[1].experiments["fh"]["alignment_job"].out_alignment_bundle, "40ms-FFsall-v8"),
    ]
    for a, a_name in alignments_40ms:
        config_21_diphone_mpc1x4_40ms.run(returnn_root=returnn_root, alignment=a, a_name=a_name)


def main():
    the_plan()
    quant_data_util()

    viterbi_30ms()
    viterbi_40ms()
    viterbi_60ms()

    alignment_eval()
