"""Development sweeps for the official Parakeet TDT v3 checkpoint."""

import json
from pathlib import Path

from sisyphus import Job, Task, tk
from sisyphus.delayed_ops import DelayedFormat

from . import parakeet_tdt as tdt


MODEL = 'nvidia/parakeet-tdt-0.6b-v3'
REVISION = '541d1f99c6b0c3cd0b11a95167540bb8edefd82b'
RASR_COMMIT = 'b0e610cb24c0fa54e1f21ad3e46c92f8f42162cc'


def recognition_config(export, *, beam, kind, scale=0., eos=tdt.EOS):
    from i6_core.rasr.config import RasrConfig

    config = RasrConfig()
    root = config.lib_rasr
    root.lexicon.file = DelayedFormat('vocab-text:{}', export.out_vocab)
    scorer = root.label_scorer
    scorer.type = 'stateful-transducer-onnx'
    scorer.durations = list(tdt.DURATIONS)
    scorer.blank_label_index = tdt.BLANK
    scorer.max_batch_size = 32
    for key, name in (('init', 'state-initializer-model'), ('update', 'state-updater-model'), ('joint', 'scorer-model')):
        session = scorer[name].session
        session.file = export.out_models[key]
        session.execution_provider_type = 'cpu'
        session.intra_op_num_threads = 2
        session.inter_op_num_threads = 1
    scorer.state_updater_model.io_map.token = 'token'
    scorer.scorer_model.io_map.input_feature = 'encoder_state'
    scorer.scorer_model.io_map.scores = 'scores'
    search = root.search_algorithm
    search.type = ('lexiconfree-rnnt-timesync-beam-search' if kind == 'am'
                   else 'llm-rnnt-timesync-beam-search')
    search.blank_label_index = tdt.BLANK
    if eos is not None:
        search.sentence_end_label_index = eos
        # The transducer preset does not score SENTENCE_END. Stock v3 predicts
        # EOS in the joint network, so allowing a free EOS truncates hypotheses.
        scorer.extra_transition_types = 'sentence-end'
    search.sentence_end_fall_back = True
    search.collapse_repeated_labels = False
    search.max_labels_per_timeframe = 10
    search.recombination_mode = 'viterbi'
    search.max_beam_size = [beam]
    search.score_threshold = [10.] if kind == 'am' else [3.]
    search.log_stepwise_statistics = False
    if kind != 'am':
        search.pre_llm_max_beam_size = 32
        search.pre_llm_score_threshold = 3.
        search.word_piece_convention = 'word-start-marked'
        search.word_piece_marker = '▁'
        search.llm.type = 'parakeet-tdt-auxiliary'
        search.llm_scale = scale
        search.word_penalty = 0.
    return config


class TdtDevReportJob(Job):
    # Previously serialized report jobs predate the explicit build revision.
    rasr_commit = 'd4aa61a12fa4f33e873512678a4d204a90329748'

    def __init__(self, rows, onnx_gate, probe, *, version, rasr_commit):
        self.rows, self.onnx_gate, self.probe, self.version = rows, onnx_gate, probe, version
        self.rasr_commit = rasr_commit
        self.out_report = self.output_path('report.md')
        self.out_summary = self.output_path('summary.json')

    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        def load(path):
            return json.loads(Path(path.get_path()).read_text())

        rows = []
        for row in self.rows:
            score = load(row['score'])
            timing = load(row['runtime']) if row['runtime'] is not None else {}
            rows.append(dict(dataset=row['dataset'], condition=row['condition'], score=score, timing=timing))
        probe, onnx = load(self.probe), load(self.onnx_gate)
        lines = ['# Parakeet TDT v3 development experiments', '',
                 f"ONNX cost parity max absolute error {onnx['max_cost_difference']:.2g}.",
                 f"Beam-1 native transcript agreement {probe['exact_matches']}/{len(probe['segments'])}.",
                 'HF greedy and RASR make different choices when the preferred blank duration is zero. '
                 'The probe JSON records encoder lengths and traceback endpoints.', '',
                 '| Dataset | Condition | WER % | I / D / S | RTF |',
                 '|---|---|---:|---:|---:|']
        for row in rows:
            score = row['score']
            rtf = row['timing'].get('rtf')
            timing = f'{rtf:.3f}' if rtf is not None else '—'
            lines.append(f"| {row['dataset']} | {row['condition']} | {score['wer_percent']:.3f} | "
                         f"{score['ins']} / {score['del']} / {score['sub']} | {timing} |")
        Path(self.out_report.get_path()).write_text('\n'.join(lines)+'\n')
        Path(self.out_summary.get_path()).write_text(json.dumps(dict(model=MODEL, revision=REVISION,
            rasr_commit=self.rasr_commit, onnx=onnx, probe=probe, rows=rows), indent=2)+'\n')


def run(output_root):
    from i6_core.rasr.config import WriteRasrConfigJob
    from i6_core.tools.compile import CMakeJob
    from i6_core.tools.git import CloneGitRepositoryJob

    from ...data.huggingface import DownloadHuggingFaceRepoJob
    from ...data.llm_fusion import QWEN_BASE_MODELS
    from . import benchmark_scoring, dynamic_llm, dynamic_qwen_tuned, hf_benchmark
    from . import parakeet_tdt_experiments, qwen_speech, rnnt_llm
    from .experiment_context import ExperimentContext
    from .job_reports import register_job_report
    from .public_pilots import analysis_version, completed, frozen_protocol, sha

    protocol, _, digest = frozen_protocol(output_root)
    version = analysis_version(tdt, parakeet_tdt_experiments, benchmark_scoring, dynamic_llm,
                               dynamic_qwen_tuned, hf_benchmark, qwen_speech, rnnt_llm,
                               compatible_modules=(parakeet_tdt_experiments,))
    scoring_version = analysis_version(benchmark_scoring)
    prefix = f'parakeet-tdt/{digest[:16]}'
    selected = protocol['selected']
    speech = selected['Speech Qwen 1.7B | None | standalone | fast']['dev']
    stages = {dataset: next(s['parameters'] for s in r['stages'] if s['job_type'] == 'SpeechQwenBeamJob')
              for dataset, r in speech.items()}
    speech_repo, = {s['speech_repo'] for s in stages.values()}
    gate_dataset = completed(stages['librispeech-dev-other']['dataset'])
    with ExperimentContext(prefix + '/tools'):
        checkout = CloneGitRepositoryJob('https://github.com/rwth-i6/rasr.git', commit=RASR_COMMIT,
                                        checkout_folder_name='rasr')
        checkout.add_alias('rasr-source')
        build = CMakeJob(source_folder=checkout.out_repository, num_processes=8, mem_rqmt=8,
                         cmake_opts=['-DMODULE_TENSORFLOW=Off', '-DMODULE_LM_TFRNN=Off',
                                     '-DMODULE_ONNX=On', '-DMARCH=x86-64-v3'])
        build.add_alias('rasr-cmake')
        patterns = ['*.json', '*.safetensors', '*.model', '*.txt']
        download = DownloadHuggingFaceRepoJob(MODEL, revision=REVISION, allow_patterns=patterns)
        download.add_alias('model')
        text_repo, text_revision = QWEN_BASE_MODELS['1.7b']
        text = DownloadHuggingFaceRepoJob(text_repo, revision=text_revision, allow_patterns=patterns)
        text.add_alias('text-qwen')
        repo = download.out_repo_dir
        export = tdt.ExportParakeetTdtJob(repo, gate_dataset, version=analysis_version(tdt, hf_benchmark))
        export.add_alias('onnx-export')
        tk.register_output('onnx-parity.json', export.out_gate)
        tk.register_output('vocab.txt', export.out_vocab)
        for key, path in export.out_models.items():
            tk.register_output(f'{key}.onnx', path)
        probe_config = WriteRasrConfigJob(recognition_config(export, beam=1, kind='am'), None).out_config
        probe = tdt.ParakeetTdtRecogJob(repo, gate_dataset, version=version, config=probe_config,
                                      binary=build.out_install_dir, vocabulary=export.out_vocab,
                                      onnx_gate=export.out_gate, kind='am', probe_size=8)
        probe.add_alias('real-audio-beam1-probe')
        tk.register_output('probe.json', probe.out_probe)
    rows = []
    conditions = [('native', 'native', 0., 1), ('rasr-beam1', 'am', 0., 1), ('rasr-beam8', 'am', 0., 8)]
    for kind in ('text', 'speech'):
        conditions.extend((f'{kind}-qwen-1.7b-scale-{scale:g}', kind, scale, 8) for scale in (.1, .25))
    auxiliaries = {'text': text.out_repo_dir, 'speech': completed(speech_repo)}
    for dataset in protocol['dev_datasets']:
        primary = speech[dataset]
        audio = completed(stages[dataset]['dataset'])
        refs = completed(primary['references'])
        rnnt = selected['Parakeet RNN-T | None | standalone | fast']['dev'][dataset]
        for name, record in (('Speech-Qwen 1.7B standalone', primary), ('Parakeet RNN-T matched control', rnnt)):
            if (sha(record['score']) != record['score_sha256']
                    or sha(record['references']) != primary['reference_sha256']):
                raise ValueError(f'Frozen TDT comparison differs for {dataset}')
            rows.append(dict(dataset=dataset, condition=name, score=completed(record['score']), runtime=None))
        for name, kind, scale, beam in conditions:
            with ExperimentContext(f'{prefix}/{dataset}/{name}'):
                config = None
                if kind != 'native':
                    config = WriteRasrConfigJob(recognition_config(export, beam=beam, kind=kind, scale=scale),
                                                None).out_config
                job = tdt.ParakeetTdtRecogJob(repo, audio, version=version, config=config,
                    binary=build.out_install_dir if config else None, vocabulary=export.out_vocab if config else None,
                    onnx_gate=export.out_gate if config else None, search_gate=probe.out_probe if config else None,
                    auxiliary=auxiliaries.get(kind), kind=kind)
                job.add_alias('recognition')
                score = benchmark_scoring.ScoreBenchmarkJob(refs, job.out_hypotheses, code_version=scoring_version)
                score.add_alias('scoring')
                tk.register_output('hypotheses.jsonl', job.out_hypotheses)
                tk.register_output('score.json', score.out_summary)
                tk.register_output('runtime.json', job.out_runtime)
                rows.append(dict(dataset=dataset, condition=name, score=score.out_summary, runtime=job.out_runtime))
    with ExperimentContext(prefix):
        report = TdtDevReportJob(rows, export.out_gate, probe.out_probe, version=version, rasr_commit=RASR_COMMIT)
        report.add_alias('report')
        tk.register_output('report.md', report.out_report)
        tk.register_output('summary.json', report.out_summary)
    register_job_report('reports/asr-lm/parakeet-tdt.md', report.out_report, title='Parakeet TDT development')
    return report
