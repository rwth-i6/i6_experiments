"""Paired waveform integrity and descriptive-analysis semantics before GPU runs."""

import io
import json

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.berger.data.noise_pilot import noise_versions, select_ids, PrepareNoisePilotJob
from i6_experiments.users.berger.model_pipelines.common.convention_audit import surface_features, paired_feature_rows


def test_subset_ignores_source_order_and_balances_chapters_before_reusing_them():
    ids = [f"{speaker}-1-{i:04d}" for speaker in range(1, 5) for i in range(10)]
    chosen = select_ids(ids, 7, 19)
    assert chosen == select_ids(ids[::-1], 7, 19)
    assert len({s.rsplit('-', 1)[0] for s in chosen[:4]}) == 4
    assert len(set(chosen)) == len(chosen)
    with pytest.raises(ValueError):
        select_ids(ids + [ids[0]], 7, 19)


def test_noise_realization_snr_and_common_gain_are_preserved():
    waveform = (0.9 * np.sin(np.arange(32000) * 0.1)).astype('float32')
    versions, gain = noise_versions(waveform, seed=83, snrs=(20., 10.))
    again, again_gain = noise_versions(waveform, seed=83, snrs=(20., 10.))
    assert gain == again_gain
    for level, signal in versions.items():
        assert np.array_equal(signal, again[level])
        assert np.max(np.abs(signal)) <= 0.990001
    assert np.allclose(versions['clean'], waveform * gain)
    clean = versions['clean'].astype('float64')
    residuals = {}
    for level, snr in [('snr20', 20), ('snr10', 10)]:
        residual = versions[level].astype('float64') - clean
        measured = 10 * np.log10(np.mean(clean ** 2) / np.mean(residual ** 2))
        assert measured == pytest.approx(snr, abs=1e-4)
        residuals[level] = residual
    assert np.allclose(residuals['snr10'], residuals['snr20'] * np.sqrt(10), atol=3e-7)
    with pytest.raises(ValueError, match='silent'):
        noise_versions(np.zeros(100), seed=1)


def test_saved_noise_dataset_contains_lossless_audio_after_scratch_is_deleted(tmp_path):
    import soundfile as sf
    from datasets import Audio, Dataset, Features, Value, load_from_disk

    original = (0.1 * np.sin(np.arange(4000) * .1)).astype('float32')
    source_audio = tmp_path / 'source.wav'
    sf.write(source_audio, original, 16000, subtype='FLOAT')
    dataset = tmp_path / 'source'
    Dataset.from_list([dict(id='1-2-0000', text='hello', audio=str(source_audio))],
                      features=Features(dict(id=Value('string'), text=Value('string'), audio=Audio(sampling_rate=16000)))).save_to_disk(str(dataset))
    source_audio.unlink()
    refs = tmp_path / 'refs.jsonl'
    reference = dict(id='1-2-0000', group_id='1-2-0000', chunk_index=0, reference='hello')
    refs.write_text(json.dumps(reference) + '\n')
    job = PrepareNoisePilotJob(tk.Path(str(dataset)), tk.Path(str(refs)), count=1, code_version='test')
    job.out_references = tk.Path(str(tmp_path / 'out-refs.jsonl'))
    job.out_manifest = tk.Path(str(tmp_path / 'manifest.json'))
    job.out_datasets = {k: tk.Path(str(tmp_path / k)) for k in job.levels}
    job.run()
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    expected, gain = noise_versions(original, seed=manifest['units'][0]['seed'])
    for level, path in job.out_datasets.items():
        row = load_from_disk(path.get_path()).cast_column('audio', Audio(decode=False))[0]
        audio = row['audio']
        assert audio['bytes'] is not None  # no dangling temporary WAV dependency
        decoded, rate = sf.read(io.BytesIO(audio['bytes']), dtype='float32')
        assert rate == 16000
        assert np.array_equal(decoded, expected[level])
        assert row['id'] == reference['id'] and row['text'] == 'hello'
    assert json.loads((tmp_path / 'out-refs.jsonl').read_text()) == reference


def test_surface_flags_do_not_conflate_digits_with_semantic_numbers():
    f = surface_features("Dr. O'Neil’s twenty-five U.S. contracts aren't 20 percent.")
    assert f['title'] == 1 and f['dotted_abbreviation'] == 1
    assert f['hyphen'] == 1 and f['digit'] == 1 and f['ambiguous_s'] == 1
    assert f['contraction'] == 2
    assert surface_features('oh no')['number_word'] == 1  # explicitly a lexical flag, not a true number label


def test_output_conditioned_slices_retain_system_specific_denominators():
    def unit(errors, ref):
        return dict(counts=dict(total=errors, ref_len=ref, ins=0, sub=errors, **{'del': 0}))
    am = {'a': unit(1, 4), 'b': unit(0, 4)}
    speech = {'a': unit(0, 5), 'b': unit(1, 5)}
    fusion = {'a': unit(1, 5), 'b': unit(0, 4)}
    rows = paired_feature_rows(am, speech, fusion, {'a': "it's done", 'b': 'all done'})
    yes = next(r for r in rows if r['slice'] == 'ambiguous_s/yes')
    assert yes['wer']['am'] == 25 and yes['wer']['fusion'] == 20
    assert yes['delta_vs_better'] == 20  # not an improvement merely because it beats the AM
    assert yes['worsened_vs_s'] == 1 and yes['improved_vs_s'] == 0
    with pytest.raises(ValueError, match='coverage'):
        paired_feature_rows(am, speech, fusion, {'a': "it's done"})


def test_noise_report_uses_the_better_constituent_at_each_snr_and_rejects_missing_audio(tmp_path):
    from i6_experiments.users.berger.model_pipelines.common.noise_pilot import NoisePilotReportJob
    from i6_experiments.users.berger.model_pipelines.common.controlled_comparison import NORMALIZER

    def write(name, value, jsonl=False):
        p = tmp_path / name
        p.write_text(''.join(json.dumps(v) + '\n' for v in value) if jsonl else json.dumps(value))
        return tk.Path(str(p))

    ids = ['1-1-0000', '1-1-0001', '2-2-0000', '2-2-0001']
    manifest = write('manifest', dict(units=[dict(id=i) for i in ids], audio_seconds=8, seed=17))
    protocol = write('protocol', {})
    results = {}
    for condition, errors in [('clean/parakeet', 0), ('clean/speech', 1), ('clean/fusion', 0),
                              ('snr10/parakeet', 2), ('snr10/speech', 0), ('snr10/fusion', 1)]:
        slug = condition.replace('/', '-')
        counts = dict(total=errors, ref_len=100, ins=0, sub=errors, **{'del': 0})
        results[condition] = dict(
            segments=write(slug+'-units', [dict(id=i, counts=counts) for i in ids], jsonl=True),
            score=write(slug+'-score', dict(normalizer_revision=NORMALIZER,
                                           **{k: v*len(ids) for k, v in counts.items()})),
            runtime=write(slug+'-runtime', dict(audio_seconds=8, num_segments=4, rtf=.2)))
    job = NoisePilotReportJob(results, manifest, protocol, code_version='test', bootstrap_samples=100)
    job.out_report = tk.Path(str(tmp_path/'report.md'))
    job.out_summary = tk.Path(str(tmp_path/'summary.json'))
    job.run()
    rows = {r['condition']: r for r in json.loads((tmp_path/'summary.json').read_text())['rows']}
    assert rows['clean/fusion']['delta'] == 0
    assert rows['snr10/fusion']['delta'] == 1  # beats/no worse than AM is insufficient
    job.results['snr10/speech']['segments'] = write('incomplete', [dict(id=ids[0], counts=counts)], jsonl=True)
    with pytest.raises(ValueError, match='coverage'):
        job.run()
