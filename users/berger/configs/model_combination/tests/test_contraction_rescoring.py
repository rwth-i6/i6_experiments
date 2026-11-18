"""Protect AM/LM separation, cached controls, and safe finite text alternatives."""

import json

import pytest
from sisyphus import tk

from i6_experiments.users.berger.model_pipelines.common.contraction_variants import variants, choose, EXPAND
from i6_experiments.users.berger.model_pipelines.common.contraction_rescoring import plans, attach_scores, check_literal_scores
from i6_experiments.users.berger.model_pipelines.common.contraction_analysis import SelectContractionNbestJob


def cached(text, cost):
    return dict(text=text, am_cost=0, lm_cost=0, first_pass_cost=0,
                speech={'initial-capital-period': dict(nll=cost, token_costs=[cost], num_tokens=0)})


def test_phrase_renderings_preserve_case_and_leave_ambiguities_untouched():
    v = variants("YOU ARE HERE AND WE WILL STAY")
    assert v['contracted']['text'] == "YOU'RE HERE AND WE'LL STAY"
    assert variants("You're here and we won't leave")['expanded']['text'] == 'You are here and we will not leave'
    assert variants("I’m ready")['expanded']['text'] == 'I am ready'
    text = "John's account, I'd say, Dr. O'Neil-Smith's twenty-five shares."
    assert all(r['text'] == text for r in variants(text).values())
    assert variants("They cannot leave")['contracted']['text'] == "They can't leave"
    assert variants('They will not leave')['contracted']['text'] == "They won't leave"
    assert variants('the donkey and candy')['contracted']['text'] == 'the donkey and candy'


def test_core_aliases_preserve_hf_normalized_content_in_a_sentence():
    from i6_experiments.users.berger.model_pipelines.common.hf_scoring import normalize_hf_english
    for contracted, expanded in EXPAND.items():
        a = normalize_hf_english('They said ' + contracted + ' today.').replace(' ', '')
        b = normalize_hf_english('They said ' + expanded + ' today.').replace(' ', '')
        assert a == b, (contracted, expanded)


def test_original_am_words_and_cached_scores_survive_transformed_lm_strings():
    candidates = [cached('YOU ARE RIGHT', 10)]
    p, cache = plans(candidates, 'initial-capital-period')
    scored = {"You're right.": dict(nll=1, token_costs=[1], num_tokens=0),
              'You are right.': dict(nll=9.999, token_costs=[9.999], num_tokens=0)}
    augmented = attach_scores(candidates, p, cache, scored)
    assert augmented[0]['contraction_scores']['literal']['nll'] == 10
    selected = choose(augmented, mode='best', n=50, scale=1, word_penalty=-1)
    assert selected['rendering'] == 'contracted'
    assert selected['word_cost'] == -3  # AM has three words; LM rendering has two
    assert augmented[0]['text'] == 'YOU ARE RIGHT'
    assert augmented[0]['am_cost'] == 0 and augmented[0]['first_pass_cost'] == 0


def test_identical_renderings_reuse_the_same_literal_score():
    candidates = [cached('all done', 3)]
    p, cache = plans(candidates, 'initial-capital-period')
    augmented = attach_scores(candidates, p, cache, {})
    assert len(cache) == 1
    for mode in ('literal', 'contracted', 'expanded', 'best'):
        assert choose(augmented, mode=mode, n=1, scale=.3, word_penalty=-1)['cost'] == pytest.approx(-1.1)


def test_literal_gate_catches_tokenization_or_model_score_changes():
    cache = {'a': dict(token_costs=[1, 2])}
    assert check_literal_scores(cache, {'a': dict(token_costs=[1.001, 2])}, ['a']) == pytest.approx(.001)
    with pytest.raises(ValueError, match='tokenization'):
        check_literal_scores(cache, {'a': dict(token_costs=[3])}, ['a'])
    with pytest.raises(AssertionError, match='Literal score replay'):
        check_literal_scores(cache, {'a': dict(token_costs=[1.1, 2])}, ['a'])


def test_selection_changes_candidate_but_never_outputs_the_alternate_lm_text(tmp_path):
    candidates = [cached('you are right', 10), cached('you were right', 2)]
    p, cache = plans(candidates, 'initial-capital-period')
    augmented = attach_scores(candidates, p, cache, {"You're right.": dict(nll=1, token_costs=[1], num_tokens=0)})

    def file(name, row):
        path = tmp_path / name
        path.write_text(json.dumps(row)+'\n')
        return tk.Path(str(path))

    job = SelectContractionNbestJob(file('candidates', dict(id='x', candidates=augmented)),
        file('gate', dict(passed=True, literal_cache_preserved=True)), file('frozen', dict(id='x', text='you were right')),
        n=50, scale=1, word_penalty=-1, code_version='test')
    job.out_hypotheses = {m: tk.Path(str(tmp_path/(m+'.jsonl'))) for m in job.out_hypotheses}
    job.out_validation = tk.Path(str(tmp_path/'validation.json'))
    job.out_selections = tk.Path(str(tmp_path/'selections.jsonl'))
    job.run()
    assert json.loads((tmp_path/'best.jsonl').read_text())['text'] == 'you are right'
    assert json.loads((tmp_path/'literal.jsonl').read_text())['text'] == 'you were right'
    (tmp_path/'frozen').write_text(json.dumps(dict(id='x', text='tampered'))+'\n')
    with pytest.raises(ValueError, match='exactly reproduce'):
        job.run()


def test_report_distinguishes_better_ranking_from_beating_the_standalone(tmp_path):
    from i6_experiments.users.berger.model_pipelines.common.contraction_analysis import ContractionReportJob, MODES
    from i6_experiments.users.berger.model_pipelines.common.controlled_comparison import NORMALIZER

    def file(name, value, jsonl=False):
        path = tmp_path/name
        path.write_text(''.join(json.dumps(v)+'\n' for v in value) if jsonl else json.dumps(value))
        return tk.Path(str(path))

    ids = ['1-1-0000', '1-1-0001', '2-2-0000', '2-2-0001']
    scores = {}
    for mode in list(MODES)+['am_first_pass', 'speech_standalone']:
        rows = []
        for j, seq_id in enumerate(ids):
            errors = 0 if mode == 'speech_standalone' or (mode == 'best' and j % 2 == 0) else 1
            rows.append(dict(id=seq_id, counts=dict(total=errors, sub=errors, ins=0, ref_len=100, **{'del': 0})))
        summary = {k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
        scores[mode] = dict(segments=file(mode+'-units',rows,True),
                            summary=file(mode+'-summary',dict(normalizer_revision=NORMALIZER, **summary)))
    ledger = [dict(id=i, eligible=j%2==0, selections={m:dict(rendering='literal', changed_output=m=='best' and j%2==0)
                                                    for m in MODES}) for j,i in enumerate(ids)]
    experiment = dict(model='Granite CTC', dataset='librispeech-dev-other', settings={}, scores=scores,
        selection_validation=file('selection-gate',dict(passed=True, units=4, eligible_units=2)),
        score_validation=file('score-gate',dict(passed=True)), selections=file('ledger',ledger,True),
        extra_runtime=file('runtime',dict(rtf=.1)), speech_hypotheses=file('speech',[dict(id=i,text='all done') for i in ids],True))
    job = ContractionReportJob([experiment],file('protocol',{}),code_version='test')
    job.out_report = tk.Path(str(tmp_path/'report.md'))
    job.out_summary = tk.Path(str(tmp_path/'report.json'))
    job.run()
    panel = json.loads((tmp_path/'report.json').read_text())['panels'][0]
    best = next(r for r in panel['rows'] if r['mode']=='best')
    assert best['delta_literal'] == -.5
    assert best['delta_components'] == .5  # improves ranking but still loses to Speech alone
    untouched = next(r for r in panel['slices'] if r['mode']=='best' and r['slice']=='untouched_pool')
    assert untouched['corrected_units'] == untouched['worsened_units'] == 0
