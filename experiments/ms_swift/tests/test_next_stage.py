import copy

import pytest

from cpt.core import digest, file_hash, write_json, write_jsonl


def test_longbench_primary_length_boundaries_retain_official_output_budget():
    from pilot_eval.next_stage import eligible
    row = {'benchmark': 'longbench-v1', 'task': 'hotpotqa',
           'native_input_tokens': 8192, 'max_new_tokens': 32, 'total_budget_tokens': 8224}
    assert eligible(row)
    assert not eligible({**row, 'native_input_tokens': 8191, 'total_budget_tokens': 8223})
    assert eligible({**row, 'native_input_tokens': 16352, 'total_budget_tokens': 16384})
    assert not eligible({**row, 'native_input_tokens': 16353, 'total_budget_tokens': 16385})
    assert eligible({**row, 'task': 'qasper', 'native_input_tokens': 4000, 'total_budget_tokens': 4032})
    with pytest.raises(ValueError, match='budget'):
        eligible({**row, 'total_budget_tokens': 8192})


def test_final_training_sources_must_be_exact_subset_of_audited_scope():
    from pilot_eval.next_stage import bind_training_sources
    sources = [{'id': 'one', 'blocks': [{'text': 'Frozen source'}]},
               {'id': 'two', 'blocks': [{'text': 'Other source'}]}]
    scope = {r['id']: digest(r) for r in sources}
    receipt = bind_training_sources(scope, [sources[0]])
    assert receipt['passed'] and receipt['selected_sources'] == {'one': digest(sources[0])}
    changed = copy.deepcopy(sources[0]); changed['blocks'][0]['text'] = 'Edited'
    for final in [[changed], [{'id': 'new', 'blocks': []}], [sources[0], sources[0]], []]:
        with pytest.raises(ValueError, match='source'):
            bind_training_sources(scope, final)


def test_input_manifest_rehashes_its_partition_source_lock(tmp_path):
    from pilot_eval.next_stage import write_inputs
    from gemma_eval.chat import validate_input_manifest
    lock = {'training_source_scope': {'fixture': 'sha'}}
    write_json(tmp_path / 'source.lock.json', lock)
    records = [{'id': 'fixture'}]
    write_inputs(tmp_path / 'development', records, lock)
    validate_input_manifest(tmp_path / 'development/records.jsonl', records)
    write_json(tmp_path / 'source.lock.json', {'changed': True})
    with pytest.raises(ValueError, match='source evidence'):
        validate_input_manifest(tmp_path / 'development/records.jsonl', records)


def test_preseal_gate_binds_final_sources_and_confirmation_before_any_model_results(tmp_path):
    from pilot_eval.next_stage import verify_preseal
    source = {'id': 'fixture', 'blocks': [{'text': 'Frozen'}]}
    scope = {'fixture': digest(source)}
    write_json(tmp_path / 'source.lock.json', {'training_source_scope': scope})
    (tmp_path / 'confirmation').mkdir()
    write_jsonl(tmp_path / 'confirmation/records.jsonl', [{'id': 'confirm', 'evaluation_only': True}])
    for role in ['development', 'demonstration', 'calibration']:
        (tmp_path / role).mkdir(); write_jsonl(tmp_path / role / 'records.jsonl', [{'id': role}])
    write_json(tmp_path / 'confirmation.preseal.json', {
        'source_lock_sha256': file_hash(tmp_path / 'source.lock.json'),
        'records_sha256': file_hash(tmp_path / 'confirmation/records.jsonl'),
        'partition_artifacts': {f'{role}/records.jsonl': file_hash(tmp_path / role / 'records.jsonl')
                                for role in ['development', 'demonstration', 'calibration', 'confirmation']}})
    assert verify_preseal(tmp_path, [source])['passed']
    with pytest.raises(ValueError, match='source'):
        verify_preseal(tmp_path, [{**source, 'blocks': []}])
    write_jsonl(tmp_path / 'confirmation/records.jsonl', [{'id': 'changed'}])
    with pytest.raises(ValueError, match='preseal'):
        verify_preseal(tmp_path, [source])


def test_partitioned_development_or_demonstrations_cannot_change_before_seal(tmp_path):
    from pilot_eval.next_stage import verify_preseal
    source = {'id': 'fixture', 'blocks': []}
    write_json(tmp_path / 'source.lock.json', {'training_source_scope': {'fixture': digest(source)}})
    artifacts = {}
    for role in ['calibration', 'confirmation', 'demonstration', 'development']:
        (tmp_path / role).mkdir(); path = tmp_path / role / 'records.jsonl'
        write_jsonl(path, [{'id': role}]); artifacts[f'{role}/records.jsonl'] = file_hash(path)
    write_json(tmp_path / 'confirmation.preseal.json', {
        'source_lock_sha256': file_hash(tmp_path / 'source.lock.json'),
        'records_sha256': artifacts['confirmation/records.jsonl'], 'partition_artifacts': artifacts})
    write_jsonl(tmp_path / 'demonstration/records.jsonl', [{'id': 'replacement'}])
    with pytest.raises(ValueError, match='preseal'):
        verify_preseal(tmp_path, [source])
