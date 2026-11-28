import json
import pytest

from cpt.core import digest, file_hash, write_json, write_jsonl


def example(i):
    return {'id': f'boolq:{i}', 'benchmark': 'boolq', 'task': 'yes_no', 'cluster_id': f'passage:{i}',
            'raw_record': {'passage': f'Fixture passage {i}.', 'question': 'Is this a fixture?', 'answer': True},
            'answers': ['yes'], 'evaluation_only': True}


def test_profiles_use_only_development_and_disjoint_demonstrations():
    from pilot_eval.calibration import build_calibration_records
    dev = [example(1), example(2)]
    demos = [example(3), example(4)]
    rows = build_calibration_records(dev, demos)
    assert len(rows) == 8
    assert len({r['id'] for r in rows}) == 8
    assert {r['base_id'] for r in rows} == {'boolq:1', 'boolq:2'}
    assert {r['max_new_tokens'] for r in rows} == {8, 32, 64}
    assert all(r['evaluation_only'] for r in rows)
    with pytest.raises(ValueError, match='demonstration'):
        build_calibration_records(dev, [example(1), example(4)])


def test_complete_profile_selection_uses_format_threshold_and_fixed_priority_not_accuracy():
    from pilot_eval.calibration import build_calibration_records, assess_calibration
    rows = build_calibration_records([example(i) for i in range(10)], [example(20), example(21)])
    predictions = [{'id': r['id'], 'status': 'ok', 'prediction': 'No' if r['profile'] == 'zero_8' else 'Yes'} for r in rows]
    report = assess_calibration(rows, predictions, model_loaded=True)
    assert report['calibration_ready'] and report['selected_profile'] == 'zero_8'
    assert report['profiles']['zero_8']['valid_correct'] == 0
    assert report['profiles']['zero_8']['valid_wrong'] == 10
    assert report['confirmation_results_observed'] == 0


def test_failures_and_format_errors_remain_in_denominator_and_cannot_grant_readiness():
    from pilot_eval.calibration import build_calibration_records, assess_calibration
    rows = build_calibration_records([example(i) for i in range(10)], [example(20), example(21)])
    predictions = [{'id': r['id'], 'status': 'ok', 'prediction': 'The answer is Yes.'} for r in rows]
    predictions[0].update(status='empty_prediction', prediction='')
    report = assess_calibration(rows, predictions, model_loaded=True)
    assert not report['calibration_ready'] and report['selected_profile'] is None
    assert sum(v['total'] for v in report['profiles'].values()) == 40
    assert sum(v['empty_or_failed'] for v in report['profiles'].values()) == 1
    for prediction in predictions:
        prediction['prediction'] = 'Yes'
    assert not assess_calibration(rows, predictions, model_loaded=False)['calibration_ready']


def test_missing_reordered_and_duplicate_ids_are_not_a_complete_calibration():
    from pilot_eval.calibration import build_calibration_records, assess_calibration
    rows = build_calibration_records([example(1)], [example(20), example(21)])
    predictions = [{'id': r['id'], 'status': 'ok', 'prediction': 'Yes'} for r in rows]
    for changed in [predictions[:-1], list(reversed(predictions)), predictions + [predictions[0]]]:
        with pytest.raises(ValueError, match='ID'):
            assess_calibration(rows, changed, model_loaded=True)


def test_ninety_percent_boundary_and_nonempty_failed_rows_are_not_valid_answers():
    from pilot_eval.calibration import build_calibration_records, assess_calibration
    rows = build_calibration_records([example(i) for i in range(10)], [example(20), example(21)])
    predictions = [{'id': r['id'], 'status': 'ok', 'prediction': 'Yes'} for r in rows]
    predictions[0]['prediction'] = 'Explanation: yes'
    report = assess_calibration(rows, predictions, model_loaded=True)
    assert report['selected_profile'] == 'zero_8'
    predictions[1]['status'] = 'model_load_failed'
    report = assess_calibration(rows, predictions, model_loaded=True)
    assert not report['profiles']['zero_8']['eligible']


@pytest.fixture
def run_files(tmp_path):
    from pilot_eval.calibration import build_calibration_records
    from cpt.schema import MODEL_REPO, MODEL_REVISION
    from gemma_eval.chat import PROTOCOL_VERSION, PROTOCOL
    records = build_calibration_records([example(1)], [example(20), example(21)])
    (tmp_path / 'calibration').mkdir()
    inputs = tmp_path / 'calibration/records.jsonl'; write_jsonl(inputs, records)
    requests = tmp_path / 'requests.jsonl'
    write_jsonl(requests, [{'id': r['id'], 'example_sha256': digest(r), 'max_new_tokens': r['max_new_tokens']} for r in records])
    model = {'repo': MODEL_REPO, 'revision': MODEL_REVISION}
    contract = {'protocol': PROTOCOL_VERSION, 'model': model, 'context_length': 16384,
                'quantization': 'bf16', 'adapter': None}
    plan = {**contract, 'kind': 'chat', 'chat_protocol': PROTOCOL, 'do_sample': False,
            'input': {'path': str(inputs), 'sha256': file_hash(inputs)},
            'requests': {'path': str(requests), 'sha256': file_hash(requests)},
            'selected_ids': [r['id'] for r in records]}
    plan_path = tmp_path / 'plan.json'; write_json(plan_path, plan)
    out = tmp_path / 'run'; (out / 'run').mkdir(parents=True)
    predictions = [{'id': r['id'], 'status': 'ok', 'prediction': 'Yes', 'example_sha256': digest(r),
                    'model': model['repo'], 'model_revision': model['revision'],
                    'quantization': 'bf16', 'protocol': PROTOCOL_VERSION, 'max_new_tokens': r['max_new_tokens'],
                    'transport_called': True} for r in records]
    write_jsonl(out / 'predictions.jsonl', predictions)
    write_jsonl(out / 'run/predictions.jsonl', predictions)
    child = {'plan_sha256': digest(plan), 'executed': True, 'model_loaded': True,
             'evaluation_contract': contract, 'selected_ids': plan['selected_ids'], 'total': len(records),
             'runtime': {'quantization': 'bf16', 'chat_template_verification': {'verified': True}},
             'load_error': None, 'predictions_sha256': file_hash(out / 'run/predictions.jsonl')}
    write_json(out / 'run/summary.json', child)
    summary = {'schema_version': 'gemma-bounded-summary-v1', 'status': 'completed', 'kind': 'chat',
               'executed': True, 'plan_sha256': digest(plan), 'plan_file_sha256': file_hash(plan_path),
               'worker_returncode': 0, 'error': None, 'child_output_errors': [], 'cleanup': {'worker_exited': True},
               'child_summary_sha256': file_hash(out / 'run/summary.json'),
               'predictions_sha256': file_hash(out / 'predictions.jsonl'),
               'total': len(records), 'child_completed_records': len(records), 'selected_ids': plan['selected_ids'],
               'statuses': {'ok': len(records)}}
    write_json(out / 'summary.json', summary)
    return inputs, plan_path, out


def test_only_complete_hash_bound_base_bf16_run_can_select_profile(run_files):
    from pilot_eval.calibration import assess_run
    inputs, plan, out = run_files
    report = assess_run(inputs, plan, out)
    assert report['calibration_ready'] and report['model_scope'] == 'pinned_base_bf16_no_adapter'
    assert report['plan_file_sha256'] == file_hash(plan)


@pytest.mark.parametrize('field,value', [('adapter', {'files': {'fake': 'sha'}}),
                                      ('quantization', 'nf4'), ('model', {'repo': 'other', 'revision': 'other'})])
def test_adapter_quantization_or_model_cannot_be_used_for_selection(run_files, field, value):
    from pilot_eval.calibration import assess_run
    inputs, plan_path, out = run_files
    plan = json.loads(plan_path.read_text()); plan[field] = value; write_json(plan_path, plan)
    with pytest.raises(ValueError, match='Base|contract'):
        assess_run(inputs, plan_path, out)


def test_mixed_prediction_metadata_and_changed_receipts_are_rejected(run_files):
    from pilot_eval.calibration import assess_run
    inputs, plan, out = run_files
    rows = [json.loads(x) for x in (out / 'predictions.jsonl').read_text().splitlines()]
    rows[0]['quantization'] = 'nf4'; write_jsonl(out / 'predictions.jsonl', rows)
    summary = json.loads((out / 'summary.json').read_text())
    summary['predictions_sha256'] = file_hash(out / 'predictions.jsonl'); write_json(out / 'summary.json', summary)
    with pytest.raises(ValueError, match='prediction contract'):
        assess_run(inputs, plan, out)
    rows[0]['quantization'] = 'bf16'; write_jsonl(out / 'predictions.jsonl', rows)
    with pytest.raises(ValueError, match='hash'):
        assess_run(inputs, plan, out)


def test_outer_predictions_cannot_replace_worker_original_text_even_with_matching_ids_and_metadata(run_files):
    from pilot_eval.calibration import assess_run
    inputs, plan, out = run_files
    rows = [json.loads(x) for x in (out / 'run/predictions.jsonl').read_text().splitlines()]
    for row in rows:
        row['prediction'] = 'The answer is Yes.'
    write_jsonl(out / 'run/predictions.jsonl', rows)
    child_path = out / 'run/summary.json'
    child = json.loads(child_path.read_text()); child['predictions_sha256'] = file_hash(out / 'run/predictions.jsonl')
    write_json(child_path, child)
    summary = json.loads((out / 'summary.json').read_text())
    summary['child_summary_sha256'] = file_hash(child_path); write_json(out / 'summary.json', summary)
    with pytest.raises(ValueError, match='worker original'):
        assess_run(inputs, plan, out)


def test_confirmation_seal_reuses_base_profile_and_requires_final_source_binding(run_files, monkeypatch, tmp_path):
    from pilot_eval.next_stage import seal_confirmation
    inputs, plan, run = run_files
    prepared = inputs.parent.parent
    source = {'id': 'train-source', 'blocks': [{'text': 'Exact source'}]}
    write_json(prepared / 'source.lock.json', {'training_source_scope': {source['id']: digest(source)}})
    final = tmp_path / 'final_sources.jsonl'; write_jsonl(final, [source])
    (prepared / 'demonstration').mkdir(); write_jsonl(prepared / 'demonstration/records.jsonl', [example(20), example(21)])
    (prepared / 'confirmation').mkdir()
    confirm = [{**example(99), 'split_role': 'confirmation_evaluation_only_never_training'}]
    write_jsonl(prepared / 'confirmation/records.jsonl', confirm)
    (prepared / 'development').mkdir(); write_jsonl(prepared / 'development/records.jsonl', [example(1)])
    write_json(prepared / 'confirmation.preseal.json', {
        'source_lock_sha256': file_hash(prepared / 'source.lock.json'),
        'records_sha256': file_hash(prepared / 'confirmation/records.jsonl'),
        'partition_artifacts': {f'{role}/records.jsonl': file_hash(prepared / role / 'records.jsonl')
                                for role in ['confirmation', 'demonstration', 'calibration', 'development']}})
    monkeypatch.setattr('pilot_eval.next_stage.tokenizer_assets', lambda _: (None, {'test': True}))
    def mock_decorate(row, _):
        row.update(native_input_tokens=100, total_budget_tokens=100 + row['max_new_tokens'])
        return row
    monkeypatch.setattr('pilot_eval.next_stage.decorate', mock_decorate)
    out = tmp_path / 'sealed'
    receipt = seal_confirmation(prepared, final, plan, run, 'mock', out)
    assert receipt['selected_profile'] == 'zero_8' and receipt['confirmation_results_observed'] == 0
    sealed = [json.loads(x) for x in (out / 'confirmation/records.jsonl').read_text().splitlines()]
    assert sealed[0]['id'] == 'boolq:99' and sealed[0]['max_new_tokens'] == 8
    assert sealed[0]['split_role'] == 'confirmation_evaluation_only_never_training'
    assert receipt['apply_to'] == 'all arms and seeds; no per-arm profile selection'
    write_jsonl(final, [{**source, 'blocks': []}])
    with pytest.raises(ValueError, match='source'):
        seal_confirmation(prepared, final, plan, run, 'mock', tmp_path / 'invalid')
