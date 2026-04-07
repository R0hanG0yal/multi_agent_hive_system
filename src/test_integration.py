def test_integration_run():
    from src.environment import run_query
    out = run_query('What is a healthy snack to eat while coding?')
    assert 'fused' in out
    assert isinstance(out['responses'], list)
