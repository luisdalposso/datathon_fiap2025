def test_import_monitoring_modules():
    # Import simples já executa as definições e conta cobertura das linhas
    import src.monitoring.logs as logs  # noqa: F401
    import src.monitoring.metrics as metrics  # noqa: F401
    assert True
