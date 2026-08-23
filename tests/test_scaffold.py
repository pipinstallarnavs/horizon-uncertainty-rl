def test_package_imports_and_has_version():
    import horizon
    assert isinstance(horizon.__version__, str)
    assert horizon.__version__
