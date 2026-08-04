import builtins
import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "serving" / "app.py"


def test_app_imports_without_model_artifact(monkeypatch):
    monkeypatch.setattr("sys.modules", {})
    monkeypatch.syspath_prepend(str(ROOT))

    spec = importlib.util.spec_from_file_location("serving.app", APP_PATH)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert module.app is not None
    assert module.model is None


def test_app_imports_without_pyspark(monkeypatch):
    monkeypatch.setattr("sys.modules", {})
    monkeypatch.syspath_prepend(str(ROOT))

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pyspark"):
            raise ModuleNotFoundError("No module named 'pyspark'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location("serving.app", APP_PATH)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert module.app is not None
    assert module.model is None
