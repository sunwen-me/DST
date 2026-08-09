"""配置加载: YAML → 嵌套 SimpleNamespace，支持点号访问。"""
from types import SimpleNamespace
from pathlib import Path
import yaml

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "default.yaml")


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str = None):
    """加载 YAML 配置为嵌套 SimpleNamespace。path 为 None 时用 default.yaml。"""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ns = _to_ns(raw)
    ns._raw = raw
    return ns
