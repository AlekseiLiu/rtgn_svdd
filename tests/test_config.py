import json

import pytest

from rtgn_svdd.config import load_config, resolve_config, write_config


def test_config_round_trip_is_canonical_and_detached(tmp_path):
    source = {"schema_version": 1, "model": {"variant": "rtgn_svdd"}}
    path = tmp_path / "resolved.json"

    resolved = resolve_config(source)
    source["model"]["variant"] = "changed"
    write_config(resolved, path)

    assert load_config(path) == {
        "model": {"variant": "rtgn_svdd"},
        "schema_version": 1,
    }
    assert path.read_text(encoding="utf-8") == (
        '{\n  "model": {\n    "variant": "rtgn_svdd"\n  },\n'
        '  "schema_version": 1\n}\n'
    )


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"schema_version": ""},
        {"schema_version": 1, "invalid": float("nan")},
        {"schema_version": 1, "invalid": object()},
    ],
)
def test_resolve_config_rejects_invalid_values(config):
    with pytest.raises(ValueError):
        resolve_config(config)


def test_load_config_rejects_non_object_root(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps([{"schema_version": 1}]), encoding="utf-8")

    with pytest.raises(ValueError, match="root"):
        load_config(path)
