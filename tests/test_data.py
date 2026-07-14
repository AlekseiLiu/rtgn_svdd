import csv

import pytest
import torch

from rtgn_svdd.data import EventStream, load_cic_csv, split_chronologically


def make_events(num_events: int = 4) -> EventStream:
    return EventStream(
        src=torch.arange(num_events, dtype=torch.long),
        dst=torch.arange(1, num_events + 1, dtype=torch.long),
        t=torch.arange(10, 10 + num_events, dtype=torch.long),
        msg=torch.arange(num_events * 2, dtype=torch.float32).reshape(num_events, 2),
        is_attack=torch.tensor([False] * (num_events - 1) + [True]),
        is_injected=torch.zeros(num_events, dtype=torch.bool),
        attack_name=("normal",) * (num_events - 1) + ("example_attack",),
    )


def test_event_stream_accepts_the_public_tensor_contract():
    events = make_events()

    assert len(events) == 4
    assert events.src.shape == (4,)
    assert events.msg.shape == (4, 2)
    assert events.attack_name == ("normal", "normal", "normal", "example_attack")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("src", torch.tensor([[0, 1, 2, 3]]), "src must have shape"),
        ("dst", torch.tensor([1, 2, 3]), "dst must have shape"),
        ("t", torch.tensor([10, 12, 11, 13]), "nondecreasing timestamp"),
        ("msg", torch.ones(4, 2, dtype=torch.float64), "msg must use torch.float32"),
        ("is_attack", torch.zeros(4, dtype=torch.long), "must use torch.bool"),
        ("src", torch.tensor([0, 1, -1, 3]), "must be nonnegative"),
        ("msg", torch.tensor([[0.0], [1.0], [float("nan")], [3.0]]), "only finite"),
        (
            "is_injected",
            torch.tensor([False, False, False, True]),
            "cannot be an attack target",
        ),
        ("attack_name", ("normal",), "one value per event"),
    ],
)
def test_event_stream_rejects_contract_violations(field, value, message):
    fields = vars(make_events()).copy()
    fields[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        EventStream(**fields)


def test_fixed_split_is_exact_contiguous_and_independent():
    events = make_events(8)

    splits = split_chronologically(events, num_train=3, num_validation=2)

    assert splits.train.src.tolist() == [0, 1, 2]
    assert splits.validation.src.tolist() == [3, 4]
    assert splits.test.src.tolist() == [5, 6, 7]
    assert splits.test.attack_name == ("normal", "normal", "example_attack")
    splits.train.src[0] = 99
    assert events.src[0].item() == 0


@pytest.mark.parametrize(
    ("num_train", "num_validation", "message"),
    [(0, 2, "positive"), (2, 0, "positive"), (5, 3, "at least one test")],
)
def test_fixed_split_rejects_invalid_boundaries(num_train, num_validation, message):
    with pytest.raises(ValueError, match=message):
        split_chronologically(
            make_events(8), num_train=num_train, num_validation=num_validation
        )


def write_csv(path, rows, fieldnames=None):
    fieldnames = fieldnames or [
        "src_ip",
        "dst_ip",
        "bidirectional_first_seen_ms",
        "label",
        "attack",
        "ip_version",
        "feature_a",
        "feature_b",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def valid_rows():
    return [
        {
            "src_ip": "0",
            "dst_ip": "2",
            "bidirectional_first_seen_ms": "10",
            "label": "0",
            "attack": "normal",
            "ip_version": "4",
            "feature_a": "1.5",
            "feature_b": "2",
        },
        {
            "src_ip": "1",
            "dst_ip": "3",
            "bidirectional_first_seen_ms": "11",
            "label": "1",
            "attack": "example_attack",
            "ip_version": "4",
            "feature_a": "3.5",
            "feature_b": "4",
        },
    ]


def test_csv_loader_infers_numeric_features_and_preserves_metadata(tmp_path):
    path = tmp_path / "events.csv"
    write_csv(path, valid_rows())

    events = load_cic_csv(path)

    assert events.src.tolist() == [0, 1]
    assert events.dst.tolist() == [2, 3]
    assert events.t.tolist() == [10, 11]
    assert torch.equal(events.msg, torch.tensor([[1.5, 2.0], [3.5, 4.0]]))
    assert events.is_attack.tolist() == [False, True]
    assert events.is_injected.tolist() == [False, False]
    assert events.attack_name == ("normal", "example_attack")
    assert torch.equal(events.msg, load_cic_csv(path).msg)


def test_csv_loader_honors_explicit_feature_order(tmp_path):
    path = tmp_path / "events.csv"
    write_csv(path, valid_rows())

    events = load_cic_csv(path, feature_columns=("feature_b",))

    assert torch.equal(events.msg, torch.tensor([[2.0], [4.0]]))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ((0, "label", "2"), "label.*0 or 1"),
        ((0, "feature_a", "not-a-number"), "feature_a.*numeric"),
        ((0, "feature_a", "nan"), "feature_a.*finite"),
        ((1, "bidirectional_first_seen_ms", "9"), "nondecreasing timestamp"),
    ],
)
def test_csv_loader_rejects_invalid_values(tmp_path, change, message):
    rows = valid_rows()
    row, column, value = change
    rows[row][column] = value
    path = tmp_path / "invalid.csv"
    write_csv(path, rows)

    with pytest.raises(ValueError, match=message):
        load_cic_csv(path)


def test_csv_loader_reports_missing_required_and_requested_columns(tmp_path):
    path = tmp_path / "missing.csv"
    rows = valid_rows()
    for row in rows:
        del row["attack"]
    write_csv(path, rows, fieldnames=[key for key in valid_rows()[0] if key != "attack"])

    with pytest.raises(ValueError, match="missing.csv.*attack"):
        load_cic_csv(path)

    valid_path = tmp_path / "valid.csv"
    write_csv(valid_path, valid_rows())
    with pytest.raises(ValueError, match="valid.csv.*unknown_feature"):
        load_cic_csv(valid_path, feature_columns=("unknown_feature",))


def test_csv_loader_reports_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="does-not-exist.csv"):
        load_cic_csv(tmp_path / "does-not-exist.csv")


def test_csv_loader_rejects_a_header_only_file(tmp_path):
    path = tmp_path / "empty.csv"
    write_csv(path, [])

    with pytest.raises(ValueError, match="empty.csv.*no event rows"):
        load_cic_csv(path)
