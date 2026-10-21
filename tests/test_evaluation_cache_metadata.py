from evaluation import _normalize_export_metadata


def test_previous_modernbert_metadata_remains_readable():
    old = {"legacy": False, "rows": 123, "fingerprint": "test"}
    assert _normalize_export_metadata(old) == {
        "model_type": "modernbert", "rows": 123, "fingerprint": "test",
    }
    assert old["legacy"] is False


def test_old_roformer_cache_is_not_relabelled_as_modernbert():
    old = {"legacy": True, "rows": 123}
    assert _normalize_export_metadata(old) == old
    assert _normalize_export_metadata(old) != {"model_type": "modernbert", "rows": 123}


def test_current_metadata_is_unchanged():
    metadata = {"model_type": "modernbert", "rows": 123}
    assert _normalize_export_metadata(metadata) == metadata
