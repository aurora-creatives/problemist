from datetime import datetime, timedelta, timezone

from problemist.mock.data_loader import MockDataLoader


def _newest(loader: MockDataLoader) -> datetime:
    newest = max(t.created_at for t in loader.get_all_tickets())
    return newest if newest.tzinfo else newest.replace(tzinfo=timezone.utc)


def test_window_is_anchored_to_newest_fixture_ticket():
    """The time window must not depend on today's date, or the demo empties as time passes."""
    loader = MockDataLoader()
    assert len(loader.get_tickets(days=90)) == len(loader.get_all_tickets())


def test_days_filter_narrows_results():
    loader = MockDataLoader()
    narrow = loader.get_tickets(days=7)
    assert 0 < len(narrow) < len(loader.get_all_tickets())
    cutoff = _newest(loader) - timedelta(days=7)
    for t in narrow:
        created = t.created_at if t.created_at.tzinfo else t.created_at.replace(tzinfo=timezone.utc)
        assert created >= cutoff


def test_empty_fixture_set_falls_back_to_now(tmp_path):
    (tmp_path / "tickets.json").write_text("[]", encoding="utf-8")
    loader = MockDataLoader(fixtures_dir=tmp_path)
    assert loader.get_tickets(days=30) == []
