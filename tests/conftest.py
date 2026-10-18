"""Initialize clean-checkout test data from the approved deployment seed only."""
from __future__ import annotations

from src.dynamic_risk import current_state


def pytest_sessionstart(session):
    # Preserve every existing local snapshot. No collection or refresh is run.
    if current_state.bootstrap_current_data():
        # Read-only scenario tests hash this optional file. The deployment seed
        # intentionally contains no raw news, so represent its absence explicitly.
        (current_state.CURRENT_DIR / "current_news.json").write_text(
            '{"articles": []}\n', encoding="utf-8"
        )
