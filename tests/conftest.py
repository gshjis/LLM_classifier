import pandas as pd
import pytest


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        {
            "text": ["win a prize now", "claim free money", "meeting moved to monday", "project update attached"] * 4,
            "label": ["spam", "spam", "not_spam", "not_spam"] * 4,
        }
    )
