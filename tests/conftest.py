import pytest
import pandas as pd

import settings

STATIC_DIR = settings.BASE_DIR / 'tests' / 'static'


@pytest.fixture
def dataset():
    return pd.read_csv(STATIC_DIR / 'example_20packets.csv', na_filter='')
