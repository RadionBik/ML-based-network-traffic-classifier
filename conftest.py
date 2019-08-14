import os

import pytest


ROOT = os.path.dirname(__file__)

@pytest.fixture
def filename():
    def _filename(local_filename):
        return os.path.join(ROOT, local_filename)
    return _filename


@pytest.fixture
def static(filename):
    def _static(local_filename):
        return filename('tests/static/'+local_filename)
    return _static