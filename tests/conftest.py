import sys
from pathlib import Path
import pytest
from PIL import Image

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
collect_ignore = ['fixtures/tmp-pytest-2']


@pytest.fixture
def knowledge():
    from rulz_optimizer.knowledge import load
    return load(APP / 'tests' / 'fixtures' / 'knowledge')


@pytest.fixture
def dataset(tmp_path):
    Image.new('RGB', (32, 40), 'red').save(tmp_path / 'one.png')
    (tmp_path / 'one.txt').write_text('person, portrait', encoding='utf-8')
    return tmp_path
