"""Private screenshot fixtures are optional and never published."""
from pathlib import Path
from unittest import SkipTest
from PIL import Image


def open_capture(path):
    if not Path(path).is_file():
        raise SkipTest('Optional local screenshot unavailable: ' + Path(path).name)
    return Image.open(path)
