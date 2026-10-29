"""Keep the executable examples in the published README in sync with the API."""

from pathlib import Path
import re

import pytest


README = Path(__file__).resolve().parents[1] / "README.md"
EXAMPLES = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def test_readme_contains_examples():
    assert len(EXAMPLES) == 2


@pytest.mark.parametrize("source", EXAMPLES, ids=["quickstart", "application"])
def test_readme_example(source):
    exec(compile(source, str(README), "exec"), {"__name__": "__main__"})
