"""Offline tests for the mechanical checks in build_humanizer_pack."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_humanizer_pack import em_dash_added, preservation_checks  # noqa: E402

ROW = {"source": "We moved the deploy to Thursday. It took eighteen minutes to fix.", "must_preserve": ["Thursday"],
       "protected": [], "audience": "team", "cls": "good"}


def test_em_dash_added_is_a_problem():
    assert em_dash_added("We moved the deploy \u2014 to Thursday.", ROW["source"])
    assert em_dash_added("We moved the deploy -- to Thursday.", ROW["source"])
    assert "em dash added" in preservation_checks("Deploy moved \u2014 Thursday. Eighteen minutes to fix.", ROW)


def test_a_dash_the_author_wrote_is_kept():
    src = "We moved the deploy \u2014 to Thursday."
    assert not em_dash_added("We moved the deploy \u2014 to Thursday, as planned.", src)
    assert not em_dash_added("We moved the deploy to Thursday.", src)
    assert "em dash added" not in preservation_checks("We moved the deploy to Thursday. It took eighteen minutes to fix.", ROW)
