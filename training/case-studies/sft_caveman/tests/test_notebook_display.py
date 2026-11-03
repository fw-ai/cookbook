"""Offline smoke checks: python -m unittest discover -s tests -p test_notebook_display.py -v."""

from pathlib import Path as _TestPath
import sys as _test_sys
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1]))

import builtins
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from notebook_display import answers, figure


class NotebookDisplayTest(unittest.TestCase):
    def test_figure_saves_and_preserves_run_label(self):
        rows = [dict(arm=a, id="one", draw=0, family="technical", **{"pass": True}, words=5)
                for a in ["base", "skill", "adapter"]]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "figure.png"
            fig = figure(rows, {"one": {"group_id": "g"}}, title="My development run", path=path, show=False)
            try:
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") and
                                fig._suptitle.get_text().startswith("My development run"))
            finally:
                plt.close(fig)

    def test_answers_without_ipython(self):
        original = builtins.__import__
        def without_ipython(name, *args, **kwargs):
            if name == "IPython" or name.startswith("IPython."):
                raise ImportError("IPython intentionally unavailable")
            return original(name, *args, **kwargs)
        sample = {"fixture": {"question": "Which port?", "facts": ["Port is 8080."]},
                  "answers": [{"arm": a, "words": 2, "pass": True, "text": "Port 8080."}
                              for a in ["base", "skill", "adapter"]]}
        output = io.StringIO()
        with patch("builtins.__import__", side_effect=without_ipython), contextlib.redirect_stdout(output):
            answers(sample)
        self.assertTrue(all(text in output.getvalue() for text in
                            ["Which port?", "Given: Port is 8080.", "Base", "Skill prompt", "Adapter", "Port 8080."]))


if __name__ == "__main__":
    unittest.main()
