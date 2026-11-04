
from pathlib import Path as _TestPath
import sys as _test_sys
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1]))

import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
from study_data import ensure_data
ROOT = Path(__file__).resolve().parents[2]

class BundleTests(unittest.TestCase):
    def test_shipped_bundles_restore_exact_bytes_idempotently(self):
        catalog = json.loads((ROOT / 'bundles/manifest.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'bundles', root / 'bundles')
            for study, entry in catalog['bundles'].items():
                with contextlib.redirect_stdout(io.StringIO()):
                    ensure_data(study, root)
                    ensure_data(study, root)
                for name, expected in entry['files'].items():
                    self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)

    def fixture(self, root, members):
        (root / 'bundles').mkdir()
        archive = root / 'bundles/test.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            for name, data in members.items(): z.writestr(name, data)
        entry = {'archive': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                 'files': {k: hashlib.sha256(v).hexdigest() for k,v in members.items()}}
        (root / 'bundles/manifest.json').write_text(json.dumps({'version':1,'bundles':{'test':entry}}))
        return archive

    def test_preserves_modified_data_without_partial_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, {'study/new.json':b'{}','study/old.json':b'[]'})
            (root / 'study').mkdir()
            target = root / 'study/old.json'; target.write_text('user edits')
            with self.assertRaises(FileExistsError): ensure_data('test', root)
            self.assertEqual(target.read_text(), 'user edits')
            self.assertFalse((root / 'study/new.json').exists())

    def test_rejects_corrupt_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive = self.fixture(root, {'study/data.json':b'{}'})
            archive.write_bytes(archive.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'Checksum'): ensure_data('test', root)
            self.assertFalse((root / 'study').exists())

    def test_rejects_traversal_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root, {'../escape.json':b'{}'})
            with self.assertRaisesRegex(ValueError, 'Unsafe'): ensure_data('test', root)

if __name__ == '__main__': unittest.main()
