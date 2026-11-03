# Case-study data bundles

Each study has one versioned download. JSONL training rows, train/development/test splits, and recorded evaluations retain their original format and paths inside the archive.

| Study | Download |
| --- | --- |
| Caveman | [caveman-v1.zip](caveman-v1.zip) |
| Humanizer | [humanizer-v1.zip](humanizer-v1.zip) |
| Anti-slop | [anti-slop-v1.zip](anti-slop-v1.zip) |
| E-vasion | [e-vasion-v1.zip](e-vasion-v1.zip) |
| Taste | [taste-v2.zip](taste-v2.zip) |
| Ask Before Acting | [ask-before-acting-v1.zip](ask-before-acting-v1.zip) |

A full Git checkout includes the archives. The notebook's first cell verifies and restores its bundle automatically. To use a script directly, run from `training/case-studies/`:

```bash
python common/study_data.py caveman
```

If downloading files individually, use GitHub's **Download raw file** button for the ZIP and save it in this `bundles/` directory. Keep the archive, `manifest.json`, notebook and helper from the same repository revision. Downloading the data alone does not install the cookbook's code dependencies.

The helper checks the archive SHA-256 and each member's checksum against `manifest.json`, rejects unsafe paths, and refuses to replace changed files. It uses only the Python standard library, makes no API requests and needs no key. Running it again verifies the existing data. To work with modified data, copy it to a separate directory and point your experiment there.

E-vasion's bundle includes the earlier lipogram study's data because the studies share evaluation and training code. That earlier notebook uses the same archive. Taste's bundle contains recorded comparison counts, briefs and rendering assets; explanatory tables remain readable in GitHub.

Extracted files are ignored by Git. Read any data links in the study guides after running setup locally; these paths are inside the bundle rather than individually hosted GitHub files. Source attribution and license documents remain alongside each study. Bundling does not change their terms.
