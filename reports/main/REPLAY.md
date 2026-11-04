# Reanalyze saved outputs without loading a model

From the repository root:

```sh
.venv/bin/watermark-lab detect --config reports/main/replay.json
.venv/bin/watermark-lab evaluate --config reports/main/replay.json
.venv/bin/watermark-lab plot --config reports/main/replay.json
```

The original run manifest retains original paths and settings. The replay configuration changes only artifact paths. Source hashes describe the implementation at archival time. C4 attribution and original document URLs are retained in the prompt records; see `docs/protocol.md`.
