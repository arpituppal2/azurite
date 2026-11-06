# Molab execution runbook

GPU host: pinned URL per session (sessions die, e.g. HTTP 410; workspace volume
may or may not survive rotation — treat as disposable, re-stage from local manifest).

## File transfer

- `scripts/molab_put_direct.py --put LOCAL:REMOTE` reports success unreliably.
- **Rule (2026-11-03 incident): a transfer tool's success response is not evidence.
  Before launching anything transmitted to the host, verify by reading the remote
  file back through the execute channel (`execute-code.sh`) and comparing its
  content hash against the local `sha256sum`. Launch if and only if they match.**
- Incident: put of `scripts/diag_dual_random.py` reported `done` twice while the
  remote bytes stayed at the old sha (`6554c9b1...` vs local `dbe39660...`).
  Workaround that succeeded: base64-embed the file in a small deliver script run
  via `execute-code.sh`, which writes and hashes the bytes in the same step.
- Same rule applies in reverse: after a run, pull artifacts back (base64 pack)
  and confirm line counts / hashes locally before concluding anything.

## Launch pattern

- Write logs to a path inside an existing directory (a fresh script's own
  `mkdir` runs too late for the shell redirect). `mkdir -p` the output dir first.
- Detach with `nohup ... &`, poll with small status scripts; keep an internal
  wall guard in the script plus a SYNC timeout above it.
- Preserve off-host: script (hash), config, logs. No checkpoint writes unless
  the approval explicitly includes them.
