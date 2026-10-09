# Legacy KB inventory and extraction boundary

The KB runtime has been copied into `src/aiq_kb`. The original
`ArchitectureIQ/aiq_rl` files and persistent runs remain in place.

## Source mapping

| Original source | Independent package |
|---|---|
| `aiq_rl/tools/kb_science_loop.py`, `kb_jobs.py`, `kb_lab.py`, `kb_sandbox.py` | Same filenames in `src/aiq_kb` |
| `cc_agent.py`, `codex_agent.py`, `kb_mcp_bridge.py`, `soa_index.py` | Same filenames, package imports |
| `kb_publish.py`, `kb_site.py`, `kb_loop_viewer.py`, `kb_loop_status.py` | Same filenames, configurable paths |
| `aiq_rl/docs/kb_viewer/` renderer and HTML/JS | Package files, included in wheel |
| `aiq_rl/docs/KB_SETTING.md` | `src/aiq_kb/KB_SETTING.md` |
| Seed, repeat-eval, growth, pilot and predictor scripts | Same filenames, external data and benchmark paths |
| Existing KB regression tests | `tests/`, with package imports and temporary fixtures |

## Ownership

- Code: this repository.
- New runs, jobs, research repos and reports: `AIQ_KB_DATA_ROOT/runs`.
- Benchmark implementation, release pool and runtime keys: external paths.
- Historical run and publisher: their original absolute paths.
- RL trainers, checkpoints, data and remote jobs: the RL project.

Do not point `Loop`, `Jobs` or `publish_once` at the historical run: they write
state and locks. Use `aiq-kb-inspect` or renderer `--exclude-test` with an output
outside that run. Extracted-source changes cover package imports, path
configuration, truthful external-seed provenance, packaged resources and
read-only inspection/export support.

## Verification

Package installation, fake-CLI async recovery, transaction tests and the legacy
read-only export have passed. Exported legacy config/split/KB/commits/metrics
hashes were unchanged. No solver, science or benchmark evaluation was started.

The source copy manifest is saved outside Git at
`LOCAL_CODEX_HOME/artifacts/aiq-kb-extraction-20261006/verification/source-copy-manifest.json`.
It records original/new hashes, including the pre-existing viewer fixes.
Current state and test commands are in [HANDOFF_2026-10-06.md](HANDOFF_2026-10-06.md).
