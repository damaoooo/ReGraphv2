# Archived Baselines

HermesSim, CLAP, and DEJINA were archived on 2026-10-21. Their full local directories were removed only after SHA-256 verification, tar-to-source comparison, and a source metadata stability check succeeded.

## NAS Location

`/mnt/nas/ReLL-archives/2026-10-21-baselines/`

- Archive: `HermesSim-CLAP-DEJINA.tar.zst` (76.99 GiB).
- Guide: `README.md`.
- Integrity: `SHA256SUMS`.
- Inventory and space accounting: `source-manifest.json`, `summary.json`.
- Historical deduplication and input-provenance evidence: `baseline-dedup-provenance.jsonl`.

Measured local free-space increase: approximately 331.5 GiB. Shared inodes still referenced by the remaining local baselines were not freed.

## Restore

Ensure `HermesSim`, `CLAP`, and `DEJINA` do not already exist in the destination. From the repository root:

```bash
(cd /mnt/nas/ReLL-archives/2026-10-21-baselines && sha256sum -c SHA256SUMS)
tar --zstd -xpf /mnt/nas/ReLL-archives/2026-10-21-baselines/HermesSim-CLAP-DEJINA.tar.zst -C baseline
```

The archive restores all three directories and the `CLAP/jina -> ../DEJINA` compatibility link. No extra copy of the data remains at the former Downloads locations.

## Old Lifter Results

These are historical experiment artifacts. At least 1,518 groups of HermesSim binary inputs were byte-identical to explicit `.re` outputs in `binary_function_similarity/Binaries/Dataset-1-origin`. Removing the suffix did not remove their dependency on custom lifting/reoptimization. Results produced from those binaries need regeneration for the new lifter. CLAP and DEJINA input provenance and evaluation-pool alignment must also be checked before reuse.

The remaining local `binary_function_similarity` and `ReLL` projects, including their explicit `.re` files, were not removed by this archival operation.

## Old-Lifter Data (2026-10-25)

The old-lifter data in `binary_function_similarity` and `ReLL` was archived to
`/mnt/nas/ReLL-archives/2026-10-25-baseline-old-ir/` and removed locally. That covers the
`.re`/`.ll`/`.bc` inputs, IDBs, DBs, large Models results, `ReLL/data` datasets and
`ReLL/experiments`. Code, `.git`, uncommitted edits, `Dataset-Vulnerability` and the Qwen
models stay here. See that directory's README for package contents and restore commands.
