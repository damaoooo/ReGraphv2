"""
Provenance gate and data-selection policy for dataset building.

Integrity (not configurable): a results.db row is used only if its function file belongs to a
task3 function directory whose state is ok, the file still has the recorded sha256, and the
row was generated from that same content (input_sha256). Legacy rows/directories without
state need --allow-legacy-inputs and stay marked "legacy-unverified".

Selection (configurable, versioned): a policy file decides which verified functions enter a
dataset build, by lifter status and approximation tags. The policy, its hash and per-reason
counts are written to selection.json; every candidate is listed in provenance.jsonl with the
decision, so retained data and the data used for one training/evaluation run stay separable.
"""
import hashlib
import json
import os
import sqlite3
from collections import Counter

from Scripts import artifact_state

DEFAULT_POLICY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selection_policies", "keep_all_v1.json")


def load_policy(path: str) -> dict:
    with open(path, "rb") as handle:
        raw = handle.read()
    policy = json.loads(raw)
    policy["_path"] = os.path.abspath(path)
    policy["_sha256"] = hashlib.sha256(raw).hexdigest()
    return policy


def policy_decision(policy: dict, info: dict) -> str | None:
    """None if the function is selected, else the exclusion reason."""
    lift = info.get("lift") or {}
    if lift.get("status") in policy.get("exclude_lift_status", []):
        return f"lift_status:{lift.get('status')}"
    for tag in lift.get("approx", []) or []:
        if tag in policy.get("exclude_approx_tags", []):
            return f"approx:{tag}"
        for prefix in policy.get("exclude_approx_prefixes", []):
            if tag.startswith(prefix):
                return f"approx:{prefix}*"
    return None


def select_inputs(input_dir: str, db_file_origin: str, policy: dict, allow_legacy: bool = False):
    """Returns (selected paths, provenance rows, summary)."""
    selected_files, rejected_dirs = artifact_state.select_function_files(input_dir, allow_legacy=allow_legacy)
    with sqlite3.connect(f"file:{db_file_origin}?mode=ro", uri=True) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
        has_hash = "input_sha256" in columns
        query = "SELECT input_path, input_sha256 FROM results" if has_hash else "SELECT input_path, NULL FROM results"
        rows = conn.execute(query).fetchall()

    integrity = Counter({f"function_dir:{reason}": n for reason, n in rejected_dirs.items()})
    excluded, approx_selected, approx_all = Counter(), Counter(), Counter()
    provenance, chosen = [], []
    for path, row_hash in rows:
        info = selected_files.get(os.path.abspath(path))
        if info is None:
            integrity["row_not_in_valid_function_dir"] += 1
            continue
        if row_hash is None:
            if not (allow_legacy and info["input_state"] == "legacy-unverified"):
                integrity["row_without_input_sha256"] += 1
                continue
        elif row_hash != info["sha256"]:
            integrity["row_generated_from_other_content"] += 1
            continue
        tags = (info.get("lift") or {}).get("approx", []) or []
        approx_all.update(tags)
        reason = policy_decision(policy, info)
        if reason is None:
            chosen.append(path)
            approx_selected.update(tags)
        else:
            excluded[reason] += 1
        provenance.append({
            "input_path": path,
            "sha256": info["sha256"],
            "llvm_name": info.get("llvm_name"),
            "lift": info.get("lift"),
            "input_state": info.get("input_state"),
            "memory_contract": info.get("memory_contract"),
            "selected": reason is None,
            "exclusion": reason,
        })

    selection_digest = hashlib.sha256("".join(sorted(f"{r['input_path']}:{r['sha256']};" for r in provenance
                                                     if r["selected"])).encode()).hexdigest()
    summary = {
        "policy": {k: v for k, v in policy.items() if not k.startswith("_")},
        "policy_path": policy.get("_path"),
        "policy_sha256": policy.get("_sha256"),
        "allow_legacy_inputs": allow_legacy,
        "db_rows": len(rows),
        "rejected_for_integrity": dict(integrity),
        "candidates": len(provenance),
        "selected": len(chosen),
        "excluded_by_policy": dict(excluded),
        "approx_tags_candidates": dict(approx_all.most_common()),
        "approx_tags_selected": dict(approx_selected.most_common()),
        "input_states": dict(Counter(r["input_state"] for r in provenance)),
        "memory_contracts": dict(Counter(r["memory_contract"] for r in provenance)),
        "selection_digest": selection_digest,
    }
    return chosen, provenance, summary


def write_selection(output_dir: str, provenance: list, summary: dict):
    os.makedirs(output_dir, exist_ok=True)
    tmp = artifact_state.tmp_path_for(os.path.join(output_dir, "provenance.jsonl"))
    with open(tmp, "w") as handle:
        for row in provenance:
            handle.write(json.dumps(row) + "\n")
    os.replace(tmp, os.path.join(output_dir, "provenance.jsonl"))
    artifact_state.write_json_atomic(os.path.join(output_dir, "selection.json"), summary)
