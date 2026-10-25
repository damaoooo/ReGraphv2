"""Split the lifted .ll files of a dataset into N size-balanced shards (greedy, biggest first).

Writes <out>/shards/shard_<k>.txt with .ll paths relative to <ll_root>. Each opt_level
shard job runs task2 and task3 on exactly one list, so a node never waits on another.
Existing lists are kept unless --force, so a resubmitted run sees the same partition.

    python Scripts/shaheen/plan_shards.py /scratch/zhoul0e/Dataset-1-ll /scratch/zhoul0e/Dataset-1-O2-fused 6
"""
import heapq
import os
import sys


def main():
    ll_root, out_root, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    force = "--force" in sys.argv[4:]
    shard_dir = os.path.join(out_root, "shards")
    existing = sorted(f for f in os.listdir(shard_dir) if f.startswith("shard_")) if os.path.isdir(shard_dir) else []
    if existing and not force:
        if len(existing) != n:
            sys.exit(f"{shard_dir} already has {len(existing)} shard lists, not {n}; pass --force to re-plan")
        print(f"keeping existing {n} shard lists in {shard_dir}")
        return

    files = []
    for d, _, names in os.walk(ll_root):
        for name in names:
            if name.endswith(".ll") and os.path.exists(os.path.join(d, name + ".lift.json")):
                p = os.path.join(d, name)  # the lift state is published last: only finished lifts
                files.append((os.path.getsize(p), os.path.relpath(p, ll_root)))
    files.sort(reverse=True)
    heap = [(0, k) for k in range(n)]
    shards = [[] for _ in range(n)]
    for size, rel in files:
        load, k = heapq.heappop(heap)
        shards[k].append(rel)
        heapq.heappush(heap, (load + size, k))

    os.makedirs(shard_dir, exist_ok=True)
    for k, rels in enumerate(shards):
        with open(os.path.join(shard_dir, f"shard_{k:03d}.txt"), "w") as f:
            f.write("".join(r + "\n" for r in sorted(rels)))
    loads = sorted(load for load, _ in heap)
    print(f"{len(files)} .ll into {n} shards; bytes per shard min={loads[0] / 1e9:.2f}G max={loads[-1] / 1e9:.2f}G")


if __name__ == "__main__":
    main()
