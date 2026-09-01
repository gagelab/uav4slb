#!/usr/bin/env python3
"""
select_cpu_affinity.py
=======================
Pick a NUMA-local, currently-idle set of physical CPU cores for a training
job pinned to a given GPU, based on a live snapshot of the machine (not a
hardcoded topology) — so the same launcher works on any multi-socket host.

Steps:
  1. Map the requested GPU to its local NUMA node via `nvidia-smi topo -m`.
  2. Enumerate physical cores on that node via `lscpu -p` (one representative
     CPU per core, so hyperthread siblings aren't double-counted).
  3. Rank those CPUs by idle % from a short /proc/stat sample.
  4. Warn (to stderr) if the target node is low on free memory — binding
     there could OOM even if its CPUs are idle.

Prints two shell-assignment lines to stdout for `eval "$(...)"`:
  AFFINITY_PREFIX="numactl --cpunodebind=N --membind=N taskset -c c1,c2,..."
  AFFINITY_OMP_THREADS=<n>

On any failure (tool missing, GPU row not found, single-node machine, etc.)
prints an empty AFFINITY_PREFIX so the caller falls back to unpinned
execution rather than aborting the launch.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def gpu_numa_node(gpu_id: int) -> int:
    topo = sh(["nvidia-smi", "topo", "-m"])
    for line in topo.splitlines():
        if not re.match(rf"^GPU{gpu_id}\b", line):
            continue
        cpu_lists = re.findall(r"\d+(?:,\d+)+", line)
        if not cpu_lists:
            break
        first_cpu = int(cpu_lists[0].split(",")[0])
        return cpu_of_node(first_cpu)
    raise RuntimeError(f"GPU{gpu_id} not found in `nvidia-smi topo -m` output")


def cpu_of_node(cpu: int) -> int:
    for cpu_id, _core, node in lscpu_topology():
        if cpu_id == cpu:
            return node
    raise RuntimeError(f"CPU{cpu} not found in `lscpu -p` output")


def lscpu_topology():
    """Yield (cpu, core, node) for every logical CPU."""
    out = sh(["lscpu", "-p=CPU,CORE,NODE"])
    for line in out.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, node = (int(x) for x in line.split(","))
        yield cpu, core, node


def physical_cores_on_node(node: int):
    """One representative (lowest-numbered) CPU per physical core on `node`."""
    by_core = {}
    for cpu, core, cpu_node in lscpu_topology():
        if cpu_node != node:
            continue
        by_core.setdefault(core, cpu)
        by_core[core] = min(by_core[core], cpu)
    return sorted(by_core.values())


def idle_pct_by_cpu(sample_secs: float):
    def read():
        stats = {}
        with open("/proc/stat") as f:
            for line in f:
                if not line.startswith("cpu") or line.startswith("cpu "):
                    continue
                parts = line.split()
                cpu = int(parts[0][3:])
                fields = [int(x) for x in parts[1:]]
                idle = fields[3] + fields[4]  # idle + iowait
                total = sum(fields)
                stats[cpu] = (idle, total)
        return stats

    before = read()
    time.sleep(sample_secs)
    after = read()

    idle_pct = {}
    for cpu, (idle0, total0) in before.items():
        idle1, total1 = after[cpu]
        d_idle, d_total = idle1 - idle0, total1 - total0
        idle_pct[cpu] = 100.0 * d_idle / d_total if d_total > 0 else 100.0
    return idle_pct


def node_free_gb(node: int):
    out = sh(["numactl", "--hardware"])
    m = re.search(rf"node {node} free: (\d+) MB", out)
    return int(m.group(1)) / 1024.0 if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--num-cores", type=int, default=8)
    ap.add_argument("--min-free-gb", type=float, default=50.0)
    ap.add_argument("--sample-secs", type=float, default=0.3)
    args = ap.parse_args()

    for tool in ("nvidia-smi", "lscpu", "numactl", "taskset"):
        if shutil.which(tool) is None:
            print(f"# select_cpu_affinity: '{tool}' not found — skipping CPU pinning", file=sys.stderr)
            print('AFFINITY_PREFIX=""')
            print("AFFINITY_OMP_THREADS=1")
            return

    try:
        node = gpu_numa_node(args.gpu)
        candidates = physical_cores_on_node(node)
        if not candidates:
            raise RuntimeError(f"no physical cores found on NUMA node {node}")

        idle = idle_pct_by_cpu(args.sample_secs)
        ranked = sorted(candidates, key=lambda c: idle.get(c, 0.0), reverse=True)
        chosen = sorted(ranked[: args.num_cores])

        free_gb = node_free_gb(node)
        if free_gb is not None and free_gb < args.min_free_gb:
            print(
                f"# WARNING: NUMA node {node} has only {free_gb:.0f} GB free "
                f"(< {args.min_free_gb:.0f} GB threshold) — binding here risks OOM. "
                f"Consider a GPU on the other NUMA node.",
                file=sys.stderr,
            )

        print(
            f"# GPU{args.gpu} -> NUMA node {node} | "
            f"cores {chosen} | idle% {[round(idle.get(c, 0.0)) for c in chosen]} | "
            f"node free {free_gb:.0f} GB" if free_gb is not None else "",
            file=sys.stderr,
        )

        omp_threads = max(1, len(chosen) // 4)
        cpu_list = ",".join(str(c) for c in chosen)
        print(f'AFFINITY_PREFIX="numactl --cpunodebind={node} --membind={node} taskset -c {cpu_list}"')
        print(f"AFFINITY_OMP_THREADS={omp_threads}")

    except (subprocess.CalledProcessError, RuntimeError) as e:
        print(f"# select_cpu_affinity: {e} — skipping CPU pinning", file=sys.stderr)
        print('AFFINITY_PREFIX=""')
        print("AFFINITY_OMP_THREADS=1")


if __name__ == "__main__":
    main()
