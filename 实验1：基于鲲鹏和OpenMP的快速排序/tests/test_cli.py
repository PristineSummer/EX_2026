"""Executable contract and experiment statistics tests (standard library only)."""
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
EXE = Path(sys.argv.pop(1)).resolve()
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.experiment import choose_candidate, profile_config, summarize


def run(*args):
    return subprocess.run([str(EXE), *map(str, args)], capture_output=True,
                          encoding="utf-8", errors="replace", timeout=30,
                          cwd=PROJECT.parent)


class InterfaceTests(unittest.TestCase):
    def records(self, *args):
        result = run(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertTrue(all(record["verified"] for record in records))
        return records

    def test_legacy(self):
        result = run(2, 1024)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout.strip(), r"^\d+\.\d+ 1024$")

    def test_repeated_inputs_and_threads(self):
        fingerprints = set()
        for algorithm in ("serial", "omp-task", "omp-partition", "std", "teaching"):
            for p in (1, 2, 4):
                rows = self.records("--algorithm", algorithm, "--threads", p, "--size", 4097,
                    "--warmups", 1, "--repeats", 3, "--seed", 18446744073709551615)
                self.assertEqual([r["iteration"] for r in rows], [1, 2, 3])
                self.assertTrue(all(r["actual_threads"] == (1 if algorithm in ("serial", "std") else p) for r in rows))
                self.assertTrue(all(not r["diagnostics_enabled"] for r in rows))
                fingerprints.update(r["input_hash"] for r in rows)
        self.assertEqual(len(fingerprints), 1)

    def test_empty_and_singleton(self):
        for algorithm in ("serial", "omp-task", "omp-partition", "std", "teaching"):
            for n in (0, 1): self.records("--algorithm", algorithm, "--size", n, "--threads", 2)

    def test_small_sort_timer_resolution(self):
        rows = self.records("--algorithm", "serial", "--size", 1000, "--repeats", 5)
        self.assertTrue(all(row["seconds"] > 0 for row in rows))

    def test_forced_partition_diagnostics(self):
        row = self.records("--algorithm", "omp-partition", "--threads", 4, "--size", 200003,
            "--task-cutoff", 32, "--partition-cutoff", 64, "--diagnostics")[0]
        self.assertGreater(row["parallel_partitions"], 0)
        self.assertGreater(row["tasks"], 0)
        self.assertEqual(row["auxiliary_bytes"], 200003 * 4)

    def test_invalid_arguments(self):
        for args in (("--threads", "0"), ("--threads", "-1"), ("--size", "-1"),
                     ("--size", "18446744073709551616"), ("--algorithm", "missing"),
                     ("--distribution", "missing"), ("--task-cutoff", "0"),
                     ("--partition-levels", "65"), ("--repeats", "0"),
                     ("--seed", "-1"), ("--size",), ("--unknown", "2")):
            result = run(*args)
            self.assertNotEqual(result.returncode, 0, args)
            self.assertIn("ERROR", result.stderr)

    def test_help_and_build_info(self):
        self.assertEqual(run("--help").returncode, 0)
        info = json.loads(run("--build-info").stdout)
        self.assertGreaterEqual(info["openmp"], 201307)
        self.assertEqual(info["value_bits"], 32)

    def test_tuning_tie_break(self):
        rows = [dict(median_seconds=1.0, task_cutoff=4096, partition_cutoff=1048576, partition_levels=2),
                dict(median_seconds=1.009, task_cutoff=65536, partition_cutoff=4194304, partition_levels=1),
                dict(median_seconds=1.1, task_cutoff=262144, partition_cutoff=4194304, partition_levels=1)]
        self.assertEqual(choose_candidate(rows)["task_cutoff"], 65536)

    def test_seed_matched_baselines(self):
        rows = []
        for seed, serial_time in ((1024, 2.0), (4096, 6.0)):
            for algorithm, p, duration in (("serial", 1, serial_time), ("omp-task", 2, serial_time / 2)):
                rows.append(dict(stage="main", algorithm=algorithm, distribution="random", size=1000,
                    seed=seed, requested_threads=p, actual_threads=p, task_cutoff=65536,
                    partition_cutoff=1048576, partition_levels=2, seconds=duration, auxiliary_bytes=0))
        for row in summarize(rows):
            if row["algorithm"] == "omp-task":
                self.assertEqual(row["speedup"], 2.0); self.assertEqual(row["efficiency"], 1.0)
        self.assertEqual(profile_config("course", 18)["threads"], [1, 2, 3, 4, 5, 6, 8, 12, 14, 18])
        self.assertEqual(len(profile_config("research", 18)["seeds"]), 3)

    def test_standard_sort_is_not_parallel_efficiency(self):
        common = dict(stage="main", distribution="random", size=1000, seed=1024,
            requested_threads=1, actual_threads=1, task_cutoff=65536,
            partition_cutoff=1048576, partition_levels=2, auxiliary_bytes=0)
        rows = summarize([dict(common, algorithm="serial", seconds=2), dict(common, algorithm="std", seconds=1)])
        standard = next(row for row in rows if row["algorithm"] == "std")
        self.assertEqual(standard["speedup"], 2)
        self.assertIsNone(standard["efficiency"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
