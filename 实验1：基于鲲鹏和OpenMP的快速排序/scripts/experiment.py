"""Build, verify, tune, measure, summarize. No simultaneous benchmark processes."""
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

PARALLEL = ("omp-task", "omp-partition")
DISTRIBUTIONS = ("random", "ascending", "descending", "nearly-sorted", "equal", "duplicates")
CONFIG_KEYS = ("task_cutoff", "partition_cutoff", "partition_levels")
GROUP_KEYS = ("stage", "algorithm", "distribution", "size", "seed", "requested_threads",
              "actual_threads", *CONFIG_KEYS)
RAW_FIELDS = ("case_id", *GROUP_KEYS, "iteration", "warmups", "seconds", "elapsed_ms",
              "input_hash", "verified", "auxiliary_bytes", "diagnostics_enabled",
              "tasks", "parallel_partitions", "heap_fallbacks")
DEFAULT_CONFIG = {"task_cutoff": 65536, "partition_cutoff": 1048576, "partition_levels": 2}


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def profile_config(name, cpu_count):
    threads = sorted({p for p in (1, 2, 3, 4, 5, 6, 8, 12, 14, 18) if p <= cpu_count})
    threads = threads or [1]
    profile = {"name": name, "threads": threads, "main_size": 8000000,
               "seeds": [1024], "training_seed": 2026, "warmups": 2, "repeats": 7,
               "supplement_repeats": 5, "tuning_warmups": 1, "tuning_repeats": 3,
               "task_candidates": [4096, 16384, 65536, 262144],
               "partition_candidates": [262144, 1048576, 4194304], "level_candidates": [1, 2],
               "scales": [10000, 100000, 1000000, 32000000], "shuffle_seed": 4102026}
    if name == "research":
        profile["seeds"] = [1024, 4096, 65537]
        profile["scales"] += [64000000]
    if name == "smoke":
        profile.update(threads=[p for p in (1, 2, 4) if p <= cpu_count] or [1],
                       main_size=200003, warmups=1, repeats=3, supplement_repeats=2,
                       tuning_repeats=2, task_candidates=[64, 1024, 4096, 16384],
                       partition_candidates=[65536, 262144], scales=[1000, 10000, 100000, 500000])
    return profile


def choose_candidate(candidates):
    """Within 1% of the best median, prefer fewer scheduling/memory passes."""
    best = min(row["median_seconds"] for row in candidates)
    eligible = [row for row in candidates if row["median_seconds"] <= best * 1.01]
    return max(eligible, key=lambda row: (row["task_cutoff"], -row["partition_levels"],
                                         row["partition_cutoff"]))


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in GROUP_KEYS), []).append(row)
    output = []
    for key, samples in groups.items():
        values = sorted(row["seconds"] for row in samples)
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else (values[0],) * 3
        median = statistics.median(values)
        item = dict(zip(GROUP_KEYS, key))
        item.update(samples=len(values), median_seconds=median, median_ms=median * 1000,
                    q1_ms=q1 * 1000, q3_ms=q3 * 1000, iqr_ms=(q3 - q1) * 1000,
                    min_ms=min(values) * 1000, max_ms=max(values) * 1000,
                    throughput_mitems_s=item["size"] / median / 1e6,
                    auxiliary_bytes=max(row["auxiliary_bytes"] for row in samples))
        output.append(item)
    baselines = {(row["stage"], row["distribution"], row["size"], row["seed"]): row["median_seconds"]
                 for row in output if row["algorithm"] == "serial"}
    for row in output:
        baseline = baselines.get((row["stage"], row["distribution"], row["size"], row["seed"]))
        row["speedup"] = baseline / row["median_seconds"] if baseline else None
        # std::sort is a different serial algorithm, not a parallel efficiency
        # result. Its relative performance remains available in speedup.
        row["efficiency"] = row["speedup"] / row["actual_threads"] if baseline and row["algorithm"] != "std" else None
    return output


def make_case(stage, algorithm, threads, size, seed, distribution, config, warmups, repeats):
    return dict(stage=stage, algorithm=algorithm, requested_threads=threads, size=size, seed=seed,
                distribution=distribution, warmups=warmups, repeats=repeats, **config)


class Experiment:
    def __init__(self, project, workspace, profile):
        self.project, self.workspace, self.profile = project, workspace, profile
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        self.result = project / "results" / f"{stamp}_{profile['name']}"
        self.result.mkdir(parents=True)
        self.logs = self.result / "logs"
        self.logs.mkdir()
        self.build = workspace / "build" / "lab1" / ("windows-release" if os.name == "nt" else "linux-release")
        self.exe = self.build / ("quicksort.exe" if os.name == "nt" else "quicksort")
        self.env = os.environ.copy()
        self.env["OMP_DYNAMIC"] = "FALSE"
        self.rows = []
        self.case_number = 0
        self.tuned = {}
        self.raw = (self.result / "raw.csv").open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.raw, fieldnames=RAW_FIELDS)
        self.writer.writeheader()
        self.raw.flush()
        self.metadata = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                         "profile": profile, "python": sys.executable, "python_version": sys.version,
                         "platform": platform.platform(), "machine": platform.machine(),
                         "logical_processors": os.cpu_count(),
                         "timer": "QueryPerformanceCounter" if os.name == "nt" else "omp_get_wtime",
                         "timed": "sorting, team setup, task scheduling and synchronization",
                         "untimed": "generation, reference sort, input copy, scratch allocation and exact verification",
                         "openmp_environment": {k: v for k, v in self.env.items() if k.startswith(("OMP_", "GOMP_"))},
                         "source_sha256": {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(project.rglob("*")) if p.is_file()
                             and not {"build", "results", ".cache"}.intersection(p.parts)
                             and (p.suffix in (".cpp", ".hpp", ".py", ".sh") or p.name in ("CMakeLists.txt", "Makefile"))},
                         "dependencies": {name: importlib.import_module(name).__version__
                                          for name in ("numpy", "pandas", "matplotlib")}}
        save_json(self.result / "metadata.json", self.metadata)

    def command(self, args, log_name, timeout=300):
        args = [str(arg) for arg in args]
        with (self.logs / log_name).open("a", encoding="utf-8") as log:
            log.write("COMMAND: " + subprocess.list2cmdline(args) + "\n")
            log.flush()
            process = subprocess.run(args, cwd=self.workspace, env=self.env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=timeout,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if process.returncode:
            raise RuntimeError(f"命令失败（退出码 {process.returncode}），详见 {self.logs / log_name}")

    def capture(self, args, timeout=30):
        result = subprocess.run([str(a) for a in args], cwd=self.workspace, env=self.env,
                                capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"退出码 {result.returncode}: {args[0]}")
        return result.stdout.strip()

    def build_and_test(self):
        tools = {name: shutil.which(name) for name in ("g++", "cmake", "ninja", "ctest")}
        if not all(tools.values()):
            raise RuntimeError("缺少构建工具：" + ", ".join(name for name, path in tools.items() if not path))
        self.env["PATH"] = str(Path(tools["g++"]).parent) + os.pathsep + self.env.get("PATH", "")
        print("[1/5] 构建 C++ / OpenMP Release 程序")
        self.command([tools["cmake"], "-S", self.project, "-B", self.build, "-G", "Ninja",
                      "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON",
                      "-DCMAKE_CXX_COMPILER=" + tools["g++"], "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], "build.log")
        self.command([tools["cmake"], "--build", self.build, "--parallel", str(min(os.cpu_count() or 1, 8))], "build.log")
        print("[2/5] 验证排序正确性、回退路径及命令行接口")
        self.command([tools["ctest"], "--test-dir", self.build, "--output-on-failure"], "tests.log")
        self.command([sys.executable, "-B", self.project / "tests" / "test_cli.py", self.exe], "tests.log")
        self.metadata.update(tools=tools, tool_versions={k: self.capture([v, "--version"]).splitlines()[0]
                                                        for k, v in tools.items()},
                             binary_sha256=hashlib.sha256(self.exe.read_bytes()).hexdigest(),
                             build_info=json.loads(self.capture([self.exe, "--build-info"])))
        commands = self.build / "compile_commands.json"
        if commands.exists():
            shutil.copyfile(commands, self.result / "compile_commands.json")
        if os.name == "nt":
            try:
                info = self.capture(["powershell.exe", "-NoProfile", "-Command",
                    "$p=Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors; "
                    "$m=Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory; "
                    "@{processor=$p;memory=$m} | ConvertTo-Json -Depth 4"])
                self.metadata["hardware"] = json.loads(info)
                self.metadata["power_scheme"] = self.capture(["powercfg", "/GETACTIVESCHEME"])
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                self.metadata["hardware_query_note"] = str(error)
        else:
            self.metadata["processor"] = platform.processor()
            if shutil.which("lscpu"):
                self.metadata["lscpu"] = self.capture(["lscpu"])
        save_json(self.result / "metadata.json", self.metadata)

    def measure(self, case):
        self.case_number += 1
        case_id = f"case_{self.case_number:04d}"
        args = [str(self.exe), "--json"]
        for key in ("algorithm", "distribution", "size", "seed", "warmups", "repeats", *CONFIG_KEYS):
            args += ["--" + key.replace("_", "-"), str(case[key])]
        args += ["--threads", str(case["requested_threads"])]
        stdout_path, stderr_path = self.logs / f"{case_id}.jsonl", self.logs / f"{case_id}.stderr.log"
        print(f"  {case_id} {case['stage']}: {case['algorithm']}, P={case['requested_threads']}, "
              f"N={case['size']:,}, {case['distribution']}")
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            stderr.write("COMMAND: " + subprocess.list2cmdline(args) + "\n"); stderr.flush()
            process = subprocess.Popen(args, cwd=self.workspace, env=self.env, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                code = process.wait(timeout=1800)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                process.kill(); process.wait(); raise
        if code:
            raise RuntimeError(f"{case_id} 失败，退出码 {code}；详见 {stderr_path}")
        samples = [json.loads(line) for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(samples) != case["repeats"]:
            raise RuntimeError(f"{case_id} 输出次数与配置不一致")
        expected_threads = 1 if case["algorithm"] in ("serial", "std") else case["requested_threads"]
        hashes = set()
        for iteration, sample in enumerate(samples, 1):
            if sample["verified"] is not True or sample["diagnostics_enabled"] or sample["actual_threads"] != expected_threads:
                raise RuntimeError(f"{case_id} 校验、实际线程数或计时模式不符合要求")
            for key in ("algorithm", "distribution", "size", "seed", "requested_threads", *CONFIG_KEYS):
                if sample[key] != case[key]:
                    raise RuntimeError(f"{case_id} 参数不一致：{key}")
            if sample["iteration"] != iteration or not math.isfinite(sample["seconds"]) or sample["seconds"] <= 0:
                raise RuntimeError(f"{case_id} 无效计时记录")
            hashes.add(sample["input_hash"])
            sample.update(case_id=case_id, stage=case["stage"])
            self.rows.append(sample); self.writer.writerow(sample)
        if len(hashes) != 1:
            raise RuntimeError(f"{case_id} 重复实验的输入不一致")
        self.raw.flush()
        return summarize(samples)[0]

    def run_cases(self, cases):
        rng = random.Random(self.profile["shuffle_seed"] + self.case_number)
        rng.shuffle(cases)
        return [self.measure(case) for case in cases]

    def tune(self):
        print("[3/5] 使用独立训练种子自动调参")
        profile = self.profile
        for p in profile["threads"]:
            if p == 1:
                for algorithm in PARALLEL: self.tuned[(algorithm, p)] = DEFAULT_CONFIG.copy()
                continue
            task_cases = []
            for cutoff in profile["task_candidates"]:
                config = dict(DEFAULT_CONFIG, task_cutoff=cutoff)
                task_cases.append(make_case("tuning-task", "omp-task", p, profile["main_size"],
                    profile["training_seed"], "random", config, profile["tuning_warmups"], profile["tuning_repeats"]))
            selected = choose_candidate(self.run_cases(task_cases))
            config = {key: selected[key] for key in CONFIG_KEYS}
            self.tuned[("omp-task", p)] = config
            partition_cases = []
            for cutoff in profile["partition_candidates"]:
                for levels in profile["level_candidates"]:
                    candidate = dict(config, partition_cutoff=cutoff, partition_levels=levels)
                    partition_cases.append(make_case("tuning-partition", "omp-partition", p, profile["main_size"],
                        profile["training_seed"], "random", candidate, profile["tuning_warmups"], profile["tuning_repeats"]))
            selected = choose_candidate(self.run_cases(partition_cases))
            self.tuned[("omp-partition", p)] = {key: selected[key] for key in CONFIG_KEYS}
            self.save_tuning()
        self.save_tuning()

    def save_tuning(self):
        save_json(self.result / "tuned_parameters.json", {algorithm: {str(p): config for (a, p), config
            in self.tuned.items() if a == algorithm} for algorithm in PARALLEL})

    def case(self, stage, algorithm, p, size, seed, distribution, repeats=None):
        return make_case(stage, algorithm, p, size, seed, distribution,
            self.tuned.get((algorithm, p), DEFAULT_CONFIG), self.profile["warmups"],
            repeats or self.profile["repeats"])

    def benchmark(self):
        print("[4/5] 正式评测：独立种子、原始数据恢复与逐元素验证")
        main = []
        for seed in self.profile["seeds"]:
            for algorithm in ("serial", "std"):
                main.append(self.case("main", algorithm, 1, self.profile["main_size"], seed, "random"))
            for algorithm in PARALLEL:
                for p in self.profile["threads"]:
                    main.append(self.case("main", algorithm, p, self.profile["main_size"], seed, "random"))
            for p in self.profile["threads"]:
                if p <= 6:
                    main.append(self.case("main", "teaching", p, self.profile["main_size"], seed, "random", repeats=3))
        main_summary = self.run_cases(main)
        best = {}
        for algorithm in PARALLEL:
            candidates = [row for row in main_summary if row["algorithm"] == algorithm]
            # Research chooses the geometric mean across seeds, rather than a lucky seed.
            scores = {p: math.exp(statistics.mean(math.log(row["median_seconds"]) for row in candidates
                                                if row["requested_threads"] == p)) for p in self.profile["threads"]}
            best[algorithm] = min(scores, key=lambda p: (scores[p], p))
        self.metadata["best_threads"] = best
        save_json(self.result / "metadata.json", self.metadata)
        supplements = []
        for seed in self.profile["seeds"]:
            scenarios = [("scale", n, "random") for n in self.profile["scales"]]
            scenarios += [("distribution", self.profile["main_size"], d) for d in DISTRIBUTIONS]
            if self.profile["name"] == "research":
                scenarios = [("scale", n, d) for n in self.profile["scales"] for d in DISTRIBUTIONS]
                scenarios += [("distribution", self.profile["main_size"], d) for d in DISTRIBUTIONS]
            for stage, n, distribution in scenarios:
                for algorithm in ("serial", "std"):
                    supplements.append(self.case(stage, algorithm, 1, n, seed, distribution,
                                                 self.profile["supplement_repeats"]))
                for algorithm in PARALLEL:
                    for p in sorted({1, best[algorithm]} if self.profile["name"] != "research" else set(self.profile["threads"])):
                        supplements.append(self.case(stage, algorithm, p, n, seed, distribution,
                                                     self.profile["supplement_repeats"]))
        self.run_cases(supplements)

    def finish(self):
        print("[5/5] 汇总统计结果、性能图表和中文分析")
        summary = summarize(self.rows)
        with (self.result / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
        # Verify that all algorithms/threads in a scenario sorted exactly the same input.
        scenario_hashes = {}
        for row in self.rows:
            key = (row["distribution"], row["size"], row["seed"])
            scenario_hashes.setdefault(key, set()).add(row["input_hash"])
        if any(len(values) != 1 for values in scenario_hashes.values()):
            raise RuntimeError("算法之间的输入指纹不一致")
        from .plot_results import generate_outputs
        generate_outputs(self.result, summary, self.metadata)
        self.metadata.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                             measured_cases=self.case_number, verified_measurements=len(self.rows))
        save_json(self.result / "metadata.json", self.metadata)
        save_json(self.project / "results" / "latest.json", {"directory": str(self.result), "profile": self.profile["name"]})
        print(f"完成：{len(self.rows)} 次正式计时均通过精确校验。\n结果目录：{self.result}\n分析摘要：{self.result / 'analysis.md'}")


def main(project, workspace):
    parser = argparse.ArgumentParser(description="自动构建、验证、调参、评测并生成快排实验图表")
    parser.add_argument("--profile", choices=("course", "research", "smoke"), default="course",
                        help="course：课程完整实验；research：扩展研究；smoke：开发流程验证")
    args = parser.parse_args()
    experiment = None
    start = time.monotonic()
    try:
        experiment = Experiment(project, workspace, profile_config(args.profile, os.cpu_count() or 1))
        print(f"实验配置：{args.profile}；结果保存在 {experiment.result}")
        experiment.build_and_test(); experiment.tune(); experiment.benchmark(); experiment.finish()
        print(f"总运行时间：{(time.monotonic() - start) / 60:.1f} 分钟")
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if experiment is not None:
            experiment.metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error) or "用户中断", elapsed_seconds=time.monotonic() - start)
            save_json(experiment.result / "metadata.json", experiment.metadata)
            (experiment.logs / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"实验停止，已完成的数据保留在：{experiment.result}", file=sys.stderr)
        print(f"错误：{error or '用户中断'}", file=sys.stderr)
        return 1
    finally:
        if experiment is not None: experiment.raw.close()
