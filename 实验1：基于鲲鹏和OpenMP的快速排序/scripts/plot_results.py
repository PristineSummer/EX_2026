"""Standalone publication-friendly PNG/SVG figures and a Chinese findings note."""
import math
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"serial": "#9C6644", "std": "#596579", "omp-task": "#2563EB",
          "omp-partition": "#0D9488", "teaching": "#DC6464"}
LABELS = {"serial": "Serial introspective quicksort", "std": "std::sort",
          "omp-task": "OpenMP task quicksort", "omp-partition": "OpenMP parallel partition",
          "teaching": "Teaching task strategy"}


def generate_outputs(result, summary, metadata):
    data = pd.DataFrame(summary)
    figures = result / "figures"
    figures.mkdir(exist_ok=True)
    tuned = json.loads((result / "tuned_parameters.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.2,
        "figure.facecolor": "white", "savefig.facecolor": "white", "svg.fonttype": "none"})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    # A research point is the median of seed-specific medians. Quartile bands
    # are descriptive spread, not confidence intervals.
    main = data[data.stage == "main"]
    for metric, title, ylabel, filename in (
        ("median_ms", "Sorting time vs. threads", "Sorting time (ms)", "time_vs_threads"),
        ("speedup", "Speedup over optimized serial quicksort", "Speedup", "speedup"),
        ("efficiency", "Parallel efficiency", "Speedup / actual threads", "efficiency"),
        ("throughput_mitems_s", "Sorting throughput", "Million integers / second", "throughput")):
        fig, ax = plt.subplots(figsize=(9.4, 5.4))
        for algorithm in COLORS:
            if metric == "efficiency" and algorithm == "std": continue
            subset = main[main.algorithm == algorithm]
            if subset.empty: continue
            grouped = subset.groupby("actual_threads", sort=True).median(numeric_only=True)
            if algorithm in ("serial", "std"):
                ax.axhline(float(grouped[metric].iloc[0]), color=COLORS[algorithm], linestyle="--", label=LABELS[algorithm])
            else:
                ax.plot(grouped.index, grouped[metric], "o-", color=COLORS[algorithm], label=LABELS[algorithm], linewidth=2)
                if metric == "median_ms":
                    ax.fill_between(grouped.index, grouped.q1_ms, grouped.q3_ms, color=COLORS[algorithm], alpha=0.13)
        ax.set(title=title, xlabel="Actual OpenMP threads", ylabel=ylabel)
        ax.legend(fontsize=8)
        ax.set_xticks(metadata["profile"]["threads"])
        save(fig, filename)

    best = metadata["best_threads"]
    scale = data[(data.stage == "scale") | ((data.stage == "main") & (data.algorithm != "teaching"))]
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for algorithm in ("serial", "std", "omp-task", "omp-partition"):
        p = best.get(algorithm, 1)
        subset = scale[(scale.algorithm == algorithm) & (scale.actual_threads == p) & (scale.distribution == "random")]
        grouped = subset.groupby("size").median(numeric_only=True)
        ax.loglog(grouped.index, grouped.median_ms, "o-", color=COLORS[algorithm], linewidth=2,
                  label=f"{LABELS[algorithm]} (P={p})")
    ax.set(title="Scaling with input size", xlabel="Number of integers", ylabel="Sorting time (ms)")
    ax.legend(fontsize=8); save(fig, "size_scaling")

    distributions = ["random", "ascending", "descending", "nearly-sorted", "equal", "duplicates"]
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    x, width = np.arange(len(distributions)), 0.19
    for index, algorithm in enumerate(("serial", "std", "omp-task", "omp-partition")):
        p = best.get(algorithm, 1)
        subset = data[(data.stage == "distribution") & (data.algorithm == algorithm) & (data.actual_threads == p)]
        values = subset.groupby("distribution").median(numeric_only=True).reindex(distributions)
        ax.bar(x + (index - 1.5) * width, values.median_ms, width, color=COLORS[algorithm],
               label=f"{LABELS[algorithm]} (P={p})")
    ax.set_xticks(x, ["Random", "Ascending", "Descending", "Nearly sorted", "All equal", "16 distinct"])
    ax.set_yscale("log")
    ax.set(title="Sensitivity to input distribution", ylabel="Sorting time (ms, log scale)")
    ax.legend(fontsize=8); save(fig, "input_distributions")

    for stage, filename, title in (("tuning-task", "task_granularity", "Task cutoff calibration"),
                                   ("tuning-partition", "partition_calibration", "Parallel partition calibration")):
        subset = data[data.stage == stage].copy()
        if subset.empty: continue
        subset["configuration"] = subset.task_cutoff.map(lambda n: f"{n:,}") if stage == "tuning-task" else (
            subset.partition_cutoff.map(lambda n: f"{n:,}") + "; L=" + subset.partition_levels.astype(str))
        table = subset.pivot_table(index="configuration", columns="actual_threads", values="median_ms", aggfunc="median")
        ordering = ["task_cutoff"] if stage == "tuning-task" else ["partition_cutoff", "partition_levels"]
        order = subset.sort_values(ordering).drop_duplicates("configuration").configuration.tolist()
        table = table.reindex(order)
        fig, ax = plt.subplots(figsize=(max(7.4, 0.8 * len(table.columns)), 5.2))
        relative = table.divide(table.min(axis=0), axis=1)
        image = ax.imshow(relative.to_numpy(), aspect="auto", cmap="viridis_r")
        for y, configuration in enumerate(table.index):
            for x, p in enumerate(table.columns):
                selected = tuned["omp-task" if stage == "tuning-task" else "omp-partition"][str(p)]
                selected_label = f"{selected['task_cutoff']:,}" if stage == "tuning-task" else (
                    f"{selected['partition_cutoff']:,}; L={selected['partition_levels']}")
                value = float(table.iloc[y, x])
                ink = "white" if image.norm(float(relative.iloc[y, x])) > 0.55 else "#17202A"
                ax.text(x, y, f"{value:.1f}" + (" *" if configuration == selected_label else ""),
                        ha="center", va="center", color=ink, fontsize=7)
        ax.grid(False)
        ax.set_xticks(range(len(table.columns)), table.columns)
        ax.set_yticks(range(len(table.index)), table.index)
        ax.set(title=title + " (cells: ms; * selected)", xlabel="Actual OpenMP threads", ylabel="Cutoff (integers)")
        fig.colorbar(image, ax=ax, label="Time / best time at the same thread count")
        save(fig, filename)

    def geometric(values):
        return math.exp(statistics.mean(math.log(float(v)) for v in values))

    recommended = []
    for algorithm in ("serial", "std", "omp-task", "omp-partition", "teaching"):
        subset = main[main.algorithm == algorithm]
        if subset.empty: continue
        scores = {int(p): geometric(group.median_seconds) for p, group in subset.groupby("actual_threads")}
        p = min(scores, key=lambda key: (scores[key], key))
        chosen = subset[subset.actual_threads == p]
        efficiency = geometric(chosen.efficiency) if algorithm != "std" else None
        recommended.append((algorithm, p, scores[p] * 1000, geometric(chosen.speedup), efficiency))
    optimized = [row for row in recommended if row[0] in ("omp-task", "omp-partition")]
    winner = min(optimized, key=lambda row: row[2])
    serial_ms = next(row[2] for row in recommended if row[0] == "serial")
    teaching_ms = next((row[2] for row in recommended if row[0] == "teaching"), None)
    lines = ["# 快排实验分析", "", f"运行配置：`{metadata['profile']['name']}`；平台：`{metadata['platform']}`。",
        f"主实验规模：{metadata['profile']['main_size']:,} 个 32 位整数；评测种子：{metadata['profile']['seeds']}。",
        "", "## 主实验结果", "", "| 算法 | 最佳实际线程数 | 中位耗时 / ms | 相对串行加速比 | 并行效率 |",
        "|---|---:|---:|---:|---:|"]
    for a, p, ms, speed, eff in recommended:
        efficiency_text = f"{eff:.3f}" if eff is not None else "—"
        lines.append(f"| {a} | {p} | {ms:.3f} | {speed:.3f} | {efficiency_text} |")
    lines += ["", f"在本次随机输入主实验中，推荐 `{winner[0]}`、{winner[1]} 个线程，耗时 {winner[2]:.3f} ms。",
        f"相对优化串行快排（{serial_ms:.3f} ms），加速比为 {winner[3]:.3f}。"]
    if teaching_ms:
        lines.append(f"相对教学版的最佳线程配置，耗时比为 {teaching_ms / winner[2]:.3f}；这个比值不作为并行加速比。")
    if winner[3] <= 1:
        lines.append("本次并行方案未超过优化串行基准；结果如实保留，不作并行必然更快的结论。")
    lines += ["", "## 测量口径与限制", "",
        f"- 使用 `{metadata['timer']}`，计入排序、线程团队进入、任务调度及同步；生成、原始数据恢复、参考排序、辅助数组分配和逐元素校验在计时之外。",
        "- 每次测量都与同一份参考排序逐元素比较，跨算法输入指纹也必须一致。",
        "- 图中的阴影为四分位范围，不是置信区间；多种子图取各种子中位数的中位数，表格使用各种子中位数的几何平均。",
        "- 标准库只作为串行算法性能参考，不报告并行效率；调参图颜色表示相同线程数下相对最佳耗时，格内数字为毫秒，星号标记所选参数。",
        "- 调参种子为 2026，正式评测使用独立种子；参数来自本机，不能直接当作其他硬件的最佳值。",
        "- 任务快排的上层分区为串行；并行分区减少这部分瓶颈，但增加计数、写入及复制的内存访问。",
        f"- 并行分区主实验辅助数组占 {metadata['profile']['main_size'] * 4 / 1048576:.2f} MiB，另有分块元数据；输入、参考和工作数组各占同等空间。",
        "- 当前结果来自本机，不代表鲲鹏实测；需在 ARM/openEuler 目标机器重新编译并运行。",
        "- Windows 混合核心、功耗、温度和后台负载会影响波动；记录现有电源方案，不修改系统设置。",
        "", "## 文件索引", "",
        "- `raw.csv`：逐次计时、输入指纹、实际线程数与校验状态。",
        "- `summary.csv`：每个种子和配置的中位数、四分位范围、吞吐量、加速比与效率。",
        "- `tuned_parameters.json`：每种并行算法、每个线程数的所选参数。",
        "- `metadata.json` / `compile_commands.json`：环境、代码与二进制指纹、构建参数。",
        "- `figures/`：耗时、加速比、效率、吞吐量、规模、输入分布及两种调参图的 PNG/SVG。",
        "- `logs/`：构建、测试和每个实验进程的命令及输出。", ""]
    (result / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
