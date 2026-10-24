# 基于鲲鹏和 OpenMP 的快速排序

本项目实现可复现的单机共享内存并行快排实验：Windows 原生开发与评测，源码兼容 Linux / ARM 鲲鹏。OpenMP 使用线程；`hostfile` 是原始教材附件，不参与运行。

## 一条命令运行完整实验

在 PowerShell 中直接执行，无需激活虚拟环境或先切换目录：

```powershell
& "D:\MyDocuments__\Python_modelingCourse\.venv\Scripts\python.exe" -B "D:\MyDocuments__\Python_modelingCourse\EX_2026\实验1：基于鲲鹏和OpenMP的快速排序\run_experiment.py"
```

入口依次执行构建、正确性测试、自动调参、正式评测、统计与绘图。完整实验会反复排序 800 万元素及更大数据，耗时随 CPU 和功耗状态变化；控制台显示当前实验进度。建议接通电源，避免同时执行其他大量占用 CPU 的任务。

新结果保存在本实验的 `results/日期时间_course/`；`results/latest.json` 指向最近完成的实验。结果目录包含 `analysis.md`、原始与汇总 CSV、环境信息、所选参数、构建命令和 PNG/SVG 图表。每次运行使用独立目录。

## 实现与测量方法

| 名称 | 实现 |
|---|---|
| `serial` | Bentley–McIlroy 三路分区；三数取中/九数采样；小数组插入排序；深度受限时堆排序 |
| `omp-task` | 单个线程团队与根部 taskgroup；较大子区间创建任务；任务粒度和深度限制 |
| `omp-partition` | 上层分块计数、前缀偏移、互不重叠写入和复制回原数组；下层使用任务快排 |
| `std` | `std::sort` 性能参考 |
| `teaching` | 教材中间元素 pivot 和每次两个任务的策略，配合修正后的输入生成与校验 |

排序元素为 `int32_t`。数据由种子与下标通过定义明确的无符号整数运算产生，包含负数，不受线程调度和线程数影响。升/降序由相同随机数据排序得到；近乎有序数据执行约 N/100 次确定性交换；重复分布使用 16 种数值。

每次计时前恢复原始输入；计时后与 `std::sort` 参考结果逐元素比较。计时包含排序、线程团队进入、调度和同步，排除生成、输入复制、参考排序及辅助数组分配。并行分区计数和写入所需的元数据处理在计时内。Windows 使用 `QueryPerformanceCounter` 高精度计时（本机 MinGW 的 `omp_get_wtime` 为毫秒级，会使小数组测量得到零）；Linux 使用 `omp_get_wtime`。实际时钟记录在元数据中。

自动调参使用 800 万元素、种子 2026，按每个测试线程数比较任务阈值；并行分区随后比较分区阈值和层数。候选耗时相差不超过 1% 时，优先减少任务和分区层数。正式种子为 1024，不复用训练输入。并行分区的任务阈值继承对应线程数的任务快排结果，分区阈值和层数单独选择。

默认主实验使用 1–6、8、12、14、18 线程（超过本机逻辑处理器数的值会过滤）；优化算法预热两次、测量七次。教学版仅测 1–6 线程，各三次。补充实验覆盖 1 万、10 万、100 万、3200 万元素，以及六种输入分布，比较单线程和主实验中所选最佳配置。

加速比以同一个种子和输入上的优化串行快排为基准；效率为加速比除以实际线程数，标准库仅作为串行性能参考，不报告其并行效率。图中阴影表示四分位范围，不是置信区间。调参图按相同线程数归一化颜色，格内给出毫秒值和所选参数标记。诊断计数仅供单独观察机制，默认评测不开启。

## 扩展运行

上述命令后添加 `--profile research`，可测三个独立种子、更多规模/分布/线程组合。它会明显增加运行时间。`--profile smoke` 使用较小数据验证完整流程，不能替代课程实测结果。

构建后的 Windows 程序位于 `EX_2026/build/lab1/windows-release/quicksort.exe`。保留教材调用 `quicksort THREADS N`；扩展模式示例：

```text
quicksort --algorithm omp-partition --threads 6 --size 8000000 --seed 1024 --distribution random --task-cutoff 65536 --partition-cutoff 1048576 --partition-levels 2 --warmups 2 --repeats 7 --json
```

支持 `--diagnostics` 输出任务数、并行分区次数和堆排序回退次数；诊断模式含额外原子计数开销，不用于正式性能比较。

## 环境、文件与兼容

Python 固定使用既有 `.venv`，依赖为 NumPy、pandas、Matplotlib；C++ 使用 PATH 中的 G++、CMake、Ninja、CTest。入口不安装依赖、不修改系统环境。每次运行记录实际工具路径、版本、编译命令、二进制与源代码 SHA-256、CPU 和电源方案。

构建目录为 `EX_2026/build/lab1/`，临时和 Matplotlib 等缓存为 `EX_2026/.cache/lab1/`。使用 ASCII 构建/临时目录避免本机 Windows 工具在中文工作目录中的兼容问题；实验源码和结果仍位于原实验文件夹。

原始源码保存在 `reference/quick_sort_original.cpp`。教材随附 ARM 二进制 `code/quicksort` 保持不变；更新后的 Makefile 输出到独立构建目录。

Linux/鲲鹏需要支持 C++17、OpenMP taskloop 的 GCC（本项目要求 OpenMP 4.5 能力）、CMake、Ninja 和 Python 绘图依赖。复制整个 `EX_2026` 后，可在 Linux 用 `python3 -B 实验目录/run_experiment.py` 执行相同流程；纯 C++ 部分也可在 `code` 下 `make`，使用 `bash run.sh quicksort 6` 运行教材规模。无需 MPI。ARM 平台必须重新编译，Windows 或 WSL 的结果不代表鲲鹏实测。

## 正确性与限制

启动入口自动运行 C++ 测试和 Python 接口测试，覆盖空/单元素、阈值边界、负数、极值、重复值、多个线程与种子，并强制触发堆排序和多块并行分区。任何校验或实际线程数异常都会停止实验，日志和已完成的数据保留。

并行分区额外使用 N×4 字节辅助数组，另有少量分块元数据；实验校验框架还持有输入、参考和工作数组。教材版没有深度保护，只用于随机输入对照，不推荐作为通用排序接口。

默认不强制线程绑核，环境中的 OpenMP 绑定设置会记录在元数据中；混合核心、温度、功耗和后台进程可能影响结果。项目按实测推荐配置，不预设必达加速倍数。
