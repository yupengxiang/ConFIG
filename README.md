# 课程提交版说明

这个仓库包含以下 4 个交付物相关的代码、数据、训练结果和报告资产：

- 作业 2
- 实验 1：Python-Julia 混合优化算法平台
- 实验 2：结合 ADMM 与 ASGO 的稀疏 PINN 优化实验
- 阅读报告：ConFIG 阅读报告复现

如果只需要查看结果，可以直接打开各任务对应的 PDF 和图表；如果需要复现实验结果，也可以按下面的入口脚本重新运行分析或重建报告。

## 目录总览

- `tools/Homework2/`：作业 2 的代码入口
- `tools/Experiment1_Optimization_Platform/`：实验 1 的代码入口和详细说明
- `tools/Experiment2_ADMM_ASGO/`：实验 2 的训练、分析、报告构建入口和详细说明
- `tools/Report/`：阅读报告的训练、分析、报告构建入口和详细说明
- `report_assets/`：各任务生成的图表、表格、原始 JSON/CSV
- `Latex/`：各任务的 Markdown、LaTeX 和最终 PDF
- `PINN_trained/Burgers/`：实验 2 与阅读报告所需的已训练 Burgers PINN 结果
- `conflictfree/`、`experiments/PINN/`、`external/ASGO/`：实验 2 与阅读报告运行所需的底层代码

## 环境准备

### 通用 Python 环境

作业 2、实验 2 和阅读报告共用一套 Python 依赖思路。建议使用 Python 3.10，并在仓库根目录安装：

```bash
pip install -e .
pip install torch numpy scipy matplotlib tensorboard pyyaml einops tqdm
```

### 实验 1 环境

实验 1 单独提供了 conda 环境文件：

```bash
conda env create -f tools/Experiment1_Optimization_Platform/environment.yml
conda activate config-exp1
```

### LaTeX 环境

如果需要重新生成 PDF，请确保系统安装了 `xelatex`；作业 2 也可以使用 `latexmk -xelatex`。

## 作业 2

作业 2 的代码位于 `tools/Homework2/`，其中有两个入口：

- `tools/Homework2/run_core_tasks.py`：生成基础数值结果、表格和部分原始输出
- `tools/Homework2/run_report_tasks.py`：生成报告中与深度学习优化、PINN 参数轨迹、线性方程求解相关的图表与表格

运行方式：

```bash
python3 tools/Homework2/run_core_tasks.py
python3 tools/Homework2/run_report_tasks.py --device cpu
```

输出位置：

- `report_assets/Homework2/`
- `Latex/Homework2/`

最终报告文件：

- `Latex/Homework2/homework2_report.pdf`
- `Latex/Homework2/作业2.pdf`

注意：`run_report_tasks.py` 会直接读取 `PINN_trained/Burgers/Report/` 下的已训练模型、中间 checkpoint 和 `experiments/PINN/data/burgers/simulation_data.npy`，因此这些目录属于作业 2 的必需依赖。

## 实验 1

实验 1 的详细说明在：

- `tools/Experiment1_Optimization_Platform/README.md`

主入口：

```bash
conda run -n config-exp1 python tools/Experiment1_Optimization_Platform/run_all.py
```

输出位置：

- `report_assets/Experiment1_Optimization_Platform/`
- `Latex/Experiment1_Optimization_Platform/`

最终报告文件：

- `Latex/Experiment1_Optimization_Platform/experiment1_optimization_platform.pdf`

## 实验 2

实验 2 的详细说明在：

- `tools/Experiment2_ADMM_ASGO/README_experiments.md`

常用入口：

- 训练：`tools/Experiment2_ADMM_ASGO/run_burgers_experiments.py`
- 分析：`tools/Experiment2_ADMM_ASGO/analyze_burgers_results.py`
- 生成 LaTeX/PDF：`tools/Experiment2_ADMM_ASGO/build_experiment2_latex.py`

输出位置：

- 训练结果：`PINN_trained/Burgers/Experiment2_ADMM_ASGO/`
- 图表和统计：`report_assets/Experiment2_ADMM_ASGO/`
- 报告：`Latex/Experiment2_ADMM_ASGO/`

最终报告文件：

- `Latex/Experiment2_ADMM_ASGO/experiment2_admm_asgo.pdf`

说明：实验 2 除了依赖 `experiments/PINN/` 和 `conflictfree/`，还会通过 `experiments/PINN/lib_pinns/sparse_trainers.py` 使用 `external/ASGO/` 中的外部 ASGO 实现。

## 阅读报告

阅读报告的详细说明在：

- `tools/Report/README_experiments.md`
- `Latex/ConFIG_Report/README.md`

常用入口：

- 训练：`tools/Report/run_burgers_experiments.py`
- 分析：`tools/Report/analyze_burgers_results.py`
- 生成 LaTeX：`tools/Report/build_latex_report.py`

输出位置：

- 训练结果：`PINN_trained/Burgers/Report/`
- 图表和统计：`report_assets/Report/`
- 报告：`Latex/ConFIG_Report/`

最终报告文件：

- `Latex/ConFIG_Report/config_report.pdf`

## 建议的复现顺序

如果希望验证“结果能否重新整理出来”，建议按下面顺序执行：

1. 直接查看 `Latex/` 下的 4 份 PDF。
2. 如需重建图表，先运行各任务分析脚本，输出会写回 `report_assets/`。
3. 如需重建 LaTeX/PDF，再运行各任务的 LaTeX 构建脚本。
4. 只有在需要重新训练 PINN 时，才运行实验 2 或阅读报告的训练脚本。