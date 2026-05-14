# Experiment1_Optimization_Platform 工具说明

本目录集中存放实验一“Python-Julia 混合优化算法平台”的代码和运行入口。

## 目录约定

- 实验脚本：`tools/Experiment1_Optimization_Platform/run_all.py`
- Python 算法模块：`tools/Experiment1_Optimization_Platform/src/`
- Julia/JuMP 基准脚本：`tools/Experiment1_Optimization_Platform/julia/jump_rosenbrock.jl`
- 分析图表、JSON 和 CSV：`report_assets/Experiment1_Optimization_Platform/`
- Markdown 主报告：`Latex/Experiment1_Optimization_Platform/experiment1_optimization_platform.md`
- LaTeX/PDF 输出：`Latex/Experiment1_Optimization_Platform/experiment1_optimization_platform.tex` 和 `Latex/Experiment1_Optimization_Platform/experiment1_optimization_platform.pdf`

## 环境准备

```bash
conda env create -f tools/Experiment1_Optimization_Platform/environment.yml
conda activate config-exp1
```

如果环境已经存在，可直接使用：

```bash
conda run -n config-exp1 python tools/Experiment1_Optimization_Platform/run_all.py
```

## 正式实验命令

```bash
conda run -n config-exp1 python tools/Experiment1_Optimization_Platform/run_all.py
```

脚本会生成：

- `report_assets/Experiment1_Optimization_Platform/raw/jump_baseline_status.json`
- `report_assets/Experiment1_Optimization_Platform/raw/rosenbrock_first_order.json`
- `report_assets/Experiment1_Optimization_Platform/raw/rosenbrock_second_order.json`
- `report_assets/Experiment1_Optimization_Platform/raw/rosenbrock_pso.json`
- `report_assets/Experiment1_Optimization_Platform/raw/admm_lasso.json`
- `report_assets/Experiment1_Optimization_Platform/tables/rosenbrock_summary.csv`
- `report_assets/Experiment1_Optimization_Platform/tables/admm_lasso_summary.csv`
- `report_assets/Experiment1_Optimization_Platform/figures/*.png`

## JuMP 基准

Julia/JuMP 脚本会在首次运行时自动激活 `tools/Experiment1_Optimization_Platform/julia/Project.toml` 并实例化依赖。首次运行可能需要较长时间，因为 Julia 需要下载并预编译 JuMP、Ipopt 和 JSON。

当前正式运行已经得到 JuMP/Ipopt 基准：

```text
termination_status = LOCALLY_SOLVED
x = 0.9999999999999899
y = 0.9999999999999792
objective = 1.3288608467480825e-28
```

## 构建报告

```bash
cd Latex/Experiment1_Optimization_Platform
xelatex -interaction=nonstopmode experiment1_optimization_platform.tex
xelatex -interaction=nonstopmode experiment1_optimization_platform.tex
```
