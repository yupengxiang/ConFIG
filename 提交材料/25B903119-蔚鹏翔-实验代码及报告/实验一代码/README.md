# 实验一代码说明

本目录保留了实验一“Python-Julia 混合优化算法平台”的代码、必要结果和 LaTeX 源文件。

## 目录内容

- `tools/Experiment1_Optimization_Platform/`：实验脚本、源码、环境文件和原始说明
- `report_assets/Experiment1_Optimization_Platform/`：图表、JSON、CSV 结果
- `Latex/Experiment1_Optimization_Platform/`：报告的 Markdown 和 TeX 源文件

## 环境说明

未打包 conda 环境本体。建议在本目录根下执行：

```bash
conda env create -f tools/Experiment1_Optimization_Platform/environment.yml
conda activate config-exp1
conda run -n config-exp1 python tools/Experiment1_Optimization_Platform/run_all.py
```

## 数据说明

实验一不依赖单独外部数据集，所需结果文件已经保留在 `report_assets/Experiment1_Optimization_Platform/` 中。

## 报告说明

最终 PDF 不放在本目录，而放在上一级的 `实验一报告/` 中；如果需要重建 PDF，可在本目录中对 `Latex/Experiment1_Optimization_Platform/experiment1_optimization_platform.tex` 执行 XeLaTeX。