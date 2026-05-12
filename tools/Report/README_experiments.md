# ConFIG 阅读报告实验复现说明

本目录集中存放 ConFIG 阅读报告使用的训练、分析和 LaTeX 构建工具。

## 目录约定

- 训练脚本：`tools/Report/run_burgers_experiments.py`
- 分析脚本：`tools/Report/analyze_burgers_results.py`
- LaTeX 构建脚本：`tools/Report/build_latex_report.py`
- 训练输出：`PINN_trained/Burgers/Report/`
- 分析图表与 JSON：`report_assets/Report/`
- Markdown 主报告：`Latex/ConFIG_Report/组合优化与凸优化阅读报告-ConFIG.md`
- LaTeX/PDF 输出：`Latex/ConFIG_Report/config_report.tex` 和 `Latex/ConFIG_Report/config_report.pdf`

## 正式实验命令

```bash
python3 tools/Report/run_burgers_experiments.py \
  --epochs 30000 \
  --num-run 3 \
  --methods standard pcgrad config mconfig \
  --device cuda:0 \
  --n-losses 2 \
  --name-suffix _full \
  --parallel-methods 4
```

## 分析命令

```bash
python3 tools/Report/analyze_burgers_results.py \
  --run-root PINN_trained/Burgers/Report \
  --methods standard_30000_full pcgrad_30000_full config_30000_full mconfig_30000_full \
  --out-dir report_assets/Report \
  --device cuda:0
```

分析脚本会生成：

- `burgers_experiment_summary.json`
- `burgers_validation_loss.png`
- `burgers_predictions_errors.png`
- `burgers_reference_solution.png`

## 构建报告

```bash
python3 tools/Report/build_latex_report.py
```

该脚本从 `Latex/ConFIG_Report/组合优化与凸优化阅读报告-ConFIG.md` 生成 `Latex/ConFIG_Report/config_report.tex`。

进入 `Latex/ConFIG_Report/` 后可使用 XeLaTeX 或 `xdvipdfmx` 生成 PDF：

```bash
xelatex -no-pdf config_report.tex
xelatex -no-pdf config_report.tex
xdvipdfmx -o config_report.pdf config_report.xdv
```
