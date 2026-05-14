# Experiment2_ADMM_ASGO 工具说明

本目录集中存放实验二“ADMM-ASGO 稀疏 PINN 优化实验”的训练、分析和写作工具。

## 目录约定

- 训练脚本：`tools/Experiment2_ADMM_ASGO/run_burgers_experiments.py`
- 分析脚本：`tools/Experiment2_ADMM_ASGO/analyze_burgers_results.py`
- LaTeX/PDF 构建脚本：`tools/Experiment2_ADMM_ASGO/build_experiment2_latex.py`
- 训练输出：`PINN_trained/Burgers/Experiment2_ADMM_ASGO/`
- 分析图表与 JSON：`report_assets/Experiment2_ADMM_ASGO/`
- Markdown 主报告：`Latex/Experiment2_ADMM_ASGO/experiment2_admm_asgo.md`
- LaTeX/PDF 输出：`Latex/Experiment2_ADMM_ASGO/experiment2_admm_asgo.tex` 和 `Latex/Experiment2_ADMM_ASGO/experiment2_admm_asgo.pdf`

## 正式实验命令

```bash
python3 tools/Experiment2_ADMM_ASGO/run_burgers_experiments.py \
  --epochs 30000 \
  --num-run 3 \
  --methods standard config mconfig asgo adam_l1 asgo_l1 admm_asgo \
  --device cuda:0 \
  --save-path ./PINN_trained/Burgers/Experiment2_ADMM_ASGO/ \
  --name-suffix _exp2_full \
  --save-epoch 30000 \
  --final-record-epoch 500 \
  --asgo-matalg-steps 5 \
  --admm-lambda 1e-5 \
  --admm-rho 1e-2 \
  --parallel-methods 4
```

## 分析命令

```bash
python3 tools/Experiment2_ADMM_ASGO/analyze_burgers_results.py \
  --run-root PINN_trained/Burgers/Experiment2_ADMM_ASGO \
  --methods standard_30000_exp2_full config_30000_exp2_full mconfig_30000_exp2_full asgo_30000_exp2_full adam_l1_30000_exp2_full asgo_l1_30000_exp2_full admm_asgo_30000_exp2_full \
  --out-dir report_assets/Experiment2_ADMM_ASGO \
  --device cuda:0
```

分析脚本会生成：

- `burgers_experiment_summary.json`
- `burgers_validation_loss.png`
- `burgers_accuracy_sparsity.png`
- `burgers_weight_sparsity.png`
- `burgers_layer_sparsity.png`
- `burgers_predictions_errors.png`
- `burgers_reference_solution.png`

## 构建报告

```bash
python3 tools/Experiment2_ADMM_ASGO/build_experiment2_latex.py
```

该脚本从 `Latex/Experiment2_ADMM_ASGO/experiment2_admm_asgo.md` 生成 `Latex/Experiment2_ADMM_ASGO/experiment2_admm_asgo.tex`，并调用 `xelatex` 两次生成 PDF。

## 提交版裁剪说明

当前仓库是课程提交版，不再保留外部 `external/ASGO` 仓库的嵌套 git 元数据；这里保留的是运行实验二所需的代码快照，其中 `optimizers/utils/matrix_decom.py` 会被 `experiments/PINN/lib_pinns/sparse_trainers.py` 直接调用。

为减小提交体积，`PINN_trained/Burgers/Experiment2_ADMM_ASGO/` 下每个 run 只保留分析脚本仍会读取的文件：

- `trained_network_weights.pt`
- `training_event.log`
- `records/events.out.tfevents.*`
- `configs.yaml`

原始训练时产生的 `checkpoints/`、`network_structure.pt` 和 `final_losses.txt` 已在提交版中移除，因为当前分析脚本和 LaTeX 报告构建流程不会再读取这些文件。
