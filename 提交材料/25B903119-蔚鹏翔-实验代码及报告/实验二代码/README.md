# 实验二代码说明

本目录保留了实验二“结合 ADMM 与 ASGO 的稀疏 PINN 优化实验”的代码快照、必要结果和 LaTeX 源文件。

## 目录内容

- `tools/Experiment2_ADMM_ASGO/`：训练、分析、LaTeX 构建脚本
- `conflictfree/`：ConFIG 相关核心代码
- `experiments/PINN/`：实验二运行所需的 Burgers PINN 代码、配置和依赖说明
- `external/ASGO/`：ASGO 外部实现快照
- `report_assets/Experiment2_ADMM_ASGO/`：最终图表和汇总 JSON
- `Latex/Experiment2_ADMM_ASGO/`：报告的 Markdown 和 TeX 源文件

## 环境说明

未打包 Python 环境本体。建议使用 Python 3.10，并在本目录根下安装：

```bash
pip install torch numpy scipy matplotlib tensorboard pyyaml einops tqdm
```

然后从本目录根运行相关脚本，例如：

```bash
python3 tools/Experiment2_ADMM_ASGO/build_experiment2_latex.py
```

## 数据与训练结果说明

按照老师要求，本目录未打包数据集和大体积训练结果，以下内容已省略：

- `experiments/PINN/data/burgers/simulation_data.npy`
- `PINN_trained/Burgers/Experiment2_ADMM_ASGO/`

因此：

- 重新训练和重新分析需要自行补回上述数据和训练结果
- 已生成的最终图表和汇总结果已经保留在 `report_assets/Experiment2_ADMM_ASGO/` 中
- 已生成的报告源文件已经保留在 `Latex/Experiment2_ADMM_ASGO/` 中，可直接重新编译 PDF

## 报告说明

最终 PDF 不放在本目录，而放在上一级的 `实验二报告/` 中。