# ConFIG Burgers PINN 实验完整复现流程

本文档说明如何从零开始复现 Burgers PINN 对比实验，并生成阅读报告中使用的数值表格、训练曲线、预测热力图和 LaTeX/PDF 报告。

整体流程如下：

```text
克隆 ConFIG 仓库
  -> 创建 Python 环境并安装依赖
  -> 放入本报告新增脚本和最终报告文件
  -> 运行 Burgers 训练实验
  -> 分析训练结果并生成图表
  -> 根据结果更新最终 Markdown 报告
  -> 从最终 Markdown 生成 LaTeX/PDF 报告
```

## 1. 获取代码

如果从官方原始仓库开始：

```bash
git clone https://github.com/tum-pbs/ConFIG.git
cd ConFIG
```

本实验额外新增或保留以下文件与目录：

```text
tools/run_burgers_experiments.py
tools/analyze_burgers_results.py
tools/build_latex_report.py
tools/README_experiments.md
组合优化与凸优化阅读报告-ConFIG.md
Latex/ConFIG_Report/
report_assets/
PINN_trained/
```

其中 `组合优化与凸优化阅读报告-ConFIG.md` 是唯一维护的 Markdown 正文。项目中不再维护第 1、2、3、4 章的拆分 Markdown 文件，也不需要通过脚本合并章节。

## 2. 创建实验环境

推荐使用普通 CPython 环境。下面给出 conda 创建方式：

```bash
conda create -n config-pinn python=3.10 -y
conda activate config-pinn
```

安装 PyTorch。若机器有 NVIDIA GPU，可按本机 CUDA 版本安装，例如：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

如果只用 CPU：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

安装 ConFIG 仓库和 PINN 实验依赖：

```bash
pip install -e .
pip install numpy matplotlib scipy einops tqdm tensorboard PyYAML
```

注意：官方 `experiments/PINN/requirements.txt` 中有两个容易导致安装失败的名字：`yaml` 应写为 `PyYAML`，`tesnsorboard` 应写为 `tensorboard`。所以上面直接给出了修正后的安装命令。

检查环境是否可用：

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python3 -c "import conflictfree; print('conflictfree import ok')"
```

本实验使用可正常加载 PyTorch 的 CPython 解释器运行，测试环境为 Python 3.10.12、PyTorch 2.11.0+cu130、CUDA 可用。实际复现时，只需保证当前 shell 中的 `python3` 能正常执行 `import torch`。

## 3. 脚本作用说明

### `tools/run_burgers_experiments.py`

作用：训练 Burgers 方程 PINN，并比较不同优化方法。

支持的方法：

| 方法名 | 含义 | 对应实现 |
|---|---|---|
| `standard` | 普通 PINN 标量损失求和 | `StandardTrainer()` |
| `pcgrad` | PCGrad 梯度投影基线 | `PCGradOperator()` |
| `config` | ConFIG 无冲突梯度 | `ConFIGOperator()` |
| `mconfig` | 动量版本 ConFIG | `get_momentum_trainer(..., ConFIGOperator())` |

主要输出：

```text
PINN_trained/Burgers/<method>_<epochs>/
```

每个 run 目录下会包含：

```text
configs.yaml                  训练配置
training_event.log            训练日志、最终 loss、训练速度
final_losses.txt              最后若干 epoch 的 train/validation loss
trained_network_weights.pt    最终模型权重
network_structure.pt          网络结构
records/                      TensorBoard event 文件
checkpoints/                  中间 checkpoint
```

### `tools/analyze_burgers_results.py`

作用：读取训练结果，重新评估模型，并生成报告图表。

它会做四件事：

1. 从 TensorBoard event 文件中读取 `Loss/validation` 曲线。
2. 用 `run_test` 在真实 Burgers 仿真数据 `simulation_data.npy` 上计算 MSE。
3. 生成预测解、绝对误差和真实解热力图。
4. 重新采样一批点，计算 PDE loss 与 BC/IC loss 的梯度余弦相似度。

主要输出：

```text
report_assets/burgers_experiment_summary.json
report_assets/burgers_validation_loss.png
report_assets/burgers_predictions_errors.png
report_assets/burgers_reference_solution.png
```

其中 `burgers_experiment_summary.json` 是数值表格的来源。

### `tools/build_latex_report.py`

作用：读取最终 Markdown 报告，生成 LaTeX 版本。

输入：

```text
组合优化与凸优化阅读报告-ConFIG.md
```

输出：

```text
Latex/ConFIG_Report/config_report.tex
Latex/ConFIG_Report/config_report.pdf
```

Markdown 报告中保留课程写作习惯，例如 `1.1`、`1.2` 这类手写小节号。生成 LaTeX 时会自动去掉这些手写编号，由 LaTeX 自身生成目录和章节序号。

## 4. 运行正式 Burgers 实验

正式报告使用 30000 epochs、三随机种子的完整实验。Burgers 网络规模较小，单个训练进程通常不能充分占用高端 GPU，因此使用 `--parallel-methods` 同时运行多个优化方法：

```bash
python3 tools/run_burgers_experiments.py \
  --epochs 30000 \
  --num-run 3 \
  --methods standard pcgrad config mconfig \
  --device cuda:0 \
  --n-losses 2 \
  --name-suffix _full \
  --parallel-methods 4
```

该命令会同时启动 `standard`、`pcgrad`、`config`、`mconfig` 四个独立 Python 进程。每个方法内部仍顺序运行 `num_run=3` 个随机种子，因此输出目录结构为：

```text
PINN_trained/Burgers/standard_30000_full/<时间戳>/
PINN_trained/Burgers/pcgrad_30000_full/<时间戳>/
PINN_trained/Burgers/config_30000_full/<时间戳>/
PINN_trained/Burgers/mconfig_30000_full/<时间戳>/
```

并行数量不宜超过显存可承受范围。运行过程中可使用以下命令观察显存、功耗和 GPU 利用率：

```bash
nvidia-smi
```

若显存占用接近上限，或者多个进程导致训练速度明显下降，可将 `--parallel-methods 4` 调低为 `2`。

## 5. 分析结果并生成图表

完整实验结束后运行：

```bash
python3 tools/analyze_burgers_results.py \
  --run-root PINN_trained/Burgers \
  --methods standard_30000_full pcgrad_30000_full config_30000_full mconfig_30000_full \
  --out-dir report_assets \
  --device cuda:0
```

输出文件：

```text
report_assets/burgers_experiment_summary.json
report_assets/burgers_validation_loss.png
report_assets/burgers_predictions_errors.png
report_assets/burgers_reference_solution.png
```

本次完整实验得到的关键结果为：

| 方法 | 最终 validation MSE | 最优 validation MSE | `run_test` MSE | 训练速度 s/iter |
|---|---:|---:|---:|---:|
| Standard Adam | 1.6799e-04 ± 1.1400e-05 | 1.4650e-04 ± 5.9984e-06 | 1.6799e-04 ± 1.1400e-05 | 0.01254 ± 0.00095 |
| PCGrad | 1.7527e-04 ± 2.2738e-05 | 1.3391e-04 ± 1.5436e-06 | 1.7527e-04 ± 2.2738e-05 | 0.01969 ± 0.00305 |
| ConFIG | 1.5516e-04 ± 5.1134e-06 | 1.3176e-04 ± 5.2519e-07 | 1.5516e-04 ± 5.1134e-06 | 0.02033 ± 0.00381 |
| M-ConFIG | 1.5714e-04 ± 6.9572e-06 | 1.2801e-04 ± 4.2173e-06 | 1.5714e-04 ± 6.9572e-06 | 0.01367 ± 0.00013 |

代表性最优 run 上的 PDE 梯度与 BC/IC 梯度余弦相似度为：

| 方法 | 梯度余弦相似度 |
|---|---:|
| Standard Adam | -0.0581 |
| PCGrad | 0.0290 |
| ConFIG | -0.2620 |
| M-ConFIG | 0.6344 |

## 6. 更新最终 Markdown 报告

实验重新运行后，直接更新：

```text
组合优化与凸优化阅读报告-ConFIG.md
```

更新时主要检查三类内容：

1. 第 3 章的数值表格是否与 `report_assets/burgers_experiment_summary.json` 一致。
2. 图表路径是否指向 `report_assets/` 下的最新图片。
3. 第 4 章的分析是否与最新实验结果一致。

不需要维护章节拆分文件，也不需要运行合并脚本。

## 7. 生成 LaTeX 与 PDF 报告

Markdown 报告确认无误后，在仓库根目录运行：

```bash
python3 tools/build_latex_report.py
```

输出目录：

```text
Latex/ConFIG_Report/
```

其中 `config_report.tex` 是主文件，图片从 `report_assets/` 读取。编译 PDF 时进入该目录：

```bash
cd Latex/ConFIG_Report
xelatex config_report.tex
xelatex config_report.tex
```

如果系统安装了 `latexmk`，也可以运行：

```bash
latexmk -xelatex config_report.tex
```

如果本机 XeLaTeX 只生成 `config_report.xdv` 而没有生成 PDF，可使用：

```bash
xelatex -no-pdf config_report.tex
xelatex -no-pdf config_report.tex
xdvipdfmx -o config_report.pdf config_report.xdv
```

## 8. 快速连通性检查

如果只想检查环境和脚本是否能跑通，可以先运行短训练：

```bash
python3 tools/run_burgers_experiments.py \
  --epochs 1000 \
  --num-run 1 \
  --methods standard pcgrad config \
  --device cuda:0 \
  --n-losses 2
```

短训练目录只用于检查流程，不作为最终报告的数据来源。

## 9. 常见问题

### 找不到 `tensorboard`

安装：

```bash
pip install tensorboard
```

### `yaml` 安装失败

不要安装 `yaml`，应安装：

```bash
pip install PyYAML
```

### 没有 GPU

训练命令和分析命令都把 `--device cuda:0` 改成：

```bash
--device cpu
```

CPU 可以跑通流程，但训练会慢很多。

### 分析脚本找不到目录

检查训练输出目录名：

```bash
find PINN_trained/Burgers -maxdepth 2 -type d | sort
```

然后把真实目录名传给 `--methods`。例如：

```bash
python3 tools/analyze_burgers_results.py \
  --methods standard_30000_full pcgrad_30000_full config_30000_full mconfig_30000_full
```

### 查看训练日志

每次训练的日志都在：

```text
PINN_trained/Burgers/<method>/<timestamp>/training_event.log
```

其中包含最终训练损失、验证损失、最优验证损失、总运行时间和每步训练速度。
