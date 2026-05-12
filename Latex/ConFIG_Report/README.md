# LaTeX 报告目录说明

本目录用于存放 ConFIG 课程阅读报告的 LaTeX 版本。

## 文件说明

```text
config_report.tex      主 LaTeX 文件，由 tools/Report/build_latex_report.py 生成
README.md              本说明文件
```

报告中的图片来自仓库根目录下的 `report_assets/Report/`：

```text
../../report_assets/Report/
```

## 生成 LaTeX 文件

在仓库根目录运行：

```bash
python3 tools/Report/build_latex_report.py
```

该命令会读取：

```text
Latex/ConFIG_Report/组合优化与凸优化阅读报告-ConFIG.md
```

并生成：

```text
Latex/ConFIG_Report/config_report.tex
```

## 编译 PDF

进入本目录：

```bash
cd Latex/ConFIG_Report
```

使用 XeLaTeX 编译：

```bash
xelatex config_report.tex
xelatex config_report.tex
```

或者使用 `latexmk`：

```bash
latexmk -xelatex config_report.tex
```

如果 XeLaTeX 只生成 `config_report.xdv` 而没有生成 PDF，可手动转换：

```bash
xelatex -no-pdf config_report.tex
xelatex -no-pdf config_report.tex
xdvipdfmx -o config_report.pdf config_report.xdv
```

若重新生成了实验图表或修改了 Markdown 正文，需要先在仓库根目录重新运行：

```bash
python3 tools/Report/build_latex_report.py
```
