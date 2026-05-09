"""Build a LaTeX report from the polished Markdown report.

The converter intentionally supports only the Markdown subset used by the
course report: headings, paragraphs, display math, simple lists, simple
tables, images, and reference lines.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "组合优化与凸优化阅读报告-ConFIG.md"
LATEX_DIR = ROOT / "Latex" / "ConFIG_Report"
REPORT_TEX = LATEX_DIR / "config_report.tex"


PREAMBLE = r"""\documentclass[UTF8,a4paper,12pt,fontset=none]{ctexart}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{float}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{setspace}
\setCJKmainfont{Noto Serif CJK SC}
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans Mono CJK SC}
\setstretch{1.35}
\setlength{\parskip}{0.35em}
\emergencystretch=2em
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}
\graphicspath{{../../}{../../report_assets/}}

\title{ConFIG 无冲突梯度方法在物理信息神经网络训练中的优化建模与实验分析}
\author{}
\date{}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""

ENDING = r"""
\end{document}
"""


def escape_text(text: str) -> str:
    pieces = re.split(r"(`[^`]*`|\$[^$]*\$)", text)
    escaped = []
    for piece in pieces:
        if not piece:
            continue
        if piece.startswith("$") and piece.endswith("$"):
            escaped.append(piece)
        elif piece.startswith("`") and piece.endswith("`"):
            code = piece[1:-1].replace("\\", r"\textbackslash{}")
            code = code.replace("_", r"\_").replace("%", r"\%")
            escaped.append(r"\texttt{" + code + "}")
        else:
            piece = piece.replace("\\", r"\textbackslash{}")
            piece = piece.replace("&", r"\&")
            piece = piece.replace("%", r"\%")
            piece = piece.replace("_", r"\_")
            piece = piece.replace("#", r"\#")
            piece = piece.replace("{", r"\{")
            piece = piece.replace("}", r"\}")
            piece = piece.replace("~", r"\textasciitilde{}")
            piece = piece.replace("^", r"\textasciicircum{}")
            escaped.append(piece)
    return "".join(escaped)


def clean_heading(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[一二三四五六七八九十]+、\s*", "", text)
    text = re.sub(r"^\d+(?:\.\d+)*\s+", "", text)
    return text


def image_to_latex(line: str, pending_caption: str | None) -> str:
    match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
    if not match:
        return escape_text(line)
    alt, path = match.groups()
    caption = pending_caption or alt
    return (
        "\\begin{figure}[H]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.92\\textwidth]{{../../{path}}}\n"
        f"\\caption{{{escape_text(caption)}}}\n"
        "\\end{figure}\n"
    )


def table_to_latex(lines: list[str]) -> str:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    header = rows[0]
    body = [row for row in rows[2:] if row]
    if len(header) >= 6:
        first_width = 0.12
        total_width = 0.82
        font_size = "\\footnotesize"
    elif len(header) >= 4:
        first_width = 0.18
        total_width = 0.82
        font_size = "\\small"
    else:
        first_width = 0.30
        total_width = 0.88
        font_size = "\\small"
    other_width = (total_width - first_width) / max(1, len(header) - 1)
    col_spec = f"p{{{first_width:.2f}\\textwidth}}" + (
        f"p{{{other_width:.2f}\\textwidth}}" * (len(header) - 1)
    )
    out = ["\\begingroup", font_size, "\\begin{longtable}{" + col_spec + "}", "\\toprule"]
    out.append(" & ".join(escape_text(cell) for cell in header) + r" \\")
    out.append("\\midrule")
    for row in body:
        out.append(" & ".join(escape_text(cell) for cell in row) + r" \\")
    out.append("\\bottomrule")
    out.append("\\end{longtable}")
    out.append("\\endgroup")
    return "\n".join(out) + "\n"


def convert(md_text: str) -> str:
    lines = md_text.splitlines()
    out: list[str] = [PREAMBLE]
    in_math = False
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line:
            out.append("")
            i += 1
            continue

        if line == "---" or line.startswith("题目："):
            i += 1
            continue

        if line.startswith("$$"):
            out.append(r"\[" if not in_math else r"\]")
            content = line.strip("$").strip()
            if content:
                out.append(content)
                out.append(r"\]")
            in_math = not in_math if not content else False
            i += 1
            continue

        if in_math:
            out.append(line)
            i += 1
            continue

        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            out.append(table_to_latex(table_lines))
            continue

        if line.startswith("# "):
            i += 1
            continue
        if line.startswith("### "):
            heading = clean_heading(line[4:])
            if heading == "参考文献":
                out.append("\\newpage")
            out.append("\\section{" + escape_text(heading) + "}")
            i += 1
            continue
        if line.startswith("#### "):
            out.append("\\subsection{" + escape_text(clean_heading(line[5:])) + "}")
            i += 1
            continue

        if line.startswith("!["):
            caption = None
            if i + 2 < len(lines) and lines[i + 1].strip() == "" and re.match(r"图\\s*\\d+", lines[i + 2].strip()):
                caption = lines[i + 2].strip()
                i += 2
            out.append(image_to_latex(line, caption))
            i += 1
            continue

        if re.match(r"^\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            out.append("\\begin{enumerate}")
            out.extend("\\item " + escape_text(item) for item in items)
            out.append("\\end{enumerate}")
            continue

        if line.startswith("*   ") or line.startswith("- "):
            items = []
            while i < len(lines) and (lines[i].startswith("*   ") or lines[i].startswith("- ")):
                items.append(re.sub(r"^(\*\s+|-\s+)", "", lines[i].strip()))
                i += 1
            out.append("\\begin{itemize}")
            out.extend("\\item " + escape_text(item) for item in items)
            out.append("\\end{itemize}")
            continue

        out.append(escape_text(line) + "\n")
        i += 1

    out.append(ENDING)
    return "\n".join(out)


def main() -> None:
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    tex = convert(REPORT_MD.read_text(encoding="utf-8"))
    REPORT_TEX.write_text(tex, encoding="utf-8")
    print(REPORT_TEX)


if __name__ == "__main__":
    main()
