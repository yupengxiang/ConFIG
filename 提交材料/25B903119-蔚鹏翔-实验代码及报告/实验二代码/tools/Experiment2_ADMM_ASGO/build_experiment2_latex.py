"""Build the experiment-2 ADMM-ASGO LaTeX/PDF report from Markdown."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_MD = ROOT / "Latex" / "Experiment2_ADMM_ASGO" / "experiment2_admm_asgo.md"
LATEX_DIR = ROOT / "Latex" / "Experiment2_ADMM_ASGO"
REPORT_TEX = LATEX_DIR / "experiment2_admm_asgo.tex"


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
\usepackage{listings}
\usepackage{enumitem}
\setCJKmainfont{Noto Serif CJK SC}
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans Mono CJK SC}
\setstretch{1.32}
\setlength{\parskip}{0.35em}
\setlist{itemsep=0.15em, topsep=0.25em}
\emergencystretch=2em
\ctexset{
    section = {name = {}, number = \arabic{section}, aftername = \quad},
    subsection = {name = {}, number = \arabic{section}.\arabic{subsection}, aftername = \quad}
}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}
\lstset{
    basicstyle=\ttfamily\footnotesize,
    breaklines=true,
    frame=single,
    columns=fullflexible,
    keepspaces=true
}
\graphicspath{{../../}{../../report_assets/Experiment2_ADMM_ASGO/}}

\title{结合 ADMM 与 ASGO 的稀疏 PINN 优化实验}
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


def escape_latex(text: str) -> str:
    pieces = re.split(r"(`[^`]*`|\$[^$]*\$|https?://\S+)", text)
    escaped: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if piece.startswith("$") and piece.endswith("$"):
            escaped.append(piece)
        elif piece.startswith("`") and piece.endswith("`"):
            code = piece[1:-1]
            code = code.replace("\\", r"\textbackslash{}")
            code = code.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            escaped.append(r"\texttt{" + code + "}")
        elif piece.startswith("http://") or piece.startswith("https://"):
            escaped.append(r"\url{" + piece + "}")
        else:
            replacements = {
                "\\": r"\textbackslash{}",
                "&": r"\&",
                "%": r"\%",
                "_": r"\_",
                "#": r"\#",
                "{": r"\{",
                "}": r"\}",
                "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}",
            }
            for raw, replacement in replacements.items():
                piece = piece.replace(raw, replacement)
            piece = piece.replace("≤", r"$\le$")
            piece = piece.replace("≥", r"$\ge$")
            piece = piece.replace("×", r"$\times$")
            escaped.append(piece)
    return "".join(escaped)


def clean_heading(text: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", text.strip())


def table_to_latex(lines: list[str]) -> str:
    rows = []
    for line in lines:
        protected = line.replace(r"\|", "__LATEX_NORM_BAR__")
        cells = [cell.strip().replace("__LATEX_NORM_BAR__", r"\|") for cell in protected.strip().strip("|").split("|")]
        rows.append(cells)
    header = rows[0]
    body = [row for row in rows[2:] if row]
    if len(header) >= 7:
        first_width = 0.15
        total_width = 0.95
        font_size = r"\scriptsize"
    elif len(header) >= 5:
        first_width = 0.18
        total_width = 0.92
        font_size = r"\footnotesize"
    else:
        first_width = 0.25
        total_width = 0.90
        font_size = r"\small"
    other_width = (total_width - first_width) / max(1, len(header) - 1)
    col_spec = f"p{{{first_width:.2f}\\textwidth}}" + f"p{{{other_width:.2f}\\textwidth}}" * (len(header) - 1)
    out = [r"\begingroup", font_size, r"\begin{longtable}{" + col_spec + "}", r"\toprule"]
    out.append(" & ".join(escape_latex(cell) for cell in header) + r" \\")
    out.append(r"\midrule")
    for row in body:
        row = row + [""] * (len(header) - len(row))
        out.append(" & ".join(escape_latex(cell) for cell in row[: len(header)]) + r" \\")
    out.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup"])
    return "\n".join(out) + "\n"


def image_to_latex(line: str) -> str:
    match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
    if not match:
        return escape_latex(line)
    alt, path = match.groups()
    return (
        r"\begin{figure}[H]" + "\n"
        r"\centering" + "\n"
        f"\\includegraphics[width=0.92\\textwidth,height=0.58\\textheight,keepaspectratio]{{../../{path}}}\n"
        f"\\caption{{{escape_latex(alt)}}}\n"
        r"\end{figure}" + "\n"
    )


def convert(md_text: str) -> str:
    lines = md_text.splitlines()
    out: list[str] = [PREAMBLE]
    in_math = False
    in_code = False
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("```"):
            out.append(r"\begin{lstlisting}" if not in_code else r"\end{lstlisting}")
            in_code = not in_code
            i += 1
            continue

        if in_code:
            out.append(line)
            i += 1
            continue

        if not line:
            out.append("")
            i += 1
            continue

        if line.startswith("$$"):
            content = line.strip("$").strip()
            if content:
                out.extend([r"\[", content, r"\]"])
            else:
                out.append(r"\[" if not in_math else r"\]")
                in_math = not in_math
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
        if line.startswith("## "):
            heading = clean_heading(line[3:])
            if heading in {"附录：关键算法代码", "参考文献"}:
                out.append(r"\newpage")
            out.append(r"\section{" + escape_latex(heading) + "}")
            i += 1
            continue
        if line.startswith("### "):
            out.append(r"\subsection{" + escape_latex(clean_heading(line[4:])) + "}")
            i += 1
            continue

        if line.startswith("!["):
            out.append(image_to_latex(line))
            i += 1
            continue

        if re.match(r"^\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            out.append(r"\begin{enumerate}")
            out.extend(r"\item " + escape_latex(item) for item in items)
            out.append(r"\end{enumerate}")
            continue

        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:].strip())
                i += 1
            out.append(r"\begin{itemize}")
            out.extend(r"\item " + escape_latex(item) for item in items)
            out.append(r"\end{itemize}")
            continue

        out.append(escape_latex(line))
        i += 1

    out.append(ENDING)
    return "\n".join(out)


def main() -> None:
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TEX.write_text(convert(REPORT_MD.read_text(encoding="utf-8")), encoding="utf-8")
    print(REPORT_TEX)
    subprocess.run(["xelatex", "-interaction=nonstopmode", REPORT_TEX.name], cwd=LATEX_DIR, check=True)
    subprocess.run(["xelatex", "-interaction=nonstopmode", REPORT_TEX.name], cwd=LATEX_DIR, check=True)
    for path in LATEX_DIR.glob(f"{REPORT_TEX.stem}.*"):
        if path.suffix not in {".md", ".tex", ".pdf"}:
            path.unlink()


if __name__ == "__main__":
    main()
