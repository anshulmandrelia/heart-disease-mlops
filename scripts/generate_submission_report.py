"""Render the assignment report with its available, authentic evidence images.

Uses only the Python standard library so it can run on the supplied project
environment. Convert the generated HTML to DOCX with macOS ``textutil``.
"""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "FINAL_REPORT.md"
OUTPUT = ROOT / "Heart_Disease_MLOps_Assignment_Report.html"

FIGURES = {
    "Architecture diagram": [],
    "EDA evidence": [
        "eda_histograms.png",
        "eda_boxplots.png",
        "eda_correlation_heatmap.png",
        "eda_pairplot.png",
        "eda_class_distribution.png",
        "eda_outliers.png",
    ],
    "evaluation evidence": [
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "feature_importance.png",
    ],
    "MLflow evidence": ["mlflow_experiment.png", "mlflow_run_metrics.png"],
    "API evidence": ["swagger_docs.png", "prometheus_metrics.png"],
    "CI evidence": ["github_actions.png"],
}


def inline(value: str) -> str:
    value = html.escape(value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)


def evidence_for(line: str) -> list[str]:
    return next((files for title, files in FIGURES.items() if title in line), [])


def render() -> str:
    source = SOURCE.read_text(encoding="utf-8").splitlines()
    parts: list[str] = []
    in_code = False
    paragraph: list[str] = []
    in_list = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            parts.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    for raw in source:
        line = raw.strip()
        if line.startswith("```"):
            flush_paragraph()
            close_list()
            parts.append("<pre><code>" if not in_code else "</code></pre>")
            in_code = not in_code
            continue
        if in_code:
            parts.append(html.escape(raw) + "\n")
            continue
        if not line or line == "---":
            flush_paragraph()
            close_list()
            continue
        if line.startswith("#"):
            flush_paragraph()
            close_list()
            level = min(len(line) - len(line.lstrip("#")), 3)
            parts.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
            continue
        if line.startswith("- "):
            flush_paragraph()
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{inline(line[2:])}</li>")
            continue
        close_list()
        flush_paragraph()
        parts.append(f"<p>{inline(line)}</p>")
        images = evidence_for(line)
        if images:
            parts.append('<div class="figures">')
            for image in images:
                parts.append(
                    f'<figure><img src="screenshots/{image}" alt="{image}">'
                    f"<figcaption>{image.replace('_', ' ').replace('.png', '').title()}</figcaption></figure>"
                )
            parts.append("</div>")

    flush_paragraph()
    close_list()
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Heart Disease MLOps Assignment Report</title>
<style>
@page { size: A4; margin: 2cm; }
body { font-family: Arial, sans-serif; color: #1d1d1f; font-size: 11pt; line-height: 1.45; }
h1 { color: #102a43; font-size: 23pt; margin-top: 0; }
h2 { color: #102a43; font-size: 16pt; margin-top: 28px; page-break-after: avoid; }
p { margin: 0 0 11px; text-align: justify; }
code, pre { font-family: Menlo, monospace; font-size: 8.5pt; }
pre { background: #f4f6f8; padding: 10px; white-space: pre-wrap; }
li { margin-bottom: 5px; }
.figures { display: block; text-align: center; page-break-inside: avoid; margin: 12px 0 18px; }
figure { display: inline-block; width: 46%; margin: 8px 1%; vertical-align: top; }
figure img { max-width: 100%; max-height: 290px; border: 1px solid #cbd5e0; }
figcaption { font-size: 8pt; color: #4a5568; margin-top: 3px; }
</style></head><body>""" + "\n".join(parts) + "</body></html>"


if __name__ == "__main__":
    OUTPUT.write_text(render(), encoding="utf-8")
    print(OUTPUT)
