"""Embed existing PNG evidence in the DOCX without third-party packages."""

from __future__ import annotations

import html
import shutil
import struct
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Heart_Disease_MLOps_Assignment_Report.docx"
EVIDENCE = [
    ("Exploratory data analysis", "eda_histograms.png"),
    ("Exploratory data analysis", "eda_boxplots.png"),
    ("Exploratory data analysis", "eda_correlation_heatmap.png"),
    ("Exploratory data analysis", "eda_pairplot.png"),
    ("Exploratory data analysis", "eda_class_distribution.png"),
    ("Exploratory data analysis", "eda_outliers.png"),
    ("Model evaluation", "confusion_matrix.png"),
    ("Model evaluation", "roc_curve.png"),
    ("Model evaluation", "precision_recall_curve.png"),
    ("Model evaluation", "feature_importance.png"),
    ("MLflow experiment tracking", "mlflow_experiment.png"),
    ("MLflow experiment tracking", "mlflow_run_metrics.png"),
    ("API and Prometheus validation", "swagger_docs.png"),
    ("API and Prometheus validation", "prometheus_metrics.png"),
    ("Continuous integration", "github_actions.png"),
]

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as image:
        image.read(16)
        return struct.unpack(">II", image.read(8))


def paragraph(text: str, page_break: bool = False) -> str:
    escaped = html.escape(text)
    break_xml = '<w:r><w:br w:type="page"/></w:r>' if page_break else ""
    return f'<w:p>{break_xml}<w:r><w:t>{escaped}</w:t></w:r></w:p>'


def image_paragraph(rel_id: int, docpr_id: int, filename: str, width_px: int, height_px: int) -> str:
    max_width, max_height = 5.8 * 914400, 4.2 * 914400
    width = width_px * 914400 / 144
    height = height_px * 914400 / 144
    scale = min(max_width / width, max_height / height, 1)
    width, height = int(width * scale), int(height * scale)
    return f'''<w:p><w:r><w:drawing xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{width}" cy="{height}"/><wp:docPr id="{docpr_id}" name="{filename}"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="{filename}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rId{rel_id}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width}" cy="{height}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'''


def main() -> None:
    backup = DOCX.with_suffix(".pre_evidence.docx")
    if not backup.exists():
        shutil.copy2(DOCX, backup)
    with zipfile.ZipFile(backup) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}

    document = files["word/document.xml"].decode("utf-8")
    relationships = files["word/_rels/document.xml.rels"].decode("utf-8")
    content_types = files["[Content_Types].xml"].decode("utf-8")

    extra = [paragraph("Appendix A — Captured Runtime and Training Evidence", page_break=True)]
    last_group = ""
    rel_id, docpr_id = 3, 1
    for group, filename in EVIDENCE:
        path = ROOT / "screenshots" / filename
        if group != last_group:
            extra.append(paragraph(group))
            last_group = group
        extra.append(paragraph(filename.replace("_", " ").replace(".png", "").title()))
        width, height = png_size(path)
        extra.append(image_paragraph(rel_id, docpr_id, filename, width, height))
        relationships = relationships.replace(
            "</Relationships>",
            f'<Relationship Id="rId{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{filename}"/></Relationships>',
        )
        files[f"word/media/{filename}"] = path.read_bytes()
        rel_id += 1
        docpr_id += 1

    document = document.replace("</w:body>", "".join(extra) + "</w:body>")
    if 'Extension="png"' not in content_types:
        content_types = content_types.replace(
            "</Types>", '<Default Extension="png" ContentType="image/png"/></Types>'
        )
    files["word/document.xml"] = document.encode("utf-8")
    files["word/_rels/document.xml.rels"] = relationships.encode("utf-8")
    files["[Content_Types].xml"] = content_types.encode("utf-8")

    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    print(f"Embedded {len(EVIDENCE)} evidence images in {DOCX.name}")


if __name__ == "__main__":
    main()
