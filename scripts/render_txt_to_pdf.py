"""
Simple script to render .txt files to .pdf using ReportLab.
Run: python scripts\render_txt_to_pdf.py
Installs: pip install reportlab
"""
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Preformatted
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def register_font():
    # Try to register DejaVuSans if available for better unicode support.
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
        return "DejaVuSans"
    except Exception:
        return "Helvetica"


FONT_NAME = register_font()


def txt_to_pdf(txt_path: Path, pdf_path: Path):
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            leftMargin=40, rightMargin=40,
                            topMargin=40, bottomMargin=40)
    try:
        text = txt_path.read_text(encoding="utf-8")
    except Exception:
        text = txt_path.read_text(encoding="latin-1")

    styles = getSampleStyleSheet()
    pre_style = ParagraphStyle(
        "pre",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=10,
        leading=12,
    )
    flow = [Preformatted(text, pre_style)]
    doc.build(flow)


def batch_render_txt_to_pdf(docs_dir: str = "data/docs", out_dir: str = "data/docs_rendered"):
    docs_path = Path(docs_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    count = 0
    for txt_file in sorted(docs_path.glob("*.txt")):
        pdf_file = out_path / (txt_file.stem + ".pdf")
        txt_to_pdf(txt_file, pdf_file)
        print(f"Rendered: {txt_file.name} -> {pdf_file}")
        count += 1
    if count == 0:
        print(f"No .txt files found in {docs_path.resolve()}")
    else:
        print(f"Rendered {count} files to {out_path.resolve()}")


if __name__ == "__main__":
    batch_render_txt_to_pdf()
