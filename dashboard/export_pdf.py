"""
WhitePaperExport: generates a clean, formatted PDF transcript for a run.

Uses reportlab Platypus for layout. Produces:
  - Cover section: run metadata + scores
  - Full conversation transcript with turn numbers and speaker labels
  - Score summary table

Download via Streamlit as bytes.
"""

from __future__ import annotations

import io
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]
METRIC_LABELS = {
    "identity_consistency": "Identity Consistency",
    "cultural_authenticity": "Cultural Authenticity",
    "naturalness": "Naturalness",
    "information_yield": "Information Yield",
}

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_NAVY = colors.HexColor("#0d2137")
_STEEL = colors.HexColor("#1a4a6e")
_ACCENT = colors.HexColor("#2a7fba")
_LIGHT = colors.HexColor("#e8f1f8")
_INTERVIEWER_BG = colors.HexColor("#f0f4f8")
_SUBJECT_BG = colors.HexColor("#dceefb")
_GREY = colors.HexColor("#666666")
_DIVIDER = colors.HexColor("#ccddee")


def build_run_pdf(run_data: dict, manual_scores_df=None) -> bytes:
    """
    Generate a PDF transcript for a single run and return the bytes.

    Args:
        run_data:          Full run log dict.
        manual_scores_df:  Optional DataFrame of manual per-turn scores.

    Returns:
        PDF as bytes (suitable for st.download_button).
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title=f"Run Report — {run_data.get('run_id', 'unknown')}",
        author="DTIC Offset Evaluation Framework",
    )

    styles = _build_styles()
    story = []

    _add_cover(story, styles, run_data)
    story.append(PageBreak())
    _add_scores_section(story, styles, run_data, manual_scores_df)
    story.append(HRFlowable(width="100%", thickness=1, color=_DIVIDER, spaceAfter=12))
    _add_transcript(story, styles, run_data)

    doc.build(story, onFirstPage=_page_header_footer, onLaterPages=_page_header_footer)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Page decorations
# ---------------------------------------------------------------------------

def _page_header_footer(canvas, doc):
    canvas.saveState()
    w, h = letter

    # Header bar
    canvas.setFillColor(_NAVY)
    canvas.rect(0, h - 0.55 * inch, w, 0.55 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(0.85 * inch, h - 0.35 * inch, "DTIC Offset Evaluation Framework")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(w - 0.85 * inch, h - 0.35 * inch, "RESEARCH USE ONLY")

    # Footer
    canvas.setFillColor(_GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.85 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.drawCentredString(w / 2, 0.45 * inch, "Georgia Institute of Technology — Offset Labs")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _add_cover(story, styles, run_data: dict) -> None:
    story.append(Spacer(1, 0.3 * inch))

    # Title
    story.append(Paragraph("Evaluation Run Report", styles["h1"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("DTIC Offset — Persona Consistency Study", styles["subtitle"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(HRFlowable(width="100%", thickness=2, color=_ACCENT, spaceAfter=16))

    meta = run_data.get("metadata", {})
    scenario = run_data.get("_scenario", {})
    identity = scenario.get("identity", {})

    # Metadata table
    meta_data = [
        ["Run ID", run_data.get("run_id", "—")],
        ["Scenario", run_data.get("scenario_id", "—")],
        ["Subject Model", run_data.get("subject_model", run_data.get("model", "—"))],
        ["Interviewer Model", run_data.get("interviewer_model", "—")],
        ["Timestamp", run_data.get("timestamp", "—")],
        ["Total Turns", str(meta.get("total_turns", "—"))],
        ["Stop Reason", meta.get("stop_reason", "—")],
        ["Context Trims", str(meta.get("context_trims", "—"))],
    ]

    t = Table(meta_data, colWidths=[1.8 * inch, 4.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _LIGHT),
        ("TEXTCOLOR", (0, 0), (0, -1), _NAVY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f5f9fd")]),
        ("GRID", (0, 0), (-1, -1), 0.5, _DIVIDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2 * inch))

    # Identity description if present
    if identity:
        story.append(Paragraph("Subject Identity", styles["h2"]))
        id_rows = [
            ["Name", identity.get("name", "—")],
            ["Background", identity.get("background", "—")],
            ["Persona", identity.get("persona", "—")],
            ["Language Style", identity.get("language_style", "—")],
        ]
        id_t = Table(id_rows, colWidths=[1.4 * inch, 4.9 * inch])
        id_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), _LIGHT),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, _DIVIDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f5f9fd")]),
        ]))
        story.append(id_t)


def _add_scores_section(story, styles, run_data: dict, manual_scores_df) -> None:
    story.append(Paragraph("Scores", styles["h2"]))
    story.append(Spacer(1, 6))

    llm_judge = run_data.get("scores", {}).get("llm_judge", {})
    llm_scores = llm_judge if isinstance(llm_judge, dict) else {}

    has_manual = manual_scores_df is not None and not manual_scores_df.empty

    header = ["Metric", "LLM Judge"]
    if has_manual:
        header.append("Manual (avg)")
    header.append("Rubric Descriptor")

    rows = [header]
    for m in METRICS:
        llm_val = llm_scores.get(m) or llm_scores.get("scores", {}).get(m)
        row = [METRIC_LABELS[m], str(llm_val) if llm_val is not None else "—"]
        if has_manual and m in manual_scores_df.columns:
            import pandas as pd
            vals = pd.to_numeric(manual_scores_df[m], errors="coerce").dropna()
            row.append(f"{vals.mean():.2f}" if not vals.empty else "—")
        elif has_manual:
            row.append("—")

        try:
            from evaluation.rubric import RUBRIC
            desc = RUBRIC.get(m, {}).get("scale", {}).get(int(float(str(llm_val))), "—") if llm_val else "—"
        except Exception:
            desc = "—"
        row.append(str(desc)[:80])
        rows.append(row)

    col_widths = [1.8 * inch, 0.9 * inch]
    if has_manual:
        col_widths.append(1.0 * inch)
    col_widths.append(3.5 * inch if not has_manual else 2.8 * inch)

    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, _DIVIDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)

    # Judge reasoning
    reasoning = llm_scores.get("reasoning") or (llm_scores.get("scores") or {}).get("reasoning")
    if reasoning:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Judge Reasoning", styles["h3"]))
        story.append(Paragraph(_safe_text(str(reasoning)), styles["body"]))

    story.append(Spacer(1, 0.15 * inch))


def _add_transcript(story, styles, run_data: dict) -> None:
    story.append(Paragraph("Conversation Transcript", styles["h2"]))
    story.append(Spacer(1, 6))

    conversation = run_data.get("conversation", [])
    if not conversation:
        story.append(Paragraph("No conversation turns recorded.", styles["body"]))
        return

    for turn in conversation:
        speaker = turn.get("speaker", "unknown")
        turn_num = turn.get("turn", 0)
        text = turn.get("text", "")
        timestamp = turn.get("timestamp", "")

        is_subject = speaker == "subject"
        bg = _SUBJECT_BG if is_subject else _INTERVIEWER_BG
        label = "SUBJECT" if is_subject else "INTERVIEWER"
        label_color = _STEEL if is_subject else _NAVY

        # Turn header
        header_data = [[
            Paragraph(f"<b>{label}</b> — Turn {turn_num}", ParagraphStyle(
                "th", fontSize=8, textColor=label_color, fontName="Helvetica-Bold"
            )),
            Paragraph(timestamp[:19] if timestamp else "", ParagraphStyle(
                "ts", fontSize=7, textColor=_GREY, alignment=TA_RIGHT, fontName="Helvetica"
            )),
        ]]
        header_t = Table(header_data, colWidths=[4 * inch, 2.3 * inch])
        header_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (0, -1), 8),
            ("RIGHTPADDING", (-1, 0), (-1, -1), 8),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, _DIVIDER),
        ]))
        story.append(header_t)

        # Turn text
        text_data = [[Paragraph(_safe_text(text), styles["turn_text"])]]
        text_t = Table(text_data, colWidths=[6.3 * inch])
        text_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, _DIVIDER),
        ]))
        story.append(text_t)

    story.append(Spacer(1, 0.2 * inch))


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", fontSize=20, fontName="Helvetica-Bold",
                              textColor=_NAVY, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", fontSize=11, fontName="Helvetica",
                                   textColor=_STEEL, spaceAfter=4),
        "h2": ParagraphStyle("h2", fontSize=13, fontName="Helvetica-Bold",
                              textColor=_NAVY, spaceBefore=12, spaceAfter=4),
        "h3": ParagraphStyle("h3", fontSize=10, fontName="Helvetica-Bold",
                              textColor=_STEEL, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", fontSize=9, fontName="Helvetica",
                               textColor=colors.black, leading=13, spaceAfter=4),
        "turn_text": ParagraphStyle("turn_text", fontSize=9.5, fontName="Helvetica",
                                    textColor=colors.HexColor("#111111"), leading=14),
    }


def _safe_text(text: str) -> str:
    """Escape XML special chars for ReportLab Paragraph."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\x00", "")
    )
