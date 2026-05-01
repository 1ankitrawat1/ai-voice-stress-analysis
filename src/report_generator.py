"""PDF report generation for final test results."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.platypus.flowables import HRFlowable

SAFETY_NOTE = "This result is only an estimate. It does not prove that the person is lying."
TRANSCRIPT_NOTE = "Transcript text is generated from the recorded voice and may contain minor recognition errors."
PROJECT_TITLE = "AI Based Voice Stress Analysis System for Lie Possibility Detection"


def _clean_text(value: Any, fallback: str = "Not available") -> str:
    """Return a safe single-line text value for PDF paragraphs."""
    text = str(value or "").strip()
    if not text:
        return fallback
    return " ".join(text.split())


def _build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            alignment=TA_LEFT,
            spaceAfter=10,
            textColor=colors.HexColor("#111827"),
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=8,
            spaceAfter=7,
            textColor=colors.HexColor("#111827"),
        ),
        "body": ParagraphStyle(
            "BodyTextClean",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            spaceAfter=6,
            textColor=colors.HexColor("#1f2937"),
        ),
        "small": ParagraphStyle(
            "SmallTextClean",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#374151"),
        ),
        "small_bold": ParagraphStyle(
            "SmallTextBold",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#111827"),
        ),
    }


def _paragraph(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_clean_text(text), style)


def _make_summary_table(analysis: Dict[str, object], styles: Dict[str, ParagraphStyle]) -> Table:
    component_scores = analysis.get("component_scores", {}) if isinstance(analysis, dict) else {}
    cards = analysis.get("cards", {}) if isinstance(analysis, dict) else {}
    transcript_hesitation = analysis.get("transcript_hesitation_score", component_scores.get("pause_and_hesitation", 0))
    rows = [
        ["Area", "Level", "Score"],
        ["Voice Stress", str(cards.get("Voice Stress", "Low")), f"{component_scores.get('voice_stress', 0)}%"],
        ["Pause and Hesitation", str(cards.get("Pause Level", "Low")), f"{component_scores.get('pause_and_hesitation', 0)}%"],
        ["Hesitation", str(cards.get("Hesitation", "Low")), f"{transcript_hesitation}%"],
        ["Answer Matching", str(cards.get("Answer Matching", "Matched")), f"{component_scores.get('answer_matching', 0)}%"],
        ["Speaking Speed", str(cards.get("Speaking Speed", "Stable")), "-"],
        ["Voice Change", str(cards.get("Voice Change", "Stable")), f"{component_scores.get('difference_from_normal', 0)}%"],
    ]
    table_data = []
    for row_index, row in enumerate(rows):
        style = styles["small_bold"] if row_index == 0 else styles["small"]
        table_data.append([Paragraph(_clean_text(cell, ""), style) for cell in row])

    table = Table(table_data, colWidths=[7.1 * cm, 5.0 * cm, 3.3 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _make_qa_table(analysis: Dict[str, object], styles: Dict[str, ParagraphStyle]) -> Table:
    qa_pairs: List[Dict[str, str]] = []
    raw_pairs = analysis.get("question_answer_pairs", []) if isinstance(analysis, dict) else []
    if isinstance(raw_pairs, list):
        for item in raw_pairs:
            if isinstance(item, dict):
                qa_pairs.append(
                    {
                        "question": _clean_text(item.get("question"), "Question not available"),
                        "answer": _clean_text(item.get("answer"), "Answer text not available"),
                    }
                )

    if not qa_pairs:
        questions = analysis.get("generated_questions", []) if isinstance(analysis, dict) else []
        answers = analysis.get("answer_texts", []) if isinstance(analysis, dict) else []
        if isinstance(questions, list) and isinstance(answers, list):
            for question, answer in zip(questions, answers[1:]):
                qa_pairs.append({"question": _clean_text(question), "answer": _clean_text(answer)})

    rows: List[List[Paragraph]] = [
        [
            Paragraph("No.", styles["small_bold"]),
            Paragraph("Question Asked", styles["small_bold"]),
            Paragraph("User Answer", styles["small_bold"]),
        ]
    ]
    if qa_pairs:
        for index, item in enumerate(qa_pairs, start=1):
            rows.append(
                [
                    Paragraph(str(index), styles["small"]),
                    Paragraph(item["question"], styles["small"]),
                    Paragraph(item["answer"], styles["small"]),
                ]
            )
    else:
        rows.append(
            [
                Paragraph("-", styles["small"]),
                Paragraph("Question data not available", styles["small"]),
                Paragraph("Answer data not available", styles["small"]),
            ]
        )

    table = Table(rows, colWidths=[1.2 * cm, 7.0 * cm, 7.2 * cm], hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table



def _make_voice_stress_chart(analysis: Dict[str, object]) -> Drawing:
    """Create a fixed 0-100 voice stress bar chart for the PDF report."""
    raw_scores = analysis.get("voice_stress_scores", []) if isinstance(analysis, dict) else []
    scores: List[float | None] = []
    if isinstance(raw_scores, list):
        for value in raw_scores:
            try:
                scores.append(max(0.0, min(100.0, float(value))))
            except (TypeError, ValueError):
                scores.append(None)

    raw_labels = analysis.get("question_labels", []) if isinstance(analysis, dict) else []
    labels = [str(item) for item in raw_labels] if isinstance(raw_labels, list) else []
    generated_questions = analysis.get("generated_questions", []) if isinstance(analysis, dict) else []
    generated_count = len(generated_questions) if isinstance(generated_questions, list) else 0
    expected_count = max(len(labels), len(scores), generated_count + 1, 5)

    if not labels:
        labels = ["Statement"] + [f"Q{i}" for i in range(1, expected_count)]
    labels = labels[:expected_count]
    while len(labels) < expected_count:
        labels.append(f"Q{len(labels)}")
    scores = scores[:expected_count]
    while len(scores) < expected_count:
        scores.append(None)

    width = 15.4 * cm
    height = 6.2 * cm
    left = 1.4 * cm
    bottom = 1.0 * cm
    top = 0.45 * cm
    right = 0.35 * cm
    plot_width = width - left - right
    plot_height = height - bottom - top

    drawing = Drawing(width, height)
    axis_color = colors.HexColor("#6b7280")
    grid_color = colors.HexColor("#e5e7eb")
    bar_color = colors.HexColor("#60a5fa")
    text_color = colors.HexColor("#111827")
    muted_color = colors.HexColor("#4b5563")

    # Grid and y-axis labels. The scale is always fixed at 0-100.
    for tick in range(0, 101, 20):
        y = bottom + (tick / 100.0) * plot_height
        drawing.add(Line(left, y, width - right, y, strokeColor=grid_color, strokeWidth=0.55))
        drawing.add(String(0.15 * cm, y - 3, str(tick), fontName="Helvetica", fontSize=7, fillColor=muted_color))

    drawing.add(Line(left, bottom, left, bottom + plot_height, strokeColor=axis_color, strokeWidth=0.8))
    drawing.add(Line(left, bottom, width - right, bottom, strokeColor=axis_color, strokeWidth=0.8))

    count = max(1, len(labels))
    slot_width = plot_width / count
    bar_width = min(slot_width * 0.48, 0.95 * cm)

    for index, (label, score) in enumerate(zip(labels, scores)):
        center_x = left + slot_width * index + slot_width / 2
        label_text = "Stmt" if label.lower().startswith("statement") else label
        drawing.add(String(center_x - 0.33 * cm, bottom - 0.45 * cm, label_text, fontName="Helvetica", fontSize=7, fillColor=text_color))
        if score is None:
            drawing.add(String(center_x - 0.35 * cm, bottom + 0.15 * cm, "No data", fontName="Helvetica", fontSize=6.5, fillColor=muted_color))
            continue
        bar_height = (score / 100.0) * plot_height
        x = center_x - bar_width / 2
        drawing.add(Rect(x, bottom, bar_width, bar_height, fillColor=bar_color, strokeColor=bar_color))
        drawing.add(String(center_x - 0.18 * cm, bottom + bar_height + 0.12 * cm, str(int(round(score))), fontName="Helvetica-Bold", fontSize=7, fillColor=text_color))

    drawing.add(String(left, height - 0.18 * cm, "Voice Stress Level per Answer", fontName="Helvetica-Bold", fontSize=9, fillColor=text_color))
    drawing.add(String(width - 1.55 * cm, bottom + plot_height + 0.08 * cm, "Scale: 0-100", fontName="Helvetica", fontSize=7, fillColor=muted_color))
    return drawing


def generate_pdf_report(analysis: Dict[str, object], test_datetime: datetime | None = None) -> bytes:
    """Generate a clean PDF report that includes transcripts, questions, and results."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
        title=PROJECT_TITLE,
    )
    styles = _build_styles()
    dt = test_datetime or datetime.now()
    main_reasons: List[str] = list(analysis.get("main_reasons", [])) if isinstance(analysis, dict) else []

    story: List[Any] = []
    story.append(Paragraph(PROJECT_TITLE, styles["title"]))
    story.append(Paragraph(f"Test date/time: {dt.strftime('%d %B %Y, %I:%M %p')}", styles["body"]))
    story.append(Paragraph("Test flow: Normal voice sample, initial statement, adaptive follow-up answers.", styles["body"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#d1d5db")))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Initial Statement", styles["section"]))
    story.append(_paragraph(analysis.get("initial_statement", "Not available"), styles["body"]))

    story.append(Paragraph("Adaptive Questions and Answers", styles["section"]))
    story.append(_make_qa_table(analysis, styles))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Analysis Summary", styles["section"]))
    story.append(_make_summary_table(analysis, styles))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Analysis Chart", styles["section"]))
    story.append(_make_voice_stress_chart(analysis))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Main Reasons", styles["section"]))
    if main_reasons:
        for reason in main_reasons:
            story.append(Paragraph(f"- {_clean_text(reason, '')}", styles["body"]))
    else:
        story.append(Paragraph("- No major risk reason was detected.", styles["body"]))

    story.append(Paragraph("Final Result", styles["section"]))
    story.append(Paragraph(f"Final Lie Possibility Score: {analysis.get('final_score', 0)}%", styles["body"]))
    story.append(Paragraph(f"Result Level: {analysis.get('result_level', 'Low')}", styles["body"]))

    story.append(Paragraph("Safety Note", styles["section"]))
    story.append(Paragraph(SAFETY_NOTE, styles["body"]))
    story.append(Paragraph(TRANSCRIPT_NOTE, styles["small"]))

    document.build(story)
    report_bytes = buffer.getvalue()
    buffer.close()
    return report_bytes
