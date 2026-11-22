"""
Generate six rich PDF documents for the company knowledge base.

Each PDF contains a mix of narrative text, ReportLab tables, and
embedded Matplotlib charts (rasterised as images → tests OCR / vision).

Documents
---------
1. employee_handbook.pdf        – mission, code of conduct, performance tables
2. onboarding_policy.pdf        – 30/60/90 plan, checklists, training table
3. return_to_office_policy.pdf  – hybrid schedule, capacity chart, safety table
4. salary_bands_report.pdf      – bar charts, salary-range tables
5. benefits_guide.pdf           – multi-row comparison tables, coverage text
6. leave_policy_report.pdf      – pie/bar charts, policy tables
"""

import io
import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DOCS_DIR, EXCEL_PATH

STYLES = getSampleStyleSheet()

# ── Custom paragraph styles ──────────────────────────────────────────────────
STYLES.add(ParagraphStyle(
    "DocTitle", parent=STYLES["Title"], fontSize=28, spaceAfter=30,
    textColor=colors.HexColor("#1a237e"), alignment=TA_CENTER,
))
STYLES.add(ParagraphStyle(
    "DocSubtitle", parent=STYLES["Heading2"], fontSize=16, spaceAfter=20,
    textColor=colors.HexColor("#37474f"), alignment=TA_CENTER,
))
STYLES.add(ParagraphStyle(
    "SectionHead", parent=STYLES["Heading2"], fontSize=16, spaceBefore=18,
    spaceAfter=8, textColor=colors.HexColor("#1565c0"),
))
STYLES.add(ParagraphStyle(
    "BodyJ", parent=STYLES["BodyText"], alignment=TA_JUSTIFY,
    fontSize=10, leading=14, spaceAfter=8,
))

# Standard table style
STD_TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565c0")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME",  (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE",  (0, 0), (-1, 0), 10),
    ("ALIGN",     (0, 0), (-1, -1), "CENTER"),
    ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
    ("GRID",      (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ("FONTSIZE",  (0, 1), (-1, -1), 9),
    ("TOPPADDING",    (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
])


def _load_employees() -> pd.DataFrame | None:
    if os.path.exists(EXCEL_PATH):
        return pd.read_excel(EXCEL_PATH, engine="openpyxl")
    return None


def _fig_to_rl_image(fig, width=6*inch, height=3.5*inch) -> RLImage:
    """Render a Matplotlib figure to a ReportLab Image flowable."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=width, height=height)


def _title_page(story, title, subtitle="NovaCorp Industries", date="October 2026"):
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph(title, STYLES["DocTitle"]))
    story.append(Paragraph(subtitle, STYLES["DocSubtitle"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(f"<i>Effective Date: {date}</i>", STYLES["DocSubtitle"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="60%", thickness=2,
                             color=colors.HexColor("#1565c0")))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("CONFIDENTIAL — Internal Use Only", STYLES["DocSubtitle"]))
    story.append(PageBreak())


def _new_doc(filename: str) -> tuple[str, SimpleDocTemplate]:
    path = os.path.join(DOCS_DIR, filename)
    doc = SimpleDocTemplate(
        path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    return path, doc


def _scan_font(size: int, bold: bool = False):
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_scan_text(draw, text: str, x: int, y: int, font, max_width: int, line_gap: int = 12):
    words = text.split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if line and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            draw.text((x, y), line, fill=(35, 35, 35), font=font)
            y += font.size + line_gap
            line = word
        else:
            line = candidate
    if line:
        draw.text((x, y), line, fill=(35, 35, 35), font=font)
        y += font.size + line_gap
    return y


def _draw_scan_table(draw, table: list[list[str]], margin: int, width: int, y: int) -> int:
    if not table:
        return y
    cell_width = (width - margin * 2) // len(table[0])
    font = _scan_font(26)
    for row_number, row in enumerate(table):
        top, bottom = y, y + 64
        fill = (222, 231, 228) if row_number == 0 else (247, 246, 240)
        draw.rectangle((margin, top, width - margin, bottom), fill=fill,
                       outline=(125, 133, 130), width=2)
        for col_number, cell in enumerate(row):
            left = margin + col_number * cell_width
            draw.text((left + 12, top + 16), str(cell)[:26],
                      fill=(30, 30, 30), font=font)
            if col_number:
                draw.line((left, top, left, bottom), fill=(125, 133, 130), width=2)
        y = bottom
    return y


def _draw_scan_stamp(draw, width: int, height: int, margin: int, stamp: str) -> None:
    box = (margin, height - 410, margin + 760, height - 280)
    draw.rectangle(box, outline=(153, 59, 50), width=7)
    draw.text((margin + 24, height - 382), stamp, fill=(153, 59, 50),
              font=_scan_font(38, bold=True))


def _render_scanned_page_image(page_content: dict) -> Image.Image:
    """Render a page as a slightly imperfect high-resolution scanned image."""
    width, height = 2550, 3300
    rng = np.random.default_rng(2026)
    grain = rng.normal(0, 1.15, (height, width, 1))
    paper = np.clip(np.full((height, width, 3), [249, 247, 239]) + grain, 0, 255)
    image = Image.fromarray(paper.astype("uint8"), "RGB")
    draw = ImageDraw.Draw(image)
    margin = 190
    y = 170
    body_font = _scan_font(32)
    title_font = _scan_font(48, bold=True)
    y = _draw_scan_text(draw, page_content["title"], margin, y, title_font,
                        width - margin * 2, line_gap=18)
    draw.line((margin, y, width - margin, y), fill=(145, 153, 151), width=3)
    y += 45

    for paragraph in page_content.get("paragraphs", []):
        y = _draw_scan_text(draw, paragraph, margin, y, body_font,
                            width - margin * 2, line_gap=18) + 18

    _draw_scan_table(draw, page_content.get("table", []), margin, width, y)
    _draw_scan_stamp(draw, width, height, margin,
                     page_content.get("stamp", "SIGNED / APPROVED"))
    return image.rotate(0.4, resample=Image.Resampling.BICUBIC, expand=False,
                        fillcolor=(249, 247, 239))


def _render_scanned_pdf(output_path: str, pages_content: list[dict]) -> None:
    """Create an image-only PDF; its pages contain no selectable text objects."""
    pdf = canvas.Canvas(output_path, pagesize=letter)
    for page_content in pages_content:
        image = _render_scanned_page_image(page_content)
        pdf.drawImage(ImageReader(image), 0, 0, width=letter[0], height=letter[1])
        pdf.showPage()
    pdf.save()


def _append_scanned_page(pdf_path: str, page_content: dict) -> None:
    from pypdf import PdfReader, PdfWriter

    scan_pdf = io.BytesIO()
    scan_canvas = canvas.Canvas(scan_pdf, pagesize=letter)
    scan_canvas.drawImage(ImageReader(_render_scanned_page_image(page_content)),
                          0, 0, width=letter[0], height=letter[1])
    scan_canvas.save()
    scan_pdf.seek(0)

    writer = PdfWriter()
    for page in PdfReader(pdf_path).pages:
        writer.add_page(page)
    writer.add_page(PdfReader(scan_pdf).pages[0])
    merged = io.BytesIO()
    writer.write(merged)
    Path(pdf_path).write_bytes(merged.getvalue())


def _long_table(rows: list[list[str]], widths: list[float], color: str,
                row_height: float = 0.32 * inch) -> Table:
    table = Table(rows, colWidths=widths, rowHeights=[0.36 * inch] +
                  [row_height] * (len(rows) - 1), repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f6")]),
    ]))
    return table


def _coverage_matrix_rows() -> list[list[str]]:
    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
    plans = ["Standard", "Premium", "Family Plus", "Global Care", "Wellness Select"]
    rows = [["Matrix ID", "Region", "Plan", "Primary Copay", "Wellness Credit", "Coverage Note"]]
    for index in range(1, 76):
        copay = 30 if index % 3 == 1 else 20 if index % 3 == 2 else 10
        rows.append([
            f"BEN-{index:03}", regions[(index - 1) % len(regions)],
            plans[(index - 1) % len(plans)], f"${copay}",
            f"${400 + (index % 7) * 100}",
            "In-network annual review" if index != 63 else "Emergency waiver within 48 hours",
        ])
    return rows


def _travel_schedule_rows() -> list[list[str]]:
    regions = ["US East", "US West", "UK", "Germany", "Japan", "Singapore", "Brazil"]
    tiers = ["Standard", "Manager", "Executive"]
    rows = [["Schedule ID", "Destination", "Travel Tier", "Daily Per Diem", "Lodging Cap", "Control"]]
    for index in range(1, 76):
        rows.append([
            f"TRV-{index:03}", regions[(index - 1) % len(regions)],
            tiers[(index - 1) % len(tiers)], f"${85 + (index % 8) * 15}",
            f"${180 + (index % 9) * 25}",
            "Receipt required at $25+" if index != 68 else "Foreign currency converted on purchase date",
        ])
    return rows


def _compensation_directory_rows() -> list[list[str]]:
    roles = [
        "Software Engineer", "Data Analyst", "Product Manager", "UX Designer",
        "Security Engineer", "Technical Writer", "Finance Partner", "People Partner",
        "Solutions Architect", "Machine Learning Engineer", "Customer Success Manager",
        "Operations Analyst", "Research Scientist", "Staff Software Engineer",
        "Director, Product", "Principal Security Engineer", "Senior Staff ML Engineer",
        "Regional Sales Director", "Lead Data Platform Engineer", "VP, Customer Experience",
    ]
    locations = ["US", "Canada", "UK", "Germany", "Singapore"]
    rows = [["Role Code", "Role Title", "Level", "Market", "Band Minimum", "Midpoint", "Band Maximum"]]
    for index in range(1, 101):
        base = 62000 + (index % 20) * 4300
        if index == 87:
            role, level, market, base = "Senior Staff ML Engineer", "L7", "US", 178000
        else:
            role = roles[(index - 1) % len(roles)]
            level = f"L{(index - 1) % 7 + 1}"
            market = locations[(index - 1) % len(locations)]
        rows.append([
            f"NC-{index:03}", role, level, market,
            f"${base:,}", f"${base + 18000:,}", f"${base + 36000:,}",
        ])
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EMPLOYEE HANDBOOK
# ═══════════════════════════════════════════════════════════════════════════════

def generate_employee_handbook(df: pd.DataFrame | None):
    path, doc = _new_doc("employee_handbook.pdf")
    story = []
    _title_page(story, "Employee Handbook")

    # --- Company Overview ---
    story.append(Paragraph("1. Company Overview", STYLES["SectionHead"]))
    story.append(Paragraph(textwrap.dedent("""\
        NovaCorp Industries is a global technology and innovation company founded in 2005.
        Headquartered in San Francisco, California, NovaCorp employs over 2,000 professionals
        across 10 departments. Our mission is to empower businesses through cutting-edge
        technology solutions while fostering a diverse and inclusive workplace.
        <br/><br/>
        <b>Vision:</b> To be the world's most innovative and employee-centric technology company.
        <br/><b>Mission:</b> Deliver transformative solutions that drive growth for our clients
        and create meaningful careers for our people.
        <br/><b>Core Values:</b> Innovation, Integrity, Inclusion, Impact, and Continuous Learning.
    """), STYLES["BodyJ"]))

    # --- Org structure table ---
    story.append(Paragraph("1.1 Organisational Structure", STYLES["SectionHead"]))
    if df is not None:
        dept_counts = df["department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Headcount"]
        dept_counts = dept_counts.sort_values("Headcount", ascending=False)
        dept_data = [["Department", "Headcount", "% of Company"]]
        for _, r in dept_counts.iterrows():
            pct = f"{r['Headcount'] / len(df) * 100:.1f}%"
            dept_data.append([r["Department"], str(int(r["Headcount"])), pct])
    else:
        dept_data = [
            ["Department", "Headcount", "% of Company"],
            ["Engineering", "420", "21.0%"], ["Sales", "310", "15.5%"],
            ["Marketing", "250", "12.5%"], ["Finance", "200", "10.0%"],
            ["Operations", "180", "9.0%"], ["HR", "160", "8.0%"],
            ["Product", "150", "7.5%"], ["Customer Support", "130", "6.5%"],
            ["R&D", "110", "5.5%"], ["Legal", "90", "4.5%"],
        ]
    t = Table(dept_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    # Department pie chart (image-based → needs OCR for data)
    if df is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        dept_counts_sorted = df["department"].value_counts()
        wedges, texts, autotexts = ax.pie(
            dept_counts_sorted.values, labels=dept_counts_sorted.index,
            autopct="%1.1f%%", startangle=140,
            colors=plt.cm.Set3(np.linspace(0, 1, len(dept_counts_sorted))),
        )
        for t in texts + autotexts:
            t.set_fontsize(8)
        ax.set_title("Department Distribution", fontsize=14, fontweight="bold")
        story.append(_fig_to_rl_image(fig, width=5.5*inch, height=3.5*inch))
        story.append(Spacer(1, 0.2*inch))

    story.append(PageBreak())

    # --- Code of Conduct ---
    story.append(Paragraph("2. Code of Conduct", STYLES["SectionHead"]))
    story.append(Paragraph(textwrap.dedent("""\
        All employees of NovaCorp Industries are expected to uphold the highest standards
        of professional and ethical behaviour. The following guidelines apply to every
        team member regardless of role or seniority:
        <br/><br/>
        <b>2.1 Professional Behaviour</b><br/>
        • Treat all colleagues, clients, and partners with respect and dignity.<br/>
        • Maintain confidentiality of proprietary information and trade secrets.<br/>
        • Avoid conflicts of interest and disclose any potential conflicts to your manager.<br/>
        • Use company resources responsibly and only for authorised business purposes.<br/>
        <br/>
        <b>2.2 Anti-Discrimination &amp; Harassment</b><br/>
        NovaCorp maintains a zero-tolerance policy for discrimination and harassment based
        on race, colour, religion, gender, sexual orientation, national origin, age,
        disability, or any other protected characteristic. All complaints will be
        investigated promptly and confidentially.
        <br/><br/>
        <b>2.3 Data Privacy &amp; Security</b><br/>
        • Employees must complete annual data-privacy training.<br/>
        • Access to sensitive data is granted on a need-to-know basis.<br/>
        • Report any suspected data breach to the IT Security team within 24 hours.<br/>
        • Personal devices used for work must have approved endpoint protection software.
    """), STYLES["BodyJ"]))

    # --- Disciplinary action table ---
    story.append(Paragraph("2.4 Disciplinary Action Framework", STYLES["SectionHead"]))
    disc_data = [
        ["Level", "Offence Type", "Action", "Duration on Record"],
        ["1 – Verbal Warning", "Minor policy violation\n(e.g., tardiness)",
         "Documented conversation\nwith direct manager", "3 months"],
        ["2 – Written Warning", "Repeated minor violation\nor moderate offence",
         "Formal written notice;\nperformance improvement plan", "6 months"],
        ["3 – Final Warning", "Serious policy breach\n(e.g., data mishandling)",
         "Final notice; mandatory\nretraining; probation period", "12 months"],
        ["4 – Suspension", "Severe misconduct\n(e.g., harassment)",
         "Unpaid suspension pending\ninvestigation", "Permanent"],
        ["5 – Termination", "Gross misconduct\n(e.g., fraud, violence)",
         "Immediate termination;\nlegal referral if applicable", "Permanent"],
    ]
    t = Table(disc_data, colWidths=[1.4*inch, 1.6*inch, 1.8*inch, 1.3*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(PageBreak())

    # --- Performance review ---
    story.append(Paragraph("3. Performance Review Process", STYLES["SectionHead"]))
    story.append(Paragraph(textwrap.dedent("""\
        NovaCorp uses a semi-annual performance review cycle (January and July).
        Each review consists of self-assessment, peer feedback (360°), manager evaluation,
        and a calibration session.
    """), STYLES["BodyJ"]))
    story.append(Paragraph(textwrap.dedent("""\
        Managers set goals at the start of each review period and explain how individual
        results connect to team commitments. Employees receive a written outcome after
        calibration and agree on development steps with their manager. A rating does not
        guarantee a particular salary adjustment; final decisions also follow approved
        budget, internal equity, and local employment requirements.
        <br/><br/>
        Concerns should first be raised with the manager within ten business days. If the
        issue involves the manager or remains unresolved, an employee may request a
        confidential People Partner review. Allegations of discrimination or retaliation
        are escalated to Employee Relations for independent follow-up.
    """), STYLES["BodyJ"]))

    perf_data = [
        ["Rating", "Label", "Description", "Salary Impact", "Bonus Range"],
        ["5", "Exceptional", "Consistently exceeds all\nobjectives; role model", "+8-12%", "20-30%"],
        ["4", "Exceeds\nExpectations", "Exceeds most objectives;\nstrong contributor", "+5-8%", "12-20%"],
        ["3", "Meets\nExpectations", "Meets all core objectives;\nreliable performer", "+2-5%", "5-12%"],
        ["2", "Needs\nImprovement", "Partially meets objectives;\nPIP recommended", "0%", "0-5%"],
        ["1", "Unsatisfactory", "Fails to meet objectives;\nimmediate PIP required", "0%", "0%"],
    ]
    t = Table(perf_data, colWidths=[0.7*inch, 1.1*inch, 2*inch, 1*inch, 1*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    # --- Work Hours ---
    story.append(Paragraph("4. Work Hours & Attendance", STYLES["SectionHead"]))
    story.append(Paragraph(textwrap.dedent("""\
        <b>Standard Hours:</b> Monday–Friday, 9:00 AM – 6:00 PM (local time), with a
        one-hour lunch break. Flexible scheduling is available with manager approval.
        <br/><br/>
        <b>Remote Work:</b> Eligible employees may work remotely up to 3 days per week
        in accordance with the Return-to-Office Policy (see separate document).
        <br/><br/>
        <b>Overtime:</b> Non-exempt employees are eligible for overtime pay at 1.5× the
        regular hourly rate for hours exceeding 40 per week.
        <br/><br/>
        <b>Time Tracking:</b> All employees must log their hours daily in the NovaCorp
        Time Portal. Persistent failure to log time may result in disciplinary action.
    """), STYLES["BodyJ"]))

    story.append(PageBreak())
    story.append(Paragraph("5. Global Travel Per-Diem, Lodging Cap & Expense Compliance Schedule",
                           STYLES["SectionHead"]))
    story.append(Paragraph(
        "Use the destination and traveler tier to identify the daily meal per-diem and "
        "nightly lodging cap. This 75-row schedule is repeated across pages with its "
        "column headings on every continuation page.", STYLES["BodyJ"],
    ))
    travel_table = _long_table(
        _travel_schedule_rows(),
        [0.85*inch, 1.0*inch, 0.95*inch, 1.0*inch, 1.0*inch, 2.2*inch],
        "#1b5e20", row_height=0.32*inch,
    )
    story.append(travel_table)
    story.append(Paragraph(
        "Itemized receipts are required for each expense of $25 or more; smaller tips "
        "and transit fares may be grouped when local rules permit. The traveler must "
        "submit the itemized record and business purpose within ten business days of return.",
        STYLES["BodyJ"],
    ))
    story.append(Paragraph(
        "For foreign-currency purchases, Finance converts the posted amount using the "
        "card-network rate on the transaction date. Cash advances use the published "
        "central-bank rate for the purchase date, with the rate source attached to the claim.",
        STYLES["BodyJ"],
    ))
    story.append(Paragraph(
        "Personal entertainment, companion travel, traffic fines, minibar purchases, and "
        "upgrades chosen for convenience are not reimbursable. A travel disruption can "
        "justify an exception only when the manager documents the business need before "
        "submission; repeated exceptions are escalated to Finance Compliance.",
        STYLES["BodyJ"],
    ))

    doc.build(story)
    print(f"  📄 {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ONBOARDING POLICY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_onboarding_policy():
    path, doc = _new_doc("onboarding_policy.pdf")
    story = []
    _title_page(story, "Onboarding Policy &\nNew Hire Guide")

    story.append(Paragraph("1. Welcome to NovaCorp", STYLES["SectionHead"]))
    story.append(Paragraph(textwrap.dedent("""\
        Congratulations on joining NovaCorp Industries! This onboarding guide will help
        you navigate your first 90 days. Your onboarding buddy and manager will support
        you every step of the way. Please complete all required items within the specified
        time-frames to ensure a smooth start.
    """), STYLES["BodyJ"]))

    # 30/60/90 day plan
    story.append(Paragraph("2. 30 / 60 / 90 Day Plan", STYLES["SectionHead"]))
    plan_data = [
        ["Phase", "Time-frame", "Focus Areas", "Key Milestones", "Owner"],
        ["Phase 1:\nFoundation", "Days 1–30",
         "• Complete all onboarding tasks\n• Meet your team & key stakeholders\n"
         "• Understand team processes & tools\n• Complete mandatory trainings",
         "• All systems access granted\n• 1-on-1 with skip-level manager\n"
         "• First team presentation", "Manager +\nHR Partner"],
        ["Phase 2:\nIntegration", "Days 31–60",
         "• Take ownership of first project\n• Shadow senior team members\n"
         "• Attend cross-functional meetings\n• Begin contributing to sprints",
         "• First project deliverable\n• Mid-point check-in with HR\n"
         "• Peer feedback collected", "Manager"],
        ["Phase 3:\nAcceleration", "Days 61–90",
         "• Lead a small project independently\n• Propose process improvements\n"
         "• Present learnings to the team\n• Set 6-month goals with manager",
         "• 90-day review completed\n• Goals documented in HR portal\n"
         "• Probation sign-off", "Manager +\nHR Partner"],
    ]
    t = Table(plan_data, colWidths=[1*inch, 0.9*inch, 2.1*inch, 1.8*inch, 0.9*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME",  (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, 0), 9),
        ("ALIGN",     (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",    (0, 0), (-1, -1), "TOP"),
        ("GRID",      (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#e8f5e9"), colors.white]),
        ("FONTSIZE",  (0, 1), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(PageBreak())

    # Onboarding checklist
    story.append(Paragraph("3. Onboarding Checklist", STYLES["SectionHead"]))
    check_data = [
        ["#", "Task", "Category", "Deadline", "Status"],
        ["1", "Sign employment contract & NDA", "Legal", "Day 1", "☐"],
        ["2", "Submit tax forms (W-4 / W-9)", "Finance", "Day 1", "☐"],
        ["3", "Collect laptop & peripherals from IT", "IT Setup", "Day 1", "☐"],
        ["4", "Set up email, Slack, and VPN access", "IT Setup", "Day 1", "☐"],
        ["5", "Enrol in health insurance & benefits", "HR", "Week 1", "☐"],
        ["6", "Complete cybersecurity awareness training", "Training", "Week 1", "☐"],
        ["7", "Complete diversity & inclusion training", "Training", "Week 1", "☐"],
        ["8", "Complete data privacy (GDPR/CCPA) training", "Training", "Week 2", "☐"],
        ["9", "Meet with onboarding buddy (first meeting)", "People", "Week 1", "☐"],
        ["10", "Attend new-hire orientation session", "HR", "Week 1", "☐"],
        ["11", "1-on-1 with direct manager (goals discussion)", "People", "Week 1", "☐"],
        ["12", "Review team wiki & documentation", "Self-study", "Week 2", "☐"],
        ["13", "Complete department-specific technical onboarding", "Training", "Month 1", "☐"],
        ["14", "Submit 30-day self-assessment", "Performance", "Day 30", "☐"],
        ["15", "90-day review & probation sign-off", "Performance", "Day 90", "☐"],
    ]
    t = Table(check_data, colWidths=[0.4*inch, 2.8*inch, 1*inch, 0.9*inch, 0.6*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(PageBreak())

    # Required trainings table
    story.append(Paragraph("4. Required Training Modules", STYLES["SectionHead"]))
    story.append(Paragraph(
        "All new hires must complete the following training modules via the NovaCorp "
        "Learning Management System (LMS). Failure to complete within the deadline "
        "may delay probation sign-off.",
        STYLES["BodyJ"],
    ))
    train_data = [
        ["Module", "Duration", "Format", "Deadline", "Certification"],
        ["Cybersecurity Fundamentals", "2 hours", "E-learning + Quiz", "Week 1", "Yes – 90% pass"],
        ["Data Privacy & GDPR/CCPA", "1.5 hours", "E-learning + Quiz", "Week 2", "Yes – 85% pass"],
        ["Diversity, Equity & Inclusion", "1 hour", "Video + Discussion", "Week 1", "Attendance"],
        ["Anti-Harassment & Workplace Safety", "1 hour", "E-learning + Quiz", "Week 1", "Yes – 90% pass"],
        ["Code of Conduct & Ethics", "45 min", "E-learning", "Week 1", "Acknowledgement"],
        ["IT Systems & Tools Orientation", "2 hours", "Live workshop", "Week 1", "Attendance"],
        ["Project Management (Agile/Scrum)", "3 hours", "E-learning + Lab", "Month 1", "Optional cert"],
        ["Emergency Procedures & First Aid", "30 min", "Video", "Week 2", "Acknowledgement"],
    ]
    t = Table(train_data, colWidths=[1.8*inch, 0.8*inch, 1.2*inch, 0.8*inch, 1.2*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    # IT setup
    story.append(Paragraph("5. IT Setup Guide", STYLES["SectionHead"]))
    it_data = [
        ["System / Tool", "Purpose", "Access Request", "SLA"],
        ["Email (Google Workspace)", "Corporate email & calendar", "Auto-provisioned", "Day 1"],
        ["Slack", "Team messaging & channels", "Auto-provisioned", "Day 1"],
        ["Jira / Confluence", "Project management & wiki", "IT ticket required", "Day 1-2"],
        ["GitHub Enterprise", "Source code & CI/CD", "Manager approval", "Day 1-3"],
        ["VPN (GlobalProtect)", "Secure remote access", "IT ticket required", "Day 1"],
        ["HR Portal (Workday)", "Leave, payslips, profile", "Auto-provisioned", "Day 1"],
        ["LMS (Cornerstone)", "Training & certifications", "Auto-provisioned", "Day 1"],
        ["Expense System (Concur)", "Travel & expense claims", "Finance approval", "Week 1"],
    ]
    t = Table(it_data, colWidths=[1.6*inch, 1.6*inch, 1.3*inch, 0.8*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(Paragraph(textwrap.dedent("""\
        The manager and onboarding buddy review progress at days 7, 30, 60, and 90. A
        required security, safety, or conduct module must be completed before independent
        access to the related system is granted. If equipment or account access is delayed,
        the new hire records the blocker in the onboarding checklist and the manager opens
        an IT service request rather than sharing another employee's credentials.
        <br/><br/>
        An accommodation, leave, or immigration question should be referred to the People
        Partner before a start date or assignment is changed. The People Partner coordinates
        with Legal or Security when needed and gives the employee a written next step and
        expected response date. Exceptions to the standard 30/60/90 schedule require a
        documented manager reason and HR approval.
    """), STYLES["BodyJ"]))

    doc.build(story)
    print(f"  📄 {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RETURN-TO-OFFICE POLICY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_rto_policy():
    path, doc = _new_doc("return_to_office_policy.pdf")
    story = []
    _title_page(story, "Return-to-Office\n& Hybrid Work Policy")

    story.append(Paragraph("1. Policy Overview", STYLES["SectionHead"]))
    story.append(Paragraph(textwrap.dedent("""\
        Effective January 2026, NovaCorp Industries adopts a structured hybrid work model.
        This policy balances the benefits of in-person collaboration with the flexibility
        that our employees value. All full-time employees are expected to be in the office
        a minimum of 2 days per week (Tuesday and Thursday are anchor days), with the option
        to work remotely the remaining days.
        <br/><br/>
        <b>Scope:</b> This policy applies to all full-time and part-time employees globally.
        Contract workers and interns follow department-specific guidelines.
        <br/><br/>
        <b>Exceptions:</b> Employees with approved medical accommodations, those in
        fully-remote roles (as indicated in their offer letter), and employees based
        in locations without a NovaCorp office are exempt from mandatory in-office days.
    """), STYLES["BodyJ"]))

    # Hybrid schedule table
    story.append(Paragraph("2. Weekly Hybrid Schedule", STYLES["SectionHead"]))
    sched_data = [
        ["Day", "Status", "Notes"],
        ["Monday", "Flexible (Remote / Office)", "Team-optional; recommended for focus work"],
        ["Tuesday", "MANDATORY In-Office", "Anchor day – all-hands, team syncs, workshops"],
        ["Wednesday", "Flexible (Remote / Office)", "Manager discretion; cross-team collabs"],
        ["Thursday", "MANDATORY In-Office", "Anchor day – 1:1s, sprint reviews, socials"],
        ["Friday", "Flexible (Remote / Office)", "Focus day; early finish at 4 PM encouraged"],
    ]
    t = Table(sched_data, colWidths=[1.2*inch, 2*inch, 3*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e65100")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME",  (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN",     (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",      (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#fff3e0")),
        ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#fff3e0")),
        ("FONTSIZE",  (0, 1), (-1, -1), 9),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    # Office capacity chart (image-based)
    fig, ax = plt.subplots(figsize=(8, 4))
    months_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    capacity = [45, 52, 58, 63, 68, 72, 65, 60, 75, 78, 80, 70]
    target = [75] * 12
    ax.bar(months_labels, capacity, color="#ff8f00", alpha=0.8, label="Actual Utilisation %")
    ax.plot(months_labels, target, "r--", linewidth=2, label="Target (75%)")
    ax.set_ylabel("Office Utilisation (%)")
    ax.set_title("Monthly Office Utilisation Rate – 2025", fontsize=13, fontweight="bold", pad=12)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    story.append(Paragraph("2.1 Office Utilisation Trends", STYLES["SectionHead"]))
    story.append(_fig_to_rl_image(fig, width=6*inch, height=3*inch))

    # Text summary table so PyPDFLoader can retrieve utilisation numbers
    story.append(Paragraph("2.2 Monthly Utilisation Data – 2025", STYLES["SectionHead"]))
    util_table_data = [["Month", "Utilisation (%)", "vs. Target (75%)"]]
    for m, v in zip(months_labels, capacity):
        delta = v - 75
        delta_str = f"+{delta}%" if delta >= 0 else f"{delta}%"
        util_table_data.append([m, f"{v}%", delta_str])
    t = Table(util_table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(PageBreak())

    # Department rotation
    story.append(Paragraph("3. Department Rotation Schedule", STYLES["SectionHead"]))
    story.append(Paragraph(
        "To manage office capacity, departments follow a rotation schedule for the "
        "flexible days (Monday, Wednesday, Friday). Each department is assigned "
        "priority access to specific floors and meeting rooms on designated days.",
        STYLES["BodyJ"],
    ))
    rot_data = [
        ["Department", "Priority Day", "Floor", "Meeting Rooms", "Desk Zone"],
        ["Engineering", "Monday", "3rd & 4th", "Rooms 3A-3D", "Zone A (120 desks)"],
        ["Product", "Monday", "4th", "Rooms 4A-4B", "Zone B (40 desks)"],
        ["Marketing", "Wednesday", "2nd", "Rooms 2A-2C", "Zone C (60 desks)"],
        ["Sales", "Wednesday", "2nd & 5th", "Rooms 5A-5B", "Zone D (80 desks)"],
        ["Finance", "Friday", "5th", "Rooms 5C-5D", "Zone E (40 desks)"],
        ["Human Resources", "Friday", "1st", "Rooms 1A-1B", "Zone F (30 desks)"],
        ["Operations", "Monday", "1st", "Rooms 1C-1D", "Zone G (35 desks)"],
        ["Legal", "Friday", "5th", "Room 5E", "Zone H (20 desks)"],
        ["Customer Support", "Wednesday", "1st", "Rooms 1E-1F", "Zone I (30 desks)"],
        ["R&D", "Monday", "3rd", "Lab & Room 3E", "Zone J (25 desks)"],
    ]
    t = Table(rot_data, colWidths=[1.2*inch, 0.9*inch, 0.9*inch, 1.2*inch, 1.4*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    # Health & safety
    story.append(Paragraph("4. Health & Safety Guidelines", STYLES["SectionHead"]))
    safety_data = [
        ["Guideline", "Requirement", "Responsible Party"],
        ["Desk Sanitation", "Wipe down desk & keyboard at end of day", "All employees"],
        ["Air Quality Monitoring", "HEPA filters maintained quarterly", "Facilities"],
        ["Ergonomic Assessment", "Annual workstation assessment", "HR + Facilities"],
        ["First Aid Kits", "Stocked on every floor; checked monthly", "Facilities"],
        ["Emergency Exits", "Clearly marked; monthly fire drills", "Safety Committee"],
        ["COVID-19 Protocol", "Stay home if symptomatic; masks optional", "All employees"],
        ["Mental Health Room", "Quiet rooms available on floors 2 & 4", "HR"],
        ["Visitor Policy", "All visitors must sign in; escorted access only", "Reception + Host"],
    ]
    t = Table(safety_data, colWidths=[1.5*inch, 2.5*inch, 1.3*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(Paragraph(textwrap.dedent("""\
        Employees coordinate their office days with their team lead at the beginning of
        each quarter. A team may designate shared anchor days for planning and mentoring,
        while individual schedule changes remain available through the manager. Employees
        working from another country must request separate approval before travel because
        payroll, privacy, and immigration rules may differ by jurisdiction.
        <br/><br/>
        A medical accommodation or temporary safety concern can support a schedule exception.
        Contact the People Partner with the requested dates and supporting details; managers
        should not ask for unnecessary medical information. If a building issue prevents
        safe access, Facilities may close a site and notify employees to work remotely until
        the location is cleared.
    """), STYLES["BodyJ"]))

    doc.build(story)
    print(f"  📄 {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SALARY BANDS REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_salary_report(df: pd.DataFrame | None):
    path, doc = _new_doc("salary_bands_report.pdf")
    story = []
    _title_page(story, "Salary Bands &\nCompensation Report")

    if df is None:
        df = pd.DataFrame({
            "department": ["Engineering", "Product", "Finance", "Human Resources"],
            "salary": [118000, 104000, 92000, 88000],
            "years_of_expertise": [6, 4, 8, 5],
        })

    # Avg salary by department bar chart
    story.append(Paragraph("1. Average Salary by Department", STYLES["SectionHead"]))
    avg_sal = df.groupby("department")["salary"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(avg_sal.index, avg_sal.values, color=plt.cm.viridis(
        np.linspace(0.2, 0.9, len(avg_sal))))
    ax.set_xlabel("Average Salary ($)")
    ax.set_title("Average Salary by Department", fontsize=14, fontweight="bold", pad=12)
    for bar, val in zip(bars, avg_sal.values):
        ax.text(val + 500, bar.get_y() + bar.get_height()/2,
                f"${val:,.0f}", va="center", fontsize=8)
    # Expand x-axis by 15% on the right so labels don't clip
    xmax = avg_sal.max()
    ax.set_xlim(0, xmax * 1.15)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    story.append(_fig_to_rl_image(fig, width=6*inch, height=3.5*inch))

    # Text summary table so PyPDFLoader can extract numbers without OCR
    story.append(Paragraph("1.1 Average Salary by Department — Summary", STYLES["SectionHead"]))
    sal_table_data = [["Department", "Average Salary ($)"]]
    for dept, val in avg_sal.sort_values(ascending=False).items():
        sal_table_data.append([dept, f"${val:,.0f}"])
    t = Table(sal_table_data, colWidths=[3*inch, 2*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(PageBreak())

    # Salary distribution histogram
    story.append(Paragraph("2. Salary Distribution", STYLES["SectionHead"]))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(df["salary"], bins=40, color="#1565c0", alpha=0.8, edgecolor="white")
    ax.axvline(df["salary"].mean(), color="red", linestyle="--", linewidth=2,
               label=f'Mean: ${df["salary"].mean():,.0f}')
    ax.axvline(df["salary"].median(), color="green", linestyle="--", linewidth=2,
               label=f'Median: ${df["salary"].median():,.0f}')
    ax.set_xlabel("Salary ($)")
    ax.set_ylabel("Number of Employees")
    ax.set_title("Company-Wide Salary Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    story.append(_fig_to_rl_image(fig, width=6*inch, height=3.2*inch))
    story.append(Spacer(1, 0.2*inch))

    # Summary statistics table
    story.append(Paragraph("2.1 Statistical Summary", STYLES["SectionHead"]))
    stats = df["salary"].describe()
    stats_data = [
        ["Metric", "Value"],
        ["Count", f"{stats['count']:,.0f}"],
        ["Mean", f"${stats['mean']:,.2f}"],
        ["Std Dev", f"${stats['std']:,.2f}"],
        ["Min", f"${stats['min']:,.2f}"],
        ["25th Percentile", f"${stats['25%']:,.2f}"],
        ["Median (50th)", f"${stats['50%']:,.2f}"],
        ["75th Percentile", f"${stats['75%']:,.2f}"],
        ["Max", f"${stats['max']:,.2f}"],
    ]
    t = Table(stats_data, colWidths=[2*inch, 2*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(PageBreak())

    # Salary by seniority table
    story.append(Paragraph("3. Salary Bands by Seniority Level", STYLES["SectionHead"]))
    story.append(Paragraph(
        "The following table shows the approved salary bands by seniority level. "
        "Actual salaries may vary based on location, performance, and market adjustments.",
        STYLES["BodyJ"],
    ))
    band_data = [
        ["Seniority", "Min Salary", "Max Salary", "Target Midpoint", "Employees"],
        ["Junior (0-1 yrs)", "$38,000", "$65,000", "$51,500",
         str(len(df[df["years_of_expertise"] < 2]))],
        ["Mid (2-4 yrs)", "$60,000", "$95,000", "$77,500",
         str(len(df[(df["years_of_expertise"] >= 2) & (df["years_of_expertise"] < 5)]))],
        ["Senior (5-9 yrs)", "$90,000", "$130,000", "$110,000",
         str(len(df[(df["years_of_expertise"] >= 5) & (df["years_of_expertise"] < 10)]))],
        ["Lead/Manager (10-14 yrs)", "$110,000", "$165,000", "$137,500",
         str(len(df[(df["years_of_expertise"] >= 10) & (df["years_of_expertise"] < 15)]))],
        ["Director/VP (15+ yrs)", "$150,000", "$230,000", "$190,000",
         str(len(df[df["years_of_expertise"] >= 15]))],
    ]
    t = Table(band_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.1*inch, 0.9*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    story.append(PageBreak())
    story.append(Paragraph("5. Role-by-Role Compensation & Equity Band Directory",
                           STYLES["SectionHead"]))
    story.append(Paragraph(textwrap.dedent("""\
        The directory uses a consistent job architecture and market reference date of
        October 2026. Compensation analysts match each role to scope, decision authority,
        skills, and sustained impact before assigning a level; geographic differentials
        are then applied to the market midpoint. Minimum and maximum values represent
        approved base-pay ranges and do not include bonus, equity awards, or benefits.
        <br/><br/>
        Equity reviews compare employees performing substantially similar work while
        accounting for relevant experience, documented performance, and location. Proposed
        offers outside the listed range require written justification, budget confirmation,
        and review by the People Compensation team before a recruiter makes a commitment.
    """), STYLES["BodyJ"]))
    story.append(_long_table(
        _compensation_directory_rows(),
        [0.65*inch, 2.0*inch, 0.6*inch, 0.65*inch, 1.0*inch, 1.0*inch, 1.1*inch],
        "#4a148c", row_height=0.31*inch,
    ))
    story.append(Paragraph(
        "Footnote: Salary bands are internal planning ranges, not guarantees or individual pay disclosures.",
        STYLES["BodyJ"],
    ))

    # Salary by dept+seniority heatmap (complex chart → OCR)
    story.append(Paragraph("4. Salary Heatmap: Department × Experience",
                           STYLES["SectionHead"]))
    bins = [0, 2, 5, 10, 15, 31]
    labels = ["0-1y", "2-4y", "5-9y", "10-14y", "15y+"]
    df["exp_bin"] = pd.cut(df["years_of_expertise"], bins=bins, labels=labels, right=False)
    pivot = df.pivot_table(values="salary", index="department",
                           columns="exp_bin", aggfunc="mean", observed=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"${val/1000:.0f}k", ha="center", va="center",
                        fontsize=7, color="black" if val < 120000 else "white")
    ax.set_title("Average Salary Heatmap ($k) by Department & Experience",
                 fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, label="Salary ($)", shrink=0.8)
    fig.tight_layout()
    story.append(_fig_to_rl_image(fig, width=6.2*inch, height=3.8*inch))

    # Clean up temp column
    df.drop(columns=["exp_bin"], inplace=True, errors="ignore")

    doc.build(story)
    print(f"  📄 {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BENEFITS GUIDE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_benefits_guide():
    path, doc = _new_doc("benefits_guide.pdf")
    story = []
    _title_page(story, "Employee Benefits\n& Wellness Guide")

    story.append(Paragraph("1. Health Insurance Plans", STYLES["SectionHead"]))
    story.append(Paragraph(
        "NovaCorp offers three health insurance tiers. All plans include dental and "
        "vision coverage. The company covers 80% of premiums for the Standard plan, "
        "70% for Premium, and 60% for Premium Plus. Dependents can be added at "
        "additional cost.",
        STYLES["BodyJ"],
    ))
    health_data = [
        ["Feature", "Standard Plan", "Premium Plan", "Premium Plus"],
        ["Monthly Premium\n(Employee)", "$150/month", "$280/month", "$420/month"],
        ["Company Subsidy", "80%", "70%", "60%"],
        ["Annual Deductible", "$2,000", "$1,000", "$500"],
        ["Out-of-Pocket Max", "$6,000", "$4,000", "$2,500"],
        ["Doctor Visit Copay", "$30", "$20", "$10"],
        ["Specialist Copay", "$60", "$40", "$25"],
        ["Emergency Room", "$300", "$200", "$100"],
        ["Prescription (Generic)", "$10", "$5", "$0"],
        ["Prescription (Brand)", "$40", "$25", "$15"],
        ["Mental Health Sessions", "12/year", "24/year", "Unlimited"],
        ["Dental Coverage", "Basic", "Comprehensive", "Comprehensive+Ortho"],
        ["Vision Coverage", "Exam only", "Exam + $150 frames", "Exam + $300 frames"],
        ["Telehealth", "Included", "Included", "Included"],
        ["HSA Eligible", "Yes", "No", "No"],
        ["Network", "Regional", "National", "National + International"],
    ]
    t = Table(health_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00695c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME",  (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, 0), 9),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e0f2f1")),
        ("FONTNAME",  (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN",     (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",      (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f5f5")]),
        ("FONTSIZE",  (0, 1), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(PageBreak())

    # Retirement
    story.append(Paragraph("2. Retirement & Financial Benefits", STYLES["SectionHead"]))
    retire_data = [
        ["Benefit", "Details", "Eligibility", "Vesting"],
        ["401(k) Plan", "Up to 6% company match\n(dollar-for-dollar)", "All FTEs after\n30 days", "3-year graded\n(33%/66%/100%)"],
        ["Employee Stock\nPurchase Plan (ESPP)", "15% discount on\ncompany stock", "All FTEs after\n6 months", "Immediate"],
        ["Restricted Stock\nUnits (RSUs)", "Granted based on\nrole & performance", "Senior level\nand above", "4-year cliff\n(25% annually)"],
        ["Financial Planning", "Free access to\ncertified financial advisor", "All employees", "Immediate"],
        ["Student Loan\nAssistance", "$100/month toward\nstudent loan payments", "All FTEs", "Immediate\n(max 5 years)"],
    ]
    t = Table(retire_data, colWidths=[1.3*inch, 1.6*inch, 1.3*inch, 1.3*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    # Wellness
    story.append(Paragraph("3. Wellness & Lifestyle Benefits", STYLES["SectionHead"]))
    well_data = [
        ["Benefit", "Details", "Annual Value"],
        ["Gym Membership", "Subsidized membership at partner gyms\nor $50/month fitness stipend", "$600"],
        ["Mental Health Support", "Free access to Talkspace/BetterHelp;\nEAP (Employee Assistance Program)", "$2,400"],
        ["Wellness Stipend", "For meditation apps, ergonomic equipment,\nhealth screenings", "$500"],
        ["Commuter Benefits", "Pre-tax transit & parking; $75/month\nbike-to-work stipend", "$900"],
        ["Meals & Snacks", "Free lunch on in-office days;\nhealthy snacks available daily", "$3,000"],
        ["Learning & Development", "Annual learning budget for courses,\nconferences, and certifications", "$3,000"],
        ["Parental Support", "Backup childcare (20 days/year);\nlactation rooms available", "$4,000"],
        ["Volunteer Time Off", "16 hours/year paid volunteer time", "2 days"],
    ]
    t = Table(well_data, colWidths=[1.3*inch, 2.8*inch, 1*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(PageBreak())

    # Education reimbursement
    story.append(Paragraph("4. Education & Professional Development",
                           STYLES["SectionHead"]))
    story.append(Paragraph(textwrap.dedent("""\
        NovaCorp invests in your growth. The following education benefits are available
        to all full-time employees who have completed their probation period:
    """), STYLES["BodyJ"]))
    edu_data = [
        ["Program", "Coverage", "Max Amount", "Approval", "Conditions"],
        ["Tuition\nReimbursement", "Degree programs\n(accredited institutions)", "$10,000/year", "HR + Manager", "Maintain B average;\n2-year commitment"],
        ["Professional\nCertifications", "Industry certifications\n(AWS, PMP, CPA, etc.)", "$3,000/year", "Manager", "Must be role-relevant;\npass on first attempt"],
        ["Conference\nAttendance", "Industry conferences\n& summits", "$2,500/year", "Manager", "Present learnings\nto team after"],
        ["Online Learning", "Coursera, Udemy,\nLinkedIn Learning", "$1,500/year", "Self-service", "Unlimited access\nto partner platforms"],
        ["Book Allowance", "Technical & business\nbooks", "$300/year", "Self-service", "Submit receipts\nvia expense portal"],
    ]
    t = Table(edu_data, colWidths=[1*inch, 1.2*inch, 1*inch, 0.9*inch, 1.4*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)

    # Benefits value chart (image-based)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("5. Total Benefits Value Estimate", STYLES["SectionHead"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    categories = ["Health\nInsurance", "401(k)\nMatch", "Learning\n& Dev",
                  "Wellness", "Meals", "Parental\nSupport", "Commuter", "Other"]
    values = [8500, 7200, 3000, 3500, 3000, 4000, 900, 1400]
    clrs = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    bars = ax.bar(categories, values, color=clrs, edgecolor="grey", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"${val:,}", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("Annual Value ($)")
    ax.set_title(f"Estimated Total Benefits Value: ${sum(values):,}/year",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    story.append(_fig_to_rl_image(fig, width=6*inch, height=3.3*inch))

    story.append(PageBreak())
    story.append(Paragraph("6. Global Benefits, Copay & Wellness Coverage Matrix",
                           STYLES["SectionHead"]))
    story.append(Paragraph(
        "Use the region and plan tier to compare routine coverage and annual wellness "
        "credits. Enrollment is available to eligible employees and dependents during "
        "the annual election window or after a qualifying life event.", STYLES["BodyJ"],
    ))
    story.append(_long_table(
        _coverage_matrix_rows(),
        [0.8*inch, 1.05*inch, 1.0*inch, 1.0*inch, 1.0*inch, 2.15*inch],
        "#00695c", row_height=0.32*inch,
    ))
    story.append(Paragraph(
        "A marriage, birth, adoption, loss of other coverage, or qualifying relocation "
        "opens a 30-day election window. Submit the event record through the Benefits "
        "Portal; coverage changes take effect on the event date when submitted on time.",
        STYLES["BodyJ"],
    ))
    story.append(Paragraph(
        "If a claim is denied, request an internal appeal within 60 days and include the "
        "explanation of benefits, relevant receipts, and any clinical documentation. The "
        "Benefits team acknowledges an appeal within five business days and sends a written "
        "decision or status update within 30 days.", STYLES["BodyJ"],
    ))
    story.append(Paragraph(
        "Subtle emergency exception: when an out-of-network emergency makes advance "
        "authorization impracticable, notify the insurer and NovaCorp Benefits within "
        "48 hours after stabilization. The pre-authorization requirement is waived for "
        "that emergency episode, but routine follow-up care still needs approval.",
        STYLES["BodyJ"],
    ))

    doc.build(story)
    print(f"  📄 {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LEAVE POLICY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_leave_report(df: pd.DataFrame | None):
    path, doc = _new_doc("leave_policy_report.pdf")
    story = []
    _title_page(story, "Leave Policy &\nAbsence Analytics Report")

    # Leave types table
    story.append(Paragraph("1. Leave Types & Entitlements", STYLES["SectionHead"]))
    leave_data = [
        ["Leave Type", "Entitlement", "Carry-Over", "Approval", "Documentation"],
        ["Paid Time Off\n(PTO)", "15-25 days/year\n(based on tenure)", "Max 5 days\nto next year",
         "Manager\n(3+ days advance)", "None"],
        ["Sick Leave", "Up to 15 days/year\n(as needed)", "Non-\ncarryover",
         "Self-certified\n(≤3 days)", "Doctor's note\nif >3 consecutive"],
        ["Maternity Leave", "Up to 90 days\n(fully paid)", "N/A",
         "HR Partner\n(4 weeks advance)", "Medical certificate\nrequired"],
        ["Paternity Leave", "Up to 15 days\n(fully paid)", "N/A",
         "HR Partner\n(2 weeks advance)", "Birth certificate\nor hospital letter"],
        ["Mental Health\nLeave", "Up to 5 days/year\n(fully paid)", "Non-\ncarryover",
         "Self-certified\nor EAP referral", "EAP referral for\nextended leave"],
        ["Bereavement", "5 days (immediate\nfamily); 3 days other", "N/A",
         "Manager\n(immediate notice)", "Death certificate\n(if requested)"],
        ["Jury Duty", "As required\n(fully paid)", "N/A",
         "Manager\n(with summons)", "Court summons"],
        ["Volunteer\nTime Off", "2 days/year\n(fully paid)", "Non-\ncarryover",
         "Manager\n(1 week advance)", "Volunteer org\nconfirmation"],
        ["Unpaid Leave\nof Absence", "Up to 30 days\n(case-by-case)", "N/A",
         "HR Director\napproval required", "Written request\nwith justification"],
    ]
    t = Table(leave_data, colWidths=[1.1*inch, 1.2*inch, 0.9*inch, 1.1*inch, 1.2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a148c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME",  (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, 0), 9),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3e5f5")),
        ("FONTNAME",  (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN",     (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",    (0, 0), (-1, -1), "TOP"),
        ("GRID",      (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE",  (0, 1), (-1, -1), 7.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(PageBreak())

    if df is not None:
        # Leave distribution pie chart
        story.append(Paragraph("2. Company-Wide Leave Distribution",
                               STYLES["SectionHead"]))
        total_sick = df["sick_leave_days"].sum()
        total_mat = df["maternity_leave_days"].sum()
        total_pat = df["paternity_leave_days"].sum()
        total_pto = df["paid_time_off_days"].sum()
        total_mh = df["mental_health_leave_days"].sum()

        # ── Pie chart (standalone, full width) ──────────────────────────────
        fig_pie, ax_pie = plt.subplots(figsize=(7, 4))
        leave_labels = ["Sick Leave", "Maternity", "Paternity", "PTO", "Mental Health"]
        leave_vals = [total_sick, total_mat, total_pat, total_pto, total_mh]
        pie_colors = ["#ef5350", "#ab47bc", "#42a5f5", "#66bb6a", "#ffa726"]
        wedges, texts, autotexts = ax_pie.pie(
            leave_vals, labels=leave_labels, autopct="%1.1f%%",
            colors=pie_colors, startangle=90,
        )
        for t in texts + autotexts:
            t.set_fontsize(8)
        ax_pie.set_title("Leave Distribution by Type", fontsize=12, fontweight="bold", pad=12)
        story.append(_fig_to_rl_image(fig_pie, width=5*inch, height=3*inch))
        story.append(Spacer(1, 0.15*inch))

        # ── Bar chart: avg leave by department (standalone, full width) ──────
        dept_leave = df.groupby("department").agg({
            "sick_leave_days": "mean",
            "paid_time_off_days": "mean",
            "mental_health_leave_days": "mean",
        }).sort_values("sick_leave_days", ascending=True)
        x_pos = np.arange(len(dept_leave))
        w = 0.25
        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        ax_bar.barh(x_pos - w, dept_leave["sick_leave_days"], w,
                    label="Sick", color="#ef5350")
        ax_bar.barh(x_pos, dept_leave["paid_time_off_days"], w,
                    label="PTO", color="#66bb6a")
        ax_bar.barh(x_pos + w, dept_leave["mental_health_leave_days"], w,
                    label="Mental Health", color="#ffa726")
        ax_bar.set_yticks(x_pos)
        ax_bar.set_yticklabels(dept_leave.index, fontsize=8)
        ax_bar.set_xlabel("Average Days")
        ax_bar.set_title("Average Leave by Department", fontsize=12, fontweight="bold", pad=12)
        # Expand x-axis by 15% to prevent label clipping
        ax_bar.set_xlim(0, dept_leave[["sick_leave_days", "paid_time_off_days",
                                        "mental_health_leave_days"]].max().max() * 1.15)
        ax_bar.legend(fontsize=8, loc="lower right")
        ax_bar.grid(axis="x", alpha=0.3)
        fig_bar.tight_layout()
        story.append(_fig_to_rl_image(fig_bar, width=6*inch, height=3.5*inch))
        story.append(Spacer(1, 0.2*inch))

        # Summary table
        story.append(Paragraph("2.1 Leave Summary Statistics", STYLES["SectionHead"]))
        summary_data = [
            ["Leave Type", "Total Days", "Avg/Employee", "Max", "Employees Using"],
            ["Sick Leave", f"{total_sick:,}", f"{df['sick_leave_days'].mean():.1f}",
             str(df["sick_leave_days"].max()),
             str((df["sick_leave_days"] > 0).sum())],
            ["Maternity Leave", f"{total_mat:,}", f"{df[df['sex']=='Female']['maternity_leave_days'].mean():.1f} (F)",
             str(df["maternity_leave_days"].max()),
             str((df["maternity_leave_days"] > 0).sum())],
            ["Paternity Leave", f"{total_pat:,}", f"{df[df['sex']=='Male']['paternity_leave_days'].mean():.1f} (M)",
             str(df["paternity_leave_days"].max()),
             str((df["paternity_leave_days"] > 0).sum())],
            ["PTO", f"{total_pto:,}", f"{df['paid_time_off_days'].mean():.1f}",
             str(df["paid_time_off_days"].max()),
             str((df["paid_time_off_days"] > 0).sum())],
            ["Mental Health", f"{total_mh:,}", f"{df['mental_health_leave_days'].mean():.1f}",
             str(df["mental_health_leave_days"].max()),
             str((df["mental_health_leave_days"] > 0).sum())],
        ]
        t = Table(summary_data, colWidths=[1.2*inch, 1*inch, 1.1*inch, 0.7*inch, 1.2*inch])
        t.setStyle(STD_TABLE_STYLE)
        story.append(t)
        story.append(PageBreak())

        # Monthly absence trend (image-based chart)
        story.append(Paragraph("3. Monthly Absence Trends", STYLES["SectionHead"]))
        months = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"]
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_avg = [df[f"absent_days_{m}"].mean() for m in months]
        monthly_total = [df[f"absent_days_{m}"].sum() for m in months]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
        ax1.plot(month_labels, monthly_avg, "o-", color="#1565c0",
                 linewidth=2, markersize=8)
        ax1.fill_between(month_labels, monthly_avg, alpha=0.15, color="#1565c0")
        ax1.set_ylabel("Average Absent Days / Employee")
        ax1.set_title("Average Absence per Employee by Month",
                      fontsize=13, fontweight="bold", pad=12)
        ax1.grid(alpha=0.3)

        ax2.bar(month_labels, monthly_total, color="#ff8f00", alpha=0.85)
        for i, v in enumerate(monthly_total):
            ax2.text(i, v + 20, str(int(v)), ha="center", fontsize=8)
        ax2.set_ylabel("Total Absent Days (All Employees)")
        ax2.set_title("Total Company Absence by Month",
                      fontsize=13, fontweight="bold", pad=12)
        ax2.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        story.append(_fig_to_rl_image(fig, width=6*inch, height=4.2*inch))

    # Policy text
    story.append(PageBreak())
    story.append(Paragraph("4. Leave Request Procedure", STYLES["SectionHead"]))
    story.append(Paragraph(textwrap.dedent("""\
        <b>Step 1 – Submit Request:</b> Log in to the NovaCorp HR Portal → Leave Management
        → New Request. Select the leave type, dates, and provide a reason if required.
        <br/><br/>
        <b>Step 2 – Manager Approval:</b> Your direct manager will receive a notification
        and must approve or decline within 3 business days. For leave longer than 5 days,
        HR Partner approval is also required.
        <br/><br/>
        <b>Step 3 – Confirmation:</b> Once approved, you will receive an email confirmation
        and a calendar event will be automatically created. Your out-of-office status will
        be updated in Slack and email.
        <br/><br/>
        <b>Step 4 – Return:</b> On your return, update the HR Portal to confirm your
        return date. If you return earlier than planned, adjust your leave dates accordingly.
        <br/><br/>
        <b>Emergency Leave:</b> In case of unforeseen circumstances, contact your manager
        by phone or Slack immediately. Submit the formal request within 48 hours of your
        return.
    """), STYLES["BodyJ"]))
    story.append(Paragraph(textwrap.dedent("""\
        Employees should submit planned leave as soon as dates are known so teams can arrange
        coverage. Managers review eligibility, accrued balance, and operational coverage in
        that order, then record an approval or a specific reason for denial. Requests longer
        than five consecutive workdays also go to the HR Partner; protected leave is handled
        under applicable local law when that law provides greater protection.
        <br/><br/>
        If a request is urgent, tell the manager by phone or the on-call channel and submit
        the portal form as soon as reasonably possible. A disagreement about an eligibility
        decision may be escalated to the People Partner, who should acknowledge it within
        two business days. Medical details are kept in the restricted leave file and are not
        shared with the wider team.
    """), STYLES["BodyJ"]))

    doc.build(story)
    print(f"  📄 {path}")
    _append_scanned_page(path, {
        "title": "2026 Executive Leave Policy Sign-Off & Bereavement Addendum",
        "paragraphs": [
            "Effective October 1, 2026. This signed addendum supersedes the earlier "
            "bereavement entitlement where the employee is eligible for the greater benefit.",
            "Employees receive 10 paid bereavement days following the death of an immediate "
            "family member and 5 paid days for an extended family member. Notify the manager "
            "as soon as practical; documentation is requested only when needed to administer leave.",
        ],
        "table": [
            ["Relationship", "Paid Days", "Approval Contact"],
            ["Immediate family", "10", "People Manager"],
            ["Extended family", "5", "People Manager"],
        ],
        "stamp": "SIGNED / APPROVED",
    })


def generate_scanned_documents() -> None:
    remote_path = os.path.join(DOCS_DIR, "scanned_remote_work_addendum_2026.pdf")
    _render_scanned_pdf(remote_path, [
        {
            "title": "Signed 2026 Executive Addendum — International Temporary Remote Work",
            "paragraphs": [
                "Effective October 1, 2026. Eligible employees may work temporarily from an "
                "approved country for a maximum of 30 calendar days per calendar year.",
                "Submit the request to the manager and People Operations at least 14 days "
                "prior to departure. Approval must be written and must cover the destination, "
                "dates, work duties, and data-handling plan.",
            ],
            "table": [
                ["Approved country", "Country", "Approval"],
                ["Canada", "Germany", "Required"],
                ["France", "Japan", "Required"],
                ["Netherlands", "Singapore", "Required"],
                ["United Kingdom", "Australia", "Required"],
            ],
            "stamp": "SIGNED / APPROVED",
        },
        {
            "title": "International Remote Work — Equipment and Compliance",
            "paragraphs": [
                "The employee remains subject to security, privacy, tax, payroll, and "
                "immigration controls for the destination. NovaCorp may shorten or revoke "
                "an approval if local law or client obligations require an earlier return.",
                "This addendum replaces the older 2025 ergonomic allowance: an employee "
                "with an approved international remote-work request may receive a $150 "
                "ergonomic chair top-up once per calendar year. Submit the receipt within "
                "30 days of purchase; no other furniture is covered by this top-up.",
            ],
            "table": [
                ["Requirement", "Rule"],
                ["Annual international limit", "30 calendar days"],
                ["Request lead time", "14 days before departure"],
                ["Chair top-up", "$150; supersedes 2025 allowance"],
            ],
            "stamp": "SIGNED / APPROVED",
        },
    ])

    audit_path = os.path.join(DOCS_DIR, "scanned_hr_audit_memo_2026.pdf")
    _render_scanned_pdf(audit_path, [{
        "title": "Confidential HR Compensation Audit Memo — Q4 2026",
        "paragraphs": [
            "The Q4 2026 merit pool is 4.5% of eligible base payroll. Business-unit "
            "allocations must be supported by calibrated performance records and reviewed "
            "for internal equity before an offer or adjustment is communicated.",
            "Any executive compensation adjustment at $180,000 or more requires written "
            "VP of People approval before final sign-off. Finance records the approved "
            "decision and the compensation team retains the audit trail in the restricted file.",
        ],
        "table": [
            ["Control", "Threshold", "Approval"],
            ["Q4 merit pool", "4.5%", "Compensation Committee"],
            ["Executive adjustment", "$180,000+", "VP of People"],
        ],
        "stamp": "CONFIDENTIAL — SIGNED / APPROVED",
    }])
    print(f"  📄 {remote_path}")
    print(f"  📄 {audit_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    df = _load_employees()
    if df is None:
        print("⚠️  data/employees.xlsx not found – using placeholder data for docs.")
    else:
        print(f"📊 Loaded {len(df)} employees for chart generation.")

    print("\nGenerating documents...")
    generate_employee_handbook(df)
    generate_onboarding_policy()
    generate_rto_policy()
    generate_salary_report(df)
    generate_benefits_guide()
    generate_leave_report(df)
    generate_scanned_documents()
    print("\n✅ Six policy PDFs and scanned addenda generated in data/documents/")


if __name__ == "__main__":
    main()
