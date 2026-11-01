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
import textwrap

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)

DOCS_DIR = os.path.join("data", "documents")
DATA_DIR = "data"
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
    path = os.path.join(DATA_DIR, "employees.xlsx")
    if os.path.exists(path):
        return pd.read_excel(path, engine="openpyxl")
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


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EMPLOYEE HANDBOOK
# ═══════════════════════════════════════════════════════════════════════════════

def generate_employee_handbook(df: pd.DataFrame | None):
    path = os.path.join(DOCS_DIR, "employee_handbook.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
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

    doc.build(story)
    print(f"  📄 {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ONBOARDING POLICY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_onboarding_policy(df: pd.DataFrame | None):
    path = os.path.join(DOCS_DIR, "onboarding_policy.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
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

    doc.build(story)
    print(f"  📄 {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RETURN-TO-OFFICE POLICY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_rto_policy(df: pd.DataFrame | None):
    path = os.path.join(DOCS_DIR, "return_to_office_policy.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
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

    # Office capacity chart (image-based → OCR needed)
    fig, ax = plt.subplots(figsize=(8, 4))
    months_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    capacity = [45, 52, 58, 63, 68, 72, 65, 60, 75, 78, 80, 70]
    target = [75] * 12
    ax.bar(months_labels, capacity, color="#ff8f00", alpha=0.8, label="Actual Utilisation %")
    ax.plot(months_labels, target, "r--", linewidth=2, label="Target (75%)")
    ax.set_ylabel("Office Utilisation (%)")
    ax.set_title("Monthly Office Utilisation Rate – 2025", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    story.append(Paragraph("2.1 Office Utilisation Trends", STYLES["SectionHead"]))
    story.append(_fig_to_rl_image(fig, width=6*inch, height=3*inch))
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

    doc.build(story)
    print(f"  📄 {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SALARY BANDS REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_salary_report(df: pd.DataFrame | None):
    path = os.path.join(DOCS_DIR, "salary_bands_report.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []
    _title_page(story, "Salary Bands &\nCompensation Report")

    if df is None:
        story.append(Paragraph("Data not available. Run generate_employees.py first.",
                               STYLES["BodyJ"]))
        doc.build(story)
        return

    # Avg salary by department bar chart
    story.append(Paragraph("1. Average Salary by Department", STYLES["SectionHead"]))
    avg_sal = df.groupby("department")["salary"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(avg_sal.index, avg_sal.values, color=plt.cm.viridis(
        np.linspace(0.2, 0.9, len(avg_sal))))
    ax.set_xlabel("Average Salary ($)")
    ax.set_title("Average Salary by Department", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, avg_sal.values):
        ax.text(val + 500, bar.get_y() + bar.get_height()/2,
                f"${val:,.0f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    story.append(_fig_to_rl_image(fig, width=6*inch, height=3.5*inch))
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
    ax.set_title("Company-Wide Salary Distribution", fontsize=14, fontweight="bold")
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
        ["Junior (0-1 yrs)", "$35,000", "$55,000", "$45,000",
         str(len(df[df["years_of_expertise"] < 2]))],
        ["Mid (2-4 yrs)", "$55,000", "$85,000", "$70,000",
         str(len(df[(df["years_of_expertise"] >= 2) & (df["years_of_expertise"] < 5)]))],
        ["Senior (5-9 yrs)", "$85,000", "$120,000", "$102,500",
         str(len(df[(df["years_of_expertise"] >= 5) & (df["years_of_expertise"] < 10)]))],
        ["Lead/Manager (10-14 yrs)", "$100,000", "$160,000", "$130,000",
         str(len(df[(df["years_of_expertise"] >= 10) & (df["years_of_expertise"] < 15)]))],
        ["Director/VP (15+ yrs)", "$140,000", "$220,000", "$180,000",
         str(len(df[df["years_of_expertise"] >= 15]))],
    ]
    t = Table(band_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.1*inch, 0.9*inch])
    t.setStyle(STD_TABLE_STYLE)
    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    # Salary by dept+seniority heatmap (complex chart → OCR)
    story.append(Paragraph("4. Salary Heatmap: Department × Experience",
                           STYLES["SectionHead"]))
    bins = [0, 2, 5, 10, 15, 31]
    labels = ["0-1y", "2-4y", "5-9y", "10-14y", "15y+"]
    df["exp_bin"] = pd.cut(df["years_of_expertise"], bins=bins, labels=labels, right=False)
    pivot = df.pivot_table(values="salary", index="department",
                           columns="exp_bin", aggfunc="mean")
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
                 fontsize=13, fontweight="bold")
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

def generate_benefits_guide(df: pd.DataFrame | None):
    path = os.path.join(DOCS_DIR, "benefits_guide.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
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
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    story.append(_fig_to_rl_image(fig, width=6*inch, height=3.3*inch))

    doc.build(story)
    print(f"  📄 {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LEAVE POLICY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_leave_report(df: pd.DataFrame | None):
    path = os.path.join(DOCS_DIR, "leave_policy_report.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
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

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # Pie chart
        leave_labels = ["Sick Leave", "Maternity", "Paternity", "PTO", "Mental Health"]
        leave_vals = [total_sick, total_mat, total_pat, total_pto, total_mh]
        pie_colors = ["#ef5350", "#ab47bc", "#42a5f5", "#66bb6a", "#ffa726"]
        wedges, texts, autotexts = ax1.pie(
            leave_vals, labels=leave_labels, autopct="%1.1f%%",
            colors=pie_colors, startangle=90,
        )
        for t in texts + autotexts:
            t.set_fontsize(8)
        ax1.set_title("Leave Distribution by Type", fontsize=12, fontweight="bold")

        # Bar chart: avg leave by department
        dept_leave = df.groupby("department").agg({
            "sick_leave_days": "mean",
            "paid_time_off_days": "mean",
            "mental_health_leave_days": "mean",
        }).sort_values("sick_leave_days", ascending=True)
        x_pos = np.arange(len(dept_leave))
        w = 0.25
        ax2.barh(x_pos - w, dept_leave["sick_leave_days"], w,
                 label="Sick", color="#ef5350")
        ax2.barh(x_pos, dept_leave["paid_time_off_days"], w,
                 label="PTO", color="#66bb6a")
        ax2.barh(x_pos + w, dept_leave["mental_health_leave_days"], w,
                 label="Mental Health", color="#ffa726")
        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(dept_leave.index, fontsize=8)
        ax2.set_xlabel("Average Days")
        ax2.set_title("Avg Leave by Department", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        story.append(_fig_to_rl_image(fig, width=6.5*inch, height=3.2*inch))
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
                      fontsize=13, fontweight="bold")
        ax1.grid(alpha=0.3)

        ax2.bar(month_labels, monthly_total, color="#ff8f00", alpha=0.85)
        for i, v in enumerate(monthly_total):
            ax2.text(i, v + 20, str(int(v)), ha="center", fontsize=8)
        ax2.set_ylabel("Total Absent Days (All Employees)")
        ax2.set_title("Total Company Absence by Month",
                      fontsize=13, fontweight="bold")
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

    doc.build(story)
    print(f"  📄 {path}")


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
    generate_onboarding_policy(df)
    generate_rto_policy(df)
    generate_salary_report(df)
    generate_benefits_guide(df)
    generate_leave_report(df)
    print("\n✅ All 6 documents generated in data/documents/")


if __name__ == "__main__":
    main()
