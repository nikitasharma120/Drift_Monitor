"""
create_presentation.py — Generates a formal, professional 16:9 widescreen
PowerPoint slide deck (Drift_Monitor_Presentation.pptx) for the DS635 MLSE viva defense.

Team Members:
  - Sanjana Nathani (Slides 1–3)
  - Nikita Sharma (Slides 4–6)
  - Anish Karmakar (Slides 7–9)
  - Srishti Lamba (Slides 10–13)
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------------------
# Colors & Typography Constants (Academic High-Contrast Theme)
# ---------------------------------------------------------------------------
C_BG = RGBColor(255, 255, 255)            # Pure White
C_CARD_BG = RGBColor(248, 250, 252)       # Slate 50
C_CARD_BORDER = RGBColor(226, 232, 240)   # Slate 200
C_TEXT_DARK = RGBColor(15, 23, 42)        # Slate 900
C_TEXT_MUTED = RGBColor(71, 85, 105)      # Slate 600
C_PRIMARY = RGBColor(29, 78, 216)         # Blue 700
C_PRIMARY_LIGHT = RGBColor(239, 246, 255) # Blue 50
C_EMERALD = RGBColor(16, 149, 93)         # Emerald 700
C_EMERALD_BG = RGBColor(236, 253, 245)    # Emerald 50
C_CORAL = RGBColor(190, 24, 93)           # Rose 700
C_CORAL_BG = RGBColor(255, 241, 242)      # Rose 50
C_AMBER = RGBColor(180, 83, 9)            # Amber 700
C_AMBER_BG = RGBColor(254, 243, 199)      # Amber 50
FONT_HEADING = "Arial"
FONT_BODY = "Arial"


def add_header(slide, category: str, title: str, speaker: str, time_window: str):
    """Draws a standardized academic slide header with breadcrumb and speaker badge."""
    # Category breadcrumb
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8.5), Inches(0.35))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category.upper()
    p_c.font.name = FONT_HEADING
    p_c.font.size = Pt(11)
    p_c.font.bold = True
    p_c.font.color.rgb = C_PRIMARY

    # Main Assertion Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(8.8), Inches(0.8))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title
    p_t.font.name = FONT_HEADING
    p_t.font.size = Pt(21)
    p_t.font.bold = True
    p_t.font.color.rgb = C_TEXT_DARK

    # Speaker Pill Badge (Top Right)
    badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.8), Inches(0.45), Inches(2.75), Inches(0.55))
    badge.fill.solid()
    badge.fill.fore_color.rgb = C_PRIMARY_LIGHT
    badge.line.color.rgb = C_CARD_BORDER
    tf_b = badge.text_frame
    tf_b.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_b = tf_b.paragraphs[0]
    p_b.alignment = PP_ALIGN.CENTER
    p_b.text = f"{speaker}\n{time_window}"
    p_b.font.name = FONT_BODY
    p_b.font.size = Pt(10)
    p_b.font.bold = True
    p_b.font.color.rgb = C_PRIMARY


def add_footer(slide, current_slide: int, total_slides: int = 13):
    """Adds formal academic footer bar with course title and slide counter."""
    footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(8.0), Inches(0.3))
    tf = footer_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Drift Monitor — DS635 Machine Learning System Engineering (MLSE) Term Project"
    p.font.name = FONT_BODY
    p.font.size = Pt(9)
    p.font.color.rgb = C_TEXT_MUTED

    counter_box = slide.shapes.add_textbox(Inches(11.0), Inches(7.0), Inches(1.5), Inches(0.3))
    tf_c = counter_box.text_frame
    p_c = tf_c.paragraphs[0]
    p_c.alignment = PP_ALIGN.RIGHT
    p_c.text = f"Slide {current_slide} of {total_slides}"
    p_c.font.name = FONT_BODY
    p_c.font.size = Pt(9)
    p_c.font.bold = True
    p_c.font.color.rgb = C_TEXT_MUTED


def add_card(slide, left, top, width, height, fill_color=C_CARD_BG, border_color=C_CARD_BORDER):
    """Adds a structured background card container."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1)
    return shape


def add_speaker_notes(slide, notes_text: str):
    """Attaches verbatim speaker notes visible in PowerPoint Presenter View."""
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = notes_text


def build_presentation():
    prs = pptx.Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide
    # Speaker: Sanjana Nathani (00:00 – 00:45)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    # Background accent band
    band = add_card(s1, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.8), C_PRIMARY_LIGHT, C_CARD_BORDER)
    
    tb = s1.shapes.add_textbox(Inches(1.2), Inches(1.3), Inches(11.0), Inches(1.8))
    tf = tb.text_frame
    p0 = tf.paragraphs[0]
    p0.text = "DRIFT MONITOR"
    p0.font.name = FONT_HEADING
    p0.font.size = Pt(38)
    p0.font.bold = True
    p0.font.color.rgb = C_PRIMARY

    p1 = tf.add_paragraph()
    p1.text = "Detecting Silent Distribution Shift in Deployed Machine Learning Models"
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(20)
    p1.font.color.rgb = C_TEXT_DARK
    p1.space_before = Pt(8)

    # Team Box
    tb_team = s1.shapes.add_textbox(Inches(1.2), Inches(3.3), Inches(10.8), Inches(1.2))
    tf_team = tb_team.text_frame
    pt1 = tf_team.paragraphs[0]
    pt1.text = "TERM PROJECT TEAM & ROLES:"
    pt1.font.size = Pt(11)
    pt1.font.bold = True
    pt1.font.color.rgb = C_TEXT_MUTED

    pt2 = tf_team.add_paragraph()
    pt2.text = "• Sanjana Nathani: Problem Context, Distribution Shift Taxonomy & Theoretical Visibility (Slides 1–3)\n" \
               "• Nikita Sharma: Experimental Controls, 3 Benchmark Datasets, Injections & Baselines (Slides 4–6)\n" \
               "• Anish Karmakar: Master Recommendation Matrix & The 4 Quantitative Trade-Off Curves (Slides 7–9)\n" \
               "• Srishti Lamba: Deep Mechanics, Mathematical Proofs, Live Reproduction & Limitations (Slides 10–13)"
    pt2.font.size = Pt(12)
    pt2.font.color.rgb = C_TEXT_DARK
    pt2.line_spacing = 1.3
    pt2.space_before = Pt(4)

    # Metadata tag
    tb_meta = s1.shapes.add_textbox(Inches(1.2), Inches(5.3), Inches(10.8), Inches(0.8))
    tf_meta = tb_meta.text_frame
    pm = tf_meta.paragraphs[0]
    pm.text = "DS635 Machine Learning System Engineering (MLSE) | Viva Defense Presentation (15 Min + 5 Min Q&A)\n" \
              "Repository: nikitasharma120/Drift_Monitor (Verified on branch: Srishti)"
    pm.font.size = Pt(11)
    pm.font.color.rgb = C_TEXT_MUTED

    add_speaker_notes(s1,
        "Good morning, Professor. Today, our team—Nikita, Anish, Srishti, and myself, Sanjana—presents 'Drift Monitor: "
        "Detecting Silent Distribution Shift in Deployed Machine Learning Models.'\n\n"
        "In traditional software systems, failures announce themselves loudly through unhandled exceptions, crash logs, or "
        "HTTP 500 errors. In production ML, failure is silent. A model continues returning valid, confident predictions while "
        "the world underneath it changes.\n\n"
        "Over the next 14 minutes, we will present our experimental setup across three real-world datasets, our four quantitative "
        "trade-off curves, the exact mathematical mechanics of why standard detectors fail, and demonstrate a single-command "
        "reproduction of our entire pipeline."
    )

    # =========================================================================
    # SLIDE 2: Problem Context — The Silent Degradation Dilemma
    # Speaker: Sanjana Nathani (00:45 – 02:00)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Context & Problem Statement", "The Silent Degradation Dilemma: Why Standard Ops Telemetry Fails", "Sanjana Nathani", "00:45 – 02:00")
    add_footer(s2, 2)

    # Left card: Traditional Ops (Red outline)
    add_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(4.5), C_CORAL_BG, C_CORAL)
    tb_l = s2.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.0))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    pl0 = tf_l.paragraphs[0]
    pl0.text = "Traditional DevOps Telemetry (Blind)"
    pl0.font.size = Pt(16)
    pl0.font.bold = True
    pl0.font.color.rgb = C_CORAL

    pl_items = [
        ("HTTP 200 OK Status", "Microservices respond with valid payloads; zero unhandled exceptions."),
        ("Low Query Latency", "Model executes within SLA (<15 ms per request)."),
        ("Healthy Infrastructure", "CPU, memory, and GPU utilization remain normal (<40%)."),
        ("Silent Failure Mode", "A credit scoring model predicts 0.05 default probability for bankrupt applicants because economic rules shifted."),
        ("Reactive Discovery", "Failure is only caught months later through revenue collapse, manual audits, or loan defaults.")
    ]
    for h, b in pl_items:
        p = tf_l.add_paragraph()
        p.text = f"• {h}: {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(8)

    # Right card: Drift Monitor Solution (Emerald outline)
    add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(4.5), C_EMERALD_BG, C_EMERALD)
    tb_r = s2.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.0))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    pr0 = tf_r.paragraphs[0]
    pr0.text = "Drift Monitor: Proactive Statistical Observability"
    pr0.font.size = Pt(16)
    pr0.font.bold = True
    pr0.font.color.rgb = C_EMERALD

    pr_items = [
        ("Input Distribution Tracking", "Monitors P(X) feature histograms via cached Population Stability Index (PSI)."),
        ("Model Output Tracking", "Tracks model predicted probabilities P(f(X)) to catch label shift without waiting for labels."),
        ("Supervisory Trickle", "Tracks Brier score degradation on an asynchronous 20% labeled trickle to detect concept drift."),
        ("Zero Alert Fatigue", "Debounces streaming alerts (k=2 consecutive batches) to ensure 0.0% pre-onset false alarms."),
        ("Lightweight & Deployable", "Runs in <60 ms per batch on standard CPUs without heavy observability infrastructure.")
    ]
    for h, b in pr_items:
        p = tf_r.add_paragraph()
        p.text = f"• {h}: {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(8)

    add_speaker_notes(s2,
        "The core problem in production machine learning is that operational monitoring tools are structurally blind to "
        "statistical degradation. If an applicant pool for credit scoring changes, the scoring service still responds in "
        "15 milliseconds with HTTP 200 OK and returns a float between 0 and 1. Nothing in standard infrastructure turns red.\n\n"
        "Most organizations only discover model degradation reactively: through an unexpected quarterly revenue drop, a wave "
        "of bad loans, or compliance audits months down the road.\n\n"
        "Our goal in Drift Monitor is to build a proactive, statistically-grounded monitoring layer that tracks streaming "
        "batches against a fixed reference distribution, detecting silent shift before downstream business damage occurs."
    )

    # =========================================================================
    # SLIDE 3: Theoretical Foundations — Factorization & The Input-Blindness Principle
    # Speaker: Sanjana Nathani (02:00 – 03:30)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Theoretical Foundations", "Taxonomy: Input Monitors Are Structurally Blind to Concept Drift", "Sanjana Nathani", "02:00 – 03:30")
    add_footer(s3, 3)

    # Dual factorization banner
    add_card(s3, Inches(0.8), Inches(1.6), Inches(11.733), Inches(0.85), C_PRIMARY_LIGHT, C_PRIMARY)
    tb_eq = s3.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(11.3), Inches(0.7))
    tf_eq = tb_eq.text_frame
    p_eq = tf_eq.paragraphs[0]
    p_eq.alignment = PP_ALIGN.CENTER
    p_eq.text = "Joint Data-Generating Factorization:   P(X, Y)  =  P(X) · P(Y | X)  =  P(Y) · P(X | Y)"
    p_eq.font.size = Pt(17)
    p_eq.font.bold = True
    p_eq.font.color.rgb = C_PRIMARY

    # 3 Cards for the 3 shift types
    col_w = Inches(3.7)
    gap = Inches(0.3)
    
    # Covariate Shift Card
    add_card(s3, Inches(0.8), Inches(2.7), col_w, Inches(4.0))
    tb_c1 = s3.shapes.add_textbox(Inches(0.95), Inches(2.85), col_w - Inches(0.3), Inches(3.7))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    p.text = "1. Covariate Shift"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    
    c1_txt = [
        "Definition: P(X) changes, while P(Y|X) remains invariant.",
        "Real Example: Loan applicant pool skews younger, or market expansion introduces lower-income demographics.",
        "Input Monitor Visibility: DIRECTLY VISIBLE. Single-feature PSI and KL divergence detect it easily on raw features X.",
        "Invariant: The underlying decision boundary linking debt to risk holds true."
    ]
    for t in c1_txt:
        p = tf_c1.add_paragraph()
        p.text = t
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(8)

    # Label Shift Card
    add_card(s3, Inches(0.8) + col_w + gap, Inches(2.7), col_w, Inches(4.0))
    tb_c2 = s3.shapes.add_textbox(Inches(0.95) + col_w + gap, Inches(2.85), col_w - Inches(0.3), Inches(3.7))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True
    p = tf_c2.paragraphs[0]
    p.text = "2. Label Shift"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_AMBER

    c2_txt = [
        "Definition: P(Y) changes, while P(X|Y) remains invariant.",
        "Real Example: Macroeconomic recession doubles overall default rates from 7% to 15%, but defaulting borrowers look identical.",
        "Input Monitor Visibility: INDIRECT. May not move any single input feature's histogram sufficiently to breach threshold.",
        "Solution: Watched via reference model output distribution P(f(X)) using pred_prop_psi."
    ]
    for t in c2_txt:
        p = tf_c2.add_paragraph()
        p.text = t
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(8)

    # Concept Drift Card
    add_card(s3, Inches(0.8) + (col_w + gap)*2, Inches(2.7), col_w, Inches(4.0), C_CORAL_BG, C_CORAL)
    tb_c3 = s3.shapes.add_textbox(Inches(0.95) + (col_w + gap)*2, Inches(2.85), col_w - Inches(0.3), Inches(3.7))
    tf_c3 = tb_c3.text_frame
    tf_c3.word_wrap = True
    p = tf_c3.paragraphs[0]
    p.text = "3. Concept Drift"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CORAL

    c3_txt = [
        "Definition: P(Y|X) changes, while P(X) can remain identical.",
        "Real Example: Regulatory overhaul changes what counts as high risk; existing borrowers with debt ratio 0.4 now default.",
        "Input Monitor Visibility: STRUCTURALLY INVISIBLE. Inputs X can be identical while predictions are totally wrong.",
        "Load-Bearing Axiom: Ground-truth performance tracking (concept_signal) is mathematically required."
    ]
    for t in c3_txt:
        p = tf_c3.add_paragraph()
        p.text = t
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(8)

    add_speaker_notes(s3,
        "The joint distribution between inputs X and targets Y factorizes in two dual forms. This gives us three fundamental "
        "failure modes:\n\n"
        "1. Covariate shift: P(X) moves, but P(Y|X) is constant. Directly visible in input data.\n"
        "2. Label shift: P(Y) moves, but class conditionals P(X|Y) are constant. Input histograms may barely shift, but model "
        "output probabilities move.\n"
        "3. Concept drift: The relationship P(Y|X) changes. What used to mean 'safe' now means 'risky', even if input feature "
        "histograms remain identical.\n\n"
        "This third row establishes the load-bearing axiom of our project: Any detector restricted solely to input feature vectors "
        "X is mathematically incapable of detecting concept drift. To catch all three modes, an architecture must monitor inputs, "
        "model outputs, and supervisory performance.\n\n"
        "Now, I hand over to Nikita to describe our experimental setup and detector suite."
    )

    # =========================================================================
    # SLIDE 4: Experimental Controls & Setup
    # Speaker: Nikita Sharma (03:30 – 04:45)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Experimental Rigour (35%)", "Experimental Control: 1,080 Runs Across 3 Diverse Datasets", "Nikita Sharma", "03:30 – 04:45")
    add_footer(s4, 4)

    # 3 Dataset summary boxes
    d_w = Inches(3.7)
    d_gap = Inches(0.3)
    
    datasets = [
        ("Give Me Some Credit", "Credit Risk (Kaggle)", "120,269 clean rows", "10 numeric features", "6.9% base default rate", "Target: DebtRatio (outliers > 130 std)"),
        ("Taiwan Credit Default", "Banking & Finance (UCI)", "30,000 clean rows", "23 continuous features", "22.1% default rate", "Target: BILL_AMT1 (credit balance)"),
        ("Adult Census Income", "Demographics & Census", "48,842 clean rows", "6 continuous features", "23.9% high income (>50k)", "Target: hours-per-week (IQR=5 on [1,99])")
    ]
    for i, (name, domain, n_rows, n_feats, b_rate, target) in enumerate(datasets):
        x = Inches(0.8) + i * (d_w + d_gap)
        add_card(s4, x, Inches(1.6), d_w, Inches(2.2))
        tb_d = s4.shapes.add_textbox(x + Inches(0.15), Inches(1.7), d_w - Inches(0.3), Inches(2.0))
        tf_d = tb_d.text_frame
        p = tf_d.paragraphs[0]
        p.text = name
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = C_PRIMARY

        d_bullets = [f"Domain: {domain}", f"Size: {n_rows} | {n_feats}", f"Base Positive Rate: {b_rate}", f"Shift Feature: {target}"]
        for b in d_bullets:
            p = tf_d.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(10)
            p.font.color.rgb = C_TEXT_DARK
            p.space_before = Pt(3)

    # Controlled variables and grid setup card
    add_card(s4, Inches(0.8), Inches(4.1), Inches(11.733), Inches(2.7), C_CARD_BG, C_PRIMARY)
    tb_g = s4.shapes.add_textbox(Inches(1.0), Inches(4.25), Inches(11.3), Inches(2.4))
    tf_g = tb_g.text_frame
    tf_g.word_wrap = True
    p = tf_g.paragraphs[0]
    p.text = "Strict Experimental Controls & Measurement Protocol"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    g_points = [
        ("Disjoint Stream Partitioning", "Batches 0–9 form the fixed historical reference distribution. Batches 10–29 form the streaming evaluation window. Rows are strictly partitioned from a held-out pool to prevent data leakage across Monte Carlo runs."),
        ("Full Grid Multiplication", "3 Shift Types (Covariate, Label, Concept) × 3 Severities (Mild, Moderate, Severe) × 2 Onset Speeds (Abrupt, Gradual) × 20 Repetitions = 360 runs per dataset × 3 datasets = 1,080 test runs, 5,400 metric-level evaluations."),
        ("Asymmetric Error Bars (Wilson Score CIs)", "Because detection sensitivities are binomial proportions near 1.0, standard Wald normal intervals fail (producing zero width or bounds > 1.0). Exact 95% Wilson score intervals are computed for every metric across replicates."),
        ("PRNG Determinism", "Every repetition seed is generated via CRC32 hashing: seed = 1000 * rep + crc32(tag) % 1000, ensuring byte-identical replication across hardware architectures.")
    ]
    for h, b in g_points:
        p = tf_g.add_paragraph()
        p.text = f"• {h}: {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(4)

    add_speaker_notes(s4,
        "Thank you, Sanjana. To test our detectors scientifically, we evaluated them across three real-world tabular datasets "
        "spanning credit risk, credit default, and demographic census income.\n\n"
        "We enforced strict experimental controls:\n"
        "First, each stream is divided into 30 sequential batches. Batches 0 to 9 form the fixed historical reference distribution. "
        "Shift is introduced starting at Batch 10. To guarantee zero data leakage, historical reference rows are drawn from a strictly "
        "held-out partition separate from the streaming pool.\n\n"
        "Second, to eliminate anecdotal noise, we run 20 independent Monte Carlo repetitions per cell with deterministic PRNG seeding. "
        "That gives 360 runs per dataset, totaling 1,080 test runs and 5,400 metric-level evaluations.\n\n"
        "Third, because detection sensitivities are binomial proportions near 1.0, standard Wald normal intervals collapse or produce "
        "impossible numbers. We report exact 95% Wilson-score confidence intervals throughout."
    )

    # =========================================================================
    # SLIDE 5: The Four Baselines & Five Evaluated Detector Mechanisms
    # Speaker: Nikita Sharma (04:45 – 06:00)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Detector Suite & Baselines", "The 4 Baselines & 5 Evaluated Detection Mechanisms", "Nikita Sharma", "04:45 – 06:00")
    add_footer(s5, 5)

    # Table of detectors
    rows_data = [
        ("Baseline 1: Standard Ops", "None (Blind)", "CPU, Memory, Query Latency, HTTP 200", "Zero statistical visibility into data shift"),
        ("Baseline 2: Feature PSI", "Unsupervised", "Per-feature quantile histogram divergence (tau=0.10)", "Detects Covariate Shift on input features X"),
        ("Baseline 2: Feature KL", "Unsupervised", "One-directional relative entropy vs reference bins (tau=0.10)", "Detects Covariate Shift on input features X"),
        ("Baseline 3: PCA Centroid", "Unsupervised", "3D PCA centroid displacement (Winsorized [1%, 99%], tau=1.0)", "Detects joint multi-feature movement in subspace"),
        ("Proposed: pred_prop_psi", "Output-aware", "PSI on reference model predicted probabilities (tau=0.015)", "Detects Label Shift & High-Weight Covariate Shift"),
        ("Proposed: concept_signal", "Supervisory", "Brier score degradation on 20% labeled trickle (tau=0.025)", "Detects Concept Drift & confirms accuracy degradation")
    ]
    
    # Create Table shape
    t_top = Inches(1.6)
    t_left = Inches(0.8)
    t_width = Inches(11.733)
    t_height = Inches(3.6)
    table_shape = s5.shapes.add_table(len(rows_data) + 1, 4, t_left, t_top, t_width, t_height)
    tbl = table_shape.table
    tbl.columns[0].width = Inches(2.6)
    tbl.columns[1].width = Inches(1.5)
    tbl.columns[2].width = Inches(4.6)
    tbl.columns[3].width = Inches(3.033)

    headers = ["Detector / Baseline", "Paradigm", "Operational Formulation & Threshold", "Target Failure Mode"]
    for col_idx, h_text in enumerate(headers):
        cell = tbl.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.font.name = FONT_HEADING
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, r_vals in enumerate(rows_data):
        for col_idx, val in enumerate(r_vals):
            cell = tbl.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_CARD_BG if row_idx % 2 == 0 else RGBColor(255, 255, 255)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_BODY
            p.font.size = Pt(10)
            p.font.color.rgb = C_TEXT_DARK

    # Bottom Callout: The Winsorizer
    add_card(s5, Inches(0.8), Inches(5.4), Inches(11.733), Inches(1.4), C_EMERALD_BG, C_EMERALD)
    tb_win = s5.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(11.3), Inches(1.2))
    tf_win = tb_win.text_frame
    tf_win.word_wrap = True
    p = tf_win.paragraphs[0]
    p.text = "Key Engineering Fix: The Non-Parametric Winsorizer"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD

    p_w = tf_win.add_paragraph()
    p_w.text = "Extreme outliers (such as DebtRatio exceeding 130 standard deviations) severely distorted the covariance matrix " \
               "in PCA embedding drift, causing the first principal component to align almost entirely with the outlier noise. " \
               "By wrapping inputs with a non-parametric Winsorizer clipping values to reference [1%, 99%] quantiles, the subspace " \
               "was fully stabilized across all operating systems."
    p_w.font.size = Pt(10.5)
    p_w.font.color.rgb = C_TEXT_DARK
    p_w.space_before = Pt(3)

    add_speaker_notes(s5,
        "We evaluated five detector mechanisms across four distinct baselines:\n\n"
        "Baseline 1 is standard ops monitoring—which has 0% detection capability for statistical shift.\n"
        "Baseline 2 monitors input features directly via cached PSI and KL divergence across 10 quantile bins.\n"
        "Baseline 3 is PCA embedding drift, projecting inputs into 3 principal components and tracking centroid distance. "
        "Crucially, extreme outliers like DebtRatio—which reaches 130 standard deviations—distorted PCA covariance. We fixed "
        "this by introducing non-parametric reference Winsorization.\n\n"
        "To catch label shift without waiting for delayed labels, we introduced pred_prop_psi, which watches the output "
        "distribution of a frozen reference model.\n\n"
        "Finally, to catch concept drift, we introduced concept_signal, which tracks Brier score degradation on an asynchronous "
        "20% labeled trickle."
    )

    # =========================================================================
    # SLIDE 6: Shift Injection Protocols & Debouncing Logic
    # Speaker: Nikita Sharma (06:00 – 07:00)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Experimental Rigour (35%)", "Calibrated Injections, Debouncing & The Lag Priming Paradox", "Nikita Sharma", "06:00 – 07:00")
    add_footer(s6, 6)

    # Left card: Outlier-Robust IQR Scaling
    add_card(s6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(4.8))
    tb_iqr = s6.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.3))
    tf_iqr = tb_iqr.text_frame
    tf_iqr.word_wrap = True
    p = tf_iqr.paragraphs[0]
    p.text = "1. Outlier-Robust IQR Scaling"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY

    iqr_points = [
        ("The Standard Deviation Flaw", "Scaling shift by standard deviation fails when datasets contain outliers—one outlier inflates std-dev 100x, turning a 'mild' shift unrealistically massive."),
        ("The Normalized IQR Solution", "We scale shift magnitudes using normalized Interquartile Range: delta = severity * (IQR / 1.349)."),
        ("Calibrated Severities", "Mild (s=0.5), Moderate (s=1.0), Severe (s=2.0). Fallback to standard deviation is enforced only if IQR=0."),
        ("Onset Profiles", "Evaluated under Abrupt Step (instant shift at t=10) and Gradual Ramp (linear transition over 6 batches).")
    ]
    for h, b in iqr_points:
        p = tf_iqr.add_paragraph()
        p.text = f"• {h}: {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(8)

    # Right card: Debouncing & The 0.95 Lag Priming Paradox
    add_card(s6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(4.8), C_AMBER_BG, C_AMBER)
    tb_db = s6.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.3))
    tf_db = tb_db.text_frame
    tf_db.word_wrap = True
    p = tf_db.paragraphs[0]
    p.text = "2. Debouncing & The Lag Priming Fix"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_AMBER

    db_points = [
        ("Debounce Rule (k=2)", "To suppress single-batch streaming noise, an alert requires k=2 consecutive batches above threshold before officially triggering."),
        ("Pre-Onset Determinism Fix", "The original label-shift injector ran bootstrap sampling on all batches, introducing sampling noise that inflated Taiwan false alarms to 35%. Making pre-onset batches deterministic dropped false alarms to 0.0%."),
        ("The 0.95 Lag Priming Paradox", "An isolated pre-onset fluctuation at batch 9 previously primed the streak counter to 1. When true drift arrived at batch 10, the detector fired immediately with an apparent lag of 0.0."),
        ("Streak Counter Reset", "Resetting streak counters to 0 exactly at onset batch eliminated this artifact, ensuring detection lag measures true sustained post-onset response time.")
    ]
    for h, b in db_points:
        p = tf_db.add_paragraph()
        p.text = f"• {h}: {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(8)

    add_speaker_notes(s6,
        "Shift injection must be controlled and realistic. Standard-deviation scaling fails when datasets contain outliers—one "
        "outlier inflates standard deviation a hundredfold, rendering a 'mild' shift massive. We scale injection magnitudes "
        "using normalized Interquartile Range: IQR / 1.349.\n\n"
        "In production, single-batch noise causes false alarms. We enforce a debouncing window requiring k=2 consecutive "
        "threshold crossings before sounding an alarm.\n\n"
        "Two critical bugs we caught and fixed:\n"
        "First, earlier implementations bootstrapped pre-onset batches, introducing sampling noise that caused 30% false alarms "
        "on Taiwan credit. Keeping pre-onset batches deterministic dropped false alarms to 0.0%.\n"
        "Second, we resolved the '0.95 lag paradox' where a pre-onset noise spike primed the debounce counter right before onset. "
        "Resetting the counter at onset ensures lag measures true post-onset response time.\n\n"
        "Now, Anish will present our empirical results and the four trade-off curves."
    )

    # =========================================================================
    # SLIDE 7: Master Empirical Recommendation Matrix Across All 3 Datasets
    # Speaker: Anish Karmakar (07:00 – 08:15)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Results & Insight (20%)", "Master Results: 0.0% False Alarms Across Best Detectors on All Datasets", "Anish Karmakar", "07:00 – 08:15")
    add_footer(s7, 7)

    # Big Results Table
    t_top = Inches(1.6)
    t_left = Inches(0.8)
    t_width = Inches(11.733)
    t_height = Inches(3.4)
    tbl_s7 = s7.shapes.add_table(4, 4, t_left, t_top, t_width, t_height).table
    tbl_s7.columns[0].width = Inches(2.5)
    tbl_s7.columns[1].width = Inches(3.077)
    tbl_s7.columns[2].width = Inches(3.077)
    tbl_s7.columns[3].width = Inches(3.077)

    s7_headers = ["Shift Failure Mode", "Give Me Some Credit", "Taiwan Credit Default", "Adult Census Income"]
    for c_idx, h in enumerate(s7_headers):
        cell = tbl_s7.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_HEADING
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)

    s7_rows = [
        ("Covariate Shift\nP(X) changes", "kl_agg / psi_agg\nSensitivity: 1.00 [0.82, 1.00]\nFalse Alarm Rate: 0.0%", "kl_agg / psi_agg\nSensitivity: 1.00 [0.82, 1.00]\nFalse Alarm Rate: 0.0%", "pred_prop_psi\nSensitivity: 0.83 [0.61, 0.94]\nFalse Alarm Rate: 0.0%"),
        ("Label Shift\nP(Y) changes", "concept_signal / pred_prop\nSensitivity: 1.00 [0.82, 1.00]\nFalse Alarm Rate: 0.0%", "pred_prop_psi\nSensitivity: 1.00 [0.82, 1.00]\nFalse Alarm Rate: 0.0%", "pred_prop_psi\nSensitivity: 1.00 [0.82, 1.00]\nFalse Alarm Rate: 0.0%"),
        ("Concept Drift\nP(Y|X) changes", "concept_signal\nSensitivity: 0.89 [0.67, 0.97]\nFalse Alarm Rate: 0.0%", "pred_prop_psi / concept_signal\nSensitivity: 1.00 [0.82, 1.00]\nFalse Alarm Rate: 0.0%", "concept_signal\nSensitivity: 1.00 [0.82, 1.00]\nFalse Alarm Rate: 0.0%")
    ]
    for r_idx, r_vals in enumerate(s7_rows):
        for c_idx, val in enumerate(r_vals):
            cell = tbl_s7.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_CARD_BG if r_idx % 2 == 0 else RGBColor(255, 255, 255)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_BODY
            p.font.size = Pt(10.5)
            p.font.color.rgb = C_TEXT_DARK

    # Bottom Callout: Key Findings
    add_card(s7, Inches(0.8), Inches(5.3), Inches(11.733), Inches(1.5), C_EMERALD_BG, C_EMERALD)
    tb_f = s7.shapes.add_textbox(Inches(1.0), Inches(5.4), Inches(11.3), Inches(1.3))
    tf_f = tb_f.text_frame
    tf_f.word_wrap = True
    p = tf_f.paragraphs[0]
    p.text = "Key Empirical Highlights:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD

    f_bullets = [
        "Zero False-Alarm Operation: Under k=2 debouncing, all recommended detectors maintain 0.0% pre-onset false alarms across all 9 cells.",
        "Model Output Generality: pred_prop_psi won in 5 of the 9 cells, proving that model predictions are highly sensitive to shifts.",
        "The Adult Income Anomaly: On Adult Income covariate shift, pred_prop_psi achieved 0.83 sensitivity while raw PSI missed completely."
    ]
    for b in f_bullets:
        p = tf_f.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(2)

    add_speaker_notes(s7,
        "Thank you, Nikita. Here is our master empirical recommendation matrix across all 1,080 test runs.\n\n"
        "Three critical patterns emerge immediately:\n"
        "First, on Give Me Some Credit and Taiwan Credit, PSI and KL achieved perfect 1.00 sensitivity on covariate shift with zero false alarms.\n"
        "Second, on label shift, watching model output probabilities via pred_prop_psi consistently achieved 1.00 sensitivity across all datasets.\n"
        "Third, on concept drift, concept_signal achieved 0.89 to 1.00 sensitivity, while input-only detectors registered zero.\n\n"
        "Notice the anomaly in cell (Adult Income, Covariate Shift): pred_prop_psi won with 0.83 sensitivity, while raw PSI and KL failed. "
        "Srishti will unpack the deep mechanics behind this shortly.\n\n"
        "Now, let us examine the four quantitative trade-off curves governing our system."
    )

    # =========================================================================
    # SLIDE 8: Trade-Off Curves 1 & 2 — Continuous ROC & Debounce Window
    # Speaker: Anish Karmakar (08:15 – 09:30)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Trade-Off Curves (20%)", "Trade-Off 1 & 2: ROC Discrimination and Why k=2 Consecutive Batches is Optimal", "Anish Karmakar", "08:15 – 09:30")
    add_footer(s8, 8)

    # Curve 1 Image & Text
    add_card(s8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1))
    s8.shapes.add_picture("report_assets_combined/tradeoff_1_roc_curves.png", Inches(1.0), Inches(1.8), Inches(5.2), Inches(2.9))
    tb_c1 = s8.shapes.add_textbox(Inches(1.0), Inches(4.8), Inches(5.2), Inches(1.8))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    p.text = "Trade-Off Curve 1: Continuous ROC Frontier"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    roc_txt = [
        "Sweeps detection threshold tau across all 5 metrics.",
        "psi_agg achieves an AUC of 0.985 (near-perfect separation).",
        "pred_prop_psi achieves an AUC of 0.942.",
        "Winsorized centroid_dist achieves 0.880 AUC (recovering substantial separating power once outliers are bounded)."
    ]
    for t in roc_txt:
        p = tf_c1.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(9.5)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(2)

    # Curve 2 Image & Text
    add_card(s8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1))
    s8.shapes.add_picture("report_assets_combined/tradeoff_2_debounce_lag_vs_far.png", Inches(7.0), Inches(1.8), Inches(5.3), Inches(2.9))
    tb_c2 = s8.shapes.add_textbox(Inches(7.0), Inches(4.8), Inches(5.3), Inches(1.8))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True
    p = tf_c2.paragraphs[0]
    p.text = "Trade-Off Curve 2: Debounce Window (k)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    db_txt = [
        "At k=1 batch: False alarm rate is 12.4% with lag = 1.0 batch.",
        "At k=2 batches: False alarms drop to 0.0% with lag = 2.0 batches.",
        "The Sweet Spot: k=2 completely suppresses noise at minimal latency cost.",
        "At k >= 3: False alarms stay at 0%, but detection lag is penalized linearly."
    ]
    for t in db_txt:
        p = tf_c2.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(9.5)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(2)

    add_speaker_notes(s8,
        "Our evaluation reveals four fundamental engineering trade-off curves.\n\n"
        "Trade-Off Curve 1 on the left is our ROC frontier. By sweeping detection thresholds across all five metrics, we see that "
        "psi_agg achieves an Area Under the Curve of 0.985, separating covariate shift from normal streaming variance almost cleanly. "
        "Output-space pred_prop_psi reaches 0.942 AUC. Notice that Winsorized embedding distance achieves 0.880—recovering substantial "
        "discriminative power once extreme outliers are clipped.\n\n"
        "Trade-Off Curve 2 on the right illustrates the debouncing trade-off. In production, false alarms cause alert fatigue. "
        "If you trigger on k=1 single batch, you get a 12.4% false alarm rate. Moving to k=2 completely eliminates pre-onset false alarms "
        "to 0.0%, incurring only a single batch of additional lag. Pushing k past 3 yields zero additional false alarm benefit while "
        "directly penalizing response latency. Thus, k=2 is the optimal production operating point."
    )

    # =========================================================================
    # SLIDE 9: Trade-Off Curves 3 & 4 — Detection Lag & Cost-Capability Frontier
    # Speaker: Anish Karmakar (09:30 – 10:45)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Trade-Off Curves (20%)", "Trade-Off 3 & 4: Sub-2-Batch Response vs. 2.4 ms/batch Feature Divergence Cost", "Anish Karmakar", "09:30 – 10:45")
    add_footer(s9, 9)

    # Curve 3 Image & Text
    add_card(s9, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1))
    s9.shapes.add_picture("report_assets_combined/tradeoff_3_lag_vs_severity.png", Inches(1.0), Inches(1.8), Inches(5.2), Inches(2.9))
    tb_c3 = s9.shapes.add_textbox(Inches(1.0), Inches(4.8), Inches(5.2), Inches(1.8))
    tf_c3 = tb_c3.text_frame
    tf_c3.word_wrap = True
    p = tf_c3.paragraphs[0]
    p.text = "Trade-Off Curve 3: Lag vs. Shift Severity"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    lag_txt = [
        "Severe Shifts: Detected in 2.0 batches (theoretical minimum under k=2).",
        "Moderate Shifts: Detected within 2.1 to 3.4 batches.",
        "Mild Shifts (Gradual Ramp): Require 5.2 to 8.6 batches before breaching threshold.",
        "Latency scales inversely with statistical signal strength."
    ]
    for t in lag_txt:
        p = tf_c3.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(9.5)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(2)

    # Curve 4 Image & Text
    add_card(s9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1))
    s9.shapes.add_picture("report_assets_combined/tradeoff_4_cost_capability_frontier.png", Inches(7.0), Inches(1.8), Inches(5.3), Inches(2.9))
    tb_c4 = s9.shapes.add_textbox(Inches(7.0), Inches(4.8), Inches(5.3), Inches(1.8))
    tf_c4 = tb_c4.text_frame
    tf_c4.word_wrap = True
    p = tf_c4.paragraphs[0]
    p.text = "Trade-Off Curve 4: Cost vs. Capability Frontier"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    cost_txt = [
        "psi_agg + kl_agg: 2.38 ms/batch (421 batches/sec) — Catches Covariate Shift.",
        "pred_prop_psi: 19.8 ms/batch (50 batches/sec) — Catches Label + High-Weight Covariate.",
        "concept_signal: 21.9 ms/batch (45 batches/sec) — Catches Concept Drift.",
        "Combined 5-Detector Pipeline: 58.7 ms/batch (~17 batches/sec on single CPU)."
    ]
    for t in cost_txt:
        p = tf_c4.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(9.5)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(2)

    add_speaker_notes(s9,
        "Trade-Off Curve 3 plots detection latency as a function of injected severity. For severe abrupt shifts, the system reacts "
        "at the theoretical minimum of 2.0 batches. Under mild gradual drift, the signal accumulates slowly, requiring 5 to 8 batches "
        "before breaching threshold.\n\n"
        "Trade-Off Curve 4 maps computational cost against capability. Cached PSI and KL execute in just 2.38 milliseconds per batch—processing "
        "over 400 batches per second. Adding model-output monitoring requires 19.8 milliseconds. Full supervisory verification runs at "
        "21.9 milliseconds.\n\n"
        "The full 5-detector pipeline completes in 58.7 milliseconds per batch—easily capable of real-time streaming monitoring at 17 batches "
        "per second on a single standard CPU.\n\n"
        "Now, Srishti will present where things break, our mathematical proofs, and demonstrate the reproduction."
    )

    # =========================================================================
    # SLIDE 10: Where Things Break (Part 1) — Jeffreys Divergence Equivalence
    # Speaker: Srishti Lamba (10:45 – 11:45)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Conceptual Understanding (30%)", "Where Things Break: Jeffreys Divergence Equivalence (PSI ≈ 2 · D_KL)", "Srishti Lamba", "10:45 – 11:45")
    add_footer(s10, 10)

    # Left card: Mathematical derivation
    add_card(s10, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), C_CARD_BG, C_PRIMARY)
    tb_m = s10.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.7))
    tf_m = tb_m.text_frame
    tf_m.word_wrap = True
    p = tf_m.paragraphs[0]
    p.text = "Mathematical Proof of Equivalence"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY

    p1 = tf_m.add_paragraph()
    p1.text = "Expanding the PSI summation:\n" \
              "PSI(P, Q) = Σ (P_b - Q_b) ln(P_b / Q_b)\n" \
              "          = Σ P_b ln(P_b / Q_b) + Σ Q_b ln(Q_b / P_b)\n" \
              "          = D_KL(P || Q) + D_KL(Q || P)"
    p1.font.size = Pt(12)
    p1.font.bold = True
    p1.font.color.rgb = C_PRIMARY
    p1.space_before = Pt(8)

    p2 = tf_m.add_paragraph()
    p2.text = "Second-Order Taylor Approximation:\n" \
              "Let P_b = Q_b + δ_b where Σ δ_b = 0:\n" \
              "• D_KL(P || Q) ≈ (1/2) Σ (δ_b² / Q_b)\n" \
              "• PSI(P, Q)    ≈ Σ (δ_b² / Q_b) ≈ 2 · D_KL(P || Q)"
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_TEXT_DARK
    p2.space_before = Pt(8)

    p3 = tf_m.add_paragraph()
    p3.text = "Why KL Never Alerts Before PSI:\n" \
              "Because PSI is the symmetrized Jeffreys divergence, PSI >= D_KL always. Setting identical thresholds (tau=0.10) " \
              "makes single-direction KL twice as strict. KL can never fire before PSI."
    p3.font.size = Pt(10.5)
    p3.font.color.rgb = C_TEXT_MUTED
    p3.space_before = Pt(8)

    # Right card: Empirical table on Adult Income
    add_card(s10, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1))
    tb_t = s10.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.7))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    p = tf_t.paragraphs[0]
    p.text = "Empirical Proof on Adult Income Batches"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY

    # Mini Table
    t_m = s10.shapes.add_table(4, 4, Inches(7.0), Inches(2.4), Inches(5.3), Inches(1.8)).table
    t_m.columns[0].width = Inches(1.2)
    t_m.columns[1].width = Inches(1.3)
    t_m.columns[2].width = Inches(1.3)
    t_m.columns[3].width = Inches(1.5)

    m_headers = ["Shift Severity", "Feature PSI", "Feature KL", "Observed Ratio"]
    for c_i, h in enumerate(m_headers):
        c = t_m.cell(0, c_i)
        c.fill.solid()
        c.fill.fore_color.rgb = C_PRIMARY
        p = c.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)

    m_rows = [
        ("Mild", "0.0058", "0.0029", "2.000 (Misses)"),
        ("Moderate", "0.0409", "0.0213", "1.920 (Misses)"),
        ("Severe", "0.1846", "0.0964", "1.915 (PSI Fires!)")
    ]
    for r_i, vals in enumerate(m_rows):
        for c_i, v in enumerate(vals):
            c = t_m.cell(r_i + 1, c_i)
            c.fill.solid()
            c.fill.fore_color.rgb = C_CARD_BG if r_i % 2 == 0 else RGBColor(255, 255, 255)
            p = c.text_frame.paragraphs[0]
            p.text = v
            p.font.size = Pt(10)
            p.font.color.rgb = C_TEXT_DARK

    # Commentary below table
    tb_c = s10.shapes.add_textbox(Inches(7.0), Inches(4.4), Inches(5.3), Inches(2.1))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    p = tf_c.paragraphs[0]
    p.text = "Key Empirical Proof:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    exp_txt = [
        "The measured ratio PSI/KL is exactly 1.92 to 2.00, matching the Taylor expansion to within 4%.",
        "At Severe shift: PSI reaches 0.1846 and crosses threshold (tau=0.10), triggering an alert.",
        "However, KL reaches only 0.0964 (< 0.10) and completely misses severe shift.",
        "Setting equal thresholds for PSI and KL is a mathematical error."
    ]
    for t in exp_txt:
        p = tf_c.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(9.5)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(2)

    add_speaker_notes(s10,
        "Thank you, Anish. A central goal of this project was to understand why the system behaves as it does, rather than "
        "just treating it as a black box.\n\n"
        "Early in our experiments, we noticed that KL divergence never alerted before PSI. Here is the mathematical proof:\n"
        "Expanding the PSI summation reveals that PSI is identically equal to D_KL(P || Q) + D_KL(Q || P)—the classic symmetrized "
        "Jeffreys divergence.\n\n"
        "Under a second-order Taylor expansion around small perturbations, D_KL scales as one-half the chi-square distance, while "
        "PSI scales as the full chi-square distance. Therefore, PSI is approximately twice D_KL.\n\n"
        "We independently measured this across all empirical batches on Adult Income: the measured ratio of PSI to KL is exactly "
        "2.000 on mild, 1.920 on moderate, and 1.915 on severe shift.\n\n"
        "When both detectors are assigned an identical threshold of 0.10, KL requires twice the statistical deviation to trip. "
        "At severe shift, PSI reached 0.184 and triggered, while KL reached 0.096 and missed. Setting equal thresholds for PSI "
        "and KL is a mathematical error."
    )

    # =========================================================================
    # SLIDE 11: Where Things Break (Part 2) — Feature Quantization vs. Model Logit Amplification
    # Speaker: Srishti Lamba (11:45 – 12:45)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Conceptual Understanding (30%)", "Where Things Break: Feature Quantization vs. Model Logit Amplification", "Srishti Lamba", "11:45 – 12:45")
    add_footer(s11, 11)

    # Left card: Raw Feature Quantization
    add_card(s11, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), C_CORAL_BG, C_CORAL)
    tb_q = s11.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.7))
    tf_q = tb_q.text_frame
    tf_q.word_wrap = True
    p = tf_q.paragraphs[0]
    p.text = "1. Raw Feature Quantization Collapse"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CORAL

    q_items = [
        ("Target Feature Scale", "hours-per-week spans [1, 99] with mean 40.42 hours."),
        ("Severe Peaking", "Over 60% of respondents sit at exactly 40 hours/week. Q25 = 40.0, Q75 = 45.0, yielding IQR = 5.0 hours."),
        ("Quantile Bin Broadness", "Because most data is concentrated at 40, quantile bins surrounding 40 are broad."),
        ("Moderate Shift (+3.7 hrs)", "A moderate shift of +3.7 hours shifts people from 40 to ~44 hours—staying inside the same broad quantile bins."),
        ("The Empirical Result", "Raw feature PSI = 0.0409 (< 0.10) and KL = 0.0213 (< 0.10). Feature histograms completely miss moderate shift.")
    ]
    for h, b in q_items:
        p = tf_q.add_paragraph()
        p.text = f"• {h}: {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(5)

    # Right card: Model Logit Amplification
    add_card(s11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), C_EMERALD_BG, C_EMERALD)
    tb_a = s11.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.7))
    tf_a = tb_a.text_frame
    tf_a.word_wrap = True
    p = tf_a.paragraphs[0]
    p.text = "2. Model Logit Amplification"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD

    a_items = [
        ("Model Architecture", "Reference LogisticRegression trained with StandardScaler (intercept = -1.4832)."),
        ("High-Leverage Weight", "hours-per-week carries a standardized coefficient of beta = +0.4993 (the 4th strongest feature in the model)."),
        ("Non-Linear Sigmoid Shift", "In logit space, z = w^T x + b. Shifting +3.7 hours moves an enormous mass of applicants across the classification threshold."),
        ("Output Probability Shift", "The model's predicted probability distribution shifts by 14.2%."),
        ("The Empirical Result", "pred_prop_psi = 0.0298 (>= 0.015). Model output divergence catches moderate shift with 1.00 sensitivity.")
    ]
    for h, b in a_items:
        p = tf_a.add_paragraph()
        p.text = f"• {h}: {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(5)

    add_speaker_notes(s11,
        "Our second major finding is the Adult Income covariate inversion. On the first two datasets, PSI was perfect. "
        "On Adult Income, PSI missed moderate covariate shift completely, while model-output pred_prop_psi caught it.\n\n"
        "We inspected the trained model and data distributions directly:\n"
        "First, over 60% of people in the census report working exactly 40 hours a week. The Interquartile Range is only 5 hours "
        "across a range of 1 to 99. A moderate shift of 3.7 hours moves instances into adjacent bins, yielding a feature PSI of "
        "only 0.0409—well below the 0.10 threshold.\n\n"
        "Second, we extracted the model's standardized weights: hours-per-week has a large positive weight of beta = +0.4993. "
        "In logit space, +3.7 hours pushes a massive cluster of applicants across the classification threshold. The output probability "
        "distribution shifted by 14%, driving pred_prop_psi to 0.0298, breaching its 0.015 threshold.\n\n"
        "This proves that when high-leverage features have narrow natural variance, output probability monitoring catches drift "
        "that raw feature histograms miss."
    )

    # =========================================================================
    # SLIDE 12: Where Things Break (Part 3) — Concept Drift in Imbalanced Data
    # Speaker: Srishti Lamba (12:45 – 13:45)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Conceptual Understanding (30%)", "Where Things Break: Localized Concept Drift in Imbalanced Datasets", "Srishti Lamba", "12:45 – 13:45")
    add_footer(s12, 12)

    # 2 Comparison cards
    add_card(s12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(4.2))
    tb_cd1 = s12.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(3.8))
    tf_cd1 = tb_cd1.text_frame
    tf_cd1.word_wrap = True
    p = tf_cd1.paragraphs[0]
    p.text = "Regime 1: Unconstrained Inversion (pure=False)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY

    cd1_txt = [
        "Mechanism: Inverts labels (1 - Y) within feature band.",
        "Compounding Shift: Baseline default rate jumps from 7.0% to 9.4% (mild), 14.2% (moderate), and 20.5% (severe).",
        "Detector Response: concept_signal breaches threshold easily (Brier delta = 0.078 >= 0.025).",
        "Analytical Diagnosis: The detector catches a compound failure: Concept Drift + Marginal Label Inflation."
    ]
    for t in cd1_txt:
        p = tf_cd1.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(6)

    add_card(s12, Inches(6.8), Inches(1.6), Inches(5.7), Inches(4.2), C_CORAL_BG, C_CORAL)
    tb_cd2 = s12.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(3.8))
    tf_cd2 = tb_cd2.text_frame
    tf_cd2.word_wrap = True
    p = tf_cd2.paragraphs[0]
    p.text = "Regime 2: Pure Swapped Inversion (pure=True)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_CORAL

    cd2_txt = [
        "Mechanism: Swaps 0 <-> 1 within band, strictly holding marginal P(Y) at 7.0%.",
        "The Minority Bottleneck: In a 4,000-row batch, only ~30 positive defaults exist in that band to swap!",
        "Trickle Attenuation: Across a 20% labeled trickle (800 rows), only ~6 swapped rows are evaluated.",
        "Detector Response: Global Brier delta moves by only 0.003 (Below 0.025 threshold -> MISSED).",
        "Analytical Diagnosis: Global batch metrics are blind to localized concept drift in low-prevalence classes."
    ]
    for t in cd2_txt:
        p = tf_cd2.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(4)

    # Bottom takeaway
    add_card(s12, Inches(0.8), Inches(6.0), Inches(11.733), Inches(0.85), C_AMBER_BG, C_AMBER)
    tb_tkw = s12.shapes.add_textbox(Inches(1.0), Inches(6.05), Inches(11.3), Inches(0.75))
    tf_tkw = tb_tkw.text_frame
    p = tf_tkw.paragraphs[0]
    p.text = "Architectural Conclusion: Global performance metrics have an inherent sensitivity floor on imbalanced data. " \
             "Catching localized concept drift mathematically requires slice-based or subgroup-level performance monitoring."
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_AMBER

    add_speaker_notes(s12,
        "Where does the concept drift detector break?\n\n"
        "In benchmark literature, concept drift is usually injected by flipping labels above a feature threshold. In Give Me Some Credit, "
        "where baseline default is 7%, flipping labels above the median causes the post-onset default rate to surge to 20.5%. "
        "concept_signal detects this easily, but it is catching a compound failure: concept drift combined with massive label inflation.\n\n"
        "We tested a pure swap where marginal P(Y) was strictly held at 7.0%. Because defaults are rare, in a batch of 4,000 instances, "
        "only about 30 defaults exist in that band. Across an asynchronous 20% labeled trickle, only 6 swapped rows are evaluated. "
        "The global Brier score changes by only 0.003—completely invisible to global batch metrics.\n\n"
        "This is a vital conceptual insight: Global performance metrics have an inherent sensitivity floor on imbalanced data. "
        "Catching pure localized concept drift mathematically requires slice-based monitoring."
    )

    # =========================================================================
    # SLIDE 13: Live Reproduction, Limitations & Conclusion
    # Speaker: Srishti Lamba (13:45 – 14:45)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Reproducibility & Report (15%)", "One-Command Reproduction & 3-Tier Production Architecture", "Srishti Lamba", "13:45 – 14:45")
    add_footer(s13, 13)

    # Left: Reproduction & Terminal Commands
    add_card(s13, Inches(0.8), Inches(1.6), Inches(5.6), Inches(4.2), C_CARD_BG, C_PRIMARY)
    tb_rep = s13.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(3.8))
    tf_rep = tb_rep.text_frame
    tf_rep.word_wrap = True
    p = tf_rep.paragraphs[0]
    p.text = "Single-Command Full Reproduction"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY

    rep_txt = [
        ("Run Unit Tests (21 passing)", "python -m pytest tests/test_core.py -v"),
        ("Full Grid Reproduction", "python reproduce_all.py"),
        ("Fast Verification Mode (~4m)", "python reproduce_all.py --fast"),
        ("Interactive Dashboard", "streamlit run drift_monitor/dashboard_app.py")
    ]
    for h, cmd in rep_txt:
        p = tf_rep.add_paragraph()
        p.text = f"• {h}:"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(4)
        
        p_c = tf_rep.add_paragraph()
        p_c.text = f"    {cmd}"
        p_c.font.size = Pt(10)
        p_c.font.color.rgb = C_PRIMARY

    # Right: Production Topology & Honest Limitations
    add_card(s13, Inches(6.8), Inches(1.6), Inches(5.7), Inches(4.2))
    tb_top = s13.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(3.8))
    tf_top = tb_top.text_frame
    tf_top.word_wrap = True
    p = tf_top.paragraphs[0]
    p.text = "Production Architecture & Limitations"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    p_top_items = [
        ("Tier 1: Unsupervised Input & Output Gate (<5 ms)", "Cached PSI on key features + pred_prop_psi with k=2 debouncing."),
        ("Tier 2: Supervised Performance Verification", "Asynchronous 20% labeled trickle tracking Brier degradation."),
        ("Tier 3: Diagnostic Precedence Engine", "Resolves compound alerts into unambiguous root causes."),
        ("Honest Limitation: Tabular Numeric Scope", "Categorical features with dynamic vocabularies require categorical PSI or Jensen-Shannon divergence."),
        ("Honest Limitation: Label Delay", "In lending, defaults take 90+ days to resolve, requiring delayed-feedback window buffers in production.")
    ]
    for h, b in p_top_items:
        p = tf_top.add_paragraph()
        p.text = f"• {h}: {b}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(4)

    # Bottom Banner: Conclusion
    add_card(s13, Inches(0.8), Inches(6.0), Inches(11.733), Inches(0.85), C_EMERALD_BG, C_EMERALD)
    tb_ccl = s13.shapes.add_textbox(Inches(1.0), Inches(6.05), Inches(11.3), Inches(0.75))
    tf_ccl = tb_ccl.text_frame
    p = tf_ccl.paragraphs[0]
    p.text = "Conclusion: No single detector is sufficient. By pairing cached feature divergences with model-output tracking " \
             "and supervisory verification, silent model degradation can be prevented with zero false alarms."
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD

    add_speaker_notes(s13,
        "To prove complete reproducibility, our entire pipeline—all three datasets, benchmarks, and the four trade-off curves—can "
        "be regenerated with one command: python reproduce_all.py.\n\n"
        "We also acknowledge honest engineering limitations: our current implementation targets numeric tabular data and assumes "
        "labeled trickle feedback within the stream buffer. In credit lending, default outcomes take 90 days to resolve, requiring "
        "delayed-feedback window buffers in production.\n\n"
        "In conclusion: no single detector is sufficient. By combining cached input PSI, model-output divergence, and debounced supervisory "
        "tracking, we achieve comprehensive coverage across all three distribution shift failure modes with zero false alarms.\n\n"
        "Thank you, Professor. We are now ready for Q&A."
    )

    out_path = "Drift_Monitor_Presentation.pptx"
    prs.save(out_path)
    print(f"Successfully generated presentation at: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    build_presentation()
