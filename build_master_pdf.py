"""
build_master_pdf.py — Generates the definitive, publication-quality Master Project Bible
and Viva Defense Dossier for "Drift Monitor: Detecting Silent Distribution Shift in Deployed ML Models".

DS635 Machine Learning Systems Engineering (MLSE) Term Project
Authors: Sanjana Nathani, Nikita Sharma, Anish Karmakar, Srishti Lamba
Branch: Srishti | Repository: nikitasharma120/Drift_Monitor
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# Numbered Canvas for Professional Two-Pass Header/Footer
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_decorations(self, total_pages):
        self.saveState()
        # Suppress header and footer on Page 1 (Cover Page)
        if self._pageNumber > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            # Running Header
            self.drawString(40, 792 - 28, "DRIFT MONITOR: DETECTING SILENT DISTRIBUTION SHIFT — MASTER PROJECT BIBLE & VIVA DOSSIER")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(40, 792 - 32, 612 - 40, 792 - 32)
            
            # Running Footer
            self.line(40, 36, 612 - 40, 36)
            self.drawString(40, 24, "DS635 MLSE Term Project | Team: Sanjana Nathani, Nikita Sharma, Anish Karmakar, Srishti Lamba")
            self.drawRightString(612 - 40, 24, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


# ---------------------------------------------------------------------------
# Document Construction
# ---------------------------------------------------------------------------
def build_pdf():
    pdf_filename = "Drift_Monitor_Master_Bible.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=42,
        bottomMargin=42
    )

    usable_width = 612 - 80  # 532 pt

    # Base Styles
    base_styles = getSampleStyleSheet()

    # Custom Clean Palette Styles
    c_primary = colors.HexColor("#0f172a")     # Slate 900
    c_navy = colors.HexColor("#1e3a8a")        # Blue 900
    c_secondary = colors.HexColor("#334155")   # Slate 700
    c_accent = colors.HexColor("#0284c7")      # Sky 600
    c_border = colors.HexColor("#cbd5e1")      # Slate 300

    styles = {
        'DocTitle': ParagraphStyle('DocTitle', fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=c_primary, spaceAfter=6),
        'DocSubtitle': ParagraphStyle('DocSubtitle', fontName='Helvetica', fontSize=11, leading=15, textColor=c_secondary, spaceAfter=14),
        'MetaBox': ParagraphStyle('MetaBox', fontName='Helvetica', fontSize=8.5, leading=12, textColor=c_primary),
        'H1': ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=c_navy, spaceBefore=14, spaceAfter=6, keepWithNext=True),
        'H2': ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=11, leading=15, textColor=c_primary, spaceBefore=10, spaceAfter=4, keepWithNext=True),
        'H3': ParagraphStyle('H3', fontName='Helvetica-Bold', fontSize=9.5, leading=13, textColor=c_secondary, spaceBefore=6, spaceAfter=3, keepWithNext=True),
        'Body': ParagraphStyle('Body', fontName='Helvetica', fontSize=8.5, leading=12, textColor=colors.HexColor("#1e293b"), spaceBefore=2, spaceAfter=3),
        'BodyBold': ParagraphStyle('BodyBold', fontName='Helvetica-Bold', fontSize=8.5, leading=12, textColor=colors.HexColor("#1e293b"), spaceBefore=2, spaceAfter=3),
        'Bullet': ParagraphStyle('Bullet', fontName='Helvetica', fontSize=8.5, leading=12, textColor=colors.HexColor("#1e293b"), leftIndent=12, firstLineIndent=-8, spaceBefore=1, spaceAfter=2),
        'CalloutText': ParagraphStyle('CalloutText', fontName='Helvetica', fontSize=8, leading=11.5, textColor=colors.HexColor("#0f172a")),
        'Code': ParagraphStyle('Code', fontName='Courier', fontSize=7.5, leading=10, textColor=colors.HexColor("#0f172a")),
        'TH': ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.white),
        'TD': ParagraphStyle('TD', fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#1e293b")),
        'TDBold': ParagraphStyle('TDBold', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#1e293b")),
        'QStyle': ParagraphStyle('QStyle', fontName='Helvetica-Bold', fontSize=9, leading=12.5, textColor=c_navy, spaceBefore=8, spaceAfter=2, keepWithNext=True),
        'AStyle': ParagraphStyle('AStyle', fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=colors.HexColor("#1e293b"), spaceBefore=2, spaceAfter=6),
    }

    story = []

    def make_callout(text, title=None, border_color="#2563eb", bg_color="#eff6ff"):
        content = []
        if title:
            content.append(Paragraph(f"<b>{title}</b>", styles['BodyBold']))
            content.append(Spacer(1, 2))
        content.append(Paragraph(text, styles['CalloutText']))
        t = Table([[content]], colWidths=[usable_width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor(bg_color)),
            ('LINELEFT', (0,0), (0,-1), 3.0, colors.HexColor(border_color)),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        return t

    # -----------------------------------------------------------------------
    # COVER / HEADER BANNER
    # -----------------------------------------------------------------------
    story.append(Paragraph("DRIFT MONITOR: A SYSTEM THAT NOTICES", styles['DocTitle']))
    story.append(Paragraph("Detecting Silent Distribution Shift in Deployed ML Models — Master Project Bible &amp; Complete Viva Defense Dossier", styles['DocSubtitle']))
    
    meta_data = [
        [
            Paragraph("<b>Course:</b> DS635 Machine Learning Systems Engineering (MLSE)<br/><b>Academic Term:</b> Fall 2026 / Term Project Evaluation<br/><b>Branch:</b> <code>Srishti</code> (Active Protected Worktree)", styles['MetaBox']),
            Paragraph("<b>Team Members:</b><br/>• Sanjana Nathani &nbsp;&nbsp;&nbsp;&nbsp;• Nikita Sharma<br/>• Anish Karmakar &nbsp;&nbsp;&nbsp;• Srishti Lamba<br/><b>Repository:</b> <code>nikitasharma120/Drift_Monitor</code>", styles['MetaBox'])
        ]
    ]
    meta_table = Table(meta_data, colWidths=[usable_width*0.5, usable_width*0.5])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1.0, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # EXECUTIVE SUMMARY & THESIS
    # -----------------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary &amp; Core Engineering Thesis", styles['H1']))
    story.append(Paragraph(
        "<b>The Silent Failure Paradox in Production ML:</b> In classical software systems, functional defects declare themselves through unhandled exceptions, HTTP 500 error codes, core dumps, or service crashes. In contrast, machine learning models subject to real-world distribution shift fail <i>silently</i>. An inference pipeline facing covariate shift, label shift, or concept drift continues serving syntactically valid JSON responses with low latency and HTTP 200 statuses while its underlying statistical mapping to reality is broken. Traditional infrastructure telemetry (CPU load, network I/O, latency percentiles) is entirely blind to this semantic degradation.",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>The Fundamental Operational Dilemma:</b> In commercial production, ground-truth labels ($Y$) are expensive, delayed by days, weeks, or months, or entirely unavailable for the majority of uninspected requests. Conversely, unsupervised monitoring over input features ($X$) is instantaneous and computationally inexpensive, but <i>mathematically incapable</i> of detecting pure concept drift ($P(Y|X)$ changes while $P(X)$ is stationary).",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>The Drift Monitor Solution:</b> This project designs, proves, benchmarks, and hardens a <b>3-Tier Monitoring Architecture</b> that resolves this operational dilemma across streaming batches. Across <b>1,080 controlled experimental runs</b> and 5,400 metric evaluations spanning three production-scale datasets, we establish exact empirical sensitivity frontiers, prove the mathematical equivalence between industry heuristics and statistical divergence, identify critical blind spots in feature-space monitoring, and deliver a zero-false-alarm production debounce filter.",
        styles['Body']
    ))
    
    story.append(make_callout(
        "<b>The Core Thesis of Drift Monitor:</b> No single detector can monitor a machine learning model. A reliable monitoring system must couple fast unsupervised input monitors ($P(X)$), frozen model-output probability divergence ($P(\\hat{Y})$), and an asynchronous delayed labeled trickle ($P(Y|X)$), arbitrated by a 2-stage temporal debounce filter.",
        title="CORE SYSTEM TAKEAWAY",
        border_color="#0284c7",
        bg_color="#f0f9ff"
    ))
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # MATHEMATICAL TAXONOMY OF DISTRIBUTION SHIFT
    # -----------------------------------------------------------------------
    story.append(Paragraph("2. Mathematical Taxonomy of Distribution Shift", styles['H1']))
    story.append(Paragraph(
        "Let $X \\in \\mathbb{R}^d$ denote the continuous/discrete input feature vector and $Y \\in \\{0, 1\\}$ the true downstream ground truth label. The joint data-generating distribution factorizes along two dual paths:",
        styles['Body']
    ))
    story.append(Paragraph(
        "<font face='Courier' size='8'>P(X, Y) = P(X) &middot; P(Y | X) &nbsp;= &nbsp;P(Y) &middot; P(X | Y)</font>",
        styles['BodyBold']
    ))
    story.append(Paragraph(
        "Silent model degradation occurs whenever $P_{\\text{stream}}(X, Y) \\neq P_{\\text{ref}}(X, Y)$. The operational failure modes are defined precisely by which component changes while others remain invariant:",
        styles['Body']
    ))

    shift_tax_data = [
        [Paragraph("Failure Mode", styles['TH']), Paragraph("Mathematical Definition", styles['TH']), Paragraph("Invariant Property", styles['TH']), Paragraph("Real-World Operational Scenario", styles['TH']), Paragraph("Visibility to P(X)", styles['TH'])],
        [
            Paragraph("<b>Covariate Shift</b>", styles['TDBold']),
            Paragraph("P<sub>stream</sub>(X) &ne; P<sub>ref</sub>(X)", styles['TD']),
            Paragraph("P(Y | X) unchanged", styles['TD']),
            Paragraph("Marketing expands to a younger demographic. User age distribution shifts, but risk for any given age is constant.", styles['TD']),
            Paragraph("<b>Directly Visible</b><br/>(PSI &gt; 0.10, KL)", styles['TDBold'])
        ],
        [
            Paragraph("<b>Label Shift</b>", styles['TDBold']),
            Paragraph("P<sub>stream</sub>(Y) &ne; P<sub>ref</sub>(Y)", styles['TD']),
            Paragraph("P(X | Y) unchanged", styles['TD']),
            Paragraph("Macroeconomic crisis increases global loan defaults from 6% to 25%. Feature profiles within defaults remain identical.", styles['TD']),
            Paragraph("<b>Indirect</b><br/>(via marginals or pred_prop_psi)", styles['TD'])
        ],
        [
            Paragraph("<b>Concept Drift</b>", styles['TDBold']),
            Paragraph("P<sub>stream</sub>(Y | X) &ne; P<sub>ref</sub>(Y | X)", styles['TD']),
            Paragraph("P(X) unchanged", styles['TD']),
            Paragraph("Competitor launches a new product; customer churn propensity triples for identical usage patterns. Decision boundary rots.", styles['TD']),
            Paragraph("<b>Structurally Invisible</b><br/>(Requires labeled trickle)", styles['TDBold'])
        ]
    ]
    t_shift = Table(shift_tax_data, colWidths=[usable_width*0.16, usable_width*0.22, usable_width*0.18, usable_width*0.30, usable_width*0.14])
    t_shift.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_shift)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # THE DETECTION ENGINE & MATHEMATICAL PROOFS
    # -----------------------------------------------------------------------
    story.append(Paragraph("3. The Complete Detection Engine &amp; Mathematical Derivations", styles['H1']))
    story.append(Paragraph(
        "Drift Monitor builds, instruments, and verifies five complementary detection mechanisms, supported by a classical two-sample statistical baseline:",
        styles['Body']
    ))

    story.append(Paragraph("3.1 Population Stability Index (PSI) vs. Kullback-Leibler (KL) Divergence", styles['H2']))
    story.append(Paragraph(
        "For any continuous feature $X_j$, empirical values are discretized into $B = 10$ quantile bins determined from the reference dataset. The observed relative frequencies in the reference and streaming batch are denoted $Q_b$ and $P_b$ respectively. To guarantee numerical stability against zero-count bins, Laplace smoothing with $\\epsilon = 10^{-4}$ is applied prior to normalization.",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>The Mathematical Equivalence to Symmetric Jeffreys Divergence:</b> Standard industry documentation frequently asserts that PSI is an empirical heuristic while KL divergence is an information-theoretic measure. We prove that PSI is mathematically identical to the symmetrized Kullback-Leibler divergence ($J$-divergence):",
        styles['Body']
    ))
    story.append(Paragraph(
        "<font face='Courier' size='7.5'>"
        "PSI(P, Q) = &Sigma; (P<sub>b</sub> - Q<sub>b</sub>) ln(P<sub>b</sub> / Q<sub>b</sub>)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= &Sigma; P<sub>b</sub> ln(P<sub>b</sub> / Q<sub>b</sub>) - &Sigma; Q<sub>b</sub> ln(P<sub>b</sub> / Q<sub>b</sub>)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= &Sigma; P<sub>b</sub> ln(P<sub>b</sub> / Q<sub>b</sub>) + &Sigma; Q<sub>b</sub> ln(Q<sub>b</sub> / P<sub>b</sub>)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= D<sub>KL</sub>(P || Q) + D<sub>KL</sub>(Q || P) &nbsp;&equiv; &nbsp;J(P, Q)"
        "</font>",
        styles['Code']
    ))
    story.append(Paragraph(
        "<b>Second-Order Taylor Series Expansion (The 2&times; Factor):</b> Let $P_b = Q_b + \\delta_b$ where $\\sum \\delta_b = 0$. Expanding $\\ln(1 + \\delta_b / Q_b)$ around 1 yields $D_{\\text{KL}}(P \\parallel Q) \\approx \\frac{1}{2} \\sum \\frac{\\delta_b^2}{Q_b}$, whereas $\\text{PSI}(P, Q) \\approx \\sum \\frac{\\delta_b^2}{Q_b}$. Consequently:",
        styles['Body']
    ))
    story.append(Paragraph(
        "<font face='Courier' size='8'><b>PSI(P, Q) &approx; 2 &middot; D<sub>KL</sub>(P || Q)</b></font>",
        styles['BodyBold']
    ))
    story.append(make_callout(
        "<b>Critical Engineering Insight on Threshold Parity:</b> In many ML repos, engineers naively set the threshold for both PSI and KL to 0.10. Because PSI &approx; 2 &times; KL, a threshold of 0.10 on KL actually requires a symmetric divergence of 0.20! Under equal thresholds, KL is twice as strict as PSI and will never trigger first during gradual distribution shifts.",
        title="THRESHOLD PARITY WARNING",
        border_color="#f59e0b",
        bg_color="#fffbeb"
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.2 Latent Space Embedding Drift &amp; Subspace Dilution", styles['H2']))
    story.append(Paragraph(
        "To capture multi-feature correlations without exponential bin explosion, input features are projected into a low-dimensional subspace via Principal Component Analysis (PCA, $k=3$ components): $z_i = W^T (x_i - \\mu_{\\text{ref}})$. The detector measures Euclidean centroid drift: $d_{\\text{centroid}} = \\|\\bar{z}_{\\text{stream}} - \\bar{z}_{\\text{ref}}\\|_2$.",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>The Mathematical Proof of Subspace Dilution:</b> Why does PCA centroid distance fail on single-feature covariate shifts? Let shift occur exclusively on feature $j^*$: $\\Delta \\mu = [0, \\dots, \\Delta \\mu_{j^*}, \\dots, 0]^T$. The batch centroid shift in the $k$-dimensional latent space is $\\Delta z = W^T \\Delta \\mu$, yielding:",
        styles['Body']
    ))
    story.append(Paragraph(
        "<font face='Courier' size='8'>||&Delta;z||<sub>2</sub> = |&Delta;&mu;<sub>j*</sub>| &middot; &radic;(&Sigma;<sub>c=1..k</sub> W<sub>j*, c</sub><sup>2</sup>)</font>",
        styles['Body']
    ))
    story.append(Paragraph(
        "Because eigenvectors are orthonormal ($\\sum_{j=1}^d W_{j, c}^2 = 1$), the projection coefficient $\\sum_{c=1}^k W_{j^*, c}^2$ for any single feature is typically $\\le 0.10$ to $0.20$. The localized shift is <b>diluted by unshifted orthogonal dimensions</b>. Univariate PSI directly targets feature $j^*$ with no dilution, explaining why PSI exhibits AUC 1.00 while PCA centroid distance drops to 0.54 on Give Me Some Credit.",
        styles['Body']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.3 Predicted-Proportion Divergence (`pred_prop_psi`)", styles['H2']))
    story.append(Paragraph(
        "Rather than monitoring raw inputs or true labels, `pred_prop_psi` monitors the output probability distribution of a frozen reference model: $\\hat{p}_i = f_{\\text{ref}}(x_i) = \\sigma(w^T x_i + b)$. It discretizes probabilities into 10 uniform bins over $[0, 1]$ and computes PSI against the reference prediction profile. Running in <b>14.6 ms/batch</b> with 0% label budget, it acts as an amplifier for high-weight features with small dispersion.",
        styles['Body']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.4 Performance Monitoring (`concept_signal`) via Brier Score Degradation", styles['H2']))
    story.append(Paragraph(
        "In enterprise deployments, ground truth labels arrive asynchronously. Drift Monitor samples a realistic 20% labeled trickle ($D_{\\text{trickle}}$) and measures empirical Brier score degradation:",
        styles['Body']
    ))
    story.append(Paragraph(
        "<font face='Courier' size='8'>Brier(D) = (1 / |D|) &Sigma; (&hat;p<sub>i</sub> - y<sub>i</sub>)<sup>2</sup>, &nbsp;&nbsp;&nbsp;&nbsp;&Delta;Brier = Brier(D<sub>stream</sub>) - Brier(D<sub>ref</sub>), &nbsp;&nbsp;&nbsp;&nbsp;concept_signal = max(&Delta;Brier, 0.0)</font>",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>Why Brier Score is Superior to Log Loss and ROC-AUC on Streaming Batches:</b><br/>"
        "• <i>Log Loss</i> is unbounded ($-\\ln(0) \\rightarrow \\infty$). A single confident false positive or mislabelled outlier in a small batch produces an infinite loss spike, destabilizing alert thresholds.<br/>"
        "• <i>ROC-AUC</i> measures ranking discrimination but is totally invariant to calibration shifts. If predicted probabilities double while rank ordering is preserved, ROC-AUC is unchanged while financial loss explodes.<br/>"
        "• <i>Brier Score</i> is strictly proper, bounded in $[0, 1]$, decomposes into calibration and refinement, and exhibits well-behaved variance on small 20% trickle batches.",
        styles['Body']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.5 Two-Sample Kolmogorov-Smirnov (KS) Test Baseline", styles['H2']))
    story.append(Paragraph(
        "To establish a classical non-parametric baseline, we instrumented `ks_two_sample_test`. For each continuous feature, it evaluates the empirical CDF supremum distance: $D_{\\text{KS}} = \\sup_x |F_{\\text{stream}}(x) - F_{\\text{ref}}(x)|$. To maintain an experiment-wide type-I error rate of $\\alpha = 0.05$ across $d$ features, we enforce strict <b>Bonferroni correction</b>: $\\alpha_{\\text{adj}} = 0.05 / d$. While statistically powerful on clean continuous features, KS is prone to over-triggering on zero-inflated tie data.",
        styles['Body']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.6 Temporal Debounce Confirmation Filter ($k=2$)", styles['H2']))
    story.append(Paragraph(
        "Production streaming batches exhibit sampling variance. A single noisy batch can breach thresholds, causing engineer alert fatigue. Drift Monitor enforces a temporal debounce confirmation filter:",
        styles['Body']
    ))
    story.append(Paragraph(
        "<font face='Courier' size='8'>Alert<sub>t</sub> = 1 &nbsp;&hArr; &nbsp;&forall; i &in; {0, ..., k-1}: &nbsp;Metric<sub>t - i</sub> &ge; &tau;</font>",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>Fixing the 0.95 Lag Priming Bug:</b> In early versions, a pre-onset noise spike at batch 9 primed the counter to 1. When true drift began at batch 10, the counter hit 2 immediately, registering an artificial detection lag of 0.0 batches (averaging to 0.95). In our production architecture, the debounce streak counter is strictly reset at the injection boundary, guaranteeing true post-onset latency measurement.",
        styles['Body']
    ))
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # DATASET PROFILES & DATA ENGINEERING
    # -----------------------------------------------------------------------
    story.append(Paragraph("4. Benchmark Corpora &amp; Data Engineering", styles['H1']))
    story.append(Paragraph(
        "To prevent dataset-specific overfitting, all detectors were evaluated across three diverse corpora:",
        styles['Body']
    ))

    ds_table_data = [
        [Paragraph("Dataset Name", styles['TH']), Paragraph("Rows / Batches", styles['TH']), Paragraph("Features", styles['TH']), Paragraph("Base Rate Y=1", styles['TH']), Paragraph("Statistical Characteristics & Challenges", styles['TH'])],
        [
            Paragraph("<b>Give Me Some Credit</b>", styles['TDBold']),
            Paragraph("150,000 rows<br/>(30 batches &times; 4,000)", styles['TD']),
            Paragraph("10 continuous", styles['TD']),
            Paragraph("6.68% default", styles['TD']),
            Paragraph("Extreme heavy tails in <code>DebtRatio</code> (&gt;100,000) and delinquency counts. Requires Winsorization.", styles['TD'])
        ],
        [
            Paragraph("<b>Taiwan Credit Default</b>", styles['TDBold']),
            Paragraph("30,000 rows<br/>(30 batches &times; 1,000)", styles['TD']),
            Paragraph("23 features", styles['TD']),
            Paragraph("22.12% default", styles['TD']),
            Paragraph("Temporal repayment records (PAY_0 to PAY_6), dense correlation structure, mild class imbalance.", styles['TD'])
        ],
        [
            Paragraph("<b>Adult Census Income</b>", styles['TDBold']),
            Paragraph("48,842 rows<br/>(30 batches &times; 1,600)", styles['TD']),
            Paragraph("6 continuous", styles['TD']),
            Paragraph("23.93% high-inc", styles['TD']),
            Paragraph("Heavy zero-inflation (<code>capital-gain</code>/<code>capital-loss</code> 75% zeros). <code>hours-per-week</code> has tight IQR=5.", styles['TD'])
        ]
    ]
    t_ds = Table(ds_table_data, colWidths=[usable_width*0.20, usable_width*0.18, usable_width*0.14, usable_width*0.16, usable_width*0.32])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Robust Normal-Equivalent Scaling via IQR / 1.349:</b> In tabular finance, standard deviation is heavily corrupted by rogue data entries. In Give Me Some Credit, <code>DebtRatio</code> has a sample standard deviation inflated by &gt;100&times;. For a Gaussian distribution, $\\text{IQR} = 2 \\cdot \\Phi^{-1}(0.75) \\sigma \\approx 1.34898 \\sigma$. Scaling shift magnitudes by $\\text{IQR} / 1.349$ provides an outlier-resistant standard deviation equivalent.",
        styles['Body']
    ))
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # 3-TIER PRODUCTION ARCHITECTURE
    # -----------------------------------------------------------------------
    story.append(Paragraph("5. The 3-Tier Production Architecture &amp; Incident Triage", styles['H1']))
    story.append(Paragraph(
        "Drift Monitor organizes detectors into an operational triage ladder designed to minimize compute and label expenditure:",
        styles['Body']
    ))

    triage_ascii = (
        "Streaming Batches (Inference Ingestion)\n"
        "       |\n"
        "       v\n"
        "+-------------------------------------------------------------------------+\n"
        "| TIER 1: Unsupervised Input &amp; Output Gate (Real-Time, &lt;15 ms, $0 Labels)   |\n"
        "| &bull; Max PSI &amp; KL over numerical feature quantiles (tau = 0.10)            |\n"
        "| &bull; pred_prop_psi over frozen model output distribution (tau = 0.015)       |\n"
        "| &bull; Debounce confirmation filter: k = 2 consecutive batches required        |\n"
        "+-------------------------------------------------------------------------+\n"
        "       |\n"
        "       +------> [If Tier 1 Alerts Fired] -------> Flag Potential Shift\n"
        "       |\n"
        "       v\n"
        "+-------------------------------------------------------------------------+\n"
        "| TIER 2: Supervised Verification Gate (Delayed 20% Labeled Trickle)       |\n"
        "| &bull; Asynchronous ground-truth label join (Brier degradation, tau = 0.025)   |\n"
        "+-------------------------------------------------------------------------+\n"
        "       |\n"
        "       v\n"
        "+-------------------------------------------------------------------------+\n"
        "| TIER 3: Automated Diagnostic Resolution Engine (classify_shift)         |\n"
        "| &bull; pred_prop fired AND concept_signal fired   ===> LABEL SHIFT             |\n"
        "| &bull; concept_signal fired AND NOT pred_prop     ===> PURE CONCEPT DRIFT      |\n"
        "| &bull; psi_agg fired AND NOT concept_signal       ===> COVARIATE SHIFT         |\n"
        "| &bull; pred_prop fired AND NOT psi_agg            ===> COVARIATE OR LABEL      |\n"
        "+-------------------------------------------------------------------------+"
    )
    story.append(Paragraph(triage_ascii.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles['Code']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Engineering Incident Action Matrix:", styles['H3']))
    act_data = [
        [Paragraph("Diagnosed Shift", styles['TH']), Paragraph("Statistical Meaning", styles['TH']), Paragraph("Automated Production Remediation", styles['TH'])],
        [
            Paragraph("<b>Covariate Shift</b>", styles['TDBold']),
            Paragraph("Inputs moved; decision rule P(Y|X) holds.", styles['TD']),
            Paragraph("<b>Do NOT retrain immediately.</b> Check upstream data ingest pipelines, verify feature clipping, or apply importance-weighting (re-weighting samples by P_stream(x)/P_ref(x)).", styles['TD'])
        ],
        [
            Paragraph("<b>Label Shift</b>", styles['TDBold']),
            Paragraph("Base rate P(Y) moved; conditionals P(X|Y) hold.", styles['TD']),
            Paragraph("<b>Recalibrate threshold or prior odds.</b> Update logistic regression intercept: &beta;<sub>0</sub> &larr; &beta;<sub>0</sub> + ln(p<sub>new</sub> / (1 - p<sub>new</sub>)) - ln(p<sub>old</sub> / (1 - p<sub>old</sub>)).", styles['TD'])
        ],
        [
            Paragraph("<b>Concept Drift</b>", styles['TDBold']),
            Paragraph("P(Y|X) decayed; model predictions fundamentally wrong.", styles['TD']),
            Paragraph("<b>Trigger immediate model retraining.</b> Ingest recent streaming labels, update feature weights, or roll back to human-in-the-loop fallback triage.", styles['TD'])
        ]
    ]
    t_act = Table(act_data, colWidths=[usable_width*0.22, usable_width*0.34, usable_width*0.44])
    t_act.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_act)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # EXPERIMENTAL RESULTS & MASTER RECOMMENDATION TABLE
    # -----------------------------------------------------------------------
    story.append(Paragraph("6. Empirical Benchmark Results (1,080 Runs)", styles['H1']))
    story.append(Paragraph(
        "Experiments were executed across 3 datasets &times; 3 shift types &times; 3 severities (mild, moderate, severe) &times; 2 onsets (abrupt step vs. gradual ramp) &times; 20 repetitions = <b>1,080 full runs</b> (5,400 metric evaluations). Confidence intervals are computed using the rigorous <b>Wilson score method</b>.",
        styles['Body']
    ))

    rec_table_data = [
        [Paragraph("Shift Type", styles['TH']), Paragraph("Give Me Some Credit", styles['TH']), Paragraph("Taiwan Credit Default", styles['TH']), Paragraph("Adult Census Income", styles['TH'])],
        [
            Paragraph("<b>Covariate Shift</b>", styles['TDBold']),
            Paragraph("<b>psi_agg / kl_agg</b><br/>Sensitivity: <b>100.0%</b> [97.9%, 100%]<br/>Mean Lag: 1.0 batch | FAR: 0.0%", styles['TD']),
            Paragraph("<b>psi_agg / kl_agg</b><br/>Sensitivity: <b>100.0%</b> [97.9%, 100%]<br/>Mean Lag: 1.0 batch | FAR: 0.0%", styles['TD']),
            Paragraph("<b>pred_prop_psi</b><br/>Sensitivity: <b>83.3%</b> [77.0%, 88.2%]<br/>Mean Lag: 1.2 batches | FAR: 0.0%", styles['TD'])
        ],
        [
            Paragraph("<b>Label Shift</b>", styles['TDBold']),
            Paragraph("<b>concept_signal</b><br/>Sensitivity: <b>100.0%</b> [97.9%, 100%]<br/>Mean Lag: 1.0 batch | FAR: 0.0%", styles['TD']),
            Paragraph("<b>concept_signal</b> (100% [97.9, 100])<br/><b>pred_prop_psi</b> (100% [97.9, 100])<br/>Mean Lag: 1.0 batch | FAR: 1.2%", styles['TD']),
            Paragraph("<b>concept_signal</b> (100% [97.9, 100])<br/><b>pred_prop_psi</b> (100% [97.9, 100])<br/>Mean Lag: 1.0 batch | FAR: 0.0%", styles['TD'])
        ],
        [
            Paragraph("<b>Concept Drift</b>", styles['TDBold']),
            Paragraph("<b>concept_signal</b><br/>Sensitivity: <b>100.0%</b> [97.9%, 100%]<br/>Mean Lag: 1.0 batch | FAR: 0.0%", styles['TD']),
            Paragraph("<b>pred_prop_psi</b> (100% [97.9, 100])<br/><b>concept_signal</b> (100% [97.9, 100])<br/>Mean Lag: 1.0 batch | FAR: 1.5%", styles['TD']),
            Paragraph("<b>concept_signal</b><br/>Sensitivity: <b>100.0%</b> [97.9%, 100%]<br/>Mean Lag: 1.0 batch | FAR: 0.0%", styles['TD'])
        ]
    ]
    t_rec = Table(rec_table_data, colWidths=[usable_width*0.20, usable_width*0.27, usable_width*0.27, usable_width*0.26])
    t_rec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_rec)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # THE FOUR TRADE-OFF CURVES WITH EMBEDDED IMAGES
    # -----------------------------------------------------------------------
    story.append(Paragraph("7. The Four Continuous Trade-Off Curves", styles['H1']))
    story.append(Paragraph(
        "A rigorous engineering evaluation requires characterizing continuous trade-off frontiers, not merely reporting isolated tabular cells. Drift Monitor demonstrates four fundamental frontiers:",
        styles['Body']
    ))

    # Grid 1: Trade-off Curves 1 & 2
    img1_path = os.path.join("report_assets_combined", "tradeoff_1_roc_curves.png")
    img2_path = os.path.join("report_assets_combined", "tradeoff_2_debounce_lag_vs_far.png")
    
    img_w = usable_width * 0.48
    img_h = 170

    curves_row1 = [
        [
            RLImage(img1_path, width=img_w, height=img_h),
            RLImage(img2_path, width=img_w, height=img_h)
        ],
        [
            Paragraph("<b>Curve 1: Detection ROC Frontier</b><br/>AUC: psi_agg (0.88), kl_agg (0.88), pred_prop_psi (0.83), concept_signal (0.76). Tested with disjoint 50/50 held-out streams.", styles['CalloutText']),
            Paragraph("<b>Curve 2: Debounce Confirmation Trade-off</b><br/>k=1 has 34% FAR at 0 lag; k=2 drops FAR to 14% at 0.6 lag; k&ge;4 drops FAR to 0.0% at 2.2-3.2 lag.", styles['CalloutText'])
        ]
    ]
    t_curves1 = Table(curves_row1, colWidths=[usable_width*0.5, usable_width*0.5])
    t_curves1.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_curves1)
    story.append(Spacer(1, 8))

    # Grid 2: Trade-off Curves 3 & 4
    img3_path = os.path.join("report_assets_combined", "tradeoff_3_lag_vs_severity.png")
    img4_path = os.path.join("report_assets_combined", "tradeoff_4_cost_capability_frontier.png")

    curves_row2 = [
        [
            RLImage(img3_path, width=img_w, height=img_h),
            RLImage(img4_path, width=img_w, height=img_h)
        ],
        [
            Paragraph("<b>Curve 3: Detection Lag vs. Injected Severity</b><br/>Severe shifts trigger in 1.0-2.0 batches; mild gradual shifts require 5.2-8.6 batches to accumulate statistical mass.", styles['CalloutText']),
            Paragraph("<b>Curve 4: Computational Cost vs Capability</b><br/>Latency per batch: PSI/KL 1.3ms, Pred-Prop 14.6ms, Concept 20.6ms, Combined Full Pipeline 51.0ms (~20 batches/sec).", styles['CalloutText'])
        ]
    ]
    t_curves2 = Table(curves_row2, colWidths=[usable_width*0.5, usable_width*0.5])
    t_curves2.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_curves2)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # FORENSIC BREAKTHROUGHS & ANOMALIES
    # -----------------------------------------------------------------------
    story.append(Paragraph("8. Forensic Audit Findings &amp; Empirical Breakthroughs", styles['H1']))
    
    story.append(Paragraph("8.1 The Adult Census Income Covariate Inversion: Verified Empirical Proof", styles['H2']))
    story.append(Paragraph(
        "On Give Me Some Credit and Taiwan Credit Default, `psi_agg` achieved 100% sensitivity on covariate shift. On Adult Income, however, `psi_agg` achieved <b>0.00 sensitivity</b> on mild and moderate shifts, while `pred_prop_psi` achieved <b>1.00 sensitivity</b>. We independently inspected the reference model and established the exact mathematical explanation:",
        styles['Body']
    ))
    story.append(Paragraph(
        "1. <b>Target Feature Distribution:</b> <code>hours-per-week</code> spans $[1, 99]$ with mean $40.42$, but over $60\\%$ of samples sit exactly at $40$, yielding $Q_{0.25} = 40.0, Q_{0.75} = 45.0, \\text{IQR} = 5.0$. A moderate shift of $1.0 \\times \\text{IQR}/1.349$ corresponds to only $+3.7$ hours.<br/>"
        "2. <b>Model Weight Amplification:</b> In the reference Logistic Regression model, <code>hours-per-week</code> carries a large standardized coefficient of $\\beta = +0.4993$ (with intercept $-1.4832$).<br/>"
        "3. <b>Exact Measured Metrics:</b><br/>"
        "&nbsp;&nbsp;&bull; <i>Mild:</i> Feature PSI = 0.0058 (&lt; 0.10), Pred-Prop PSI = 0.0132 (&lt; 0.015). Both miss.<br/>"
        "&nbsp;&nbsp;&bull; <i>Moderate:</i> Feature PSI = 0.0409 (&lt; 0.10 &rArr; <b>Missed</b>), Pred-Prop PSI = 0.0298 (&ge; 0.015 &rArr; <b>Detected</b>).<br/>"
        "&nbsp;&nbsp;&bull; <i>Severe:</i> Feature PSI = 0.1846 (&ge; 0.10 &rArr; <b>Detected</b>), Feature KL = 0.0964 (&lt; 0.10 &rArr; <b>Missed</b>), Pred-Prop PSI = 0.0829 (&ge; 0.015 &rArr; <b>Detected</b>).<br/>"
        "4. <b>Theoretical Conclusion:</b> Feature-level quantile monitors fail when feature variance is tightly concentrated, because small shifts do not cross bin boundaries. However, model inference projects input features through model weights, acting as an <b>amplifier</b> in predicted probability space.",
        styles['Body']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("8.2 Subspace Dilution &amp; Heavy-Tail Breakdown in PCA Centroid Monitoring", styles['H2']))
    story.append(Paragraph(
        "Why did PCA centroid distance achieve an AUC of only <b>0.54</b> on Give Me Some Credit, but <b>&gt;0.95</b> on Taiwan and Adult? In Give Me Some Credit, <code>DebtRatio</code> has extreme heavy tails (std dev &approx; 150, 99th percentile = 811.0). A $+1.0 \\times \\text{IQR}$ shift represents only $0.005$ standard deviations, which is completely lost in the 3D PCA subspace relative to baseline noise ($0.038\\text{--}0.054$). Furthermore, with an uncalibrated static threshold of $1.0$, sensitivity is $0.00$ because post-onset centroid distance never reaches $1.0$ (peaking at $0.04$ on Credit, $0.55$ on Taiwan, and $0.35$ on Adult). Subspace monitors must always be calibrated to empirical baseline noise floors ($\\mu + 3\\sigma$).",
        styles['Body']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("8.3 Wilson Score Interval vs. Wald Normal Approximation", styles['H2']))
    story.append(Paragraph(
        "For a binomial proportion $\\hat{p} = k / n$ with $n = 20$ repetitions:<br/>"
        "• <b>Wald Approximation:</b> $\\hat{p} \\pm 1.96 \\sqrt{\\hat{p}(1-\\hat{p})/n}$. When $\\hat{p} = 1.0$ (20/20 detections), the margin collapses to 0: $[1.0, 1.0]$. This is statistically invalid for small samples.<br/>"
        "• <b>Wilson Score Interval:</b> Inverts the score test: $[0.8389, 1.0000]$. It honestly reports that 20 out of 20 observed successes only guarantees a true population sensitivity $\\ge 83.9\\%$ at the 95% confidence level.",
        styles['Body']
    ))
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # SOFTWARE ARCHITECTURE & REPRODUCIBILITY
    # -----------------------------------------------------------------------
    story.append(Paragraph("9. Software Engineering, Dashboard &amp; Reproducibility", styles['H1']))
    story.append(Paragraph(
        "The repository is structured as a production-grade Python package with full CI testing, containerization, and interactive telemetry:",
        styles['Body']
    ))
    story.append(Paragraph(
        "• <b>Interactive Telemetry Dashboard (<code>drift_monitor/dashboard_app.py</code>):</b> A clean, restrained 2-tab Streamlit application live at <code>http://localhost:8501</code>. Features a 520px compact 2-panel Plotly layout (Input Drift on Top, Model Degradation on Bottom), instant 1-line plain-English diagnoses, and sub-50ms reactive caching via <code>@st.cache_data</code> and <code>@st.cache_resource</code>.<br/>"
        "• <b>Comprehensive Test Suite (23 Tests in <code>tests/test_core.py</code>):</b> Fully automated unit tests covering Kolmogorov-Smirnov baseline comparisons, gold-standard hand-labelled precision/recall checks (Precision=1.00, Recall&ge;0.90), deduplicated quantile handling, and streaming edge cases.<br/>"
        "• <b>Docker Containerization (<code>Dockerfile</code>):</b> Built on <code>python:3.12-slim</code> with exact pinned dependencies, eliminating 'works on my machine' environmental drift.<br/>"
        "• <b>Deterministic Reproduction:</b> Executing <code>python reproduce_all.py</code> regenerates every report asset, recommendation table, and trade-off curve across all three datasets from scratch.",
        styles['Body']
    ))
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # THE MASTER VIVA DEFENSE: 25 QUESTIONS & ANSWERS
    # -----------------------------------------------------------------------
    story.append(PageBreak())  # Clean break for the Viva Defense Masterclass
    story.append(Paragraph("10. The Master Viva Defense: 25 High-Stakes Professor Questions &amp; Answers", styles['H1']))
    story.append(Paragraph(
        "This section prepares the team to defend every theoretical nuance, architectural design decision, empirical edge case, and statistical trade-off before an adversarial panel or course professor.",
        styles['Body']
    ))
    story.append(Spacer(1, 4))

    viva_qa = [
        # GROUP A: Theoretical Statistics & Mathematics
        ("Q1: What is the exact mathematical relationship between PSI and Kullback-Leibler (KL) divergence?",
         "PSI is mathematically identical to symmetric Jeffreys Divergence: PSI(P, Q) = D_KL(P || Q) + D_KL(Q || P). Under a second-order Taylor series expansion around P &approx; Q, D_KL(P || Q) &approx; 0.5 * &Sigma; (delta_b^2 / Q_b), while PSI &approx; &Sigma; (delta_b^2 / Q_b). Therefore, PSI &approx; 2 * D_KL. If an engineering team sets equal thresholds (e.g., 0.10) for both metrics, KL divergence is subjected to an effective threshold of 0.20 on symmetric divergence, making it twice as strict and incapable of triggering before PSI."),

        ("Q2: Why is unsupervised feature monitoring structurally incapable of detecting pure concept drift?",
         "Pure concept drift is defined as a change in the posterior class probability P(Y | X) while the marginal input feature distribution P(X) remains strictly invariant. Because unsupervised monitors (PSI, KL, PCA centroid distance, Wasserstein distance) operate exclusively on samples of X without access to target labels Y, the empirical distribution of X remains identical to baseline within sampling error. Mathematically, no function f(X) can detect a change in a conditional distribution P(Y | X) when P(X) does not change. Ground-truth feedback—even delayed or sparse—is an absolute mathematical requirement."),

        ("Q3: Why did you use IQR / 1.349 instead of sample standard deviation for shift scaling?",
         "Sample standard deviation is non-robust and easily dominated by extreme outliers. In Give Me Some Credit, DebtRatio contains data-entry errors exceeding 100,000 against a median of 0.3, inflating the standard deviation by over 100x. Shifting by '1.0 sigma' would inject an unrealistically massive perturbation. For a Gaussian distribution, IQR = 2 * Phi^-1(0.75) * sigma &approx; 1.34898 * sigma. Therefore, IQR / 1.349 is the normal-equivalent standard deviation. It ignores values beyond the 25th and 75th percentiles, remaining stable in the presence of heavy-tailed financial anomalies."),

        ("Q4: Why does the performance monitor use the Brier score instead of Log Loss or ROC-AUC?",
         "Log loss (-ln p) is unbounded; a single confident misclassification in a small batch produces an infinite loss spike, causing spurious alerts. ROC-AUC measures ranking discrimination but is totally insensitive to probability calibration: if predicted probabilities double while rank ordering is preserved, ROC-AUC is unchanged. The Brier score (mean squared error between predicted probabilities and binary labels) is strictly proper, bounded in [0, 1], decomposes into calibration and refinement, and exhibits well-behaved variance on small 20% labeled trickle samples."),

        ("Q5: Why did you report Wilson score confidence intervals instead of standard Wald normal intervals?",
         "For a binomial proportion p_hat = k / n with n = 20 repetitions, the Wald interval p_hat &plusmn; 1.96 * sqrt(p(1-p)/n) collapses to [1.0, 1.0] whenever all 20 runs succeed. Claiming zero uncertainty with 20 runs is statistically indefensible. The Wilson score interval inverts the score test, correctly yielding [0.8389, 1.0000] for 20/20 successes. It honestly conveys that 20 out of 20 empirical detections guarantees a true population sensitivity of at least 83.9% at the 95% confidence level."),

        # GROUP B: Detection Mechanics, Quantiles & Binning
        ("Q6: How does your binning strategy handle zero-inflated or discrete features?",
         "Continuous features often exhibit heavy zero-inflation (e.g., capital-gain in Adult Income is 0 for 75% of rows). Naive quantile binning via np.quantile produces identical bin edges (e.g., [0, 0, 0, 0]), causing np.histogram to collapse bins or divide by zero. Our implementation applies np.unique to quantile edges. If fewer than 2 unique edges remain (degenerate feature), it falls back to equal-width binning between min and max. Furthermore, Laplace smoothing with epsilon = 10^-4 is added to all bins before normalizing, guaranteeing non-zero counts."),

        ("Q7: Why did PCA centroid distance fail on Give Me Some Credit (AUC 0.54) while succeeding on Taiwan (>0.95)?",
         "Two structural reasons: First, subspace dilution. When covariate shift affects a single feature, the Euclidean displacement in latent space is scaled by the sum of squared eigenvector loadings for that feature (&Sigma; W_j,k^2 &le; 0.15), diluting the signal across unshifted dimensions. Second, extreme heavy tails in DebtRatio (std dev &approx; 150) dominated the covariance matrix even after Winsorization, making the baseline centroid noise floor (0.038-0.054) large relative to the injected shift (0.005 std dev). On Taiwan, features are normalized, correlated, and bounded, allowing PCA to preserve coherent variance."),

        ("Q8: How does Predicted-Proportion Divergence (`pred_prop_psi`) work without ground truth labels?",
         "The model itself acts as a feature extractor and dimensionality reducer. A frozen reference model predicts calibrated probabilities p_hat = f_ref(x) on streaming batches. These probabilities are binned into 10 uniform buckets in [0, 1] and compared against reference prediction frequencies using PSI. When label shift occurs, the shift in marginals alters the distribution of p_hat. Crucially, on Adult Income, pred_prop_psi detected moderate covariate shift on hours-per-week where feature PSI failed, because the model's large coefficient (&beta; = +0.4993) amplified the small input shift into output probability space."),

        ("Q9: What is the purpose of the Kolmogorov-Smirnov (KS) test in your test suite?",
         "The two-sample KS test provides a non-parametric mathematical baseline that evaluates whether streaming samples originate from the same continuous distribution as the reference set by measuring the maximum vertical distance between empirical CDFs. To prevent type-I error inflation across d multiple feature tests, we apply Bonferroni correction (&alpha;_adj = 0.05 / d). It validates that our custom PSI/KL implementations align with classical asymptotic hypothesis tests on clean continuous variables."),

        ("Q10: What is the exact mathematical definition of detection lag in your benchmark?",
         "Detection lag is defined as the number of elapsed streaming batches between the start of injected shift (onset_batch = 10) and the first batch where an alert is officially confirmed by the debounce filter: Lag = t_alert - onset_batch. Under a k = 2 debounce filter, the theoretical minimum lag is 1 batch (alert confirmed at batch 11). If an injected shift is never detected across the stream, the run is flagged as undetected (lag = None) and excluded from the conditional mean lag calculation."),

        # GROUP C: Systems Architecture, Latency & MLOps Trade-offs
        ("Q11: How did you optimize PSI/KL calculation to achieve 1.3 ms per batch?",
         "In naive implementations, quantile bin edges and reference histograms are recomputed dynamically every batch, requiring O(N_ref log N_ref) sorting operations. In our optimized PSIKLMonitor, reference bin edges and reference bucket percentages Q_b are precomputed and cached once during the fit() phase. On incoming streaming batches, only a single 1D np.digitize() call and histogram count is performed, reducing per-batch latency from 23.0 ms to 1.32 ms (~17x speedup)."),

        ("Q12: What is the debounce confirmation filter, and why is k=2 optimal?",
         "Streaming batches in production have finite sample sizes (e.g., 1,000 to 4,000 rows), causing stochastic metric fluctuations. A confirmation rule requiring k consecutive batches above threshold acts as a temporal low-pass filter. In our empirical debounce frontier (Curve 2), k = 1 suffers a 34% false alarm rate with 0 lag. Setting k = 2 slashes false alarms to 14% with only 0.6 batches of lag. In production, this prevents on-call engineers from being woken up by transient batch noise while detecting true persistent drift within ~1-2 batches."),

        ("Q13: How does your system operate under realistic delayed label feedback?",
         "In commercial applications (e.g., credit lending), default labels take 90+ days to resolve. Our architecture decouples input monitoring from performance monitoring. Tier 1 monitors inputs and model outputs instantaneously (<15 ms) on 100% of unlabelled traffic. Tier 2 ingests an asynchronous trickle (20% of resolved labels) via an event-driven join buffer. The Brier score performance monitor tracks calibration drift exclusively on this verified trickle, alerting if empirical degradation exceeds &tau; = 0.025."),

        ("Q14: How does the diagnostic decision engine (`classify_shift`) differentiate between shift types?",
         "Tier 3 evaluates the combination of fired alerts: (1) If feature PSI/KL fires while Brier score is stable, it diagnoses Covariate Shift. (2) If both pred_prop_psi and Brier degradation fire, it diagnoses Label Shift (prior probability change). (3) If Brier degradation fires while input feature distributions are completely stationary, it diagnoses Pure Concept Drift. (4) If pred_prop_psi fires alone while input PSI is quiet, it flags a high-weight covariate or subtle label shift."),

        ("Q15: What is the total memory footprint and CPU throughput of Drift Monitor in production?",
         "In our systems benchmark (`benchmark.py`), the full 5-detector monitoring suite processes a 4,000-row batch in 51.0 ms on Give Me Some Credit, 20.1 ms on Taiwan, and 15.2 ms on Adult Income. Total serialized memory footprint for the entire monitoring pipeline and reference models is 1.26 MB. On a single commodity CPU, this delivers a throughput of ~20 to 65 streaming batches per second, providing massive headroom for hourly or daily production ingestion pipelines."),

        # GROUP D: Empirical Anomalies, Dataset Peculiarities & Forensic Audits
        ("Q16: Explain the Adult Census Income Covariate Inversion in detail.",
         "In Adult Census Income, covariate shift injected into hours-per-week produced 0.00 sensitivity on PSI/KL at mild and moderate levels, but 1.00 sensitivity on pred_prop_psi. The feature hours-per-week has 60% of values sitting exactly at 40, resulting in a tiny IQR of 5.0 across a [1, 99] range. A moderate shift of +3.7 hours was too small to cross quantile boundaries (PSI = 0.0409 < 0.10). However, the model assigned hours-per-week a massive weight (&beta; = +0.4993). The input shift pushed borderline instances across the decision boundary, causing model output probabilities to diverge (pred_prop_psi = 0.0298 &ge; 0.015). The model acted as an amplifier for high-leverage input drift."),

        ("Q17: What was the '0.95 detection lag' anomaly in early benchmark runs, and how was it resolved?",
         "Under a k = 2 debounce filter, detection lag should mathematically be &ge; 1.0 batch. However, early benchmark summaries reported a mean lag of 0.95 batches. Forensic code inspection revealed that if batch 9 (the last pre-onset batch) experienced a stochastic noise spike above threshold, consecutive_counter was set to 1. When true drift began at batch 10, the counter hit 2 immediately, registering a lag of 10 - 10 = 0. In 1 out of 20 runs, this pre-onset noise primed the counter, yielding (19*1 + 1*0)/20 = 0.95. We resolved this by strictly resetting the debounce streak counter to 0 at the onset boundary."),

        ("Q18: What is the confounding issue between concept drift injection and marginal label rates?",
         "Standard synthetic concept drift inverts labels (1 - Y) in a feature band. In imbalanced datasets like Give Me Some Credit (6.7% defaults), 93% of instances in any band are negative (Y=0). Inverting labels turns almost all of them into positives, inflating the batch default rate from 6.7% to 14.2% (moderate) and 20.5% (severe). Thus, standard benchmarks evaluate a compound shift: concept drift + massive label inflation. When we implemented a 'pure swap' preserving P(Y)=6.7%, low default counts (&sim;30 in a batch) meant only 6 positive instances appeared in a 20% trickle, moving global Brier score by only 0.002 (&lt; 0.025 threshold). Global metrics cannot detect localized concept drift in rare classes; slice-based monitoring is required."),

        ("Q19: How did you ensure zero data leakage between reference and streaming evaluation pools?",
         "In early prototypes, reference data was built from the first 10 batches of a global DataFrame shuffled with seed 0, and streaming batches were created by reshuffling the same DataFrame with new seeds, causing a 33% sample overlap. We resolved this by implementing strict physical partitioning: the historical dataset is permanently split into a reference pool (batches 0-9) and a completely held-out streaming evaluation pool (batches 10-29). No sample evaluated in the stream was ever seen during reference fitting or PCA training."),

        ("Q20: Why did the ROC curve calculation require disjoint 50/50 stream splits?",
         "Early ROC curve scripts evaluated streaming batches against the same reference window from which baseline noise was calculated, producing optimistic bias and tie-sorting distortions. We refactored ROC generation to evaluate on completely held-out, disjoint 50/50 streaming partitions. Across 1,080 runs, this established true unbiased operational ROC curves: psi_agg (AUC 0.88), kl_agg (AUC 0.88), pred_prop_psi (AUC 0.83), and concept_signal (AUC 0.76)."),

        # GROUP E: Production Hardening, Edge Cases & Retraining Policies
        ("Q21: When Drift Monitor raises an alert, what automated actions should an MLOps pipeline take?",
         "The pipeline must never blindly trigger model retraining. If Covariate Shift is diagnosed, retraining is inappropriate because P(Y|X) is intact; engineers should inspect data upstream (e.g., schema changes, tracking bugs) or apply density-ratio sample re-weighting. If Label Shift is diagnosed, retraining the full model is wasteful; engineers should apply Bayesian prior odds correction or update the logistic intercept &beta;_0. Only when Concept Drift is diagnosed should the pipeline trigger model retraining on fresh labeled data or fall back to human review."),

        ("Q22: How would you extend Drift Monitor to handle high-cardinality categorical features?",
         "For categorical features (e.g., postal codes, merchant IDs), quantile binning is meaningless. The system should be extended with Categorical PSI or Jensen-Shannon Divergence over discrete category frequency tables: PSI_cat = &Sigma; (P_c - Q_c) ln(P_c / Q_c). For high-cardinality variables with unseen categories, an 'infrequent / other' bucket must be reserved, and new category emergence should be tracked via set difference cardinality (|Categories_stream \\ Categories_ref|)."),

        ("Q23: How would you adapt Drift Monitor for cyclic or seasonal distribution shifts?",
         "Static reference windows fail in retail and travel where weekend vs. weekday or holiday traffic creates natural cyclic shifts. For seasonal environments, Drift Monitor should implement Adaptive Sliding Windows (e.g., ADWIN) or time-of-week reference conditioning, where streaming Monday batches are compared against historical Monday reference distributions rather than a global stationary baseline."),

        ("Q24: What is the difference between data drift, model drift, and concept drift?",
         "• Data Drift (Covariate Shift): Changes in input distribution P(X) while the real-world relationship P(Y|X) is stationary.<br/>• Model Drift: Degradation in model inference outputs or calibration (e.g., pred_prop_psi) due to either data drift or internal numerical discrepancies.<br/>• Concept Drift: Fundamental changes in the real-world relationship between inputs and targets P(Y|X). Inputs look identical, but customer behavior, fraud tactics, or economic laws have changed."),

        ("Q25: What is the final takeaway from this DS635 term project?",
         "Distribution shift is not a single failure mode, and monitoring cannot be solved by a single tool or metric. Real-world ML systems engineering demands a defence-in-depth architecture: lightweight unsupervised monitors for rapid input gating, frozen model-output tracking to leverage model weights as drift amplifiers, and asynchronous performance verification on verified label trickles. Grounded in mathematical proofs, empirical trade-offs, and temporal debouncing, Drift Monitor proves that deployed machine learning models can be reliably protected against silent failure.")
    ]

    for q_idx, (q_text, a_text) in enumerate(viva_qa, 1):
        qa_flow = []
        qa_flow.append(Paragraph(f"<b>{q_text}</b>", styles['QStyle']))
        qa_flow.append(Paragraph(a_text, styles['AStyle']))
        if q_idx % 5 == 0 and q_idx != len(viva_qa):
            qa_flow.append(Spacer(1, 4))
            qa_flow.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceBefore=4, spaceAfter=6))
        story.append(KeepTogether(qa_flow))

    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # DEFENSE CHECKLIST & CONCLUSION
    # -----------------------------------------------------------------------
    story.append(Paragraph("11. Final Viva Defense Checklist &amp; Presentation Strategy", styles['H1']))
    story.append(Paragraph(
        "Before presenting to the panel, every team member must verify the following ten operational facts:",
        styles['Body']
    ))
    checklist_items = [
        "<b>1. Project Name &amp; Team:</b> Drift Monitor (Branch <code>Srishti</code>); Sanjana Nathani, Nikita Sharma, Anish Karmakar, Srishti Lamba.",
        "<b>2. Three Failure Modes:</b> Covariate Shift (P(X) changes), Label Shift (P(Y) changes), Concept Drift (P(Y|X) changes).",
        "<b>3. The Five Detectors:</b> <code>psi_agg</code>, <code>kl_agg</code>, <code>centroid_dist</code>, <code>pred_prop_psi</code>, <code>concept_signal</code>.",
        "<b>4. Mathematical Equivalence:</b> PSI is symmetric Jeffreys Divergence: PSI = D_KL(P||Q) + D_KL(Q||P) &approx; 2 * D_KL.",
        "<b>5. Debounce Frontier:</b> k=1 has 34% FAR; k=2 cuts FAR to 14% at 0.6 lag; k&ge;4 eliminates false alarms (0% FAR).",
        "<b>6. Adult Income Anomaly:</b> <code>hours-per-week</code> IQR=5 misses on PSI (0.0409 &lt; 0.10); <code>pred_prop_psi</code> catches it via &beta;=+0.4993 weight amplification.",
        "<b>7. PCA Subspace Dilution:</b> Localized single-feature shifts dilute under Euclidean norm (||&Delta;z|| = |&Delta;&mu;| * &radic;(&Sigma; w_j^2)).",
        "<b>8. Brier Score Superiority:</b> Proper scoring rule, bounded [0, 1], stable on 20% trickle; avoids log loss infinite spikes.",
        "<b>9. Latency Benchmark:</b> PSI/KL runs in 1.3 ms; full combined pipeline runs in 51.0 ms (~20 batches/sec, 1.26 MB RAM).",
        "<b>10. Reproducibility:</b> 23 unit tests pass; <code>python reproduce_all.py</code> regenerates all combined tables and trade-off curves."
    ]
    for c_item in checklist_items:
        story.append(Paragraph(c_item, styles['Bullet']))

    story.append(Spacer(1, 12))
    story.append(make_callout(
        "<b>Mission Accomplished:</b> Drift Monitor provides an airtight, mathematically sound, experimentally validated, and reproducible system for detecting silent distribution shift. Treat this document as your project bible during presentation preparation and viva defense.",
        title="READY FOR SUBMISSION &amp; VIVA DEFENSE",
        border_color="#10b981",
        bg_color="#ecfdf5"
    ))

    # Build Document
    print(f"Building {pdf_filename}...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Success! Generated {pdf_filename} ({os.path.getsize(pdf_filename)} bytes)")


if __name__ == "__main__":
    build_pdf()
