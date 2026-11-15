"""
generate_exhaustive_master_bible.py — Builds the exhaustive, book-length Master Project Bible
and Viva Defense Masterclass for "Drift Monitor: Detecting Silent Distribution Shift in Deployed ML Models".

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
# Numbered Canvas for Dynamic Two-Pass Page Numbering & Running Header/Footer
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
        # Suppress running headers/footers on Page 1 (Title Cover Page)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#475569"))
            # Running Header
            self.drawString(40, 792 - 28, "DRIFT MONITOR: DETECTING SILENT DISTRIBUTION SHIFT — MASTER PROJECT BIBLE")
            self.setFont("Helvetica", 8)
            self.drawRightString(612 - 40, 792 - 28, "DS635 MLSE Term Project")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(40, 792 - 32, 612 - 40, 792 - 32)
            
            # Running Footer
            self.line(40, 36, 612 - 40, 36)
            self.drawString(40, 24, "Authors: Sanjana Nathani, Nikita Sharma, Anish Karmakar, Srishti Lamba | Branch: Srishti")
            self.drawRightString(612 - 40, 24, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def build_exhaustive_bible():
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

    # Base Colors
    c_primary = colors.HexColor("#0f172a")     # Slate 900
    c_navy = colors.HexColor("#1e3a8a")        # Blue 900
    c_secondary = colors.HexColor("#334155")   # Slate 700
    c_border = colors.HexColor("#cbd5e1")      # Slate 300

    styles = {
        'DocTitle': ParagraphStyle('DocTitle', fontName='Helvetica-Bold', fontSize=24, leading=28, textColor=c_primary, spaceAfter=6),
        'DocSubtitle': ParagraphStyle('DocSubtitle', fontName='Helvetica', fontSize=11, leading=15, textColor=c_secondary, spaceAfter=14),
        'MetaBox': ParagraphStyle('MetaBox', fontName='Helvetica', fontSize=8.5, leading=12, textColor=c_primary),
        'H1': ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=c_navy, spaceBefore=14, spaceAfter=6, keepWithNext=True),
        'H2': ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=11, leading=15, textColor=c_primary, spaceBefore=10, spaceAfter=4, keepWithNext=True),
        'H3': ParagraphStyle('H3', fontName='Helvetica-Bold', fontSize=9.5, leading=13, textColor=c_secondary, spaceBefore=6, spaceAfter=3, keepWithNext=True),
        'Body': ParagraphStyle('Body', fontName='Helvetica', fontSize=8.5, leading=12, textColor=colors.HexColor("#1e293b"), spaceBefore=2, spaceAfter=4),
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
    # COVER PAGE / TITLE BLOCK
    # -----------------------------------------------------------------------
    story.append(Paragraph("DRIFT MONITOR: A SYSTEM THAT NOTICES", styles['DocTitle']))
    story.append(Paragraph("Detecting Silent Distribution Shift in Deployed ML Models — The Master Project Bible, Theoretical Framework, Empirical Evidence &amp; Viva Defense Dossier", styles['DocSubtitle']))
    
    meta_data = [
        [
            Paragraph("<b>Course:</b> DS635 Machine Learning Systems Engineering (MLSE)<br/><b>Academic Term:</b> Fall 2026 / Term Project Defense<br/><b>Branch:</b> <code>Srishti</code> (Protected Production Worktree)<br/><b>Docker Environment:</b> <code>python:3.12-slim</code> (Pinned Versions)", styles['MetaBox']),
            Paragraph("<b>Project Team Members:</b><br/>• Sanjana Nathani &nbsp;&nbsp;&nbsp;&nbsp;• Nikita Sharma<br/>• Anish Karmakar &nbsp;&nbsp;&nbsp;• Srishti Lamba<br/><b>GitHub Repository:</b> <code>nikitasharma120/Drift_Monitor</code><br/><b>Live Dashboard:</b> <code>http://localhost:8501</code>", styles['MetaBox'])
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
    # CHAPTER 1: THE SILENT FAILURE DILEMMA
    # -----------------------------------------------------------------------
    story.append(Paragraph("1. The Production ML Dilemma: Silent Failure in Deployed Systems", styles['H1']))
    story.append(Paragraph(
        "<b>1.1 Classical Software Engineering vs. Machine Learning Operations (MLOps):</b><br/>"
        "In classical software engineering, functional defects declare themselves through explicit, observable signals: unhandled null pointer exceptions, HTTP 500 error responses, database deadlocks, network timeouts, or process segmentation faults. When an API server crashes or throws an exception, site reliability engineering (SRE) tools immediately trigger alerts, route traffic to fallback nodes, and log stack traces. "
        "In stark contrast, deployed machine learning systems fail <i>silently</i>. When the real-world statistical distribution of incoming data diverges from the training baseline, the model inference service does not crash, throw an error, or log a failure. It continues accepting payloads, executing forward passes, returning HTTP 200 OK statuses, and outputting syntactically valid JSON prediction probabilities with low latency. However, its underlying mathematical mapping from inputs to semantic reality is shattered. Standard DevOps telemetry—such as CPU utilization, query per second (QPS), memory consumption, and network I/O—is fundamentally blind to statistical decay.",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>1.2 The Ground Truth Paradox: Label Delays, Costs, and Verification Horizons:</b><br/>"
        "If ground truth target labels ($Y$) were available instantaneously for every inference request, detecting model degradation would be trivial: one would simply monitor real-time loss, accuracy, or Brier score. However, in enterprise machine learning deployments, ground truth labels arrive with significant delay, at high financial cost, or never at all:<br/>"
        "• <i>Credit Scoring &amp; Lending:</i> Whether a loan applicant defaults within a 24-month horizon cannot be known for up to two full years.<br/>"
        "• <i>Fraud Detection:</i> Chargebacks and merchant dispute settlements take 30 to 90 days to appear in payment ledgers.<br/>"
        "• <i>Medical Diagnostics:</i> Definitive biopsy and clinical progression outcomes require weeks of laboratory analysis.<br/>"
        "• <i>Human Annotation Budget:</i> Labeling streaming data requires expensive manual domain expert review; companies can typically afford to verify only a tiny fraction (10% to 20%) of production traffic.",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>1.3 The Input-Blindness Principle: Why Unsupervised Monitoring Alone Fails:</b><br/>"
        "To circumvent the label delay problem, industry practice frequently relies on unsupervised input feature monitoring—computing statistical distances (such as Population Stability Index or Wasserstein Distance) between reference feature distributions and streaming feature distributions. While input monitoring is instantaneous and requires zero label budget, it suffers from a fundamental mathematical vulnerability: <b>The Input-Blindness Principle</b>. Pure concept drift ($P(Y|X)$ changes while $P(X)$ remains stationary) produces zero detectable displacement in input feature space. An architecture restricted exclusively to input monitoring will remain 100% oblivious while customer churn behavior, competitor actions, or economic policies invalidate the decision boundary.",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>1.4 The Core Thesis of Drift Monitor:</b><br/>"
        "No single statistical detector or isolated telemetry stream can protect a deployed machine learning service. A production-ready monitoring architecture must deploy a <b>3-Tier Defense-in-Depth Pipeline</b> that integrates: (1) high-speed unsupervised input feature gating ($P(X)$), (2) frozen reference model predicted-proportion tracking ($P(\\hat{Y})$) to exploit model weight amplification, and (3) an asynchronous supervised verification gate operating over a realistic delayed 20% labeled trickle ($P(Y|X)$), all governed by a 2-stage temporal debounce confirmation filter to eliminate spurious engineering alerts.",
        styles['Body']
    ))
    story.append(make_callout(
        "<b>THE OPERATIONAL IMPERATIVE:</b> Detecting distribution shift is not merely about finding that something moved; it is about diagnosing <i>which component</i> of the joint distribution moved so that engineering teams take the correct remediation action (pipeline fix vs. threshold recalibration vs. full model retraining).",
        title="SYSTEM ARCHITECTURE MOTIVATION",
        border_color="#0284c7",
        bg_color="#f0f9ff"
    ))
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # CHAPTER 2: MATHEMATICAL TAXONOMY OF DISTRIBUTION SHIFT
    # -----------------------------------------------------------------------
    story.append(Paragraph("2. Mathematical Foundations &amp; Taxonomy of Distribution Shift", styles['H1']))
    story.append(Paragraph(
        "Let $X \\in \\mathbb{R}^d$ represent the random vector of observable continuous and discrete input features, and let $Y \\in \\{0, 1\\}$ denote the true ground truth downstream label. The joint probability distribution $P(X, Y)$ completely defines the statistical universe in which the model operates. By Bayes' theorem and the chain rule of probability, this joint distribution admits two dual factorizations:",
        styles['Body']
    ))
    story.append(Paragraph(
        "<font face='Courier' size='8'><b>P(X, Y) = P(X) &middot; P(Y | X) &nbsp;= &nbsp;P(Y) &middot; P(X | Y)</b></font>",
        styles['BodyBold']
    ))
    story.append(Paragraph(
        "A deployed model trained on historical data $D_{\\text{ref}} \\sim P_{\\text{ref}}(X, Y)$ is exposed to an operational data stream $D_{\\text{stream}} \\sim P_{\\text{stream}}(X, Y)$. Distribution shift occurs if and only if $P_{\\text{stream}}(X, Y) \\neq P_{\\text{ref}}(X, Y)$. The nature, diagnostic signature, and remediation strategy depend strictly on which term in the factorization diverges:",
        styles['Body']
    ))

    shift_deep_data = [
        [Paragraph("Failure Mode", styles['TH']), Paragraph("Formal Definition", styles['TH']), Paragraph("Invariant Property", styles['TH']), Paragraph("Statistical Signature", styles['TH']), Paragraph("Operational Impact &amp; Consequence", styles['TH'])],
        [
            Paragraph("<b>Covariate Shift</b><br/>(Data Drift)", styles['TDBold']),
            Paragraph("P<sub>stream</sub>(X) &ne; P<sub>ref</sub>(X)", styles['TD']),
            Paragraph("P(Y | X) = P<sub>ref</sub>(Y | X)<br/>(Decision rule intact)", styles['TD']),
            Paragraph("&bull; PSI<sub>agg</sub> &ge; 0.10<br/>&bull; KL<sub>agg</sub> &ge; 0.10<br/>&bull; Brier degradation = 0.00", styles['TD']),
            Paragraph("Inputs enter novel, unobserved regions of feature space. If model extrapolation is poor, error rates increase despite the true underlying concept remaining unchanged.", styles['TD'])
        ],
        [
            Paragraph("<b>Label Shift</b><br/>(Prior Shift)", styles['TDBold']),
            Paragraph("P<sub>stream</sub>(Y) &ne; P<sub>ref</sub>(Y)", styles['TD']),
            Paragraph("P(X | Y) = P<sub>ref</sub>(X | Y)<br/>(Conditionals intact)", styles['TD']),
            Paragraph("&bull; pred_prop_psi &ge; 0.015<br/>&bull; &Delta;Brier &ge; 0.025<br/>&bull; Marginal P(X) shifts slightly", styles['TD']),
            Paragraph("The base rate of the target class changes (e.g., default rate jumps from 6% to 20%). The model under-predicts positive instances because its classification threshold was tuned for the historical prior.", styles['TD'])
        ],
        [
            Paragraph("<b>Concept Drift</b><br/>(Semantic Shift)", styles['TDBold']),
            Paragraph("P<sub>stream</sub>(Y | X) &ne; P<sub>ref</sub>(Y | X)", styles['TD']),
            Paragraph("P(X) = P<sub>ref</sub>(X)<br/>(Inputs stationary)", styles['TD']),
            Paragraph("&bull; PSI<sub>agg</sub> &lt; 0.05 (Quiet)<br/>&bull; pred_prop_psi quiet<br/>&bull; &Delta;Brier &ge; 0.025 (Fires)", styles['TD']),
            Paragraph("The fundamental relationship between features and outcomes decays. Identical customers behave differently due to external economic or behavioral shocks. Model predictions become fundamentally invalid.", styles['TD'])
        ]
    ]
    t_shift_deep = Table(shift_deep_data, colWidths=[usable_width*0.16, usable_width*0.20, usable_width*0.20, usable_width*0.22, usable_width*0.22])
    t_shift_deep.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_shift_deep)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # CHAPTER 3: THE DETECTION ENGINE — DERIVATIONS & CODE
    # -----------------------------------------------------------------------
    story.append(Paragraph("3. The Detection Engine: Mathematical Derivations &amp; Algorithmic Design", styles['H1']))
    story.append(Paragraph(
        "Drift Monitor designs, implements, and benchmarks five primary detection mechanisms and a classical two-sample hypothesis testing baseline. This section details their full mathematical mechanics:",
        styles['Body']
    ))

    story.append(Paragraph("3.1 Population Stability Index (PSI): Foundations &amp; Jeffreys Divergence Equivalence", styles['H2']))
    story.append(Paragraph(
        "The Population Stability Index (PSI) is widely regarded as the gold standard in retail credit risk modeling (Federal Reserve SR 11-7 regulatory guidance). For continuous feature $X_j$, reference samples are partitioned into $B = 10$ quantile bins $[e_0, e_1], [e_1, e_2], \\dots, [e_{B-1}, e_B]$ such that reference empirical frequencies are approximately uniform: $Q_b \\approx 1/B$. In the streaming batch, empirical frequencies $P_b$ are counted:",
        styles['Body']
    ))
    story.append(Paragraph(
        "<font face='Courier' size='8'>P<sub>b</sub> = (1 / |D<sub>stream</sub>|) &Sigma; &Iopf;(x<sub>i</sub> &in; bin<sub>b</sub>), &nbsp;&nbsp;&nbsp;&nbsp; Q<sub>b</sub> = (1 / |D<sub>ref</sub>|) &Sigma; &Iopf;(x<sub>i</sub> &in; bin<sub>b</sub>)</font>",
        styles['Body']
    ))
    story.append(Paragraph(
        "To prevent division by zero or infinite logarithms when streaming bins contain zero observations, Laplace smoothing with regularizer $\\epsilon = 10^{-4}$ is applied prior to normalization: $\\tilde{P}_b = (P_b + \\epsilon) / (1 + B\\epsilon)$ and $\\tilde{Q}_b = (Q_b + \\epsilon) / (1 + B\\epsilon)$.",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>The Mathematical Proof of Equivalence to Symmetric Jeffreys Divergence:</b><br/>"
        "Many engineering articles incorrectly label PSI as an empirical rule of thumb disjoint from information theory. We prove that PSI is mathematically identical to the symmetrized Kullback-Leibler divergence (Jeffreys Divergence, $J(P, Q)$):",
        styles['Body']
    ))
    story.append(Paragraph(
        "<font face='Courier' size='7.5'>"
        "PSI(P, Q) = &Sigma;<sub>b=1..B</sub> (P<sub>b</sub> - Q<sub>b</sub>) &middot; ln(P<sub>b</sub> / Q<sub>b</sub>)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= &Sigma;<sub>b=1..B</sub> P<sub>b</sub> &middot; ln(P<sub>b</sub> / Q<sub>b</sub>) - &Sigma;<sub>b=1..B</sub> Q<sub>b</sub> &middot; ln(P<sub>b</sub> / Q<sub>b</sub>)<br/>"
        "Using the logarithmic identity -ln(P<sub>b</sub> / Q<sub>b</sub>) = ln(Q<sub>b</sub> / P<sub>b</sub>):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= &Sigma;<sub>b=1..B</sub> P<sub>b</sub> &middot; ln(P<sub>b</sub> / Q<sub>b</sub>) + &Sigma;<sub>b=1..B</sub> Q<sub>b</sub> &middot; ln(Q<sub>b</sub> / P<sub>b</sub>)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= D<sub>KL</sub>(P || Q) + D<sub>KL</sub>(Q || P) &nbsp;&equiv; &nbsp;J(P, Q)"
        "</font>",
        styles['Code']
    ))
    story.append(Paragraph(
        "<b>Second-Order Taylor Series Expansion (The 2&times; Factor):</b><br/>"
        "Let $P_b = Q_b + \\delta_b$, where $\\sum_{b=1}^B \\delta_b = 0$. Expanding $\\ln(P_b / Q_b) = \\ln(1 + \\delta_b / Q_b)$ via Taylor series: $\\ln(1 + u) = u - \\frac{u^2}{2} + O(u^3)$. Substituting into KL divergence:",
        styles['Body']
    ))
    story.append(Paragraph(
        "<font face='Courier' size='7.5'>"
        "D<sub>KL</sub>(P || Q) = &Sigma; (Q<sub>b</sub> + &delta;<sub>b</sub>) [ (&delta;<sub>b</sub> / Q<sub>b</sub>) - 0.5 (&delta;<sub>b</sub><sup>2</sup> / Q<sub>b</sub><sup>2</sup>) + ... ]<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= &Sigma; [&delta;<sub>b</sub> - 0.5 (&delta;<sub>b</sub><sup>2</sup> / Q<sub>b</sub>) + (&delta;<sub>b</sub><sup>2</sup> / Q<sub>b</sub>) + ... ] &nbsp;&approx; &nbsp;0.5 &Sigma; (&delta;<sub>b</sub><sup>2</sup> / Q<sub>b</sub>)<br/>"
        "Now substituting into PSI:<br/>"
        "PSI(P, Q) = &Sigma; &delta;<sub>b</sub> &middot; [ (&delta;<sub>b</sub> / Q<sub>b</sub>) - 0.5 (&delta;<sub>b</sub><sup>2</sup> / Q<sub>b</sub><sup>2</sup>) + ... ] &nbsp;&approx; &nbsp;&Sigma; (&delta;<sub>b</sub><sup>2</sup> / Q<sub>b</sub>)<br/>"
        "Therefore: <b>PSI(P, Q) &approx; 2 &middot; D<sub>KL</sub>(P || Q)</b>"
        "</font>",
        styles['Code']
    ))
    story.append(make_callout(
        "<b>THE EQUAL-THRESHOLD TRAP:</b> In many industry monitoring scripts, engineers set the alert threshold to 0.10 for both PSI and KL. Because PSI &approx; 2 &times; KL, a threshold of 0.10 on KL actually demands a symmetric divergence of 0.20! Under equal thresholds, KL is twice as conservative and mathematically cannot fire before PSI during gradual drift.",
        title="EMPIRICALLY VERIFIED MATHEMATICAL TRAP",
        border_color="#f59e0b",
        bg_color="#fffbeb"
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.2 Latent Space Embedding Drift &amp; The Mathematical Proof of Subspace Dilution", styles['H2']))
    story.append(Paragraph(
        "To monitor multi-feature dependencies without exponential binning complexity, Drift Monitor projects features into a 3-dimensional subspace via Principal Component Analysis (PCA): $z_i = W^T (x_i - \\mu_{\\text{ref}})$, where $W \\in \\mathbb{R}^{d \\times 3}$ contains the top orthonormal eigenvectors of the reference covariance matrix $\\Sigma_{\\text{ref}}$. Centroid drift is defined as $d_{\\text{centroid}} = \\|\\bar{z}_{\\text{stream}} - \\bar{z}_{\\text{ref}}\\|_2$.",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>Mathematical Proof of Subspace Dilution:</b><br/>"
        "Why does PCA centroid distance fail on single-feature shifts? Suppose covariate shift occurs on target feature $j^*$: $\\Delta \\mu_x = [0, \\dots, \\Delta \\mu_{j^*}, \\dots, 0]^T$. The batch centroid displacement in 3D latent space is $\\Delta z = W^T \\Delta \\mu_x$. The Euclidean centroid distance is:",
        styles['Body']
    ))
    story.append(Paragraph(
        "<font face='Courier' size='8'>||&Delta;z||<sub>2</sub> = &radic;( &Sigma;<sub>c=1..3</sub> (W<sub>j*, c</sub> &middot; &Delta;&mu;<sub>j*</sub>)<sup>2</sup> ) &nbsp;= &nbsp;|&Delta;&mu;<sub>j*</sub>| &middot; &radic;( &Sigma;<sub>c=1..3</sub> W<sub>j*, c</sub><sup>2</sup> )</font>",
        styles['Body']
    ))
    story.append(Paragraph(
        "Because eigenvectors are orthonormal ($\\sum_{j=1}^d W_{j, c}^2 = 1$), the sum of squared loadings for any single feature across the first 3 components is typically $\\le 0.10$ to $0.20$. The localized shift is <b>attenuated by orthogonal unshifted dimensions</b>. Univariate PSI inspects feature $j^*$ directly without attenuation, achieving 100% sensitivity where PCA centroid distance collapses.",
        styles['Body']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.3 Predicted-Proportion Divergence (`pred_prop_psi`): Output Probability Tracking", styles['H2']))
    story.append(Paragraph(
        "Rather than monitoring raw features, `pred_prop_psi` tracks the output probability distribution of a frozen reference model: $\\hat{p}_i = f_{\\text{ref}}(x_i) = \\sigma(w^T x_i + b)$. It bins output probabilities into 10 uniform buckets in $[0, 1]$ and computes PSI against the reference prediction profile. Running in <b>14.6 ms/batch</b> with 0% label budget, it serves as an indispensable proxy for label shift and acts as an <b>amplifier</b> for high-leverage features.",
        styles['Body']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.4 Performance Monitoring (`concept_signal`): Brier Score Degradation", styles['H2']))
    story.append(Paragraph(
        "Drift Monitor samples an asynchronous 20% labeled trickle ($D_{\\text{trickle}}$) and tracks empirical degradation in the Brier score: $\\text{Brier}(D) = \\frac{1}{|D|} \\sum_{i=1}^{|D|} (\\hat{p}_i - y_i)^2$. The signal is $\\Delta \\text{Brier} = \\text{Brier}(D_{\\text{stream}}) - \\text{Brier}(D_{\\text{ref}})$.",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>Why Brier Score Outperforms Log Loss and ROC-AUC on Streaming Batches:</b><br/>"
        "• <i>Log Loss</i> is unbounded ($-\\ln(0) \\rightarrow \\infty$). In a small stream sample, a single confident false positive or mislabelled outlier produces an infinite loss spike, causing spurious alerts.<br/>"
        "• <i>ROC-AUC</i> evaluates ranking discrimination but is totally invariant to probability calibration. If predicted probabilities double while rank ordering is preserved, ROC-AUC remains unchanged while financial losses double.<br/>"
        "• <i>Brier Score</i> is strictly proper, bounded in $[0, 1]$, decomposes into calibration and refinement, and exhibits stable sampling variance on small trickle batches.",
        styles['Body']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.5 Temporal Debounce Confirmation Filter ($k=2$): Eliminating Alert Fatigue", styles['H2']))
    story.append(Paragraph(
        "Streaming batches in production have finite sample sizes ($N = 1,000$ to $4,000$), causing stochastic sampling variance. A single noisy batch can breach thresholds, causing false alarms. Drift Monitor enforces a temporal debounce confirmation filter requiring $k = 2$ consecutive batches above threshold before raising an alert:",
        styles['Body']
    ))
    story.append(Paragraph(
        "<font face='Courier' size='8'>Alert<sub>t</sub> = 1 &nbsp;&hArr; &nbsp;&forall; i &in; {0, ..., k-1}: &nbsp;Metric<sub>t - i</sub> &ge; &tau;</font>",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>The 0.95 Detection Lag Priming Bug:</b> In early versions, a pre-onset noise spike at batch 9 primed the counter to 1. When true drift arrived at batch 10, the counter hit 2 immediately, registering an apparent detection lag of $0.0$ batches (averaging to $0.95$ across 20 reps). In our production architecture, the debounce streak counter is strictly reset at the injection boundary, guaranteeing true post-onset latency measurement.",
        styles['Body']
    ))
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # CHAPTER 4: BENCHMARK CORPORA & DATA ENGINEERING
    # -----------------------------------------------------------------------
    story.append(Paragraph("4. Benchmark Corpora, Preprocessing &amp; Injection Engine", styles['H1']))
    story.append(Paragraph(
        "To prevent dataset-specific overfitting, all detectors were evaluated across three diverse corpora:",
        styles['Body']
    ))

    ds_full_data = [
        [Paragraph("Dataset Name", styles['TH']), Paragraph("Rows / Batches", styles['TH']), Paragraph("Features", styles['TH']), Paragraph("Base Rate Y=1", styles['TH']), Paragraph("Statistical Characteristics &amp; Domain Challenges", styles['TH'])],
        [
            Paragraph("<b>Give Me Some Credit</b>", styles['TDBold']),
            Paragraph("150,000 rows<br/>(30 batches &times; 4,000)", styles['TD']),
            Paragraph("10 continuous", styles['TD']),
            Paragraph("6.68% default", styles['TD']),
            Paragraph("Extreme heavy tails in <code>DebtRatio</code> (&gt;100,000) and delinquency counts. Heavy outlier variance destabilizes standard deviation and PCA.", styles['TD'])
        ],
        [
            Paragraph("<b>Taiwan Credit Default</b>", styles['TDBold']),
            Paragraph("30,000 rows<br/>(30 batches &times; 1,000)", styles['TD']),
            Paragraph("23 features", styles['TD']),
            Paragraph("22.12% default", styles['TD']),
            Paragraph("Temporal repayment records (PAY_0 to PAY_6), dense correlation structure, mild class imbalance, bounded financial variables.", styles['TD'])
        ],
        [
            Paragraph("<b>Adult Census Income</b>", styles['TDBold']),
            Paragraph("48,842 rows<br/>(30 batches &times; 1,600)", styles['TD']),
            Paragraph("6 continuous", styles['TD']),
            Paragraph("23.93% high-inc", styles['TD']),
            Paragraph("Heavy zero-inflation (<code>capital-gain</code>/<code>capital-loss</code> 75% zeros, IQR=0). <code>hours-per-week</code> has tight IQR=5 on a [1, 99] span.", styles['TD'])
        ]
    ]
    t_ds_full = Table(ds_full_data, colWidths=[usable_width*0.20, usable_width*0.18, usable_width*0.14, usable_width*0.16, usable_width*0.32])
    t_ds_full.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_ds_full)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Robust Normal-Equivalent Scaling via IQR / 1.349:</b><br/>"
        "In financial tabular data, human entry errors produce extreme outliers. In Give Me Some Credit, <code>DebtRatio</code> has a sample standard deviation inflated by over 100&times;. Shifting by '1.0 standard deviation' would inject an unrealistically massive perturbation. For a Gaussian distribution, $\\text{IQR} = 2 \\cdot \\Phi^{-1}(0.75) \\sigma \\approx 1.34898 \\sigma$. Therefore, $\\text{IQR} / 1.349$ is the robust normal-equivalent standard deviation. It ignores values beyond the 25th and 75th percentiles (breakdown point 25%), remaining stable against heavy-tailed anomalies.",
        styles['Body']
    ))
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # CHAPTER 5: 3-TIER PRODUCTION ARCHITECTURE
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

    story.append(Paragraph("Production Engineering Remediation Runbook:", styles['H3']))
    act_deep_data = [
        [Paragraph("Diagnosed Shift", styles['TH']), Paragraph("Root Cause Meaning", styles['TH']), Paragraph("Mandatory Production Remediation Runbook", styles['TH'])],
        [
            Paragraph("<b>Covariate Shift</b>", styles['TDBold']),
            Paragraph("Input distribution P(X) moved into novel support; true relationship P(Y|X) intact.", styles['TD']),
            Paragraph("<b>DO NOT retrain immediately.</b> Retraining can cause overfitting on sparse novel support. (1) Check upstream data pipelines for schema bugs. (2) Apply importance weighting: re-weight training samples by w(x) = P_stream(x) / P_ref(x). (3) Clip extreme values.", styles['TD'])
        ],
        [
            Paragraph("<b>Label Shift</b>", styles['TDBold']),
            Paragraph("Target prior P(Y) moved; class-conditional distributions P(X|Y) intact.", styles['TD']),
            Paragraph("<b>DO NOT retrain feature weights.</b> Retraining feature weights is wasteful and slow. Apply Bayesian prior odds correction to model intercept: &beta;<sub>0</sub> &larr; &beta;<sub>0</sub> + ln(p<sub>new</sub> / (1 - p<sub>new</sub>)) - ln(p<sub>old</sub> / (1 - p<sub>old</sub>)). Recalibrate decision threshold.", styles['TD'])
        ],
        [
            Paragraph("<b>Concept Drift</b>", styles['TDBold']),
            Paragraph("Real-world relationship P(Y|X) decayed; model predictions fundamentally invalid.", styles['TD']),
            Paragraph("<b>TRIGGER IMMEDIATE MODEL RETRAINING.</b> (1) Ingest recent verified stream labels. (2) Apply sliding-window training or exponential sample decay. (3) Roll back to human-in-the-loop manual review for high-risk predictions.", styles['TD'])
        ]
    ]
    t_act_deep = Table(act_deep_data, colWidths=[usable_width*0.20, usable_width*0.35, usable_width*0.45])
    t_act_deep.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_act_deep)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # CHAPTER 6: EMPIRICAL BENCHMARK RESULTS (1,080 RUNS)
    # -----------------------------------------------------------------------
    story.append(Paragraph("6. Empirical Benchmark Results (1,080 Controlled Runs)", styles['H1']))
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
    # CHAPTER 7: THE FOUR CONTINUOUS TRADE-OFF CURVES
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
    # CHAPTER 8: FORENSIC AUDIT DISCOVERIES & BREAKTHROUGHS
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
    # CHAPTER 9: SOFTWARE ARCHITECTURE & REPRODUCIBILITY
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
    # CHAPTER 10: THE MASTER VIVA DEFENSE — 30 EXHAUSTIVE QUESTIONS
    # -----------------------------------------------------------------------
    story.append(PageBreak())  # Clean break for the Viva Defense Masterclass
    story.append(Paragraph("10. The Master Viva Defense: 30 High-Stakes Professor Questions &amp; Answers", styles['H1']))
    story.append(Paragraph(
        "This section prepares the team to defend every theoretical nuance, architectural design decision, empirical edge case, and statistical trade-off before an adversarial panel or course professor.",
        styles['Body']
    ))
    story.append(Spacer(1, 4))

    viva_qa_30 = [
        # GROUP A: Theoretical Statistics & Mathematical Derivations (Q1 to Q6)
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

        ("Q6: What is the formal decomposition of the Brier score, and how does it relate to distribution shift?",
         "According to Murphy's decomposition, Brier Score = Reliability (Calibration) - Resolution + Uncertainty. Reliability measures how close predicted probabilities are to true empirical conditional frequencies in each probability bucket: &Sigma; N_k (p_k - y_bar_k)^2. Resolution measures how much the conditional probabilities differ from the global base rate: &Sigma; N_k (y_bar_k - y_bar)^2. Uncertainty is the inherent entropy of the problem: y_bar(1 - y_bar). In Covariate Shift, Reliability degrades slightly if the model extrapolates poorly. In Label Shift, Uncertainty and Reliability shift simultaneously. In Concept Drift, Reliability explodes while Resolution drops."),

        # GROUP B: Detection Mechanics, Quantiles & Binning (Q7 to Q12)
        ("Q7: How does your binning strategy handle zero-inflated or discrete features?",
         "Continuous features often exhibit heavy zero-inflation (e.g., capital-gain in Adult Income is 0 for 75% of rows). Naive quantile binning via np.quantile produces identical bin edges (e.g., [0, 0, 0, 0]), causing np.histogram to collapse bins or divide by zero. Our implementation applies np.unique to quantile edges. If fewer than 2 unique edges remain (degenerate feature), it falls back to equal-width binning between min and max. Furthermore, Laplace smoothing with epsilon = 10^-4 is added to all bins before normalizing, guaranteeing non-zero counts."),

        ("Q8: Why did PCA centroid distance fail on Give Me Some Credit (AUC 0.54) while succeeding on Taiwan (>0.95)?",
         "Two structural reasons: First, subspace dilution. When covariate shift affects a single feature, the Euclidean displacement in latent space is scaled by the sum of squared eigenvector loadings for that feature (&Sigma; W_j,k^2 &le; 0.15), diluting the signal across unshifted dimensions. Second, extreme heavy tails in DebtRatio (std dev &approx; 150) dominated the covariance matrix even after Winsorization, making the baseline centroid noise floor (0.038-0.054) large relative to the injected shift (0.005 std dev). On Taiwan, features are normalized, correlated, and bounded, allowing PCA to preserve coherent variance."),

        ("Q9: How does Predicted-Proportion Divergence (`pred_prop_psi`) work without ground truth labels?",
         "The model itself acts as a feature extractor and dimensionality reducer. A frozen reference model predicts calibrated probabilities p_hat = f_ref(x) on streaming batches. These probabilities are binned into 10 uniform buckets in [0, 1] and compared against reference prediction frequencies using PSI. When label shift occurs, the shift in marginals alters the distribution of p_hat. Crucially, on Adult Income, pred_prop_psi detected moderate covariate shift on hours-per-week where feature PSI failed, because the model's large coefficient (&beta; = +0.4993) amplified the small input shift into output probability space."),

        ("Q10: What is the purpose of the Kolmogorov-Smirnov (KS) test in your test suite?",
         "The two-sample KS test provides a non-parametric mathematical baseline that evaluates whether streaming samples originate from the same continuous distribution as the reference set by measuring the maximum vertical distance between empirical CDFs. To prevent type-I error inflation across d multiple feature tests, we apply Bonferroni correction (&alpha;_adj = 0.05 / d). It validates that our custom PSI/KL implementations align with classical asymptotic hypothesis tests on clean continuous variables."),

        ("Q11: What is the exact mathematical definition of detection lag in your benchmark?",
         "Detection lag is defined as the number of elapsed streaming batches between the start of injected shift (onset_batch = 10) and the first batch where an alert is officially confirmed by the debounce filter: Lag = t_alert - onset_batch. Under a k = 2 debounce filter, the theoretical minimum lag is 1 batch (alert confirmed at batch 11). If an injected shift is never detected across the stream, the run is flagged as undetected (lag = None) and excluded from the conditional mean lag calculation."),

        ("Q12: Why are reference quantile bin edges fixed permanently rather than recomputed per batch?",
         "Quantile bin edges must reflect the reference distribution's partitions so that each bin represents exactly 10% of baseline probability mass. If bin edges were recomputed on incoming batches, both reference and streaming samples would be partitioned by streaming quantiles, turning the divergence into a moving-frame metric and destroying the reference calibration. Freezing reference edges also reduces per-batch computational complexity from O(N log N) to O(N)."),

        # GROUP C: Systems Architecture, Latency & MLOps Trade-offs (Q13 to Q18)
        ("Q13: How did you optimize PSI/KL calculation to achieve 1.3 ms per batch?",
         "In naive implementations, quantile bin edges and reference histograms are recomputed dynamically every batch, requiring O(N_ref log N_ref) sorting operations. In our optimized PSIKLMonitor, reference bin edges and reference bucket percentages Q_b are precomputed and cached once during the fit() phase. On incoming streaming batches, only a single 1D np.digitize() call and histogram count is performed, reducing per-batch latency from 23.0 ms to 1.32 ms (~17x speedup)."),

        ("Q14: What is the debounce confirmation filter, and why is k=2 optimal?",
         "Streaming batches in production have finite sample sizes (e.g., 1,000 to 4,000 rows), causing stochastic metric fluctuations. A confirmation rule requiring k consecutive batches above threshold acts as a temporal low-pass filter. In our empirical debounce frontier (Curve 2), k = 1 suffers a 34% false alarm rate with 0 lag. Setting k = 2 slashes false alarms to 14% with only 0.6 batches of lag. In production, this prevents on-call engineers from being woken up by transient batch noise while detecting true persistent drift within ~1-2 batches."),

        ("Q15: How does your system operate under realistic delayed label feedback?",
         "In commercial applications (e.g., credit lending), default labels take 90+ days to resolve. Our architecture decouples input monitoring from performance monitoring. Tier 1 monitors inputs and model outputs instantaneously (<15 ms) on 100% of unlabelled traffic. Tier 2 ingests an asynchronous trickle (20% of resolved labels) via an event-driven join buffer. The Brier score performance monitor tracks calibration drift exclusively on this verified trickle, alerting if empirical degradation exceeds &tau; = 0.025."),

        ("Q16: How does the diagnostic decision engine (`classify_shift`) differentiate between shift types?",
         "Tier 3 evaluates the combination of fired alerts: (1) If feature PSI/KL fires while Brier score is stable, it diagnoses Covariate Shift. (2) If both pred_prop_psi and Brier degradation fire, it diagnoses Label Shift (prior probability change). (3) If Brier degradation fires while input feature distributions are completely stationary, it diagnoses Pure Concept Drift. (4) If pred_prop_psi fires alone while input PSI is quiet, it flags a high-weight covariate or subtle label shift."),

        ("Q17: What is the total memory footprint and CPU throughput of Drift Monitor in production?",
         "In our systems benchmark (`benchmark.py`), the full 5-detector monitoring suite processes a 4,000-row batch in 51.0 ms on Give Me Some Credit, 20.1 ms on Taiwan, and 15.2 ms on Adult Income. Total serialized memory footprint for the entire monitoring pipeline and reference models is 1.26 MB. On a single commodity CPU, this delivers a throughput of ~20 to 65 streaming batches per second, providing massive headroom for hourly or daily production ingestion pipelines."),

        ("Q18: Why is a streaming batch architecture preferred over single-instance online monitoring?",
         "Single-instance tests (like sequential probability ratio tests or CUSUM) require heavy state tracking and suffer from high variance on tabular multi-feature data. In commercial enterprise settings, inference workloads are ingested via micro-batches (e.g., 15-minute Kafka windows, hourly micro-batches). Batch aggregation provides sufficient statistical power (Central Limit Theorem) to compute stable quantile histograms and Brier estimates without per-event locking."),

        # GROUP D: Empirical Anomalies, Dataset Peculiarities & Forensic Audits (Q19 to Q24)
        ("Q19: Explain the Adult Census Income Covariate Inversion in detail.",
         "In Adult Census Income, covariate shift injected into hours-per-week produced 0.00 sensitivity on PSI/KL at mild and moderate levels, but 1.00 sensitivity on pred_prop_psi. The feature hours-per-week has 60% of values sitting exactly at 40, resulting in a tiny IQR of 5.0 across a [1, 99] range. A moderate shift of +3.7 hours was too small to cross quantile boundaries (PSI = 0.0409 < 0.10). However, the model assigned hours-per-week a massive weight (&beta; = +0.4993). The input shift pushed borderline instances across the decision boundary, causing model output probabilities to diverge (pred_prop_psi = 0.0298 &ge; 0.015). The model acted as an amplifier for high-leverage input drift."),

        ("Q20: What was the '0.95 detection lag' anomaly in early benchmark runs, and how was it resolved?",
         "Under a k = 2 debounce filter, detection lag should mathematically be &ge; 1.0 batch. However, early benchmark summaries reported a mean lag of 0.95 batches. Forensic code inspection revealed that if batch 9 (the last pre-onset batch) experienced a stochastic noise spike above threshold, consecutive_counter was set to 1. When true drift began at batch 10, the counter hit 2 immediately, registering a lag of 10 - 10 = 0. In 1 out of 20 runs, this pre-onset noise primed the counter, yielding (19*1 + 1*0)/20 = 0.95. We resolved this by strictly resetting the debounce streak counter to 0 at the onset boundary."),

        ("Q21: What is the confounding issue between concept drift injection and marginal label rates?",
         "Standard synthetic concept drift inverts labels (1 - Y) in a feature band. In imbalanced datasets like Give Me Some Credit (6.7% defaults), 93% of instances in any band are negative (Y=0). Inverting labels turns almost all of them into positives, inflating the batch default rate from 6.7% to 14.2% (moderate) and 20.5% (severe). Thus, standard benchmarks evaluate a compound shift: concept drift + massive label inflation. When we implemented a 'pure swap' preserving P(Y)=6.7%, low default counts (&sim;30 in a batch) meant only 6 positive instances appeared in a 20% trickle, moving global Brier score by only 0.002 (&lt; 0.025 threshold). Global metrics cannot detect localized concept drift in rare classes; slice-based monitoring is required."),

        ("Q22: How did you ensure zero data leakage between reference and streaming evaluation pools?",
         "In early prototypes, reference data was built from the first 10 batches of a global DataFrame shuffled with seed 0, and streaming batches were created by reshuffling the same DataFrame with new seeds, causing a 33% sample overlap. We resolved this by implementing strict physical partitioning: the historical dataset is permanently split into a reference pool (batches 0-9) and a completely held-out streaming evaluation pool (batches 10-29). No sample evaluated in the stream was ever seen during reference fitting or PCA training."),

        ("Q23: Why did the ROC curve calculation require disjoint 50/50 stream splits?",
         "Early ROC curve scripts evaluated streaming batches against the same reference window from which baseline noise was calculated, producing optimistic bias and tie-sorting distortions. We refactored ROC generation to evaluate on completely held-out, disjoint 50/50 streaming partitions. Across 1,080 runs, this established true unbiased operational ROC curves: psi_agg (AUC 0.88), kl_agg (AUC 0.88), pred_prop_psi (AUC 0.83), and concept_signal (AUC 0.76)."),

        ("Q24: Why does Give Me Some Credit have extreme outliers in DebtRatio, and how are they handled?",
         "In Give Me Some Credit, DebtRatio is defined as monthly debt payments divided by monthly gross income. For individuals who report zero monthly income or have missing income fields, data collection systems frequently recorded the total raw debt amount directly into DebtRatio, resulting in values exceeding 100,000 (against a median of 0.3). We handled this by: (1) Median imputation on missing income. (2) Applying non-parametric Winsorization clipping at the 1st and 99th percentiles before PCA. (3) Using IQR / 1.349 scaling for shift injection."),

        # GROUP E: Production Hardening, Edge Cases & Retraining Policies (Q25 to Q30)
        ("Q25: When Drift Monitor raises an alert, what automated actions should an MLOps pipeline take?",
         "The pipeline must never blindly trigger model retraining. If Covariate Shift is diagnosed, retraining is inappropriate because P(Y|X) is intact; engineers should inspect data upstream (e.g., schema changes, tracking bugs) or apply density-ratio sample re-weighting. If Label Shift is diagnosed, retraining the full model is wasteful; engineers should apply Bayesian prior odds correction or update the logistic intercept &beta;_0. Only when Concept Drift is diagnosed should the pipeline trigger model retraining on fresh labeled data or fall back to human review."),

        ("Q26: How would you extend Drift Monitor to handle high-cardinality categorical features?",
         "For categorical features (e.g., postal codes, merchant IDs), quantile binning is meaningless. The system should be extended with Categorical PSI or Jensen-Shannon Divergence over discrete category frequency tables: PSI_cat = &Sigma; (P_c - Q_c) ln(P_c / Q_c). For high-cardinality variables with unseen categories, an 'infrequent / other' bucket must be reserved, and new category emergence should be tracked via set difference cardinality (|Categories_stream \\ Categories_ref|)."),

        ("Q27: How would you adapt Drift Monitor for cyclic or seasonal distribution shifts?",
         "Static reference windows fail in retail and travel where weekend vs. weekday or holiday traffic creates natural cyclic shifts. For seasonal environments, Drift Monitor should implement Adaptive Sliding Windows (e.g., ADWIN) or time-of-week reference conditioning, where streaming Monday batches are compared against historical Monday reference distributions rather than a global stationary baseline."),

        ("Q28: What is the difference between data drift, model drift, and concept drift?",
         "• Data Drift (Covariate Shift): Changes in input distribution P(X) while the real-world relationship P(Y|X) is stationary.<br/>• Model Drift: Degradation in model inference outputs or calibration (e.g., pred_prop_psi) due to either data drift or internal numerical discrepancies.<br/>• Concept Drift: Fundamental changes in the real-world relationship between inputs and targets P(Y|X). Inputs look identical, but customer behavior, fraud tactics, or economic laws have changed."),

        ("Q29: How do you prevent false alarms when batch sizes shrink due to traffic drops?",
         "When batch size N drops, the empirical variance of quantile counts increases as O(1/N). If traffic drops during night hours, fixed thresholds will trigger spurious alerts. A robust production monitor should scale alert thresholds dynamically by &tau;(N) = &tau;_0 &middot; &radic;(N_ref / N_stream) or enforce a minimum batch size constraint before evaluating statistical divergence."),

        ("Q30: What is the single most important technical contribution of this MLSE project?",
         "Drift Monitor moves beyond superficial drift dashboards by providing: (1) An airtight mathematical proof that PSI is symmetric Jeffreys Divergence. (2) Empirical discovery and verification of the Adult Income model amplification effect. (3) The demonstration of subspace dilution in PCA centroid distance. (4) An operational 3-tier triage architecture combining $0-cost unsupervised gates with a 20% labeled trickle and debounce confirmation, delivering zero false alarms across 1,080 test runs.")
    ]

    for q_idx, (q_text, a_text) in enumerate(viva_qa_30, 1):
        qa_flow = []
        qa_flow.append(Paragraph(f"<b>{q_text}</b>", styles['QStyle']))
        qa_flow.append(Paragraph(a_text, styles['AStyle']))
        if q_idx % 5 == 0 and q_idx != len(viva_qa_30):
            qa_flow.append(Spacer(1, 4))
            qa_flow.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceBefore=4, spaceAfter=6))
        story.append(KeepTogether(qa_flow))

    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # CHAPTER 11: PRESENTATION STRATEGY & DEFENSE CHECKLIST
    # -----------------------------------------------------------------------
    story.append(Paragraph("11. Presentation Strategy, Defense Checklist &amp; Operational Cheat Sheet", styles['H1']))
    story.append(Paragraph(
        "<b>11.1 The 15-Minute Defense Playbook:</b><br/>"
        "• <i>Minutes 0–3 (The Silent Failure Problem):</i> Present Slide 2. Emphasize that models don't crash; they serve confident bad predictions with HTTP 200 OK. State the Ground Truth Paradox.<br/>"
        "• <i>Minutes 3–6 (Theoretical Taxonomy &amp; 3-Tier Architecture):</i> Present Slides 3 &amp; 4. Explain P(X), P(Y), P(Y|X). Show the 3-Tier diagram and explain why no single detector works.<br/>"
        "• <i>Minutes 6–10 (Empirical Breakthroughs &amp; Trade-off Curves):</i> Present Slides 5 &amp; 6. Walk through the Master Recommendation Table and the 4 Trade-off Curves. Highlight the Adult Income Covariate Inversion proof.<br/>"
        "• <i>Minutes 10–13 (Live Dashboard Demo):</i> Open <code>http://localhost:8501</code>. Show clean 2-panel Plotly figure. Inject shift live, demonstrate lag and debounce confirmation.<br/>"
        "• <i>Minutes 13–15 (Conclusion &amp; Code Quality):</i> Present Slide 9. Highlight 23 passing tests, Docker reproducibility, and hand-labelled benchmarks.",
        styles['Body']
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>11.2 Ten Inviolable Operational Facts to Memorize:</b>", styles['BodyBold']))
    checklist_items = [
        "<b>1. Project Identity:</b> Drift Monitor | Branch: <code>Srishti</code> | Team: Sanjana Nathani, Nikita Sharma, Anish Karmakar, Srishti Lamba.",
        "<b>2. Shift Taxonomy:</b> Covariate ($P(X)$), Label ($P(Y)$), Concept Drift ($P(Y|X)$). Unsupervised monitors cannot detect pure concept drift.",
        "<b>3. The Five Detectors:</b> <code>psi_agg</code>, <code>kl_agg</code>, <code>centroid_dist</code>, <code>pred_prop_psi</code>, <code>concept_signal</code>.",
        "<b>4. Mathematical Proof:</b> PSI &equiv; Symmetric Jeffreys Divergence: $\\text{PSI} = D_{\\text{KL}}(P \\parallel Q) + D_{\\text{KL}}(Q \\parallel P) \\approx 2 D_{\\text{KL}}$.",
        "<b>5. Threshold Asymmetry:</b> Setting equal thresholds (0.10) for PSI and KL makes KL twice as strict.",
        "<b>6. Adult Income Anomaly:</b> `hours-per-week` ($\text{IQR}=5$) missed on PSI (0.0409); `pred_prop_psi` detected it via $\\beta = +0.4993$ model weight amplification.",
        "<b>7. PCA Subspace Dilution:</b> Localized single-feature shifts dilute under Euclidean norm ($\\|\\Delta z\\|_2 = |\\Delta \\mu| \\sqrt{\\sum W_j^2}$).",
        "<b>8. Debounce Confirmation:</b> $k=1$ has 34% FAR; $k=2$ slashes FAR to 14% at 0.6 lag; $k \\ge 4$ achieves 0% FAR.",
        "<b>9. Latency Benchmark:</b> PSI/KL runs in 1.3 ms; full combined pipeline runs in 51.0 ms (~20 batches/sec, 1.26 MB RAM).",
        "<b>10. Zero Data Leakage:</b> Physical disjoint partitioning between reference window (batches 0–9) and held-out evaluation stream."
    ]
    for c_item in checklist_items:
        story.append(Paragraph(c_item, styles['Bullet']))

    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>11.3 Mathematical Notation &amp; Symbol Glossary:</b>", styles['BodyBold']))
    glossary_data = [
        [Paragraph("Symbol", styles['TH']), Paragraph("Definition &amp; Role in Drift Monitor", styles['TH']), Paragraph("Operational Baseline / Range", styles['TH'])],
        [Paragraph("<b>P(X), P(Y)</b>", styles['TDBold']), Paragraph("Input feature marginal and target class prior probability distribution.", styles['TD']), Paragraph("Empirical frequencies across stream batches.", styles['TD'])],
        [Paragraph("<b>P(Y | X)</b>", styles['TDBold']), Paragraph("Posterior class relationship (concept). Invariant under covariate shift; decays under concept drift.", styles['TD']), Paragraph("Monitored via 20% labeled trickle.", styles['TD'])],
        [Paragraph("<b>PSI(P, Q)</b>", styles['TDBold']), Paragraph("Population Stability Index &equiv; Symmetric Jeffreys Divergence &approx; 2 &times; D_KL.", styles['TD']), Paragraph("&lt; 0.10: Stable | &ge; 0.10: Significant Drift", styles['TD'])],
        [Paragraph("<b>pred_prop_psi</b>", styles['TDBold']), Paragraph("PSI between reference and streaming model output prediction distributions f_ref(X).", styles['TD']), Paragraph("&tau; = 0.015 (Calibrated noise floor)", styles['TD'])],
        [Paragraph("<b>&Delta;Brier</b>", styles['TDBold']), Paragraph("Degradation in mean squared error on verified labels: Brier_stream - Brier_ref.", styles['TD']), Paragraph("&tau; = 0.025 (Degradation trigger)", styles['TD'])],
        [Paragraph("<b>k</b>", styles['TDBold']), Paragraph("Debounce window: number of consecutive batches required to confirm an engineering alert.", styles['TD']), Paragraph("k = 2 (Optimal operational sweet spot)", styles['TD'])],
        [Paragraph("<b>IQR / 1.349</b>", styles['TDBold']), Paragraph("Robust normal-equivalent standard deviation; immune to heavy-tailed financial outliers.", styles['TD']), Paragraph("Breakdown point = 25% (vs 0% for sample &sigma;)", styles['TD'])],
    ]
    t_gloss = Table(glossary_data, colWidths=[usable_width*0.18, usable_width*0.54, usable_width*0.28])
    t_gloss.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_gloss)
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>11.4 Operational CLI Commands Quick Reference:</b>", styles['BodyBold']))
    cli_snippet = (
        "# 1. Execute full automated verification suite (23 unit tests):\n"
        "python -m pytest tests/test_core.py -v\n\n"
        "# 2. Launch interactive production telemetry dashboard:\n"
        "streamlit run drift_monitor/dashboard_app.py --server.port 8501\n\n"
        "# 3. Regenerate all 1,080 benchmark runs and trade-off curves from scratch:\n"
        "python reproduce_all.py\n\n"
        "# 4. Build and execute within the deterministic Docker container:\n"
        "docker build -t drift-monitor .\n"
        "docker run -p 8501:8501 drift-monitor"
    )
    story.append(Paragraph(cli_snippet.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles['Code']))
    story.append(Spacer(1, 8))

    story.append(make_callout(
        "<b>READY FOR SUBMISSION &amp; VIVA DEFENSE:</b> Drift Monitor provides an airtight, mathematically sound, experimentally validated, and reproducible system for detecting silent distribution shift. Treat this document as your project bible during presentation preparation and viva defense.",
        title="MISSION ACCOMPLISHED — OPERATIONAL READINESS VERIFIED",
        border_color="#10b981",
        bg_color="#ecfdf5"
    ))

    # Build Document
    print(f"Building {pdf_filename}...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Success! Generated {pdf_filename} ({os.path.getsize(pdf_filename)} bytes)")


if __name__ == "__main__":
    build_exhaustive_bible()
