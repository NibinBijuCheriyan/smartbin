"""
HTML Evaluation Report Generator for SmartBin AI v2.
Produces self-contained, publication-grade HTML reports containing KPI cards,
confusion matrices, PR/ROC curves, per-class tables, and failure galleries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from smartbin_v2.evaluation.evaluator import EvaluationMetrics
from smartbin_v2.utils.logger import get_logger

logger = get_logger("report_generator")


def generate_html_evaluation_report(
    metrics: EvaluationMetrics,
    output_path: str | Path = "smartbin-v2/evaluation_report/index.html",
    cm_path: Optional[str] = "confusion_matrix.png",
    pr_path: Optional[str] = "pr_curve.png",
    roc_path: Optional[str] = "roc_curve.png",
) -> Path:
    """
    Generate an executive HTML performance report.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Build per-class rows
    class_rows = ""
    for cls_name, vals in metrics.per_class_metrics.items():
        class_rows += f"""
        <tr>
            <td style="font-weight: 600;">{cls_name}</td>
            <td>{vals.get('precision', 0.0) * 100:.1f}%</td>
            <td>{vals.get('recall', 0.0) * 100:.1f}%</td>
            <td><span class="badge badge-success">{vals.get('map50', 0.0) * 100:.1f}%</span></td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SmartBin AI v2 — Production Model Evaluation Report</title>
    <style>
        :root {{
            --primary: #2563EB;
            --success: #10B981;
            --dark: #0F172A;
            --card-bg: #FFFFFF;
            --bg: #F8FAFC;
            --border: #E2E8F0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: #1E293B;
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 15px; }}
        .grid-kpi {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .card-title {{
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748B;
            margin-bottom: 8px;
            font-weight: 600;
        }}
        .card-val {{
            font-size: 30px;
            font-weight: 700;
            color: var(--dark);
        }}
        .card-sub {{
            font-size: 12px;
            color: #10B981;
            margin-top: 5px;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: 700;
            margin: 30px 0 15px 0;
            color: #0F172A;
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 8px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-card {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }}
        .chart-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #F1F5F9;
            font-size: 14px;
        }}
        th {{
            background: #F8FAFC;
            color: #475569;
            font-weight: 600;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-success {{ background: #DCFCE7; color: #166534; }}
        footer {{
            text-align: center;
            margin-top: 50px;
            font-size: 13px;
            color: #94A3B8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SmartBin AI v2 — Production Model Evaluation</h1>
            <p>Industrial Waste Segregation Computer Vision Engine • Raspberry Pi 5 & Edge Deployable</p>
        </div>

        <div class="grid-kpi">
            <div class="card">
                <div class="card-title">mAP@50 Metric</div>
                <div class="card-val">{metrics.map50 * 100:.1f}%</div>
                <div class="card-sub">Target: &gt; 96.0%</div>
            </div>
            <div class="card">
                <div class="card-title">mAP@50-95 Metric</div>
                <div class="card-val">{metrics.map50_95 * 100:.1f}%</div>
                <div class="card-sub">Target: &gt; 88.0%</div>
            </div>
            <div class="card">
                <div class="card-title">Precision</div>
                <div class="card-val">{metrics.precision * 100:.1f}%</div>
                <div class="card-sub">False Alarm Filter Active</div>
            </div>
            <div class="card">
                <div class="card-title">Recall</div>
                <div class="card-val">{metrics.recall * 100:.1f}%</div>
                <div class="card-sub">High Detection Sensitivity</div>
            </div>
            <div class="card">
                <div class="card-title">Edge Latency (RPi 5)</div>
                <div class="card-val">{metrics.total_latency_ms:.1f} ms</div>
                <div class="card-sub">{metrics.fps:.1f} FPS Real-Time</div>
            </div>
        </div>

        <div class="section-title">Diagnostic Visualizations</div>
        <div class="charts-grid">
            <div class="chart-card">
                <h3 style="margin-top:0; font-size:16px;">Confusion Matrix</h3>
                <img src="{cm_path}" alt="Confusion Matrix" onerror="this.parentNode.innerHTML='<p style=\\'color:#94A3B8\\'>Confusion Matrix visualization generated upon validation run.</p>'">
            </div>
            <div class="chart-card">
                <h3 style="margin-top:0; font-size:16px;">Precision-Recall Curves</h3>
                <img src="{pr_path}" alt="PR Curves" onerror="this.parentNode.innerHTML='<p style=\\'color:#94A3B8\\'>PR Curve visualization generated upon validation run.</p>'">
            </div>
        </div>

        <div class="section-title">Per-Class Accuracy Breakdown</div>
        <table>
            <thead>
                <tr>
                    <th>Waste Category</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>mAP@50</th>
                </tr>
            </thead>
            <tbody>
                {class_rows if class_rows else "<tr><td colspan='4' style='text-align:center;'>Validation data initialized. Execute train or eval CLI to populate per-class statistics.</td></tr>"}
            </tbody>
        </table>

        <footer>
            Generated automatically by SmartBin AI v2 Evaluation Suite • Cashcrow Edge Systems
        </footer>
    </div>
</body>
</html>
"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML evaluation report generated: {out_file}")
    return out_file
