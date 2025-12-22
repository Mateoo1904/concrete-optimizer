"""
pdf_report_generator.py - Generate PDF reports (xuất báo cáo PDF) với ReportLab
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
from pathlib import Path
from typing import Dict


class PDFReportGenerator:
    """
    Tạo báo cáo PDF chuyên nghiệp cho kết quả MultiCementWorkflow
    """

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Khai báo style (kiểu chữ / bố cục) cho các section trong PDF"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))

        # Subsection
        self.styles.add(ParagraphStyle(
            name='Subsection',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#d62728'),
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))

        # Body text – chỉnh lại style sẵn có
        body = self.styles['BodyText']
        body.fontSize = 10
        body.leading = 14
        body.alignment = TA_LEFT

    # ======================================================================
    # PUBLIC API
    # ======================================================================
    def generate_report(
        self,
        workflow_results: Dict,
        output_dir: str
    ) -> str:
        """
        Generate báo cáo PDF tổng hợp

        Args:
            workflow_results: dict kết quả từ MultiCementWorkflow.run_optimization
            output_dir: thư mục để lưu file PDF

        Returns:
            Đường dẫn (str) tới file PDF
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pdf_path = output_path / f"optimization_report_{workflow_results['session_id']}.pdf"

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )

        story = []

        # Cover page
        story.extend(self._build_cover_page(workflow_results))
        story.append(PageBreak())

        # Executive summary
        story.extend(self._build_executive_summary(workflow_results))
        story.append(PageBreak())

        # Detailed per cement type
        for cement_type in workflow_results['optimization_results'].keys():
            story.extend(self._build_cement_section(workflow_results, cement_type))
            story.append(PageBreak())

        # Comparison (nếu có nhiều cement type)
        if len(workflow_results['optimization_results']) > 1:
            story.extend(self._build_comparison_section(workflow_results))
            story.append(PageBreak())

        # Recommendations
        story.extend(self._build_recommendations_section(workflow_results))

        # Build PDF
        doc.build(story)

        return str(pdf_path)

    # ======================================================================
    # COVER / SUMMARY
    # ======================================================================
    def _build_cover_page(self, results: Dict) -> list:
        story = []

        story.append(Spacer(1, 3*cm))
        story.append(Paragraph("🏗️ CONCRETE MIX DESIGN", self.styles['CustomTitle']))
        story.append(Paragraph("OPTIMIZATION REPORT", self.styles['CustomTitle']))
        story.append(Spacer(1, 2*cm))

        user_input = results['user_input']

        info_data = [
            ['Target Strength:',
             f"{user_input['fc_target']:.0f} MPa at {user_input['age_target']} days"],
            ['Target Slump:',
             f"{user_input['slump_target']:.0f} ± {user_input['slump_tolerance']:.0f} mm"],
            ['Cement Types:', ', '.join(results['optimization_results'].keys())],
            ['Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['Session ID:', results['session_id']]
        ]

        table = Table(info_data, colWidths=[5*cm, 10*cm])
        table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.grey),
        ]))

        story.append(table)
        return story

    def _build_executive_summary(self, results: Dict) -> list:
        story = []

        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.5*cm))

        lines = []
        for cement_type, proc_result in results['processed_results'].items():
            top_design = proc_result['ranked_designs'][0]
            lines.append(f"<b>{cement_type}</b>:")
            lines.append(f"• Best Cost: {top_design['objectives']['cost']:,.0f} VNĐ/m³")
            # strength lấy từ predictions f28 (cường độ 28 ngày)
            lines.append(f"• Predicted f28: {top_design['predictions']['f28']:.1f} MPa")
            lines.append(f"• CO₂ Emission: {top_design['objectives']['co2']:.0f} kgCO₂/m³")
            lines.append("")

        story.append(Paragraph("<br/>".join(lines), self.styles['BodyText']))

        return story

    # ======================================================================
    # PER-CEMENT SECTION
    # ======================================================================
    def _build_cement_section(self, results: Dict, cement_type: str) -> list:
        story = []
        story.append(Paragraph(f"Detailed Results: {cement_type}",
                               self.styles['SectionHeader']))
        story.append(Spacer(1, 0.5*cm))

        proc_result = results['processed_results'][cement_type]
        metrics = proc_result['metrics']

        story.append(Paragraph("Pareto Front Statistics", self.styles['Subsection']))

        stats_data = [
            ['Metric', 'Minimum', 'Maximum'],
            ['Cost (VNĐ/m³)',
             f"{metrics['cost_range'][0]:,.0f}",
             f"{metrics['cost_range'][1]:,.0f}"],
            ['Strength (MPa)',
             f"{metrics['strength_range'][0]:.1f}",
             f"{metrics['strength_range'][1]:.1f}"],
            ['CO₂ (kgCO₂/m³)',
             f"{metrics['co2_range'][0]:.0f}",
             f"{metrics['co2_range'][1]:.0f}"],
            ['Avg Slump Dev (mm)',
             '-',
             f"{metrics['avg_slump_deviation']:.1f}"],
        ]

        stats_table = Table(stats_data, colWidths=[6*cm, 4*cm, 4*cm])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.lightgrey]),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 1*cm))

        # Top 3 designs
        story.append(Paragraph("Top 3 Recommended Designs",
                               self.styles['Subsection']))
        story.append(Spacer(1, 0.3*cm))

        for idx, design in enumerate(proc_result['ranked_designs'][:3], 1):
            story.extend(self._build_design_card(design, idx))
            story.append(Spacer(1, 0.6*cm))

        return story

    def _build_design_card(self, design: Dict, rank: int) -> list:
        story = []

        header = f"<b>#{rank}. {design['profile']}</b> (Score: {design['score']:.3f})"
        story.append(Paragraph(header, self.styles['BodyText']))
        story.append(Spacer(1, 0.2*cm))

        mix = design['mix_design']
        binder = mix['cement'] + mix.get('flyash', 0) + mix.get('slag', 0) + mix.get('silica_fume', 0)
        w_b = mix['water'] / binder if binder > 0 else 0.0
        scm = mix.get('flyash', 0) + mix.get('slag', 0) + mix.get('silica_fume', 0)
        scm_frac = scm / binder if binder > 0 else 0.0

        mix_data = [
            ['Component', 'kg/m³', '', 'Property', 'Value'],
            ['Cement', f"{mix['cement']:.1f}", '', 'w/b', f"{w_b:.3f}"],
            ['Water', f"{mix['water']:.1f}", '', 'SCM %', f"{scm_frac*100:.1f}%"],
            ['Flyash', f"{mix.get('flyash', 0):.1f}", '', 'f28',
             f"{design['predictions']['f28']:.1f} MPa"],
            ['Slag', f"{mix.get('slag', 0):.1f}", '', 'Slump',
             f"{design['predictions']['slump']:.0f} mm"],
            ['Silica Fume', f"{mix.get('silica_fume', 0):.1f}", '', 'Cost',
             f"{design['objectives']['cost']:,.0f} VNĐ"],
            ['SP', f"{mix.get('superplasticizer', 0):.1f}", '', 'CO₂',
             f"{design['objectives']['co2']:.0f} kg"],
            ['Fine Agg', f"{mix['fine_agg']:.1f}", '', '', ''],
            ['Coarse Agg', f"{mix['coarse_agg']:.1f}", '', '', ''],
        ]

        table = Table(mix_data, colWidths=[3.5*cm, 2*cm, 0.5*cm, 3*cm, 3*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
            ('BACKGROUND', (3, 0), (4, 0), colors.lightgreen),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 8),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('ALIGN', (4, 0), (4, -1), 'RIGHT'),
            ('GRID', (0, 0), (1, -1), 0.5, colors.grey),
            ('GRID', (3, 0), (4, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(table)

        # Validation info
        story.append(Spacer(1, 0.1*cm))
        if design['validation']['is_valid']:
            story.append(Paragraph("✅ <b>All constraints satisfied</b>",
                                   self.styles['BodyText']))
        else:
            violations = design['validation'].get('violations', [])
            text = "⚠️ <b>Violations:</b> " + "; ".join(violations)
            story.append(Paragraph(text, self.styles['BodyText']))

        return story

    # ======================================================================
    # COMPARISON & RECOMMENDATIONS
    # ======================================================================
    def _build_comparison_section(self, results: Dict) -> list:
        story = []
        story.append(Paragraph("Cement Type Comparison",
                               self.styles['SectionHeader']))
        story.append(Spacer(1, 0.5*cm))

        comparison = results['comparison']
        if not comparison:
            story.append(Paragraph("No comparison available.",
                                   self.styles['BodyText']))
            return story

        for key, comp in comparison.items():
            ct1, ct2 = key.split('_vs_')
            story.append(Paragraph(f"<b>{ct1} vs {ct2}</b>",
                                   self.styles['Subsection']))

            comp_data = [
                ['Metric', 'Difference', '% Change'],
                ['Cost',
                 f"{comp['cost_difference']:,.0f} VNĐ",
                 f"{comp['cost_pct']:+.1f}%"],
                ['Strength',
                 f"{comp['strength_difference']:.1f} MPa",
                 f"{comp['strength_pct']:+.1f}%"],
                ['CO₂',
                 f"{comp['co2_difference']:.0f} kg",
                 f"{comp['co2_pct']:+.1f}%"],
            ]

            table = Table(comp_data, colWidths=[5*cm, 5*cm, 4*cm])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
                ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(table)
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(f"💡 <i>{comp['recommendation']}</i>",
                                   self.styles['BodyText']))
            story.append(Spacer(1, 0.5*cm))

        return story

    def _build_recommendations_section(self, results: Dict) -> list:
        story = []
        story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.5*cm))

        for rec in results['recommendations']:
            story.append(Paragraph(f"• {rec}", self.styles['BodyText']))
            story.append(Spacer(1, 0.2*cm))

        return story


if __name__ == "__main__":
    # Chạy test nhanh (smoke test – test nhanh xem import có lỗi không)
    print("✅ PDFReportGenerator ready for use")
    print("   Usage: generator.generate_report(workflow_results, output_dir)")
