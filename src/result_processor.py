"""
result_processor.py - Xử lý và format kết quả optimization
"""
import numpy as np
from typing import Dict, List
from datetime import datetime


class ResultProcessor:
    """
    Xử lý kết quả optimization: ranking, visualization prep
    """

    def __init__(self):
        """Initialize processor"""
        self.results = None
        self.comparisons = {}

    def process_results(
        self,
        optimization_results: Dict,
        user_preferences: Dict = None
    ) -> Dict:
        """
        Xử lý và ranking designs

        Args:
            optimization_results: Output từ MixDesignOptimizer.optimize()
            user_preferences: Dict với weights cho objectives

        Returns:
            Processed results với rankings
        """
        self.results = optimization_results

        # Default preferences nếu không có
        if user_preferences is None:
            user_preferences = {
                'cost': 0.3,
                'performance': 0.3,
                'sustainability': 0.2,
                'workability': 0.2
            }

        processed = {}

        for cement_type, result in optimization_results.items():
            processed[cement_type] = {
                'ranked_designs': result['top_designs'],
                'metrics': result['metrics'],
                'pareto_front': result['pareto_front']
            }

        return processed

    def generate_summary_report(self, processed_results: Dict) -> str:
        """
        Tạo summary report dạng text - ✅ ĐÃ SỬA LỖI

        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("🗏️  CONCRETE MIX DESIGN OPTIMIZATION REPORT - WEEK 2")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        for cement_type, res in processed_results.items():
            report.append("=" * 80)
            report.append(f"📊 {cement_type} - OPTIMIZATION RESULTS")
            report.append("=" * 80)

            # Metrics
            metrics = res['metrics']
            report.append(f"\n📈 Pareto Front Statistics:")
            report.append(f"   • Total solutions: {metrics['n_solutions']}")
            report.append(f"   • Cost range: {metrics['cost_range'][0]:,.0f} - {metrics['cost_range'][1]:,.0f} VNĐ/m³")
            report.append(f"   • Strength range: {metrics['strength_range'][0]:.1f} - {metrics['strength_range'][1]:.1f} MPa")
            report.append(f"   • CO2 range: {metrics['co2_range'][0]:.0f} - {metrics['co2_range'][1]:.0f} kgCO2/m³")

            # Top designs
            report.append(f"\n🏆 TOP RECOMMENDED DESIGNS:")
            for i, design in enumerate(res['ranked_designs'][:3], 1):
                report.append(f"\n{i}. {design['profile']}")
                report.append(f"   {'─' * 76}")

                # Mix proportions
                mix = design['mix_design']
                report.append(f"   Mix Design:")
                report.append(f"      Cement:          {mix['cement']:6.1f} kg/m³")
                report.append(f"      Water:           {mix['water']:6.1f} kg/m³")
                if mix.get('flyash', 0) > 0:
                    report.append(f"      Flyash:          {mix['flyash']:6.1f} kg/m³")
                if mix.get('slag', 0) > 0:
                    report.append(f"      Slag:            {mix['slag']:6.1f} kg/m³")
                if mix.get('silica_fume', 0) > 0:
                    report.append(f"      Silica Fume:     {mix['silica_fume']:6.1f} kg/m³")
                if mix.get('superplasticizer', 0) > 0:
                    report.append(f"      SP:              {mix['superplasticizer']:6.1f} kg/m³")
                report.append(f"      Fine Agg:        {mix['fine_agg']:6.1f} kg/m³")
                report.append(f"      Coarse Agg:      {mix['coarse_agg']:6.1f} kg/m³")

                # Derived properties
                binder = mix['cement'] + mix.get('flyash', 0) + mix.get('slag', 0) + mix.get('silica_fume', 0)
                w_b = mix['water'] / binder if binder > 0 else 0
                scm_frac = (mix.get('flyash', 0) + mix.get('slag', 0) + mix.get('silica_fume', 0)) / binder if binder > 0 else 0

                report.append(f"      w/b ratio:       {w_b:.3f}")
                report.append(f"      SCM fraction:    {scm_frac*100:.1f}%")

                # Performance
                pred = design['predictions']
                obj = design['objectives']
                report.append(f"\n   Performance:")
                report.append(f"      f28:             {pred['f28']:6.1f} MPa")
                report.append(f"      Slump:           {pred['slump']:6.0f} mm (deviation: {obj['slump_deviation']:.1f} mm)")
                report.append(f"      s-parameter:     {pred['s']:6.3f}")

                # Cost & CO2
                report.append(f"\n   Economics & Sustainability:")
                report.append(f"      Total Cost:      {obj['cost']:10,.0f} VNĐ/m³")
                report.append(f"      Total CO2:       {obj['co2']:10.0f} kgCO2/m³")

                # Validation
                val = design['validation']
                if val['is_valid']:
                    report.append(f"      Status:          ✅ All constraints satisfied")
                else:
                    report.append(f"      Status:          ⚠️  Violations detected:")
                    for v in val['violations']:
                        report.append(f"         - {v}")

        report.append("\n" + "=" * 80)
        report.append("✅ END OF REPORT")
        report.append("=" * 80)

        return "\n".join(report)  # ✅ QUAN TRỌNG: Return string thay vì None


# ===== TEST NGAY TẠI CHỖ =====
if __name__ == "__main__":
    # Test nhỏ để đảm bảo không lỗi
    processor = ResultProcessor()
    test_data = {
        'PC40': {
            'ranked_designs': [{
                'profile': 'Test Design',
                'mix_design': {'cement': 350, 'water': 160, 'fine_agg': 750, 'coarse_agg': 1050},
                'predictions': {'f28': 40.0, 'slump': 180, 's': 0.25},
                'objectives': {'cost': 1000000, 'slump_deviation': 10, 'co2': 300},
                'validation': {'is_valid': True, 'violations': []}
            }],
            'metrics': {
                'n_solutions': 10,
                'cost_range': (800000, 1200000),
                'strength_range': (35.0, 45.0),
                'co2_range': (250, 350)
            }
        }
    }
    
    report = processor.generate_summary_report(test_data)
    print("✅ ResultProcessor test:")
    print(f"Report type: {type(report)}")
    print(f"Report length: {len(report)}")
    print("\n--- Sample ---")
    print(report[:200] + "...")
