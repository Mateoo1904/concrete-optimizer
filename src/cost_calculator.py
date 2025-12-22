"""
cost_calculator.py - Tính toán chi phí
"""
from typing import Dict, Tuple

# =========================================================================
# KHỐI IMPORT AN TOÀN
# =========================================================================
try:
    from .material_database import MaterialDatabase
except ImportError:
    try:
        from material_database import MaterialDatabase
    except ImportError:
        # Fallback trống để tránh crash nếu chưa có file 2
        pass
# =========================================================================

class CostCalculator:
    """
    Tính toán chi phí tổng thể và phân tích chi tiết
    """

    def __init__(self, material_db):
        self.db = material_db

    def calculate_total_cost(
        self,
        mix_design: Dict[str, float],
        cement_type: str = 'PC40'
    ) -> Dict:
        """
        Tính tổng chi phí và breakdown
        """
        cost_breakdown = {}
        total_cost = 0.0

        # 1. Xi măng (xử lý riêng vì phụ thuộc loại PC40/PC50)
        if 'cement' in mix_design and mix_design['cement'] > 0:
            cement_props = self.db.get_cement_props(cement_type)
            # Kiểm tra an toàn nếu props None
            if cement_props:
                cement_mass = mix_design['cement']
                cement_cost = cement_mass * cement_props['price']

                cost_breakdown['cement'] = {
                    'mass': cement_mass,
                    'price_per_kg': cement_props['price'],
                    'total': cement_cost,
                    'percentage': 0
                }
                total_cost += cement_cost

        # 2. Các vật liệu khác
        for mix_key, mass in mix_design.items():
            if mix_key == 'cement':
                continue

            if mass > 0:
                mat_props = self.db.get_material_by_mix_key(mix_key)
                if mat_props:
                    mat_cost = mass * mat_props['price']

                    cost_breakdown[mix_key] = {
                        'mass': mass,
                        'price_per_kg': mat_props['price'],
                        'total': mat_cost,
                        'percentage': 0
                    }
                    total_cost += mat_cost

        # 3. Tính % cho từng thành phần
        if total_cost > 0:
            for key in cost_breakdown:
                cost_breakdown[key]['percentage'] = (
                    cost_breakdown[key]['total'] / total_cost * 100
                )

        return {
            'total_cost': total_cost,
            'breakdown': cost_breakdown,
            'unit': 'VNĐ/m³',
            'cement_type': cement_type
        }

    def compare_costs(
        self,
        mix_design: Dict[str, float],
        cement_types: list = ['PC40', 'PC50']
    ) -> Dict:
        """
        So sánh chi phí giữa các loại xi măng
        """
        results = {}

        for cement_type in cement_types:
            results[cement_type] = self.calculate_total_cost(mix_design, cement_type)

        # So sánh nhanh
        if len(cement_types) == 2:
            cost1 = results[cement_types[0]]['total_cost']
            cost2 = results[cement_types[1]]['total_cost']

            results['difference'] = abs(cost2 - cost1)
            results['cheaper'] = cement_types[0] if cost1 < cost2 else cement_types[1]
            results['difference_pct'] = (cost2 - cost1) / cost1 * 100 if cost1 > 0 else 0

        return results

    def get_cost_summary(self, cost_data: Dict) -> str:
        """Tạo summary text cho UI"""
        summary = f"TỔNG CHI PHÍ: {cost_data['total_cost']:,.0f} {cost_data['unit']}\n"
        summary += "="*50 + "\n"
        summary += "PHÂN TÍCH CHI TIẾT:\n"

        sorted_items = sorted(
            cost_data['breakdown'].items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )

        for comp_key, data in sorted_items:
            summary += (
                f"  {comp_key:20s}: "
                f"{data['total']:>10,.0f} VNĐ "
                f"({data['percentage']:>5.1f}%)\n"
            )

        return summary
