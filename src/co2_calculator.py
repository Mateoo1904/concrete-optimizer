"""
co2_calculator.py - Tính toán phát thải CO2
"""
from typing import Dict

# =========================================================================
# KHỐI IMPORT AN TOÀN
# =========================================================================
try:
    from .material_database import MaterialDatabase
    from .config import A3_MIXING_EMISSION
except ImportError:
    from material_database import MaterialDatabase
    from config import A3_MIXING_EMISSION
# =========================================================================

class CO2Calculator:
    """
    Tính toán phát thải CO2 theo 3 giai đoạn: A1, A2, A3
    """

    def __init__(self, material_db: MaterialDatabase, a3_emission: float = A3_MIXING_EMISSION):
        self.db = material_db
        self.a3_emission = a3_emission

    def calculate_total_emission(self, mix_design: Dict[str, float], cement_type: str = 'PC40') -> Dict:
        emission_breakdown = {
            'A1': {},  # Sản xuất
            'A2': {},  # Vận chuyển
            'A3': self.a3_emission
        }

        # ===== A1: PHÁT THẢI SẢN XUẤT VẬT LIỆU =====
        # 1. Xi măng
        if 'cement' in mix_design and mix_design['cement'] > 0:
            cement_props = self.db.get_cement_props(cement_type)
            if cement_props:
                cement_mass = mix_design['cement']
                cement_a1 = cement_mass * cement_props['co2_production']
                emission_breakdown['A1']['cement'] = {
                    'mass': cement_mass, 'factor': cement_props['co2_production'],
                    'emission': cement_a1
                }

        # 2. Các vật liệu khác
        for mix_key, mass in mix_design.items():
            if mix_key == 'cement': continue
            if mass > 0:
                mat_props = self.db.get_material_by_mix_key(mix_key)
                if mat_props:
                    mat_a1 = mass * mat_props['co2_production']
                    emission_breakdown['A1'][mix_key] = {
                        'mass': mass, 'factor': mat_props['co2_production'],
                        'emission': mat_a1
                    }

        # ===== A2: PHÁT THẢI VẬN CHUYỂN =====
        # 1. Xi măng
        if 'cement' in mix_design and mix_design['cement'] > 0:
            cement_props = self.db.get_cement_props(cement_type)
            if cement_props:
                cement_mass = mix_design['cement']
                cement_a2 = cement_mass * cement_props['transport_distance'] * cement_props['co2_transport']
                emission_breakdown['A2']['cement'] = {
                    'mass': cement_mass, 'distance': cement_props['transport_distance'],
                    'factor': cement_props['co2_transport'], 'emission': cement_a2
                }

        # 2. Các vật liệu khác
        for mix_key, mass in mix_design.items():
            if mix_key == 'cement': continue
            if mass > 0:
                mat_props = self.db.get_material_by_mix_key(mix_key)
                if mat_props:
                    mat_a2 = mass * mat_props['transport_distance'] * mat_props['co2_transport']
                    emission_breakdown['A2'][mix_key] = {
                        'mass': mass, 'distance': mat_props['transport_distance'],
                        'factor': mat_props['co2_transport'], 'emission': mat_a2
                    }

        # ===== TỔNG HỢP =====
        total_a1 = sum(item['emission'] for item in emission_breakdown['A1'].values())
        total_a2 = sum(item['emission'] for item in emission_breakdown['A2'].values())
        total_emission = total_a1 + total_a2 + self.a3_emission

        return {
            'total': total_emission,
            'A1_total': total_a1, 'A2_total': total_a2, 'A3_total': self.a3_emission,
            'breakdown': emission_breakdown,
            'unit': 'kgCO2/m³', 'cement_type': cement_type
        }

    def get_emission_summary(self, emission_data: Dict) -> Dict[str, str]:
        total = emission_data['total']
        return {
            'Sản xuất vật liệu (A1)': f"{emission_data['A1_total']:.2f} ({emission_data['A1_total']/total*100:.1f}%)",
            'Vận chuyển (A2)': f"{emission_data['A2_total']:.2f} ({emission_data['A2_total']/total*100:.1f}%)",
            'Sản xuất bê tông (A3)': f"{emission_data['A3_total']:.2f} ({emission_data['A3_total']/total*100:.1f}%)",
            'Tổng cộng (A1+A2+A3)': f"{emission_data['total']:.2f} kgCO2/m³"
        }

    def compare_emissions(self, mix_design: Dict[str, float], cement_types: list = ['PC40', 'PC50']) -> Dict:
        results = {}
        for c_type in cement_types:
            results[c_type] = self.calculate_total_emission(mix_design, c_type)

        if len(cement_types) == 2:
            em1 = results[cement_types[0]]['total']
            em2 = results[cement_types[1]]['total']
            results['difference'] = abs(em2 - em1)
            results['greener'] = cement_types[0] if em1 < em2 else cement_types[1]
            results['difference_pct'] = (em2 - em1) / em1 * 100

        return results


# ===== TEST NGAY TẠI CHỖ =====
if __name__ == "__main__":
    # SỬA LỖI INDENTATION VÀ IMPORT Ở ĐÂY
    try:
        from material_database import MaterialDatabase

        db = MaterialDatabase()
        calc = CO2Calculator(db)

        test_mix = {
            'cement': 350, 'water': 160, 'flyash': 50, 'slag': 80,
            'silica_fume': 20, 'superplasticizer': 6.5,
            'fine_agg': 750, 'coarse_agg': 1050
        }

        # Test 1: Single cement type
        print("--- KIỂM TRA TÍNH TOÁN CO2 ---")
        emission_pc40 = calc.calculate_total_emission(test_mix, 'PC40')
        print(f"Tổng phát thải PC40: {emission_pc40['total']:.2f} kgCO2/m³")

        summary = calc.get_emission_summary(emission_pc40)
        for key, value in summary.items():
            print(f"  {key}: {value}")

        # Test 2: Compare
        comparison = calc.compare_emissions(test_mix)
        print(f"\nChênh lệch: {comparison['difference']:.2f} kgCO2/m³")
        print(f"Xanh hơn: {comparison['greener']}")
        print("✅ File 4 chạy thành công!")

    except ImportError:
        print("❌ Lỗi: Chưa tìm thấy file material_database.py. Hãy chạy File 2 trước!")
