"""
cement_adjuster.py - Hiệu chỉnh predictions cho loại xi măng khác nhau
"""
from typing import Dict

# =========================================================================
# KHỐI IMPORT AN TOÀN
# =========================================================================
try:
    from .config import CEMENT_TYPES
except ImportError:
    from config import CEMENT_TYPES
# =========================================================================

class CementTypeAdjuster:
    """
    Hiệu chỉnh predictions khi dùng xi măng khác PC40 (base)
    Vì mô hình KHÔNG học loại xi măng, ta dùng correction factors
    dựa trên lý thuyết xi măng và literature data
    """

    def __init__(self):
        """Load cement factors từ config"""
        self.cement_factors = {}
        for cement_type, props in CEMENT_TYPES.items():
            self.cement_factors[cement_type] = {
                'f_early': props['early_strength_factor'],
                'f28': props['strength_factor'],
                's_curve': 1.0 / props['strength_factor'],  # PC50 phát triển nhanh hơn
                'water_demand': 1.0 + (props['fineness'] - 350) / 1000  # Mịn hơn → cần nước nhiều
            }

    def adjust_predictions(self, mix_design: Dict[str, float], predictions: Dict[str, float], cement_type: str = 'PC40') -> Dict[str, float]:
        if cement_type == 'PC40':
            return predictions.copy()

        factors = self.cement_factors.get(cement_type, self.cement_factors['PC40'])
        adjusted = predictions.copy()

        if 'f28' in adjusted:
            adjusted['f28'] = predictions['f28'] * factors['f28']

        if 's' in adjusted:
            adjusted['s'] = predictions['s'] * factors['s_curve']

        if 'slump' in adjusted:
            adjusted['slump'] = predictions['slump'] / factors['water_demand']

        return adjusted

    def adjust_mix_for_target(self, mix_design: Dict[str, float], target_fc: float, cement_type: str = 'PC50') -> Dict[str, float]:
        if cement_type == 'PC40':
            return mix_design.copy()

        factors = self.cement_factors.get(cement_type, self.cement_factors['PC40'])
        adjusted_mix = mix_design.copy()

        # Giảm lượng xi măng (vì PC50 mạnh hơn)
        cement_reduction = 1.0 / factors['f28']
        adjusted_mix['cement'] *= cement_reduction

        # Điều chỉnh nước (vì PC50 cần nước nhiều hơn)
        adjusted_mix['water'] *= factors['water_demand']

        return adjusted_mix

    def get_correction_summary(self, cement_type: str) -> Dict:
        return self.cement_factors.get(cement_type, {})

# ===== TEST NGAY TẠI CHỖ =====
if __name__ == "__main__":
    try:
        adjuster = CementTypeAdjuster()

        # Test 1: Adjust predictions
        predictions_base = {'f28': 40.0, 's': 0.25, 'slump': 180}

        print("--- KIỂM TRA HIỆU CHỈNH XI MĂNG ---")
        print("Predictions với PC40 (base):")
        print(f"  f28 = {predictions_base['f28']:.1f} MPa")
        print(f"  s = {predictions_base['s']:.3f}")
        print(f"  Slump = {predictions_base['slump']:.0f} mm")

        predictions_pc50 = adjuster.adjust_predictions({}, predictions_base, 'PC50')

        print("\nPredictions với PC50 (adjusted):")
        print(f"  f28 = {predictions_pc50['f28']:.1f} MPa (+{(predictions_pc50['f28']/predictions_base['f28']-1)*100:.1f}%)")
        print(f"  s = {predictions_pc50['s']:.3f} ({(predictions_pc50['s']/predictions_base['s']-1)*100:+.1f}%)")
        print(f"  Slump = {predictions_pc50['slump']:.0f} mm ({(predictions_pc50['slump']/predictions_base['slump']-1)*100:+.1f}%)")

        # Test 2: Adjust mix design
        mix_pc40 = {'cement': 350, 'water': 160}
        mix_pc50 = adjuster.adjust_mix_for_target(mix_pc40, 40.0, 'PC50')

        print("\nĐể đạt cùng f28=40 MPa:")
        print(f"  PC40: cement={mix_pc40['cement']:.0f} kg, water={mix_pc40['water']:.0f} kg")
        print(f"  PC50: cement={mix_pc50['cement']:.0f} kg, water={mix_pc50['water']:.0f} kg")
        print(f"  Tiết kiệm: {mix_pc40['cement']-mix_pc50['cement']:.0f} kg xi măng/m³")
        print("✅ File 5 chạy thành công!")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
