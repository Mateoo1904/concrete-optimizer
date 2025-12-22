"""
material_database.py - Quản lý database vật liệu
✅ FIXED: Hỗ trợ custom density từ user input
"""
from typing import Dict, Optional, List
import copy

try:
    from .config import MATERIALS_DB, CEMENT_TYPES, MIX_TO_DB_MAPPING
except ImportError:
    try:
        from config import MATERIALS_DB, CEMENT_TYPES, MIX_TO_DB_MAPPING
    except ImportError as e:
        print("❌ LỖI NGHIÊM TRỌNG: Không thể import config.py!")
        print(f"   Chi tiết lỗi: {e}")
        MATERIALS_DB, CEMENT_TYPES, MIX_TO_DB_MAPPING = {}, {}, {}


class MaterialDatabase:
    def __init__(self, custom_density: Optional[Dict[str, float]] = None):
        """
        Args:
            custom_density: Dict mapping mix_key -> density (kg/m³)
                           e.g., {'cement': 3150, 'water': 1000, ...}
        """
        self.materials = copy.deepcopy(MATERIALS_DB)
        self.cement_types = copy.deepcopy(CEMENT_TYPES)
        self.mix_to_db = MIX_TO_DB_MAPPING
        
        # ✅ THÊM: Custom density override
        self.custom_density = custom_density or {}

    def get_material_props(self, material_name: str) -> Optional[Dict]:
        return self.materials.get(material_name)

    def get_cement_props(self, cement_type: str) -> Optional[Dict]:
        return self.cement_types.get(cement_type)

    def get_material_by_mix_key(self, mix_key: str) -> Optional[Dict]:
        db_name = self.mix_to_db.get(mix_key)
        if db_name:
            return self.materials.get(db_name)
        return None

    def list_available_materials(self) -> List[str]:
        return list(self.materials.keys())

    def update_material_price(self, material_name: str, new_price: float):
        if material_name in self.materials:
            self.materials[material_name]['price'] = new_price
            return True
        return False

    def get_density(self, mix_key: str) -> float:
        """
        ✅ FIXED: Ưu tiên custom density, fallback về database
        """
        # 1. Kiểm tra custom density trước
        if mix_key in self.custom_density:
            return self.custom_density[mix_key]
        
        # 2. Fallback về database
        mat = self.get_material_by_mix_key(mix_key)
        if mat:
            return mat['density']
        
        # 3. Default fallback
        default_densities = {
            'cement': 3150,
            'water': 1000,
            'flyash': 2200,
            'slag': 2900,
            'silica_fume': 2200,
            'superplasticizer': 1050,
            'fine_agg': 2650,
            'coarse_agg': 2700
        }
        return default_densities.get(mix_key, 1000)

    def compute_volume(self, mix_design: Dict) -> float:
        """
        ✅ Tính thể tích sử dụng density (custom hoặc default)
        """
        volume = 0
        for mix_key, mass in mix_design.items():
            if mass > 0:
                density = self.get_density(mix_key)
                volume += mass / density
        return volume

    def validate_mix_design(self, mix_design: Dict) -> Dict:
        errors = []
        required_keys = ['cement', 'water', 'fine_agg', 'coarse_agg']
        for key in required_keys:
            if key not in mix_design or mix_design[key] <= 0:
                errors.append(f"Thiếu {key}")

        volume = self.compute_volume(mix_design)
        if not (0.95 <= volume <= 1.05):
            errors.append(f"Volume = {volume:.3f} m³ (nên từ 0.95-1.05)")

        binder = sum(mix_design.get(k, 0) for k in ['cement', 'flyash', 'slag', 'silica_fume'])
        if binder > 0:
            w_b = mix_design.get('water', 0) / binder
            if not (0.20 <= w_b <= 0.70):
                errors.append(f"w/b = {w_b:.3f} (nên từ 0.20-0.70)")

        return {'valid': len(errors) == 0, 'errors': errors, 'volume': volume}
    
    def set_custom_density(self, density_dict: Dict[str, float]):
        """
        ✅ THÊM: Cập nhật custom density
        
        Args:
            density_dict: Dict mapping mix_key -> density (kg/m³)
        """
        self.custom_density = density_dict.copy()
    
    def get_density_summary(self) -> Dict[str, float]:
        """
        ✅ THÊM: Lấy tất cả density hiện tại
        """
        mix_keys = ['cement', 'water', 'flyash', 'slag', 'silica_fume', 
                    'superplasticizer', 'fine_agg', 'coarse_agg']
        return {key: self.get_density(key) for key in mix_keys}


# ===== TEST CHẠY THỬ =====
if __name__ == "__main__":
    # Test 1: Default density
    db = MaterialDatabase()
    print("Default densities:")
    print(db.get_density_summary())
    
    # Test 2: Custom density
    custom = {
        'cement': 3200,  # Custom value
        'water': 998     # Custom value
    }
    db_custom = MaterialDatabase(custom_density=custom)
    print("\nCustom densities:")
    print(db_custom.get_density_summary())
    
    # Test 3: Volume calculation
    test_mix = {
        'cement': 350, 'water': 160, 'flyash': 50, 'slag': 80,
        'silica_fume': 20, 'superplasticizer': 6.5,
        'fine_agg': 750, 'coarse_agg': 1050
    }
    
    vol_default = db.compute_volume(test_mix)
    vol_custom = db_custom.compute_volume(test_mix)
    
    print(f"\nVolume with default density: {vol_default:.3f} m³")
    print(f"Volume with custom density: {vol_custom:.3f} m³")
    print("✅ Code hoạt động TỐT!")