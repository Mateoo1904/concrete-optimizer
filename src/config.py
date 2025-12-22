"""
config.py - Cấu hình cho môi trường LOCAL (VSCode)
"""
from pathlib import Path
import os

# ===== PATHS (LOCAL VERSION) =====
# Lấy đường dẫn project root
BASE_DIR = Path(__file__).resolve().parent.parent  # Concrete_Project/
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Tạo folder outputs nếu chưa có
OUTPUTS_DIR.mkdir(exist_ok=True)

# ===== MODEL FILES =====
F28_MODEL = MODELS_DIR / "f28_blend_retrained_bundle.pkl"
S_MODEL = MODELS_DIR / "s_model_catboost_mono_s_bundle.pkl"
SLUMP_MODEL = MODELS_DIR / "feature_builder_final.pkl"

# ===== MATERIAL DATABASE =====
MATERIALS_DB = {
    'Xi măng (Cement)': {
        'category': 'Binder',
        'density': 3150,
        'price': 1000,
        'co2_production': 0.931,
        'transport_distance': 60,
        'co2_transport': 5.18e-5,
        'required': True
    },
    'Xỉ (Slag)': {
        'category': 'SCM',
        'density': 2900,
        'price': 550,
        'max_replacement': 40,
        'co2_production': 0.0265,
        'transport_distance': 70,
        'co2_transport': 5.18e-5,
        'required': False
    },
    'Tro bay (Flyash)': {
        'category': 'SCM',
        'density': 2200,
        'price': 350,
        'max_replacement': 30,
        'co2_production': 0.0196,
        'transport_distance': 170,
        'co2_transport': 5.18e-5,
        'required': False
    },
    'Silica fume': {
        'category': 'SCM',
        'density': 2200,
        'price': 15000,
        'max_replacement': 10,
        'co2_production': 0.05,
        'transport_distance': 100,
        'co2_transport': 5.18e-5,
        'required': False
    },
    'Phụ gia siêu dẻo (SP)': {
        'category': 'Admixture',
        'density': 1050,
        'price': 25000,
        'max_dosage': 2.0,
        'co2_production': 0.25,
        'transport_distance': 16,
        'co2_transport': 2.21e-4,
        'required': False
    },
    'Đá (Coarse agg)': {
        'category': 'Aggregate',
        'density': 2700,
        'price': 275,
        'co2_production': 0.0075,
        'transport_distance': 20,
        'co2_transport': 6.3e-5,
        'required': True
    },
    'Cát (Fine agg)': {
        'category': 'Aggregate',
        'density': 2650,
        'price': 415,
        'co2_production': 0.0026,
        'transport_distance': 35,
        'co2_transport': 6.3e-5,
        'required': True
    },
    'Nước (Water)': {
        'category': 'Water',
        'density': 1000,
        'price': 20,
        'co2_production': 0.00196,
        'transport_distance': 0,
        'co2_transport': 0,
        'required': True
    }
}

# ===== XI MĂNG TYPES =====
CEMENT_TYPES = {
    'PC40': {
        'name': 'Xi măng Poóc-lăng PCB40',
        'C3S': 55,
        'C3A': 8,
        'fineness': 350,
        'strength_factor': 1.0,
        'early_strength_factor': 1.0,
        'price': 1000,
        'co2_production': 0.931,
        'transport_distance': 60,
        'co2_transport': 5.18e-5
    },
    'PC50': {
        'name': 'Xi măng Poóc-lăng PCB50',
        'C3S': 60,
        'C3A': 9,
        'fineness': 380,
        'strength_factor': 1.15,
        'early_strength_factor': 1.20,
        'price': 1100,
        'co2_production': 0.931,
        'transport_distance': 60,
        'co2_transport': 5.18e-5
    }
}

# ===== PHÁT THẢI A3 =====
A3_MIXING_EMISSION = 0.507  # kgCO2/m³

# ===== MAPPING KEYS =====
MIX_TO_DB_MAPPING = {
    'cement': 'Xi măng (Cement)',
    'water': 'Nước (Water)',
    'flyash': 'Tro bay (Flyash)',
    'slag': 'Xỉ (Slag)',
    'silica_fume': 'Silica fume',
    'superplasticizer': 'Phụ gia siêu dẻo (SP)',
    'fine_agg': 'Cát (Fine agg)',
    'coarse_agg': 'Đá (Coarse agg)'
}

# ===== KIỂM TRA MODELS =====
def check_models():
    """Kiểm tra xem models có tồn tại không"""
    print("✅ Config loaded!")
    print(f"   BASE_DIR: {BASE_DIR}")
    print(f"   MODELS_DIR: {MODELS_DIR}")
    
    for name, path in [("F28_MODEL", F28_MODEL), ("S_MODEL", S_MODEL), ("SLUMP_MODEL", SLUMP_MODEL)]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"   {name}: ✅ {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"   {name}: ❌ NOT FOUND")

if __name__ == "__main__":
    check_models()
