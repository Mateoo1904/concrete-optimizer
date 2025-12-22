"""
constraint_builder.py - Xây dựng constraints từ user inputs
✅ FIXED: Hỗ trợ custom material bounds từ UI
"""
import numpy as np
from typing import Dict, List, Tuple

try:
    from material_database import MaterialDatabase
except ImportError:
    from src.material_database import MaterialDatabase


class ConstraintBuilder:
    """
    Chuyển đổi user inputs thành constraints cho NSGA-II
    """

    def __init__(self, material_db: MaterialDatabase):
        self.db = material_db
        self.constraints = []
        self.bounds = {}

    def build_from_user_input(self, user_input: Dict) -> Dict:
        """
        Xây dựng constraints từ UI inputs

        Args:
            user_input: Dict từ UI với keys:
                - fc_target: float (MPa)
                - age_target: int (ngày)
                - slump_target: float (mm)
                - slump_tolerance: float (±mm)
                - cement_types: List[str]
                - available_materials: Dict
                - material_bounds: Dict (OPTIONAL - custom bounds)

        Returns:
            {
                'bounds': Dict[str, Tuple[float, float]],
                'constraints': List[Dict],
                'objectives': List[str],
                'user_input': Dict
            }
        """
        # ===== 1. BOUNDS (8 BIẾN) =====
        # ✅ THÊM: Kiểm tra xem user có custom bounds không
        if 'material_bounds' in user_input and user_input['material_bounds']:
            # Sử dụng custom bounds từ UI
            self.bounds = user_input['material_bounds'].copy()
            
            # Override với availability
            materials = user_input['available_materials']
            scm_mapping = {
                'flyash': 'Tro bay (Flyash)',
                'slag': 'Xỉ (Slag)',
                'silica_fume': 'Silica fume'
            }
            
            # Nếu material không available, force bounds = (0, 0)
            for key, mat_name in scm_mapping.items():
                if mat_name in materials and not materials[mat_name].get('available'):
                    self.bounds[key] = (0, 0)
            
            if 'Phụ gia siêu dẻo (SP)' in materials and not materials['Phụ gia siêu dẻo (SP)'].get('available'):
                self.bounds['superplasticizer'] = (0, 0)
        else:
            # Sử dụng default bounds
            self.bounds = self._build_bounds(user_input['available_materials'])

        # ===== 2. CONSTRAINTS =====
        self.constraints = []

        # 2a. Slump constraint
        self.constraints.append({
            'type': 'slump_tolerance',
            'target': user_input['slump_target'],
            'tolerance': user_input['slump_tolerance'],
            'description': f"Slump {user_input['slump_target']}±{user_input['slump_tolerance']} mm"
        })

        # 2b. Strength constraint
        self.constraints.append({
            'type': 'strength_min',
            'age': user_input['age_target'],
            'min_value': user_input['fc_target'],
            'description': f"f{user_input['age_target']} >= {user_input['fc_target']} MPa"
        })

        # 2c. w/b constraints
        self.constraints.append({
            'type': 'w_b_range',
            'min': 0.25,
            'max': 0.65,
            'description': "w/b trong khoảng 0.25-0.65"
        })

        # 2d. SCM limits
        materials = user_input['available_materials']
        max_scm_total = 0.50
        scm_limits = {}

        for mat_name, mat_props in materials.items():
            if mat_props.get('category') == 'SCM' and mat_props.get('available'):
                max_pct = mat_props.get('max_replacement', 30) / 100
                scm_limits[mat_name] = max_pct

        self.constraints.append({
            'type': 'scm_limits',
            'individual': scm_limits,
            'total': max_scm_total,
            'description': f"Total SCM <= {max_scm_total*100}%"
        })

        # 2e. Volume constraint
        self.constraints.append({
            'type': 'volume',
            'target': 1.0,
            'tolerance': 0.02,
            'description': "Thể tích 0.98-1.02 m³"
        })

        # 2f. Binder limits - ✅ THÊM: Động dựa trên bounds
        cement_bounds = self.bounds.get('cement', (200, 600))
        binder_min = cement_bounds[0]  # Ít nhất bằng cement min
        binder_max = cement_bounds[1] * 1.5  # Tối đa 1.5x cement max (khi có SCM)
        
        self.constraints.append({
            'type': 'binder_range',
            'min': max(300, binder_min),  # Tối thiểu 300 kg/m³
            'max': min(700, binder_max),  # Tối đa 700 kg/m³
            'description': f"Tổng binder {max(300, binder_min):.0f}-{min(700, binder_max):.0f} kg/m³"
        })

        objectives = ['cost', 'strength', 'slump_deviation', 'co2']

        return {
            'bounds': self.bounds,
            'constraints': self.constraints,
            'objectives': objectives,
            'user_input': user_input
        }

    def _build_bounds(self, materials: Dict) -> Dict[str, Tuple[float, float]]:
        """
        Xây dựng DEFAULT bounds cho 8 decision variables
        """
        bounds = {
            'cement': (200, 600),
            'water': (100, 250),
            'fine_agg': (600, 900),
            'coarse_agg': (800, 1200)
        }

        # SCM bounds dựa vào availability
        scm_mapping = {
            'flyash': 'Tro bay (Flyash)',
            'slag': 'Xỉ (Slag)',
            'silica_fume': 'Silica fume'
        }

        for key, mat_name in scm_mapping.items():
            if mat_name in materials and materials[mat_name].get('available'):
                if key == 'flyash':
                    bounds[key] = (0, 150)
                elif key == 'slag':
                    bounds[key] = (0, 200)
                elif key == 'silica_fume':
                    bounds[key] = (0, 40)
            else:
                bounds[key] = (0, 0)  # Force = 0 nếu không available

        # Superplasticizer
        if 'Phụ gia siêu dẻo (SP)' in materials and materials['Phụ gia siêu dẻo (SP)'].get('available'):
            bounds['superplasticizer'] = (0, 15)
        else:
            bounds['superplasticizer'] = (0, 0)

        return bounds

    def validate_mix_design(self, mix_design: Dict) -> Tuple[bool, List[str]]:
        """
        Kiểm tra mix design có thỏa mãn constraints không
        """
        violations = []

        # Check bounds
        for key, (min_val, max_val) in self.bounds.items():
            value = mix_design.get(key, 0)
            if not (min_val <= value <= max_val):
                violations.append(f"{key} = {value:.1f} ngoài bounds [{min_val}, {max_val}]")

        # Check volume
        volume = self.db.compute_volume(mix_design)
        if abs(volume - 1.0) > 0.02:
            violations.append(f"Volume = {volume:.3f} m³ (nên ~1.0 m³)")

        # Check w/b
        binder = sum(mix_design.get(k, 0) for k in ['cement', 'flyash', 'slag', 'silica_fume'])
        if binder > 0:
            w_b = mix_design.get('water', 0) / binder
            if not (0.25 <= w_b <= 0.65):
                violations.append(f"w/b = {w_b:.3f} ngoài [0.25, 0.65]")

        return (len(violations) == 0, violations)

    def get_constraint_summary(self) -> str:
        """
        Tạo summary text cho constraints
        """
        summary = "📋 CONSTRAINTS SUMMARY\n"
        summary += "=" * 60 + "\n"

        for i, constraint in enumerate(self.constraints, 1):
            summary += f"{i}. {constraint['description']}\n"

        summary += "\n📊 BOUNDS (Decision Variables):\n"
        for key, (min_val, max_val) in self.bounds.items():
            if max_val > 0:
                summary += f"   {key:20s}: [{min_val:6.0f}, {max_val:6.0f}]\n"

        return summary