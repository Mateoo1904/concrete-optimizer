"""
test_optimization_integration.py - Test end-to-end optimization pipeline
"""
import sys
from pathlib import Path

# Add project to path
project_path = '/content/drive/MyDrive/Concrete_Project'
if project_path not in sys.path:
    sys.path.append(project_path)

from src.material_database import MaterialDatabase
from src.predictor_unified import UnifiedPredictor
from src.nsga2_optimizer import MixDesignOptimizer
from src.result_processor import ResultProcessor
import numpy as np


def test_constraint_builder():
    """Test ConstraintBuilder"""
    print("\n" + "="*60)
    print("TEST 1: Constraint Builder")
    print("="*60)
    
    from src.constraint_builder import ConstraintBuilder
    
    db = MaterialDatabase()
    builder = ConstraintBuilder(db)
    
    user_input = {
        'fc_target': 40.0,
        'age_target': 28,
        'slump_target': 180,
        'slump_tolerance': 20,
        'cement_types': ['PC40'],
        'available_materials': {
            'Xỉ (Slag)': {'available': True, 'category': 'SCM'},
            'Tro bay (Flyash)': {'available': True, 'category': 'SCM'},
            'Phụ gia siêu dẻo (SP)': {'available': True}
        }
    }
    
    config = builder.build_from_user_input(user_input)
    
    # Validate config structure
    assert 'bounds' in config, "❌ Thiếu bounds"
    assert 'constraints' in config, "❌ Thiếu constraints"
    assert len(config['bounds']) == 8, f"❌ Bounds phải có 8 biến, nhận {len(config['bounds'])}"
    
    print(f"✅ Bounds: {len(config['bounds'])} biến")
    print(f"✅ Constraints: {len(config['constraints'])} điều kiện")
    print(builder.get_constraint_summary())


def test_optimization_problem():
    """Test OptimizationProblem setup"""
    print("\n" + "="*60)
    print("TEST 2: Optimization Problem")
    print("="*60)
    
    from src.constraint_builder import ConstraintBuilder
    from src.optimization_problem import ConcreteMixOptimizationProblem
    
    db = MaterialDatabase()
    predictor = UnifiedPredictor()
    builder = ConstraintBuilder(db)
    
    user_input = {
        'fc_target': 40.0,
        'age_target': 28,
        'slump_target': 180,
        'slump_tolerance': 20,
        'cement_types': ['PC40'],
        'available_materials': {
            'Xỉ (Slag)': {'available': True, 'category': 'SCM'},
            'Tro bay (Flyash)': {'available': True, 'category': 'SCM'},
            'Phụ gia siêu dẻo (SP)': {'available': True}
        }
    }
    
    config = builder.build_from_user_input(user_input)
    problem = ConcreteMixOptimizationProblem(predictor, config, 'PC40')
    
    # Validate problem structure
    assert problem.n_var == 8, f"❌ n_var phải = 8, nhận {problem.n_var}"
    assert problem.n_obj == 4, f"❌ n_obj phải = 4, nhận {problem.n_obj}"
    assert problem.n_constr > 0, "❌ Phải có ít nhất 1 constraint"
    
    print(f"✅ Problem setup: {problem.n_var} vars, {problem.n_obj} objs, {problem.n_constr} constraints")
    
    # Test evaluation với 1 solution
    X_test = np.array([[350, 160, 50, 80, 20, 6.5, 750, 1050]])
    out = {}
    problem._evaluate(X_test, out)
    
    assert 'F' in out, "❌ Thiếu objectives"
    assert 'G' in out, "❌ Thiếu constraints"
    assert out['F'].shape == (1, 4), f"❌ F shape sai: {out['F'].shape}"
    
    print(f"✅ Evaluation test:")
    print(f"   Cost: {out['F'][0, 0]:,.0f} VNĐ/m³")
    print(f"   f28: {-out['F'][0, 1]:.1f} MPa")
    print(f"   Slump dev: {out['F'][0, 2]:.1f} mm")
    print(f"   CO2: {out['F'][0, 3]:.0f} kgCO2/m³")


def test_mini_optimization():
    """Test optimization với quy mô nhỏ"""
    print("\n" + "="*60)
    print("TEST 3: Mini Optimization (pop=10, gen=5)")
    print("="*60)
    
    db = MaterialDatabase()
    predictor = UnifiedPredictor()
    
    optimizer = MixDesignOptimizer(
        predictor=predictor,
        material_db=db,
        pop_size=10,  # Rất nhỏ để test nhanh
        n_gen=5,
        seed=42
    )
    
    user_input = {
        'fc_target': 40.0,
        'age_target': 28,
        'slump_target': 180,
        'slump_tolerance': 20,
        'cement_types': ['PC40'],
        'available_materials': {
            'Xỉ (Slag)': {'available': True, 'category': 'SCM'},
            'Tro bay (Flyash)': {'available': True, 'category': 'SCM'},
            'Phụ gia siêu dẻo (SP)': {'available': True}
        }
    }
    
    print("⏳ Running mini optimization...")
    results = optimizer.optimize(user_input, cement_types=['PC40'], verbose=False)
    
    # Validate results
    assert 'PC40' in results, "❌ Thiếu kết quả PC40"
    assert 'top_designs' in results['PC40'], "❌ Thiếu top_designs"
    assert len(results['PC40']['top_designs']) > 0, "❌ Không tìm được design nào"
    
    print(f"✅ Found {len(results['PC40']['top_designs'])} designs")
    
    # Check design quality
    for i, design in enumerate(results['PC40']['top_designs'][:3], 1):
        pred = design['predictions']
        obj = design['objectives']
        mix = design['mix_design']
        
        print(f"\nDesign {i}: {design['profile']}")
        print(f"   f28: {pred['f28']:.1f} MPa (target: {user_input['fc_target']:.1f})")
        print(f"   Slump: {pred['slump']:.0f} mm (target: {user_input['slump_target']:.0f})")
        print(f"   Cost: {obj['cost']:,.0f} VNĐ/m³")
        print(f"   CO2: {obj['co2']:.0f} kgCO2/m³")
        
        # Basic validation
        binder = mix['cement'] + mix['flyash'] + mix['slag'] + mix['silica_fume']
        w_b = mix['water'] / binder if binder > 0 else 0
        print(f"   w/b: {w_b:.3f}")


def test_multi_cement_optimization():
    """Test optimization cho 2 loại xi măng"""
    print("\n" + "="*60)
    print("TEST 4: Multi-Cement Optimization")
    print("="*60)
    
    db = MaterialDatabase()
    predictor = UnifiedPredictor()
    
    optimizer = MixDesignOptimizer(
        predictor=predictor,
        material_db=db,
        pop_size=10,
        n_gen=5,
        seed=42
    )
    
    user_input = {
        'fc_target': 40.0,
        'age_target': 28,
        'slump_target': 180,
        'slump_tolerance': 20,
        'cement_types': ['PC40', 'PC50'],  # 2 loại
        'available_materials': {
            'Xỉ (Slag)': {'available': True, 'category': 'SCM'},
            'Tro bay (Flyash)': {'available': True, 'category': 'SCM'},
            'Phụ gia siêu dẻo (SP)': {'available': True}
        }
    }
    
    print("⏳ Running multi-cement optimization...")
    results = optimizer.optimize(user_input, cement_types=['PC40', 'PC50'], verbose=False)
    
    # Validate
    assert 'PC40' in results, "❌ Thiếu PC40"
    assert 'PC50' in results, "❌ Thiếu PC50"
    
    # Compare results
    pc40_cost = results['PC40']['top_designs'][0]['objectives']['cost']
    pc50_cost = results['PC50']['top_designs'][0]['objectives']['cost']
    
    pc40_f28 = results['PC40']['top_designs'][0]['predictions']['f28']
    pc50_f28 = results['PC50']['top_designs'][0]['predictions']['f28']
    
    print(f"\n✅ PC40 - Cost: {pc40_cost:,.0f} VNĐ/m³, f28: {pc40_f28:.1f} MPa")
    print(f"✅ PC50 - Cost: {pc50_cost:,.0f} VNĐ/m³, f28: {pc50_f28:.1f} MPa")
    
    diff_cost = abs(pc50_cost - pc40_cost)
    diff_f28 = abs(pc50_f28 - pc40_f28)
    
    print(f"\n📊 Comparison:")
    print(f"   Cost difference: {diff_cost:,.0f} VNĐ/m³")
    print(f"   Strength difference: {diff_f28:.1f} MPa")


def test_result_processor():
    """Test ResultProcessor"""
    print("\n" + "="*60)
    print("TEST 5: Result Processor")
    print("="*60)
    
    db = MaterialDatabase()
    predictor = UnifiedPredictor()
    
    optimizer = MixDesignOptimizer(
        predictor=predictor,
        material_db=db,
        pop_size=10,
        n_gen=5,
        seed=42
    )
    
    user_input = {
        'fc_target': 40.0,
        'age_target': 28,
        'slump_target': 180,
        'slump_tolerance': 20,
        'cement_types': ['PC40'],
        'available_materials': {
            'Xỉ (Slag)': {'available': True, 'category': 'SCM'},
            'Tro bay (Flyash)': {'available': True, 'category': 'SCM'},
            'Phụ gia siêu dẻo (SP)': {'available': True}
        }
    }
    
    results = optimizer.optimize(user_input, cement_types=['PC40'], verbose=False)
    
    # Process results
    processor = ResultProcessor()
    processed = processor.process_results(results)
    
    # Validate
    assert 'PC40' in processed, "❌ Thiếu PC40 trong processed results"
    assert 'ranked_designs' in processed['PC40'], "❌ Thiếu ranked_designs"
    
    # Generate report
    report = processor.generate_summary_report(processed)
    
    assert len(report) > 100, "❌ Report quá ngắn"
    assert 'OPTIMIZATION RESULTS' in report, "❌ Report thiếu header"
    
    print("✅ Report generated successfully")
    print(f"   Report length: {len(report)} chars")
    print("\n--- Sample Report ---")
    print(report[:500] + "...")


if __name__ == "__main__":
    print("\n🔬 RUNNING INTEGRATION TESTS")
    print("="*60)
    
    try:
        test_constraint_builder()
        test_optimization_problem()
        test_mini_optimization()
        test_multi_cement_optimization()
        test_result_processor()
        
        print("\n" + "="*60)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("="*60)
        print("\n🎉 WEEK 1 & 2 HOÀN TOÀN HOÀN THIỆN!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
