"""
test_integration_full.py - Full system integration test
"""
import sys
from pathlib import Path

project_path = '/content/drive/MyDrive/Concrete_Project'
if project_path not in sys.path:
    sys.path.append(project_path)

from src.multi_cement_workflow import MultiCementWorkflow
from src.pdf_report_generator import PDFReportGenerator
from src.advanced_visualizer import AdvancedVisualizer
from src.sensitivity_analyzer import SensitivityAnalyzer


def test_full_workflow():
    """Test complete workflow end-to-end"""
    print("\n" + "="*70)
    print("🧪 FULL SYSTEM INTEGRATION TEST")
    print("="*70)
    
    # User input
    user_input = {
        'fc_target': 40.0,
        'age_target': 28,
        'slump_target': 180,
        'slump_tolerance': 20,
        'available_materials': {
            'Xỉ (Slag)': {'available': True, 'category': 'SCM'},
            'Tro bay (Flyash)': {'available': True, 'category': 'SCM'},
            'Phụ gia siêu dẻo (SP)': {'available': True}
        },
        'preferences': {
            'cost': 0.4,
            'performance': 0.3,
            'sustainability': 0.2,
            'workability': 0.1
        }
    }
    
    # Initialize workflow
    workflow = MultiCementWorkflow(
        models_dir="models",
        output_dir="outputs"
    )
    
    # Run optimization
    print("\n⏳ Running optimization...")
    results = workflow.run_optimization(
        user_input=user_input,
        cement_types=['PC40', 'PC50'],
        optimization_config={'pop_size': 20, 'n_gen': 10}  # Small for testing
    )
    
    assert 'PC40' in results['optimization_results'], "❌ Missing PC40 results"
    assert 'PC50' in results['optimization_results'], "❌ Missing PC50 results"
    print("✅ Optimization complete")
    
    # Test PDF generation
    print("\n⏳ Generating PDF report...")
    pdf_gen = PDFReportGenerator()
    pdf_path = pdf_gen.generate_report(results, "outputs")
    assert Path(pdf_path).exists(), "❌ PDF not generated"
    print(f"✅ PDF generated: {pdf_path}")
    
    # Test sensitivity analysis
    print("\n⏳ Running sensitivity analysis...")
    analyzer = SensitivityAnalyzer()
    top_design = results['processed_results']['PC40']['ranked_designs'][0]
    base_mix = top_design['mix_design']
    
    df_sens = analyzer.one_at_a_time_analysis(base_mix, cement_type='PC40')
    assert len(df_sens) > 0, "❌ Sensitivity analysis failed"
    print(f"✅ Sensitivity analysis complete ({len(df_sens)} tests)")
    
    # Test export
    print("\n⏳ Testing production export...")
    export_path = workflow.export_for_production([('PC40', 0), ('PC50', 0)])
    assert Path(export_path).exists(), "❌ Export failed"
    print(f"✅ Export complete: {export_path}")
    
    print("\n" + "="*70)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("="*70)


if __name__ == "__main__":
    test_full_workflow()
